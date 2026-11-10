"""Matched engineering session controller; study collection remains gated.

The existing campaign module supplies process supervision and token accounting.
Its historical treatment and scoring entry points are deliberately not invoked.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import campaign

from aisle.harness.hidden_access_log import (
    AuthorityRoots,
    SandboxReportCollector,
    marker_command,
    process_names,
)
from aisle.harness.matched_evidence import retain_run
from aisle.harness.matched_frontend import FrontendToolBudget
from aisle.harness.matched_session import AdmissionError, admit_pair, execute_session, verify_plan
from aisle.harness.matched_tool_service import ToolService
from aisle.harness.matched_tools import ToolController
from aisle.harness.relay_reservation import ReservationError, host_addresses, reserve_relay_port
from aisle.harness.treatment_confinement import (
    SANDBOX_EXEC,
    MacOSPolicy,
    compile_macos_profile,
    wrap_verified_command,
)

COLLECTOR = "macos-sandbox-reports"
COLLECTION_DIR = "access-collection"


@contextlib.contextmanager
def session_relay_reservation(policy, destination: Path, *, addresses=host_addresses):
    """Hold a loopback session's pinned relay port on every host address for
    the session's lifetime, retaining the address-free record (ADR-69). A
    port that cannot be held exclusively refuses the launch."""
    if policy.network_policy != "loopback":
        yield None
        return
    try:
        reservation = reserve_relay_port(port=policy.loopback_port, addresses=addresses)
    except ReservationError as exc:
        raise AdmissionError(f"cannot hold the relay port exclusively: {exc}") from exc
    with reservation:
        with (destination / "relay-reservation.json").open("x") as stream:
            json.dump(reservation.record(), stream, indent=2, allow_nan=False)
            stream.write("\n")
        yield reservation


def _worker_policies(preparations) -> list[dict]:
    """Every worker policy inside the controller's per-run preparations: a
    typed run is a list of stage maps (worker name -> launch), a monolithic run
    is one launch. A dynamic preparation (a provider template, workers
    provisioned at run time) carries no policy to select, so it refuses."""
    found: list[dict] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            if "policy" in value:
                found.append(value["policy"])
            elif "provider" in value:
                raise AdmissionError(
                    "sandbox-report collection cannot select dynamically provisioned workers"
                )
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(preparations or [])
    return found


def _access_log_collector(
    plan: dict, arm: str, profile_path: Path, destination: Path, worker_preparations
):
    """The sandbox-report collector for one launch: selects every process name
    the session's policies admit (the arm, the validator, every prepared
    worker), classifies targets against the arm's own authority roots and runs
    its positive-control markers with the arm's own admitted executable under
    the arm's profile. `adapter_active` is settled by the launcher once the
    wrapped command is verified."""
    declared = plan["confinement_bindings"][arm]["policy"]
    policies = [declared]
    if "typed_validation" in plan:
        policies.append(plan["typed_validation"]["policy"])
    policies.extend(_worker_policies(worker_preparations))
    policy = MacOSPolicy.from_canonical(declared)
    marker = marker_command(declared["allowed_executables"])
    if marker is None:
        raise AdmissionError(
            "sandbox-report collection needs an admitted interpreter, shell or cat for its markers"
        )
    return SandboxReportCollector(
        names=process_names(policies),
        roots=AuthorityRoots(readable=policy.readable_roots, hidden=policy.hidden_roots),
        output=destination / COLLECTION_DIR,
        adapter_active=False,
        profile_path=Path(profile_path),
        marker=marker,
    )


def run_engineering_session(
    plan: dict,
    root: Path,
    visible_roots: dict,
    arm: str,
    output: Path,
    *,
    session_id: str,
    profile_path: Path,
    attestation: dict,
    hidden_access_log: Path | dict,
    purpose: str = "engineering",
    worker_preparations: list | None = None,
) -> dict:
    """Launch only bound process inputs through the retained capability adapter.

    This records engineering infrastructure evidence, never baseline outcomes.
    Adapter capability does not replace the independent study prerequisites.
    `hidden_access_log` is either a controller-supplied log path or
    `{"collect": "macos-sandbox-reports"}`, in which case the sandbox's own
    denial reports are streamed for the launch window and retained as the log.
    """

    authority_references = {}
    collect = isinstance(hidden_access_log, dict)
    if collect:
        if hidden_access_log != {"collect": COLLECTOR}:
            raise AdmissionError("unsupported hidden access log collector")
        hidden_access_log = Path(output) / COLLECTION_DIR / "hidden-access-log.json"

    def launch(destination: Path) -> dict:
        collector = (
            _access_log_collector(plan, arm, profile_path, destination, worker_preparations)
            if collect
            else None
        )
        with collector if collector is not None else contextlib.nullcontext():
            return _launch(destination, collector)

    def _launch(destination: Path, collector: SandboxReportCollector | None) -> dict:
        current = verify_plan(plan, root, visible_roots)
        manifest = current["arms"][arm]
        try:
            declared = current["confinement_bindings"][arm]["policy"]
            ambient = current["ambient_bindings"][arm]
            argv = current["launch_bindings"][arm]["argv"]
        except KeyError as exc:
            raise AdmissionError(
                "process requires confinement, ambient and launch bindings"
            ) from exc
        policy = MacOSPolicy.from_canonical(declared)
        compiled = compile_macos_profile(policy)
        if (
            attestation.get("adapter", {}).get("sha256")
            != manifest["confinement"]["adapter_binary_sha256"]
        ):
            raise AdmissionError("capability adapter differs from admitted binary")
        budget = manifest["budget"]
        if budget["unit"] != "provider_reported_tokens":
            raise AdmissionError("process runner does not support the declared token meter")
        retained_profile = destination / "launch-profile.sb"
        with retained_profile.open("xb") as stream:
            stream.write(Path(profile_path).read_bytes())
        with (destination / "capability.json").open("x") as stream:
            json.dump(attestation, stream, indent=2, allow_nan=False)
            stream.write("\n")
        command = wrap_verified_command(
            argv, compiled, retained_profile, attestation, policy=policy
        )
        if collector is not None:
            # only a launch the real, verified adapter wraps is confined
            collector.adapter_active = Path(command[0]).resolve() == SANDBOX_EXEC
        app_server = "app_server" in current["launch_bindings"][arm]
        with (destination / "launch.json").open("x") as stream:
            json.dump(
                {
                    "argv": argv,
                    "wrapped_argv": command,
                    "cwd": str(visible_roots[arm]),
                    "budget": budget,
                    "environment_sha256": ambient["record"]["environment_sha256"],
                    "compiled_profile_sha256": compiled.sha256,
                    "system_prompt_delivery": {
                        "transport": "app_server_thread" if app_server else "argv",
                        "argument_index": None
                        if app_server
                        else current["launch_bindings"][arm]["system_prompt_arg"],
                        "sha256": manifest["prompts"]["system_sha256"],
                        "provider_role_verified": False,
                    },
                    "research_contract_delivery": {
                        "transport": "app_server_thread" if app_server else "argv",
                        "argument_index": None
                        if app_server
                        else current["launch_bindings"][arm]["research_contract_arg"],
                        "identity_sha256": manifest["prompts"]["research_contract_sha256"],
                        "encoding": "document_bundle_json" if current.get("prompt_row") else "text",
                        "frontend_interpretation_verified": False,
                    },
                },
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")

        def run_agent():
            guard = (
                FrontendToolBudget(manifest["agent"]["kind"], budget["frontend_tool_ceiling"])
                if "frontend_tool_ceiling" in budget
                else None
            )
            try:
                return campaign.run_session(
                    manifest["agent"]["kind"],
                    command,
                    Path(visible_roots[arm]),
                    destination,
                    {
                        "prior_tokens": 0,
                        "prior_wall_s": 0,
                        "token_ceiling": budget["ceiling"],
                        "wall_ceiling_s": budget["wall_ceiling_s"],
                    },
                    env=ambient["environment"],
                    environment_record=ambient["record"],
                    line_guard=guard,
                )
            finally:
                if guard is not None:
                    with (destination / "frontend-live.json").open("x") as stream:
                        json.dump(guard.report(), stream, indent=2, allow_nan=False)
                        stream.write("\n")

        if not app_server and not {"harness.check", "harness.run"}.intersection(
            manifest["policy"]["allowed_external_tools"]
        ):
            return run_agent()
        python = current["launch_bindings"][arm].get("tool_python")
        if python is None:
            raise AdmissionError("tool-enabled session lacks an admitted interpreter path")
        controller = ToolController(
            current,
            root,
            visible_roots,
            arm,
            destination,
            session_id=session_id,
            python=Path(python),
            profile_path=retained_profile,
            attestation=attestation,
            worker_preparations=worker_preparations,
        )
        if app_server:
            from aisle.harness.matched_app_server import run_authorized_app_server

            with session_relay_reservation(policy, destination) as reservation:
                return run_authorized_app_server(
                    controller,
                    command,
                    cwd=Path(visible_roots[arm]),
                    env=ambient["environment"],
                    launch=current["launch_bindings"][arm],
                    budget=budget,
                    references=authority_references,
                    reservation=reservation,
                )
        with ToolService(controller) as service:
            result = run_agent()
        if not service.report["ok"]:
            raise campaign.SessionInfraError("tool service failed or left pending requests", result)
        return result

    def authority_evidence():
        from aisle.harness.matched_app_server import acquire_authority_evidence

        return acquire_authority_evidence(Path(output), authority_references)

    return execute_session(
        plan,
        root,
        visible_roots,
        arm,
        output,
        session_id=session_id,
        launch=launch,
        hidden_access_log=hidden_access_log,
        purpose=purpose,
        request_authority_evidence=authority_evidence,
    )


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AdmissionError(message)


def main(argv: list[str] | None = None) -> int:
    """CON-8: read controller-owned JSON inputs and emit one JSON result."""
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("admit", "run", "collect-run", "check", "heldout-eval"):
        command = commands.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
        if name != "heldout-eval":
            command.add_argument("--output", type=Path, required=True)
        if name in ("run", "check"):
            command.add_argument("--arm", choices=("typed", "monolithic"), required=True)
            command.add_argument("--session-id", required=True)
            if name == "run":
                command.add_argument("--purpose", default="engineering")
    try:
        args = parser.parse_args(argv)
        request = json.loads(args.request.read_text())
        if not isinstance(request, dict):
            raise AdmissionError("controller request must be an object")
        if args.command == "heldout-eval":
            from aisle.harness.heldout_eval import run_heldout

            result = {**run_heldout(request), "eligible_for_estimate": False}
            print(json.dumps(result, allow_nan=False))
            return 0 if result["ok"] else 2
        if args.command == "collect-run":
            result = retain_run(request["source"], args.output, run_id=request["run_id"])
            print(json.dumps(result, allow_nan=False))
            return 0 if result["ok"] else 2
        root = Path(request["root"]).resolve()
        declared_views = request["visible_roots"]
        if not isinstance(declared_views, dict):
            raise AdmissionError("visible roots must be an object")
        views = {arm: Path(path).resolve() for arm, path in declared_views.items()}
        if args.command == "admit":
            plan = admit_pair(
                root,
                request["candidates"],
                views,
                confinement=request.get("confinement"),
                ambient=request.get("ambient"),
                prompt_row=request.get("prompt_row"),
                launches=request.get("launches"),
                development=request.get("development"),
                tool_runtime=request.get("tool_runtime"),
                typed_validation=request.get("typed_validation"),
                run_controller=request.get("run_controller"),
                private_roots=request.get("private_roots"),
            )
            destination = args.output.resolve()
            if any(destination.is_relative_to(view) for view in views.values()):
                raise AdmissionError("admission evidence sink overlaps a participant view")
            with destination.open("x") as stream:
                json.dump(plan, stream, indent=2, allow_nan=False)
                stream.write("\n")
            result = {
                "ok": True,
                "plan_id": plan["immutable_id"],
                "confirmatory_ready": False,
                "output": str(destination),
            }
        elif args.command == "check":
            result = ToolController(
                request["plan"],
                root,
                views,
                args.arm,
                args.output,
                session_id=args.session_id,
                python=Path(request["python"]),
                profile_path=Path(request["profile_path"]),
                attestation=request["attestation"],
                create_output=True,
            ).check()
        else:
            if not isinstance(request["plan"], dict):
                raise AdmissionError("retained plan must be an object")
            result = run_engineering_session(
                request["plan"],
                root,
                views,
                args.arm,
                args.output,
                session_id=args.session_id,
                profile_path=Path(request["profile_path"]),
                attestation=request["attestation"],
                hidden_access_log=(
                    request["hidden_access_log"]
                    if isinstance(request["hidden_access_log"], dict)
                    else Path(request["hidden_access_log"])
                ),
                purpose=args.purpose,
                worker_preparations=request.get("worker_preparations"),
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "error": str(exc), "eligible_for_estimate": False}
    print(json.dumps(result, allow_nan=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
