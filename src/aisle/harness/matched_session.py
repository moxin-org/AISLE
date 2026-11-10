"""Controller-side matched-session admission (MON-8/MON-13).

Admission binds engineering inputs. It does not supply the independent gate
records needed to authorize pilot or confirmatory collection (CSE-10).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from datetime import UTC
from pathlib import Path, PurePosixPath

from aisle.harness import candidates as cand
from aisle.harness import monolith
from aisle.harness.matched_evidence import SCHEMA, audit_tool_journal, common_envelope
from aisle.harness.monolithic import TypedSurfaceError, validate_matched_treatment
from aisle.harness.treatment_integrity import ManifestError, create_treatment_manifest
from aisle.harness.typed_execution_bundle import CONTROLLER_FILES as TYPED_EXECUTION_FILES

ARMS = {"typed", "monolithic"}
CONTROLLER_FILES = (
    "src/aisle/__init__.py",
    "src/aisle/monolith/primitives.py",
    "src/aisle/monolith/confinement.py",
    "src/aisle/monolith/primitive_api.py",
    "src/aisle/monolith/requests.py",
    "src/aisle/monolith/proxy.py",
    "src/aisle/monolith/wire.py",
    "src/aisle/monolith/worker.py",
    "src/aisle/monolith/supervisor.py",
    "src/aisle/monolith/worker_launch.py",
    "src/aisle/monolith/worker_config.py",
    "src/aisle/nodes/monolith_broker.py",
    "src/aisle/harness/monolith.py",
    "tools/matched_campaign.py",
    "tools/campaign.py",
    "src/aisle/harness/matched_session.py",
    "src/aisle/harness/candidates.py",
    "src/aisle/harness/hidden_access_log.py",
    "src/aisle/harness/matched_evidence.py",
    "src/aisle/harness/matched_collection.py",
    "src/aisle/harness/monolithic_run_evidence.py",
    "src/aisle/harness/monolithic_run_prepare.py",
    "src/aisle/harness/matched_tools.py",
    "src/aisle/harness/matched_runtime.py",
    "src/aisle/harness/matched_run.py",
    "src/aisle/harness/matched_dynamic_run.py",
    "src/aisle/harness/typed_stage_provider.py",
    "src/aisle/harness/typed_worker_provisioning.py",
    "src/aisle/harness/worker_declaration.py",
    "src/aisle/harness/worker_capability.py",
    "src/aisle/harness/worker_authority_probe.py",
    "src/aisle/harness/worker_network_probe.py",
    "src/aisle/harness/matched_run_launch.py",
    "src/aisle/harness/typed_snapshot.py",
    "src/aisle/harness/typed_run_prepare.py",
    "src/aisle/harness/typed_validation.py",
    "src/aisle/harness/typed_node_requests.py",
    "src/aisle/harness/typed_node_worker.py",
    "src/aisle/harness/typed_node_supervisor.py",
    "src/aisle/harness/typed_execution_bundle.py",
    "src/aisle/harness/typed_worker_launch.py",
    "src/aisle/harness/typed_node_host.py",
    "src/aisle/harness/typed_graph_hosts.py",
    "src/aisle/harness/typed_graph_stage.py",
    "src/aisle/harness/typed_graph_audit.py",
    "src/aisle/turn_node.py",
    "src/aisle/turns.py",
    "src/aisle/harness/validate.py",
    "src/aisle/harness/registry.py",
    "src/aisle/harness/matched_frontend.py",
    "src/aisle/harness/matched_tool_service.py",
    "src/aisle/harness/matched_app_server.py",
    "src/aisle/harness/frontend_app_server.py",
    "src/aisle/harness/frontend_app_server_audit.py",
    "src/aisle/harness/frontend_request_authority.py",
    "src/aisle/harness/frontend_request_audit.py",
    "src/aisle/harness/frontend_dispatch.py",
    "src/aisle/harness/frontend_dispatch_audit.py",
    "src/aisle/harness/provider_response_authority.py",
    "src/aisle/harness/provider_hosted.py",
    "src/aisle/harness/provider_relay.py",
    "src/aisle/harness/provider_runner.py",
    "src/aisle/harness/relay_reservation.py",
    "src/aisle/harness/loopback_capability.py",
    "src/aisle/harness/provider_source_audit.py",
    "src/aisle/harness/mcp_harness_source.py",
    "src/aisle/harness/mcp_harness_authority.py",
    "src/aisle/harness/mcp_harness_transport.py",
    "src/aisle/harness/frontend_conformance.py",
    "src/aisle/harness/frontend_qualification.py",
    "src/aisle/harness/frontend_effects.py",
    "src/aisle/harness/code_mode_authority.py",
    "src/aisle/harness/code_mode_proxy.py",
    "src/aisle/harness/code_mode_runner.py",
    "src/aisle/harness/code_mode_audit.py",
    "src/aisle/harness/_code_mode_protocol/__init__.py",
    "src/aisle/harness/_code_mode_protocol/protocol.proto",
    "src/aisle/harness/_code_mode_protocol/descriptor.pb",
    "src/aisle/harness/cli.py",
    "src/aisle/harness/common.py",
    "src/aisle/harness/rollout.py",
    "src/aisle/harness/simulator_work.py",
    "src/aisle/harness/rollout_client.py",
    "src/aisle/harness/guard_divergence.py",
    "src/aisle/harness/traces.py",
    "src/aisle/harness/monolithic.py",
    "src/aisle/harness/treatment_integrity.py",
    "src/aisle/harness/treatment_ambient.py",
    "src/aisle/harness/treatment_confinement.py",
    "src/aisle/harness/treatment_postflight.py",
)

CONTROLLER_FILES = tuple(dict.fromkeys((*CONTROLLER_FILES, *TYPED_EXECUTION_FILES)))


class AdmissionError(ValueError):
    """The controller cannot establish the required matched session identity."""


def _digest(value: dict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _shared(
    manifest: dict,
    permission_id: str | None = None,
    ambient_id: str | None = None,
    prompt_id: str | None = None,
) -> dict:
    shared = copy.deepcopy(manifest)
    shared.pop("immutable_id")
    shared["assignment"].pop("arm")
    for key in ("visible_allowlist", "visible_files", "editable_allowlist"):
        shared["repository"].pop(key, None)
    if permission_id is not None:
        shared["confinement"]["profile_sha256"] = permission_id
        shared["confinement"]["policy_sha256"] = permission_id
    if ambient_id is not None:
        shared["state"]["environment_baseline_sha256"] = ambient_id
    if prompt_id is not None:
        shared["prompts"]["research_contract_sha256"] = prompt_id
    return shared


def _prompt_identity(root: Path, table: dict, row_id: str, manifests: dict, views: dict) -> str:
    rows = [row for row in table["rows"] if row["id"] == row_id]
    if len(rows) != 1 or rows[0]["surface"] != "documentation given to the agent":
        raise AdmissionError(
            "prompt binding must identify the declared representation document row"
        )
    row = rows[0]
    if row["class"] != "intentionally-different":
        raise AdmissionError("representation document difference is not declared")
    for arm in sorted(ARMS):
        paths = row[arm]["paths"]
        manifest = manifests[arm]
        visible = manifest["repository"]["visible_allowlist"]
        editable = manifest["repository"]["editable_allowlist"]
        if any(name not in visible or name in editable for name in paths):
            raise AdmissionError("representation documents must be visible and read-only")
        expected = monolith._set_digest(root, paths)
        if manifest["prompts"]["research_contract_sha256"] != expected:
            raise AdmissionError("representation document bundle identity is unbound")
        if monolith._set_digest(Path(views[arm]), paths) != expected:
            raise AdmissionError("visible representation document bytes differ")
    return _digest(row)


def _private_tree_hash(root: Path) -> str:
    """Fingerprint regular-file bytes and modes without retaining their contents."""
    if root.resolve() != root or not root.is_dir():
        raise AdmissionError("private baseline root is missing or redirected")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise AdmissionError("private baseline contains a redirected path")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AdmissionError("private baseline contains a nonregular file")
        files[path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mode": path.stat().st_mode & 0o777,
        }
    return _digest(files).split(":", 1)[1]


def _ambient_ids(manifests: dict, bindings: dict, ambient: dict, active_arm=None) -> dict:
    from aisle.harness.treatment_ambient import (
        _FORWARDED_VARIABLES,
        _GENERATED_PATHS,
        AmbientIsolationError,
        verify_declared_environment,
    )
    from aisle.harness.treatment_integrity import _reject_secrets

    if bindings is None or set(ambient) != ARMS:
        raise AdmissionError("ambient binding requires both confined arms")
    _reject_secrets(ambient)
    identities = {}
    allowed_keys = (
        set(_GENERATED_PATHS)
        | set(_FORWARDED_VARIABLES)
        | {"__CF_USER_TEXT_ENCODING", "AISLE_ENV_BASELINE"}
    )
    for arm in sorted(ARMS):
        environment = ambient[arm]["environment"]
        record = ambient[arm]["record"]
        if not isinstance(environment, dict) or not all(
            isinstance(v, str) for v in environment.values()
        ):
            raise AdmissionError("ambient variables must be strings")
        if set(environment) - allowed_keys:
            raise AdmissionError("ambient binding contains undeclared variable authority")
        try:
            verify_declared_environment(environment, record)
        except AmbientIsolationError as exc:
            raise AdmissionError(f"ambient identity invalid: {exc}") from exc
        if manifests[arm]["state"]["environment_baseline_sha256"] != record["environment_sha256"]:
            raise AdmissionError("ambient hash differs from the admitted manifest")
        home = Path(environment["HOME"])
        scratch = Path(bindings[arm]["scratch"])
        if (
            home.resolve() != home
            or not home.is_relative_to(scratch)
            or record["home"] != str(home)
        ):
            raise AdmissionError("ambient HOME is outside its bound scratch root")
        normalized = dict(environment)
        for name, relative in _GENERATED_PATHS.items():
            expected = home if relative == "." else home / relative
            if (
                environment.get(name) != str(expected)
                or expected.resolve() != expected
                or not expected.is_dir()
            ):
                raise AdmissionError("ambient generated path differs from its HOME role")
            normalized[name] = f"HOME/{relative}"
        if arm != active_arm:
            for variable, field in (
                ("HOME", "home_baseline_sha256"),
                ("XDG_CONFIG_HOME", "config_baseline_sha256"),
                ("XDG_CACHE_HOME", "cache_baseline_sha256"),
            ):
                if (
                    _private_tree_hash(Path(environment[variable]))
                    != manifests[arm]["state"][field]
                ):
                    raise AdmissionError(
                        f"private {variable} contents differ from declared baseline"
                    )
        identities[arm] = _digest(normalized)
    return identities


def _private_roles(root, views, bindings, declaration):
    """Normalize only named, disjoint controller roots explicitly hidden from both arms."""
    if declaration is None:
        return {}
    if type(declaration) is not dict or not declaration:
        raise AdmissionError("private root roles require a nonempty mapping")
    protected = [Path(root), *(Path(p) for p in views.values())]
    for binding in bindings.values():
        protected.append(Path(binding["scratch"]))
        for key in ("visible_roots", "output_roots", "runtime_read_roots", "allowed_executables"):
            protected.extend(Path(p) for p in binding["policy"][key])
    roles = {}
    for name, value in declaration.items():
        if (
            type(name) is not str
            or re.fullmatch("[a-z][a-z0-9_]{0,63}", name) is None
            or type(value) is not str
        ):
            raise AdmissionError("private root role names and paths are invalid")
        path = Path(value)
        if not path.is_absolute() or path.resolve() != path or not path.is_dir():
            raise AdmissionError("private root roles require canonical existing directories")
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in (*protected, *roles)):
            raise AdmissionError("private root roles overlap authority or another role")
        if any(value not in binding["policy"]["hidden_roots"] for binding in bindings.values()):
            raise AdmissionError("private root roles must be explicitly hidden from both arms")
        roles[path] = "private/" + name
    return roles


def _permission_ids(
    root: Path, manifests: dict, views: dict, bindings: dict, private_roots=None
) -> dict:
    from aisle.harness.treatment_confinement import (
        ConfinementError,
        MacOSPolicy,
        compile_macos_profile,
    )

    if set(bindings) != ARMS:
        raise AdmissionError("both confinement bindings are required")
    private_roles = _private_roles(root, views, bindings, private_roots)
    scratch = {arm: Path(bindings[arm]["scratch"]) for arm in ARMS}
    isolated = [root, *(Path(views[a]).resolve() for a in sorted(ARMS)), *scratch.values()]
    for index, first in enumerate(isolated):
        if not first.is_absolute() or first.resolve() != first:
            raise AdmissionError("confinement roots must be canonical absolute paths")
        for second in isolated[index + 1 :]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise AdmissionError("confinement role roots overlap")
    identities = {}
    for arm in sorted(ARMS):
        other = next(name for name in ARMS if name != arm)
        view, other_view = Path(views[arm]).resolve(), Path(views[other]).resolve()
        try:
            policy = MacOSPolicy.from_canonical(bindings[arm]["policy"])
        except ConfinementError as exc:
            raise AdmissionError(f"invalid confinement policy: {exc}") from exc
        editable = {view / p for p in manifests[arm]["repository"]["editable_allowlist"]}
        if set(policy.output_roots) != editable | {scratch[arm]}:
            raise AdmissionError("confinement write authority differs from declared edit grants")
        if policy.visible_roots != (view,):
            raise AdmissionError("confinement visible authority must match its single arm view")
        if not {root, other_view, scratch[other]}.issubset(policy.hidden_roots):
            raise AdmissionError("confinement must protect controller and other arm roots")
        try:
            compiled = compile_macos_profile(policy)
        except ConfinementError as exc:
            raise AdmissionError(f"invalid confinement policy: {exc}") from exc
        expected = manifests[arm]["confinement"]
        if (
            expected["profile_sha256"] != compiled.sha256
            or expected["policy_sha256"] != compiled.policy_id
        ):
            raise AdmissionError("confinement profile identity differs from the compiled policy")
        roles = {
            view: "view",
            scratch[arm]: "scratch",
            other_view: "other_view",
            scratch[other]: "other_scratch",
            root: "controller",
            **private_roles,
        }
        # Only exact role roots are normalized. Shared runtime/executable paths
        # retain their full values; added permissions therefore remain visible.
        normalized = {
            key: sorted(roles.get(path, str(path)) for path in getattr(policy, key))
            for key in (
                "visible_roots",
                "runtime_read_roots",
                "allowed_executables",
                "hidden_roots",
            )
        }
        normalized["output_roots"] = ["declared_editable_set", "scratch"]
        normalized["network_policy"] = policy.network_policy
        identities[arm] = _digest(normalized)
    return identities


def _verify_loopback_launch(policy: object, launch: dict) -> None:
    """A `loopback` grant reaches one pinned port: the owned provider relay.
    MCP harness and Code Mode listen on other ports it cannot reach (ADR-69)."""
    if type(policy) is not dict or policy.get("network_policy") != "loopback":
        return
    if "app_server" not in launch or "provider" not in launch:
        raise AdmissionError("a loopback policy requires an owned provider relay launch")
    if launch.get("mcp_harness") or "code_mode_host" in launch:
        raise AdmissionError(
            "a single relay port cannot reach the MCP harness or Code Mode listeners"
        )


def _verify_launches(
    candidates: dict,
    launches: dict,
    root: Path,
    prompt_row: str | None,
    *,
    execution=None,
    documents=None,
) -> None:
    if not isinstance(launches, dict) or set(launches) != ARMS:
        raise AdmissionError("both launch bindings are required")
    comparable = copy.deepcopy(launches)
    bindings = (execution or {}).get("confinement_bindings") or {}
    for arm in sorted(ARMS):
        launch = launches[arm]
        if not isinstance(launch, dict):
            raise AdmissionError("unsupported launch binding")
        binding = bindings.get(arm) if isinstance(bindings, dict) else None
        _verify_loopback_launch(
            binding.get("policy") if isinstance(binding, dict) else None, launch
        )
        app_server = "app_server" in launch
        if app_server:
            if (
                set(launch)
                - {
                    "tool_python",
                    "code_mode_host",
                    "provider",
                    "mcp_harness",
                    "conformance_profile",
                }
                != {"argv", "app_server"}
                or candidates[arm]["agent"]["kind"] != "codex"
                or type(launch["app_server"]) is not dict
                or set(launch["app_server"]) != {"baseInstructions", "developerInstructions"}
                or any(type(value) is not str for value in launch["app_server"].values())
            ):
                raise AdmissionError("unsupported app-server launch binding")
            if "mcp_harness" in launch and (
                launch["mcp_harness"] is not True
                or "provider" not in launch
                or not {"harness.check", "harness.run"}.intersection(
                    candidates[arm]["policy"]["allowed_external_tools"]
                )
            ):
                raise AdmissionError(
                    "MCP harness requires owned provider and admitted harness tools"
                )
            if "provider" in launch:
                from aisle.harness.provider_relay import verify_provider
                from aisle.harness.treatment_confinement import verify_relay_port

                try:
                    verify_provider(launch["provider"])
                    ceiling = candidates[arm]["budget"].get("frontend_tool_ceiling")
                    if type(ceiling) is not int or ceiling <= 0:
                        raise ValueError("provider requires a shared frontend dispatch ceiling")
                    bindings = (execution or {}).get("confinement_bindings") or {}
                    verify_relay_port(
                        bindings.get(arm, {}).get("policy") if isinstance(bindings, dict) else None,
                        launch["provider"],
                    )
                except (ValueError, TypeError) as exc:
                    raise AdmissionError("invalid provider binding: " + str(exc)) from exc
            if "code_mode_host" in launch:
                from aisle.harness.code_mode_runner import verify_host

                try:
                    verify_host(launch["code_mode_host"])
                    ceiling = candidates[arm]["budget"].get("frontend_tool_ceiling")
                    if type(ceiling) is not int or ceiling <= 0:
                        raise ValueError("host requires a shared frontend dispatch ceiling")
                except (OSError, ValueError, TypeError) as exc:
                    raise AdmissionError("invalid nested host binding: " + str(exc)) from exc
        else:
            if "system_prompt_arg" not in launch:
                raise AdmissionError("system prompt argument binding is required")
            if "research_contract_arg" not in launch:
                raise AdmissionError("research contract argument binding is required")
            if set(launch) not in (
                {"argv", "system_prompt_arg", "research_contract_arg"},
                {"argv", "system_prompt_arg", "research_contract_arg", "tool_python"},
            ):
                raise AdmissionError("unsupported launch binding")
        argv = launch["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(value, str) and value and "\0" not in value for value in argv)
        ):
            raise AdmissionError("launch arguments must be a nonempty argv")
        if app_server:
            system_prompt = launch["app_server"]["baseInstructions"]
            research_contract = launch["app_server"]["developerInstructions"]
        else:
            prompt_arg = launch["system_prompt_arg"]
            if type(prompt_arg) is not int or not 0 < prompt_arg < len(argv):
                raise AdmissionError("system prompt argument index is invalid")
            contract_arg = launch["research_contract_arg"]
            if (
                type(contract_arg) is not int
                or not 0 < contract_arg < len(argv)
                or contract_arg == prompt_arg
            ):
                raise AdmissionError("research contract argument index is invalid")
            system_prompt = argv[prompt_arg]
            research_contract = argv[contract_arg]
        if (
            hashlib.sha256(system_prompt.encode()).hexdigest()
            != candidates[arm]["prompts"]["system_sha256"]
        ):
            raise AdmissionError("system prompt argument differs from admitted bytes")
        if prompt_row is None:
            if (
                hashlib.sha256(research_contract.encode()).hexdigest()
                != candidates[arm]["prompts"]["research_contract_sha256"]
            ):
                raise AdmissionError("research contract argument differs from admitted bytes")
        else:
            documents = monolith.DEFAULT_DOCUMENTS if documents is None else documents
            rows = [
                row
                for row in monolith.load_document(root, documents["treatment_table"])["rows"]
                if row["id"] == prompt_row
            ]
            if len(rows) != 1 or rows[0]["surface"] != "documentation given to the agent":
                raise AdmissionError("research contract document row is unresolved")
            bundle = [
                {"path": name, "text": (root / name).read_bytes().decode("utf-8")}
                for name in rows[0][arm]["paths"]
            ]
            if research_contract != json.dumps(bundle, ensure_ascii=False, separators=(",", ":")):
                raise AdmissionError("research contract argument differs from declared documents")
            if app_server:
                comparable[arm]["app_server"]["developerInstructions"] = (
                    "declared representation document bundle"
                )
            else:
                comparable[arm]["argv"][contract_arg] = "declared representation document bundle"
        executable = Path(argv[0])
        if (
            not executable.is_absolute()
            or executable.resolve() != executable
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
            or hashlib.sha256(executable.read_bytes()).hexdigest()
            != candidates[arm]["agent"]["cli_binary_sha256"]
        ):
            raise AdmissionError("launch executable differs from admitted agent binary")
        wall = candidates[arm]["budget"].get("wall_ceiling_s")
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise AdmissionError("launch wall budget must be finite and positive")
        if "frontend_tool_ceiling" in candidates[arm]["budget"]:
            ceiling = candidates[arm]["budget"]["frontend_tool_ceiling"]
            if type(ceiling) is not int or ceiling <= 0:
                raise AdmissionError("frontend tool ceiling must be a positive integer")
        if "tool_python" in launch:
            python = Path(launch["tool_python"])
            if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
                raise AdmissionError("tool interpreter path is unresolved")
            expected = {
                "name": "harness-python",
                "sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
            }
            if expected not in candidates[arm]["runtime_binaries"]:
                raise AdmissionError("tool interpreter differs from admitted runtime")
        if "conformance_profile" in launch:
            from aisle.harness.frontend_conformance import (
                read_bound_profile,
                verify_fault_matrix,
                verify_route_matrix,
            )

            try:
                bound_profile = read_bound_profile(
                    root,
                    launch["conformance_profile"],
                    candidate=candidates[arm],
                    launch=launch,
                    arm=arm,
                    execution=execution,
                )
                if candidates[arm]["budget"].get("frontend_coverage") == "complete":
                    verify_route_matrix(
                        bound_profile, candidate=candidates[arm], launch=launch, arm=arm
                    )
                    verify_fault_matrix(
                        bound_profile, candidate=candidates[arm], launch=launch, arm=arm
                    )
            except ValueError as exc:
                raise AdmissionError("conformance profile binding failed: " + str(exc)) from exc
    if comparable["typed"] != comparable["monolithic"]:
        raise AdmissionError("undeclared launch arguments differ between arms")


def _verify_development(protocol: dict, root: Path | None = None) -> dict:
    """Return the retained development form: v1 unchanged after the fixed-field
    check; v2 normalized against the hashed candidate table (MON-3/MON-8)."""
    from aisle.harness import candidates as cand

    try:
        if (
            isinstance(protocol, dict)
            and protocol.get("schema_version") != cand.DEVELOPMENT_SCHEMA_V1
        ):
            return cand.normalize_development(protocol, root)
        fixed = {
            "schema_version": cand.DEVELOPMENT_SCHEMA_V1,
            "purpose": cand.DEVELOPMENT_PURPOSE,
            "tier": "T1",
            "embodiment": "franka",
            "verifier": "oracle",
            "reset": "teleport",
        }
        if not isinstance(protocol, dict) or set(protocol) != set(fixed) | set(cand.BUDGET_KEYS):
            raise AdmissionError("development protocol fields are unresolved")
        if any(protocol[key] != value for key, value in fixed.items()):
            raise AdmissionError("development protocol requests an unsupported run mode")
        cand.check_budget(protocol)
    except cand.CandidateError as exc:
        raise AdmissionError(f"development protocol refused: {exc}") from exc
    return protocol


def _verify_fresh_private_home(record):
    """Require the generated empty layout before an owning arm starts."""
    from aisle.harness.treatment_ambient import _GENERATED_PATHS

    home = Path(record["home"])
    expected = set()
    for relative in _GENERATED_PATHS.values():
        directory = home / relative
        while directory != home:
            expected.add(directory)
            directory = directory.parent
    pending = [home]
    observed = set()
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink() or path not in expected or not path.is_dir():
                raise AdmissionError("private state is not fresh")
            observed.add(path)
            pending.append(path)
    if observed != expected:
        raise AdmissionError("private state generated layout is incomplete")


def admit_pair(
    root: Path,
    candidates: dict,
    visible_roots: dict,
    *,
    confinement: dict | None = None,
    ambient: dict | None = None,
    prompt_row: str | None = None,
    launches: dict | None = None,
    development: dict | None = None,
    tool_runtime: dict | None = None,
    typed_validation: dict | None = None,
    run_controller: dict | None = None,
    private_roots: dict | None = None,
    _active_arm=None,
) -> dict:
    """Bind both SPEC 420 manifests to current launcher artifacts before execution."""
    if set(candidates) != ARMS or set(visible_roots) != ARMS:
        raise AdmissionError("both matched arms are required")
    if private_roots is not None and confinement is None:
        raise AdmissionError("private root roles require confinement bindings")
    root = Path(root).resolve()
    roots = [root, *(Path(visible_roots[arm]).resolve() for arm in sorted(ARMS))]
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise AdmissionError("controller and participant roots overlap")
    try:
        if tool_runtime is not None:
            from aisle.harness.matched_runtime import RuntimeDrift, verify_runtime

            try:
                verify_runtime(tool_runtime)
            except RuntimeDrift as exc:
                raise AdmissionError(str(exc)) from exc
            protected_runtime_roots = [*roots]
            if confinement is not None:
                protected_runtime_roots.extend(
                    Path(path).resolve()
                    for binding in confinement.values()
                    for path in binding["policy"]["output_roots"]
                )
            for runtime_root in tool_runtime["trees"]:
                runtime_path = Path(runtime_root)
                if any(
                    runtime_path.is_relative_to(p) or p.is_relative_to(runtime_path)
                    for p in protected_runtime_roots
                ):
                    raise AdmissionError(
                        "tool runtime overlaps controller or participant authority"
                    )
        if run_controller is not None:
            from aisle.harness.matched_run_launch import _verify_controller_binding_fields
            from aisle.harness.treatment_ambient import _GENERATED_PATHS

            if tool_runtime is None or confinement is None or set(run_controller) != ARMS:
                raise AdmissionError("run controller requires both arms, runtime and confinement")
            homes = []
            controller_environments = []
            for arm in sorted(ARMS):
                try:
                    _verify_controller_binding_fields(
                        run_controller[arm],
                        tool_runtime,
                        roots,
                        [binding["policy"] for binding in confinement.values()],
                    )
                    expected = {
                        "name": "harness-python",
                        "sha256": run_controller[arm]["python_sha256"],
                    }
                    if expected not in candidates[arm]["runtime_binaries"]:
                        raise ValueError("run controller interpreter is not admitted")
                    home = Path(run_controller[arm]["environment_record"]["home"])
                    if _active_arm != arm:
                        _verify_fresh_private_home(run_controller[arm]["environment_record"])
                    if any(home.is_relative_to(p) or p.is_relative_to(home) for p in homes):
                        raise ValueError("run controllers share private state")
                    homes.append(home)
                    # Binding verification established every generated path's HOME role.
                    # Preserve all other settings exactly, including PATH and locale.
                    environment = dict(run_controller[arm]["environment"])
                    for name, relative in _GENERATED_PATHS.items():
                        environment[name] = f"HOME/{relative}"
                    controller_environments.append(_digest(environment))
                except (ValueError, KeyError, TypeError) as exc:
                    raise AdmissionError(f"run controller binding invalid: {exc}") from exc
            if len(set(controller_environments)) != 1:
                raise AdmissionError("run controller environments differ between arms")
        if typed_validation is not None:
            from aisle.harness.typed_validation import _verify_validation_binding_fields

            if tool_runtime is None or confinement is None:
                raise AdmissionError("typed validation requires runtime and confinement bindings")
            try:
                _verify_validation_binding_fields(
                    typed_validation,
                    tool_runtime,
                    roots,
                    [binding["policy"] for binding in confinement.values()],
                )
                if _active_arm != "typed":
                    _verify_fresh_private_home(typed_validation["environment_record"])
                    if any(Path(typed_validation["snapshot_storage"]).iterdir()):
                        raise AdmissionError("typed validation snapshot storage is not fresh")
                expected_python = {
                    "name": "harness-python",
                    "sha256": typed_validation["python_sha256"],
                }
                if expected_python not in candidates["typed"]["runtime_binaries"]:
                    raise AdmissionError("typed validation interpreter is not admitted")
                if (
                    typed_validation["attestation"]["adapter"]["sha256"]
                    != candidates["typed"]["confinement"]["adapter_binary_sha256"]
                ):
                    raise AdmissionError("typed validation adapter differs from admitted adapter")
            except (ValueError, RuntimeError) as exc:
                raise AdmissionError(f"typed validation binding invalid: {exc}") from exc
        if development is not None:
            development = _verify_development(development, root)
        documents = cand.documents_for(development)
        for arm in sorted(ARMS):
            budget = candidates[arm]["budget"]
            if type(budget) is not dict:
                raise AdmissionError("invalid frontend coverage budget")
            coverage = budget.get("frontend_coverage", "observed")
            if type(coverage) is not str or coverage not in {"observed", "complete"}:
                raise AdmissionError("unsupported frontend coverage policy")
            if coverage == "complete" and (
                type(launches) is not dict
                or type(launches.get(arm)) is not dict
                or "conformance_profile" not in launches[arm]
            ):
                raise AdmissionError(
                    "complete frontend coverage requires a bound conformance profile"
                )
        if launches is not None:
            _verify_launches(
                candidates,
                launches,
                root,
                prompt_row,
                documents=documents,
                execution={
                    "development": development,
                    "tool_runtime": tool_runtime,
                    "typed_validation": typed_validation,
                    "run_controller": run_controller,
                    "confinement_bindings": confinement,
                    "ambient_bindings": ambient,
                    "private_roots": private_roots,
                    "prompt_row": prompt_row,
                },
            )
        table = monolith.table_report(root, write=False, documents=documents)
        interface = monolith.interface_report(root, documents=documents)
        if not table["ok"] or not interface["ok"]:
            raise AdmissionError("controller surface checks failed")
        allowlist = monolith.load_document(root, documents["allowlist"])
        source_table = monolith.load_document(root, documents["treatment_table"])
        validation_rows = [
            row
            for row in source_table["rows"]
            if row["surface"] == "static validation and diagnostics"
        ]
        if len(validation_rows) != 1 or validation_rows[0]["class"] != "intentionally-different":
            raise AdmissionError("validation declaration must identify one intentional difference")
        validation_row = validation_rows[0]
        manifests = {}
        for arm in sorted(ARMS):
            candidate = candidates[arm]
            if candidate.get("assignment", {}).get("arm") != arm:
                raise AdmissionError("assignment arm does not match its declared view")
            editable = candidate.get("repository", {}).get("editable_allowlist")
            if editable != sorted(allowlist[arm]["editable"]):
                raise AdmissionError("editable grants do not match the launcher allowlist")
            if arm == "monolithic":
                visible = candidate.get("repository", {}).get("visible_allowlist", [])
                forbidden = ("graphs", "registry", "src/aisle/harness", ".git")
                if any(
                    PurePosixPath(name).is_relative_to(prefix)
                    for name in visible
                    for prefix in forbidden
                ):
                    raise AdmissionError("monolithic view exposes typed facilities")
            manifests[arm] = create_treatment_manifest(candidate, Path(visible_roots[arm]))
        permissions = (
            _permission_ids(root, manifests, visible_roots, confinement, private_roots)
            if confinement is not None
            else {}
        )
        ambient_ids = (
            _ambient_ids(manifests, confinement, ambient, _active_arm)
            if ambient is not None
            else {}
        )
        prompt_id = (
            _prompt_identity(root, source_table, prompt_row, manifests, visible_roots)
            if prompt_row is not None
            else None
        )
        try:
            validate_matched_treatment(
                _shared(
                    manifests["typed"],
                    permissions.get("typed"),
                    ambient_ids.get("typed"),
                    prompt_id,
                ),
                _shared(
                    manifests["monolithic"],
                    permissions.get("monolithic"),
                    ambient_ids.get("monolithic"),
                    prompt_id,
                ),
                set(),
            )
        except TypedSurfaceError as exc:
            raise AdmissionError(f"shared treatment identity differs: {exc}") from exc
        paths = {path for row in source_table["rows"] for arm in ARMS for path in row[arm]["paths"]}
        paths.update(CONTROLLER_FILES)
        paths.update(
            f"docs/monolithic/{name}.json"
            for name in (
                "treatment-table",
                "interface-map",
                "allowlist",
                "parity-protocol",
                "experts",
                "candidates",
            )
        )
        paths.update(cand.retained_artifacts(development))
        artifact_hashes = {}
        for name in sorted(paths):
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise AdmissionError("controller surface path escapes the root")
            artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name in CONTROLLER_FILES:
                executing = Path(__file__).resolve().parents[3] / name
                if artifact_hashes[name] != hashlib.sha256(executing.read_bytes()).hexdigest():
                    raise AdmissionError(
                        f"declared source differs from executing controller: {name}"
                    )
        try:
            cand.verify_table_binding(development, artifact_hashes)
        except cand.CandidateError as exc:
            raise AdmissionError(str(exc)) from exc
        record = {
            "schema_version": "aisle.matched-session-plan.v1",
            "confirmatory_ready": False,
            "arms": manifests,
            "surface": {
                "treatment_table_id": table["immutable_id"],
                "interface_map_id": monolith.load_document(root, documents["interface_map"])["id"],
                "artifact_hashes": artifact_hashes,
                "common_evidence_schema": {**SCHEMA, "immutable_id": _digest(SCHEMA)},
                "validation_declaration": {
                    key: copy.deepcopy(validation_row[key]) for key in ("id", "typed", "monolithic")
                },
            },
        }
        if confinement is not None:
            record["confinement_bindings"] = copy.deepcopy(confinement)
        if private_roots is not None:
            record["private_roots"] = copy.deepcopy(private_roots)
        if ambient is not None:
            record["ambient_bindings"] = copy.deepcopy(ambient)
        if prompt_row is not None:
            record["prompt_row"] = prompt_row
        if launches is not None:
            record["launch_bindings"] = copy.deepcopy(launches)
        if development is not None:
            record["development"] = copy.deepcopy(development)
        if tool_runtime is not None:
            record["tool_runtime"] = copy.deepcopy(tool_runtime)
        if run_controller is not None:
            record["run_controller"] = copy.deepcopy(run_controller)
        if typed_validation is not None:
            record["typed_validation"] = copy.deepcopy(typed_validation)
        record["immutable_id"] = _digest(record)
        return record
    except (OSError, KeyError, TypeError, ManifestError) as exc:
        raise AdmissionError(f"unresolved admission input: {exc}") from exc


def verify_plan(record: dict, root: Path, visible_roots: dict) -> dict:
    """Recompute retained admission against current controller and participant bytes."""
    return _verify_plan(record, root, visible_roots)


def verify_active_plan(record: dict, root: Path, visible_roots: dict, arm: str) -> dict:
    """Recheck an active session, permitting only its already-declared editable files.

    The trusted controller selects the active arm. This does not authorize a
    fresh launch or resume using edited inputs; those still use verify_plan.
    """
    if arm not in ARMS:
        raise AdmissionError("active session arm is unresolved")
    return _verify_plan(record, root, visible_roots, editable_arm=arm)


def _active_comparison(record: dict, arm: str) -> dict:
    compared = copy.deepcopy(record)
    compared.pop("immutable_id")
    manifest = compared["arms"][arm]
    manifest.pop("immutable_id")
    repository = manifest["repository"]
    repository["visible_files"] = [
        {"path": row["path"]} if row["path"] in repository["editable_allowlist"] else row
        for row in repository["visible_files"]
    ]
    return compared


def _verify_plan(record: dict, root: Path, visible_roots: dict, editable_arm=None) -> dict:
    try:
        retained = copy.deepcopy(record)
        identity = retained.pop("immutable_id")
        if identity != _digest(retained):
            raise AdmissionError("retained plan identity drift")
        candidates = copy.deepcopy(retained["arms"])
        for candidate in candidates.values():
            candidate.pop("immutable_id")
            candidate["repository"].pop("visible_files")
        current = admit_pair(
            root,
            candidates,
            visible_roots,
            confinement=retained.get("confinement_bindings"),
            ambient=retained.get("ambient_bindings"),
            prompt_row=retained.get("prompt_row"),
            launches=retained.get("launch_bindings"),
            development=retained.get("development"),
            tool_runtime=retained.get("tool_runtime"),
            typed_validation=retained.get("typed_validation"),
            run_controller=retained.get("run_controller"),
            private_roots=retained.get("private_roots"),
            _active_arm=editable_arm,
        )
        if editable_arm is None:
            unchanged = current == record
        else:
            unchanged = _active_comparison(current, editable_arm) == _active_comparison(
                record, editable_arm
            )
        if not unchanged:
            raise AdmissionError("admitted plan or participant bytes drift")
        return current
    except (KeyError, TypeError, ValueError) as exc:
        raise AdmissionError(f"retained plan drift or unresolved input: {exc}") from exc


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _snapshot(manifest: dict, view: Path, destination: Path) -> dict:
    """Retain actual deliverable bytes outside participant authority."""
    snapshots = {}
    for name in manifest["repository"]["editable_allowlist"]:
        source = view / name
        if source.is_symlink() or not source.resolve().is_relative_to(view.resolve()):
            raise AdmissionError("deliverable snapshot path escapes its view")
        data = source.read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        snapshots[name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "mode": source.stat().st_mode & 0o777,
        }
    return snapshots


def _record_error(record: dict, message: str) -> None:
    record["error"] = "; ".join(value for value in (record["error"], message) if value)


def _check_process_telemetry(record: dict) -> None:
    process = record["process"]
    try:
        json.dumps(process, allow_nan=False)
    except (TypeError, ValueError):
        record["rejected_process_telemetry"] = repr(process)
        record["process"] = None
        _record_error(record, "process telemetry is not finite JSON data")
        return
    if not isinstance(process, dict) or type(process.get("rc")) is not int:
        _record_error(record, "process telemetry lacks an integer exit status")
        return
    for key in ("tokens", "tokens_generated", "wall_s"):
        if key not in process:
            continue
        value = process[key]
        types = (int, float) if key == "wall_s" else (int,)
        if (
            type(value) not in types
            or value < 0
            or (type(value) is float and not math.isfinite(value))
        ):
            _record_error(record, f"process telemetry has invalid {key}")


def execute_session(
    plan: dict,
    root: Path,
    visible_roots: dict,
    arm: str,
    output: Path,
    *,
    session_id: str,
    launch,
    hidden_access_log: Path,
    purpose: str = "engineering",
    now=_utc_now,
    request_authority_evidence=None,
) -> dict:
    """Execute a controller-owned launcher and retain an unscored common record.

    `launch` belongs to the trusted controller, not the participant or a JSON
    input. It writes raw process evidence into `output` and returns the process
    runner's observed record. The adapter's access log is likewise controller
    input; this function does not mint an external confinement attestation.
    """
    from aisle.harness.treatment_postflight import create_postflight_record

    output = Path(output).resolve()
    if output.exists():
        raise AdmissionError("existing attempt cannot be overwritten or resumed")
    for view in visible_roots.values():
        view = Path(view).resolve()
        if output.is_relative_to(view) or view.is_relative_to(output):
            raise AdmissionError("evidence sink overlaps a participant view")
    for binding in plan.get("confinement_bindings", {}).values():
        scratch = Path(binding["scratch"]).resolve()
        if output.is_relative_to(scratch) or scratch.is_relative_to(output):
            raise AdmissionError("evidence sink overlaps participant scratch")
    output.mkdir(parents=True)
    record = {
        "schema_version": "aisle.matched-session-evidence.v1",
        "session_id": session_id,
        "arm": arm,
        "purpose": purpose,
        "plan_id": plan.get("immutable_id"),
        "ok": False,
        "classification": "infrastructure_exclusion",
        "eligible_for_estimate": False,
        "lifecycle": {"started_at": now(), "finished_at": None},
        "events": [],
        "process": None,
        "rejected_process_telemetry": None,
        "snapshots": {"authored": {}, "final": {}},
        "postflight": None,
        "tool_audit": None,
        "frontend_tools": None,
        "private_state": {"initial": None, "final": None, "error": None},
        "artifacts": {},
        "error": None,
    }
    manifest = None
    validation = None
    profile_launch = None
    profile_hashes = None
    try:
        if purpose != "engineering":
            raise AdmissionError("scored collection requires independent CSE-10 gate records")
        if arm not in ARMS or not isinstance(session_id, str) or not session_id.strip():
            raise AdmissionError("session identity is unresolved")
        current = verify_plan(plan, root, visible_roots)
        manifest = current["arms"][arm]
        declaration = current["surface"]["validation_declaration"]
        validation = {
            "typed_validator": arm == "typed",
            "declaration": declaration["id"],
            "description": declaration[arm]["description"],
        }
        view = Path(visible_roots[arm])
        (output / "admission.json").write_text(json.dumps(current, indent=2) + "\n")
        profile_launch = current.get("launch_bindings", {}).get(arm)
        if profile_launch is not None and "conformance_profile" in profile_launch:
            from aisle.harness.frontend_conformance import retain_bound_profile

            record["conformance_profile"] = {
                "reference": copy.deepcopy(profile_launch["conformance_profile"]),
                "status": "invalid",
                "qualification": "not_evaluated",
            }
            profile_hashes = retain_bound_profile(
                root,
                output / "conformance",
                profile_launch["conformance_profile"],
                candidate=manifest,
                launch=profile_launch,
                arm=arm,
                execution=current,
            )
        record["snapshots"]["authored"] = _snapshot(manifest, view, output / "authored")
        # Snapshot reads are not permission to launch changed inputs.
        verify_plan(plan, root, visible_roots)
        record["events"].append({"kind": "launch", "at": now()})
        try:
            record["process"] = launch(output)
        except Exception as exc:
            # Existing campaign runner exceptions carry the retained failed
            # process record. Preserve it instead of losing the attempt.
            record["process"] = getattr(exc, "session", getattr(exc, "record", None))
            record["error"] = f"launcher failed: {exc}"
        _check_process_telemetry(record)
        process = record["process"]
        if isinstance(process, dict):
            for meter, limit in (("tokens", "ceiling"), ("wall_s", "wall_ceiling_s")):
                observed = process.get(meter)
                ceiling = manifest["budget"].get(limit)
                if (
                    type(observed) in (int, float)
                    and type(ceiling) in (int, float)
                    and observed >= ceiling
                ):
                    _record_error(record, f"final {meter} budget reached or exceeded")
        record["events"].append({"kind": "process_finished", "at": now()})
        try:
            record["snapshots"]["final"] = _snapshot(manifest, view, output / "final")
        except (OSError, AdmissionError) as exc:
            _record_error(record, f"final snapshot failed: {exc}")
        retained_access = output / "hidden-access-log.json"
        try:
            with retained_access.open("xb") as stream:
                stream.write(Path(hidden_access_log).read_bytes())
        except OSError as exc:
            _record_error(record, f"access log retention failed: {exc}")
        candidates = copy.deepcopy(current["arms"])
        for candidate in candidates.values():
            candidate.pop("immutable_id")
            candidate["repository"].pop("visible_files")
        record["postflight"] = create_postflight_record(
            manifest, candidates[arm], view, retained_access
        )
        refreshed = admit_pair(
            root,
            candidates,
            visible_roots,
            confinement=current.get("confinement_bindings"),
            ambient=current.get("ambient_bindings"),
            prompt_row=current.get("prompt_row"),
            launches=current.get("launch_bindings"),
            development=current.get("development"),
            tool_runtime=current.get("tool_runtime"),
            typed_validation=current.get("typed_validation"),
            run_controller=current.get("run_controller"),
            private_roots=current.get("private_roots"),
            _active_arm=arm,
        )
        if refreshed["surface"] != current["surface"]:
            raise AdmissionError("controller surface drift after execution")
        other = next(name for name in ARMS if name != arm)
        if refreshed["arms"][other] != current["arms"][other]:
            raise AdmissionError("other participant view drift after execution")
        process = record["process"]
        if not isinstance(process, dict) or process.get("rc") != 0:
            raise AdmissionError("process failed or did not retain an exit status")
        if process.get("classification") == "infrastructure_exclusion":
            raise AdmissionError("process runner reported infrastructure exclusion")
        if record["postflight"]["classification"] != "synthetic_pass":
            raise AdmissionError("postflight integrity checks failed")
        if record["error"] is None:
            record["ok"] = True
            record["classification"] = "engineering_execution"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        _record_error(record, str(exc))
    finally:
        if profile_hashes is not None:
            from aisle.harness.frontend_conformance import read_retained_profile

            # Preserve write-time identities even if finalization finds changed bytes.
            record["artifacts"].update(
                {"conformance/" + name: digest for name, digest in profile_hashes.items()}
            )
            try:
                read_retained_profile(
                    output / "conformance",
                    profile_launch["conformance_profile"],
                    candidate=manifest,
                    launch=profile_launch,
                    arm=arm,
                    execution=current,
                )
                record["conformance_profile"]["status"] = "bindings_verified"
            except (OSError, ValueError) as exc:
                _record_error(record, f"retained conformance profile failed: {exc}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        if manifest is not None and current.get("ambient_bindings") is not None:
            fields = (
                ("HOME", "home_baseline_sha256"),
                ("XDG_CONFIG_HOME", "config_baseline_sha256"),
                ("XDG_CACHE_HOME", "cache_baseline_sha256"),
            )
            record["private_state"]["initial"] = {
                field: manifest["state"][field] for _, field in fields
            }
            try:
                environment = current["ambient_bindings"][arm]["environment"]
                record["private_state"]["final"] = {
                    field: _private_tree_hash(Path(environment[variable]))
                    for variable, field in fields
                }
            except (OSError, ValueError) as exc:
                record["private_state"]["error"] = str(exc)
                _record_error(record, f"private state postflight failed: {exc}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        record["lifecycle"]["finished_at"] = now()
        for name in (
            "admission.json",
            "hidden-access-log.json",
            "session.jsonl",
            "session.stderr",
            "token_samples.jsonl",
            "frontend-live.json",
            "session-record.json",
            "launch.json",
            "launch-profile.sb",
            "capability.json",
            "tool-events.jsonl",
            "tool-service.json",
            "tool-request-index.jsonl",
            "frontend-authority-reference.json",
            "frontend-protocol-reference.json",
            "frontend-dispatch-reference.json",
            "code-mode-reference.json",
            "provider-reference.json",
            "relay-reservation.json",
            "mcp-harness-reference.json",
            "code-mode/host.stdout",
            "code-mode/host.stderr",
        ):
            path = output / name
            try:
                if path.is_symlink():
                    raise AdmissionError("retained artifact is a symlink")
                if path.is_file():
                    record["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, AdmissionError) as exc:
                _record_error(record, f"artifact hash failed for {name}: {exc}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        authority_evidence = None
        if request_authority_evidence is not None:
            try:
                authority_evidence = request_authority_evidence()
                if authority_evidence is not None:
                    snapshots = {"frontend-authority": authority_evidence["artifacts"]}
                    if "protocol" in authority_evidence:
                        snapshots["frontend-protocol"] = authority_evidence["protocol"]["artifacts"]
                    if "dispatch" in authority_evidence:
                        snapshots["frontend-dispatch"] = authority_evidence["dispatch"]["artifacts"]
                    if "code_mode" in authority_evidence:
                        snapshots["code-mode/rpc"] = authority_evidence["code_mode"]["artifacts"]
                    if "provider" in authority_evidence:
                        snapshots["provider"] = authority_evidence["provider"]["artifacts"]
                    if "mcp_harness" in authority_evidence:
                        snapshots["mcp-harness"] = authority_evidence["mcp_harness"]["artifacts"]
                    for directory, snapshot in snapshots.items():
                        for name, data in snapshot.items():
                            if Path(name).name != name:
                                raise ValueError("invalid authority evidence name")
                            path = output / directory / name
                            if path.resolve() != path:
                                raise ValueError("redirected authority evidence")
                            with path.open("rb") as stream:
                                if stream.read(len(data) + 1) != data:
                                    raise ValueError(
                                        "authority evidence changed during finalization"
                                    )
                            record["artifacts"][f"{directory}/{name}"] = hashlib.sha256(
                                data
                            ).hexdigest()
            except Exception as exc:
                _record_error(record, f"request authority evidence acquisition failed: {exc}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        if "tool-events.jsonl" in record["artifacts"]:
            audit = audit_tool_journal(
                output,
                session_id=session_id,
                plan_id=record["plan_id"],
                arm=arm,
                development=plan.get("development"),
                request_authority=authority_evidence,
                require_frontend_source=(
                    manifest is not None
                    and "app_server" in current.get("launch_bindings", {}).get(arm, {})
                ),
                frontend_dispatch_ceiling=(
                    manifest["budget"].get("frontend_tool_ceiling")
                    if manifest is not None
                    and "app_server" in current.get("launch_bindings", {}).get(arm, {})
                    else None
                ),
            )
            record["tool_audit"] = audit
            record["artifacts"].update(audit["files"])
            try:
                path = output / "tool-audit.json"
                with path.open("x") as stream:
                    json.dump(audit, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                record["artifacts"]["tool-audit.json"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            except OSError as exc:
                _record_error(record, f"tool journal audit retention failed: {exc}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
            if not audit["ok"]:
                _record_error(record, f"tool journal audit failed: {audit['error']}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        if manifest is not None and "frontend_tool_ceiling" in manifest["budget"]:
            if not {"session.jsonl", "frontend-live.json"}.issubset(record["artifacts"]):
                _record_error(record, "live frontend budget evidence is missing")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        if manifest is not None and "session.jsonl" in record["artifacts"]:
            from aisle.harness.matched_frontend import observe_tools, verify_live_report

            try:
                observer_kind = (
                    "codex_app_server"
                    if "app_server" in current.get("launch_bindings", {}).get(arm, {})
                    else manifest["agent"]["kind"]
                )
                data = (output / "session.jsonl").read_bytes()
                if hashlib.sha256(data).hexdigest() != record["artifacts"]["session.jsonl"]:
                    raise AdmissionError("frontend transcript changed during finalization")
                if "frontend_tool_ceiling" in manifest["budget"]:
                    live_bytes = (output / "frontend-live.json").read_bytes()
                    if hashlib.sha256(live_bytes).hexdigest() != record["artifacts"].get(
                        "frontend-live.json"
                    ):
                        raise AdmissionError("live frontend report changed during finalization")
                    process = record["process"] if isinstance(record["process"], dict) else {}
                    verify_live_report(
                        observer_kind,
                        manifest["budget"]["frontend_tool_ceiling"],
                        data.decode().splitlines(),
                        json.loads(live_bytes),
                        process.get("stopped"),
                        admission_controlled=(
                            observer_kind == "codex_app_server"
                            and "provider" in current.get("launch_bindings", {}).get(arm, {})
                        ),
                    )
                observation = observe_tools(observer_kind, data.decode().splitlines())
                observation["transcript_sha256"] = record["artifacts"]["session.jsonl"]
                if not observation["ok"]:
                    _record_error(record, f"frontend observation failed: {observation['error']}")
                    record["ok"] = False
                    record["classification"] = "infrastructure_exclusion"
                if (
                    not isinstance(record["process"], dict)
                    or record["process"].get("stream_complete") is not True
                ):
                    observation.update(
                        ok=False,
                        observed_calls=None,
                        completed_calls=None,
                        error="; ".join(
                            filter(
                                None,
                                [
                                    observation["error"],
                                    "frontend stream completeness is unverified",
                                ],
                            )
                        ),
                    )
                record["frontend_tools"] = observation
                path = output / "frontend-tools.json"
                with path.open("x") as stream:
                    json.dump(observation, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                record["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError) as exc:
                _record_error(record, f"frontend observation failed: {exc}")
                record["ok"] = False
                record["classification"] = "infrastructure_exclusion"
        record["common_evidence"] = common_envelope(record, manifest, validation)
        record["immutable_id"] = _digest(record)
        with (output / "matched-session.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return record
