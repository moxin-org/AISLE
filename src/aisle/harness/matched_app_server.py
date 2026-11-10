"""Run admitted App Server sessions with controller-owned authorization and evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

from aisle.harness.frontend_app_server import MAX_MESSAGE_BYTES, dynamic_tools, run_app_server
from aisle.harness.frontend_dispatch import DispatchBudget
from aisle.harness.frontend_request_authority import RequestAuthority
from aisle.harness.matched_frontend import FrontendToolBudget
from aisle.harness.matched_tool_service import ToolService, _directory, _write


def _json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def controller_reply(response, *, request_id):
    """Validate a controller-owned response before exposing it on the frontend pipe."""
    try:
        if (
            type(response) is not dict
            or set(response) != {"ok", "classification", "result", "error", "attempt", "request_id"}
            or type(response["ok"]) is not bool
            or response["request_id"] != request_id
            or type(response["attempt"]) is not int
            or response["attempt"] <= 0
            or response["classification"] not in {"tool_result", "infrastructure_exclusion"}
            or response["error"] is not None
            and type(response["error"]) is not str
        ):
            raise ValueError("invalid controller response fields")
        if response["classification"] == "tool_result":
            result = response["result"]
            if (
                type(result) is not dict
                or type(result.get("ok")) is not bool
                or result["ok"] is not response["ok"]
            ):
                raise ValueError("controller response verdict differs from its result")
        elif response["ok"]:
            raise ValueError("controller response promotes an infrastructure failure")
        reply = {
            "success": response["ok"],
            "contentItems": [{"type": "inputText", "text": json.dumps(response, allow_nan=False)}],
        }
        if len(json.dumps(reply, allow_nan=False).encode()) > MAX_MESSAGE_BYTES:
            result = response["result"]
            evidence = result.get("worker_evidence") if type(result) is dict else None
            if (
                type(evidence) is dict
                and evidence.get("schema_version") == "aisle.monolithic-worker-retention.v1"
            ):
                if (
                    type(evidence.get("ok")) is not bool
                    or type(evidence.get("files")) is not dict
                    or type(evidence.get("errors")) is not list
                    or any(type(error) is not str for error in evidence["errors"])
                    or type(evidence.get("worker_evidence_present")) is not bool
                    or evidence.get("eligible_for_estimate") is not False
                ):
                    raise ValueError("invalid controller response worker evidence")
                # The full index remains in the controller's immutable result.
                # Replay derives this reference from those same retained bytes.
                reference = {
                    "schema_version": "aisle.monolithic-worker-reference.v1",
                    "sha256": hashlib.sha256(
                        json.dumps(
                            evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode()
                    ).hexdigest(),
                    "file_count": len(evidence["files"]),
                    **{
                        key: evidence[key]
                        for key in (
                            "ok",
                            "errors",
                            "worker_evidence_present",
                            "eligible_for_estimate",
                        )
                    },
                }
                visible = {**response, "result": {**result, "worker_evidence": reference}}
                reply = {
                    "success": response["ok"],
                    "contentItems": [
                        {"type": "inputText", "text": json.dumps(visible, allow_nan=False)}
                    ],
                }
        if len(json.dumps(reply, allow_nan=False).encode()) > MAX_MESSAGE_BYTES:
            raise ValueError("controller response exceeds frontend frame limit")
        return reply
    except (KeyError, TypeError, RecursionError) as exc:
        raise ValueError("malformed controller response") from exc


def acquire_authority_evidence(output, references):
    """Acquire bounded snapshots against references held by the session controller."""
    if not references or not {"authority", "protocol"}.issubset(references):
        return None

    def snapshot(directory, expected, *, extra=(), frame_limit=65536, byte_limit=4 * 1024 * 1024):
        fd = _directory(directory)
        try:
            names = expected["artifacts"]
            if set(os.listdir(fd)) != set(names) | set(extra):
                raise ValueError("authorization evidence inventory changed")
            result = {}
            total = 0
            for name in names:
                if Path(name).name != name:
                    raise ValueError("invalid authorization artifact name")
                handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(handle, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or info.st_size > frame_limit
                    ):
                        raise ValueError("authorization artifact is not a bounded private file")
                    data = stream.read(frame_limit + 1)
                if len(data) > frame_limit or hashlib.sha256(data).hexdigest() != names[name]:
                    raise ValueError("authorization artifact differs from closed reference")
                total += len(data)
                if total > byte_limit:
                    raise ValueError("authorization evidence exceeds acquisition limit")
                result[name] = data
            return result
        finally:
            os.close(fd)

    authority, protocol = references["authority"], references["protocol"]
    result = {
        "artifacts": snapshot(output / "frontend-authority", authority),
        "expected": authority,
        "byte_limit": 4 * 1024 * 1024,
        "protocol": {
            "artifacts": snapshot(output / "frontend-protocol", protocol, extra=("stderr.log",)),
            "expected": protocol,
            "byte_limit": 4 * 1024 * 1024,
        },
    }
    if "dispatch" in references:
        dispatch_limit = (
            64 * 1024 * 1024 if {"code_mode", "provider"} & references.keys() else 4 * 1024 * 1024
        )
        result["dispatch"] = {
            "artifacts": snapshot(
                output / "frontend-dispatch",
                references["dispatch"],
                frame_limit=16 * 1024 * 1024,
                byte_limit=dispatch_limit,
            ),
            "expected": references["dispatch"],
            "byte_limit": dispatch_limit,
        }
    if "code_mode" in references:
        nested = references["code_mode"]
        if nested["proxy"] is None:
            raise ValueError("nested host lacks a closed proxy reference")
        result["code_mode"] = {
            "artifacts": snapshot(
                output / "code-mode" / "rpc",
                nested["proxy"],
                frame_limit=24 * 1024 * 1024,
                byte_limit=64 * 1024 * 1024,
            ),
            "expected": nested["proxy"],
            "delegated_tools": set(nested["delegated_tools"]),
            "byte_limit": 64 * 1024 * 1024,
        }
        if nested["failure"] is not None:
            # Acquisition preserves closed bytes even when qualification must fail.
            result["code_mode"]["host_failure"] = nested["failure"]
    if "provider" in references:
        result["provider"] = {
            "artifacts": snapshot(
                output / "provider",
                references["provider"],
                frame_limit=16 * 1024 * 1024,
                byte_limit=64 * 1024 * 1024,
            ),
            "expected": references["provider"],
            "byte_limit": 64 * 1024 * 1024,
        }
    if "mcp_harness" in references:
        result["mcp_harness"] = {
            "artifacts": snapshot(
                output / "mcp-harness",
                references["mcp_harness"],
                frame_limit=1024 * 1024,
                byte_limit=64 * 1024 * 1024,
            ),
            "expected": references["mcp_harness"],
            "byte_limit": 64 * 1024 * 1024,
        }
    return result


def run_authorized_app_server(
    controller, command, *, cwd, env, launch, budget, references, reservation=None
):
    """Host the fixed harness request API while the owned frontend process runs."""
    output = controller.output
    manifest = controller.plan["arms"][controller.arm]
    allowed = manifest["policy"]["allowed_external_tools"]
    operations = [op for op in ("check", "run") if "harness." + op in allowed]
    thread = {
        **launch["app_server"],
        "cwd": str(cwd),
        "ephemeral": True,
        "model": manifest["model"]["requested_identity"],
        "approvalPolicy": manifest["policy"]["approval"],
        "sandbox": "danger-full-access",
        "dynamicTools": dynamic_tools(operations),
    }
    guard = (
        FrontendToolBudget(
            "codex_app_server",
            budget["frontend_tool_ceiling"],
            admission_controlled="provider" in launch,
        )
        if "frontend_tool_ceiling" in budget
        else None
    )
    deadline = time.monotonic() + budget["wall_ceiling_s"]
    protocol_root = output / "frontend-protocol"
    primary_error = None
    dispatch = None
    authority = None
    try:
        with (
            RequestAuthority(
                output / "frontend-authority", session_id=controller.session_id
            ) as authority,
            (
                DispatchBudget(
                    output / "frontend-dispatch",
                    session_id=controller.session_id,
                    ceiling=budget["frontend_tool_ceiling"],
                )
                if "frontend_tool_ceiling" in budget
                else nullcontext(None)
            ) as dispatch,
        ):
            with ToolService(controller, request_authority=authority) as service:

                def request(call, source):
                    if call["tool_name"] not in allowed:
                        raise ValueError("frontend requested an unadmitted harness tool")
                    operation = call["tool_name"].split(".")[1]
                    identity = uuid.uuid4().hex
                    request = {
                        "schema_version": "aisle.matched-tool-request.v1",
                        "id": identity,
                        "operation": operation,
                    }
                    raw = json.dumps(request).encode() + b"\n"
                    grant = authority.authorize(call=call, request=raw)
                    fd = _directory(service.channel)
                    try:
                        _write(
                            fd,
                            f"{identity}.authorization.json",
                            {"authorization_id": grant["authorization_id"]},
                        )
                        _write(fd, f"{identity}.request.json", request)
                        while time.monotonic() < deadline:
                            if service.failed.is_set():
                                raise ValueError(
                                    "authorized tool service failed: " + str(service.error)
                                )
                            response = service.controller_response(identity)
                            if response is not None:
                                try:
                                    return controller_reply(response, request_id=identity)
                                except ValueError as exc:
                                    try:
                                        authority.retain_response_rejection(
                                            call=call, request_id=identity, response=response
                                        )
                                    except BaseException as retention_error:
                                        exc.add_note(
                                            "response rejection retention failed: "
                                            + str(retention_error)
                                        )
                                    raise
                            time.sleep(0.01)
                        raise TimeoutError("authorized tool response deadline expired")
                    finally:
                        os.close(fd)

                def handle(call, source):
                    if dispatch is None:
                        return request(call, source)
                    response = None

                    def deliver(frame):
                        nonlocal response
                        response = request(call, frame)

                    dispatch.dispatch(call, source, deliver)
                    return response

                options = dict(
                    cwd=cwd,
                    env=env,
                    output=protocol_root,
                    thread_params=thread,
                    input_items=[
                        {
                            "type": "text",
                            "text": "Perform the assigned research task.",
                            "text_elements": [],
                        }
                    ],
                    handle_call=handle,
                    timeout_s=budget["wall_ceiling_s"],
                    token_ceiling=budget["ceiling"],
                    on_message=guard,
                    on_reference=lambda reference: references.update(protocol=reference),
                )
                if "provider" in launch:
                    from aisle.harness.provider_runner import run_provider_app_server

                    delegated = {("harness", op, "function_call") for op in operations}
                    if launch.get("mcp_harness"):
                        delegated.update(
                            ("mcp__aisle_harness", op, "function_call") for op in operations
                        )
                    if "code_mode_host" in launch:
                        delegated.update(
                            {
                                (None, "exec", "custom_tool_call"),
                                ("functions", "exec", "custom_tool_call"),
                            }
                        )
                    result = run_provider_app_server(
                        binding=launch["provider"],
                        dispatch=dispatch,
                        delegated_tools=delegated,
                        provider_output=output / "provider",
                        references=references,
                        argv=command,
                        code_mode_host=launch.get("code_mode_host"),
                        code_mode_output=output / "code-mode",
                        mcp_harness=launch.get("mcp_harness", False),
                        mcp_output=output / "mcp-harness",
                        reservation=reservation,
                        **options,
                    )
                elif "code_mode_host" in launch:
                    from aisle.harness.code_mode_runner import run_code_mode_app_server

                    async def asynchronous_handle(call, source):
                        return await asyncio.to_thread(handle, call, source)

                    options["handle_call"] = asynchronous_handle
                    options["protocol_output"] = options.pop("output")
                    result = asyncio.run(
                        run_code_mode_app_server(
                            host=launch["code_mode_host"],
                            dispatch=dispatch,
                            delegated_tools={"harness." + op for op in operations},
                            output=output / "code-mode",
                            references=references,
                            argv=command,
                            **options,
                        )
                    )
                else:
                    result = run_app_server(command, **options)
            if not service.report["ok"]:
                raise ValueError("authorized tool service did not finish cleanly")
        references.setdefault(
            "protocol",
            {key: result[key] for key in ("thread_id", "turn_id", "dynamic_calls", "artifacts")},
        )
        if "mcp_calls" in result:
            references["protocol"]["mcp_calls"] = result["mcp_calls"]
    except BaseException as exc:
        primary_error = exc
        raise
    finally:

        def retain_authority_reference():
            if authority is not None:
                references["authority"] = authority.reference()
                _json(output / "frontend-authority-reference.json", references["authority"])

        def retain_protocol_reference():
            if "protocol" in references:
                _json(output / "frontend-protocol-reference.json", references["protocol"])

        def retain_dispatch():
            if dispatch is not None:
                references["dispatch"] = dispatch.reference()
                _json(output / "frontend-dispatch-reference.json", references["dispatch"])

        def retain_protocol():
            if protocol_root.is_dir():
                with (output / "session.jsonl").open("xb") as transcript:
                    for path in sorted(protocol_root.glob("*-received.json")):
                        transcript.write(path.read_bytes())
                stderr = protocol_root / "stderr.log"
                if stderr.is_file():
                    (output / "session.stderr").write_bytes(stderr.read_bytes())

        def retain_observations():
            if guard is not None:
                _json(output / "frontend-live.json", guard.report())

        def retain_code_mode():
            if "code_mode" in references:
                _json(output / "code-mode-reference.json", references["code_mode"])

        def retain_provider():
            if "provider" in references:
                _json(output / "provider-reference.json", references["provider"])

        def retain_mcp():
            if "mcp_harness" in references:
                _json(output / "mcp-harness-reference.json", references["mcp_harness"])

        failure = primary_error
        for retain in (
            retain_authority_reference,
            retain_protocol_reference,
            retain_dispatch,
            retain_protocol,
            retain_observations,
            retain_code_mode,
            retain_provider,
            retain_mcp,
        ):
            try:
                retain()
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                else:
                    failure.add_note(f"frontend finalization failed: {cleanup_error}")
        if primary_error is None and failure is not None:
            raise failure
    _json(
        output / "token_samples.jsonl",
        {
            "tokens": result["tokens"],
            "tokens_generated": result["tokens_generated"],
            "wall_s": result["wall_s"],
        },
    )
    process = {
        key: result[key]
        for key in ("rc", "tokens", "tokens_generated", "wall_s", "stopped", "stream_complete")
    }
    process["artifacts"] = {
        name: "sha256:" + hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("session.jsonl", "session.stderr", "token_samples.jsonl")
    }
    _json(output / "session-record.json", process)
    return process
