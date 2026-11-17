"""Unscored P3 participant/generated-code boundary checks, without provider calls.

This tool exercises evidence RPC under the existing macOS confinement adapter.
It neither selects a robot action surface nor authorizes measured collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from p3_evidence import digest
from p3_mock_session import EvidenceSession, encoded, write

from aisle.harness.matched_runtime import capture_runtime, verify_runtime, worker_interpreter
from aisle.harness.treatment_ambient import build_declared_environment, spawn_isolated_process
from aisle.harness.treatment_confinement import (
    MacOSPolicy,
    compile_macos_profile,
    run_macos_capability_audit,
    wrap_verified_command,
)
from aisle.monolith.supervisor import WorkerFailure, _DeadlinePipe

MAX_REQUEST = 4096
MAX_RESPONSE = 2 * 1024 * 1024
MAX_SOURCE = 65536


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def receive_frame(stream):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                raise ValueError("partial child frame")
            data.extend(chunk)
        return data

    size = int.from_bytes(exact(4), "big")
    if not 0 < size <= MAX_REQUEST:
        raise ValueError("child frame exceeds budget")
    value = json.loads(exact(size), object_pairs_hook=_object)
    if type(value) is not dict:
        raise ValueError("child frame must be an object")
    encoded(value)  # Reject NaN/Infinity, including nested values.
    return value


def send_frame(stream, value):
    data = encoded(value).encode()
    if len(data) > MAX_RESPONSE:
        raise ValueError("response exceeds budget")
    pending = memoryview(len(data).to_bytes(4, "big") + data)
    while pending:
        count = stream.write(pending)
        if type(count) is not int or count <= 0:
            raise ValueError("child pipe refused response")
        pending = pending[count:]
    stream.flush()


def supervise_client(process, session, output, *, timeout_s):
    """Retain bounded RPC; the caller must separately admit and confine the child."""
    output = Path(output)
    started = time.monotonic()
    result = dict(ok=False, requests=0, provider_calls=0, measured_sessions=0, error=None)
    created = False
    sequence = 0

    def transfer(direction, value=None, *, eof=False):
        nonlocal sequence
        pipe = _DeadlinePipe(
            process.stdin if direction == "controller" else process.stdout,
            deadline,
            time.monotonic,
        )
        complete = False
        try:
            if eof:
                if pipe.read(1):
                    raise ValueError("child emitted trailing bytes")
                response = None
            else:
                response = (
                    send_frame(pipe, value) if direction == "controller" else receive_frame(pipe)
                )
            complete = True
            return response
        finally:
            if not (eof and complete and not pipe.observed):
                sequence += 1
                name = f"{sequence:06d}-{direction}.frame"
                raw = bytes(pipe.observed)
                with (output / name).open("xb") as stream:
                    stream.write(raw)
                with (output / "messages.jsonl").open("a") as stream:
                    stream.write(
                        encoded(
                            dict(
                                file=name,
                                bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest(),
                                complete=complete,
                            )
                        )
                    )

    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        (output / "messages.jsonl").touch(exist_ok=False)
        if (
            type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 300
        ):
            raise ValueError("client timeout must be in (0, 300]")
        deadline = started + timeout_s
        while True:
            message = transfer("child")
            if message == {"kind": "finished"}:
                transfer("child", eof=True)
                if process.wait(timeout=max(0, deadline - time.monotonic())) != 0:
                    raise ValueError("child exit disagrees with completion")
                result["ok"] = True
                break
            if set(message) != {"kind", "request"} or message["kind"] != "call":
                raise ValueError("undeclared child operation")
            result["requests"] += 1
            transfer("controller", session.call(message["request"]))
    except (
        OSError,
        ValueError,
        TypeError,
        RecursionError,
        WorkerFailure,
        subprocess.SubprocessError,
    ) as exc:
        result["error"] = str(exc)
    finally:
        # Kill the owned group even when its leader already exited; descendants
        # must not preserve pipes or session state after the retained terminal event.
        process.poll()  # Reap a zombie leader before signaling its remaining group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        result.update(returncode=process.returncode, elapsed_s=time.monotonic() - started)
        if created:
            write(output / "result.json", result)
    return result


def capture_generated_source(home):
    """Capture after the participant is reaped, refusing links and oversized code."""
    path = Path(home) / "worker.py"
    try:
        if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("generated source must be a canonical regular file")
        with path.open("rb") as stream:
            source = stream.read(MAX_SOURCE + 1)
        if not 0 < len(source) <= MAX_SOURCE:
            raise ValueError("generated source exceeds budget")
        source.decode("utf-8")
        return source
    except OSError as exc:
        raise ValueError("generated source is unavailable") from exc


def launch_client(
    *,
    source,
    workspace,
    controller,
    bundle,
    manifest_sha256,
    arm,
    session_id,
    runtime,
    hidden_roots,
    timeout_s=30,
):
    """Fresh fixed-authority Python process; source bytes never execute in the host."""
    workspace, controller, bundle = (Path(p).absolute() for p in (workspace, controller, bundle))
    if bundle.is_symlink() or digest(bundle / "manifest.json") != manifest_sha256:
        raise ValueError("bound evidence changed")
    if any(p.resolve() != p for p in (workspace, controller, bundle)):
        raise ValueError("launch paths must be canonical")
    if any(
        a.is_relative_to(b)
        for a, b in (
            (workspace, controller),
            (controller, workspace),
            (workspace, bundle),
            (bundle, workspace),
        )
    ):
        raise ValueError("launch authorities overlap")
    if type(source) is not bytes or not 0 < len(source) <= MAX_SOURCE:
        raise ValueError("invalid client source")
    if (
        type(timeout_s) not in (int, float)
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 300
    ):
        raise ValueError("client timeout must be in (0, 300]")
    source.decode("utf-8")
    workspace.mkdir(parents=True, exist_ok=False)
    controller.mkdir(parents=True, exist_ok=False)
    visible, home = workspace / "visible", workspace / "home"
    visible.mkdir()
    script = visible / "client.py"
    script.write_bytes(source)
    script.chmod(0o444)
    source_hash = hashlib.sha256(source).hexdigest()
    environment, environment_record = build_declared_environment(
        home, source_env={"PATH": "/usr/bin:/bin"}
    )
    python = worker_interpreter()
    roots = tuple(Path(p) for p in runtime["trees"])
    if not any(python.is_relative_to(p) for p in roots):
        raise ValueError("interpreter is outside bound runtime")
    policy = MacOSPolicy(
        visible_roots=(visible,),
        output_roots=(home,),
        runtime_read_roots=(*roots, Path("/System"), Path("/usr/lib")),
        allowed_executables=(python,),
        hidden_roots=tuple(dict.fromkeys([*map(Path, hidden_roots), controller, bundle])),
        network_policy="deny-external",
    )
    result = dict(
        ok=False,
        mode="mock_only",
        provider_calls=0,
        measured_sessions=0,
        confinement_verified=False,
    )
    try:
        compiled = compile_macos_profile(policy)
        profile = controller / "profile.sb"
        profile.write_text(compiled.text)
        write(controller / "policy.json", policy.canonical_dict())
        write(controller / "environment.json", environment_record)
        write(controller / "runtime.json", runtime)
        attestation = run_macos_capability_audit(
            policy, sentinel_homes=(visible, home, controller), widen_probes=True
        )
        write(controller / "attestation.json", attestation)
        verify_runtime(runtime)
        if digest(script) != source_hash:
            raise ValueError("client source changed during admission")
        python_hash = digest(python)
        bootstrap = (
            "import resource,runpy;"
            f"resource.setrlimit(resource.RLIMIT_CPU,({math.ceil(timeout_s)},{math.ceil(timeout_s)}));"
            "resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576));"
            "resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
            "resource.setrlimit(resource.RLIMIT_NPROC,(0,0));"
            f"runpy.run_path({str(script)!r},run_name='__main__')"
        )
        command = [str(python), "-I", "-S", "-B", "-c", bootstrap]
        wrapped = wrap_verified_command(command, compiled, profile, attestation, policy=policy)
        write(
            controller / "launch.json",
            dict(
                argv=command,
                source_sha256=source_hash,
                python_sha256=python_hash,
                session_id=session_id,
                arm=arm,
            ),
        )
        with EvidenceSession(
            bundle=bundle,
            manifest_sha256=manifest_sha256,
            arm=arm,
            session_id=session_id,
            output=controller / "evidence",
            max_calls=16,
        ) as service:
            with (controller / "stderr.log").open("xb") as stderr:
                process = spawn_isolated_process(
                    wrapped,
                    cwd=home,
                    environment=environment,
                    environment_record=environment_record,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                )
                result["client"] = supervise_client(
                    process, service, controller / "rpc", timeout_s=timeout_s
                )
            service._verify_bundle()
        verify_runtime(runtime)
        wrap_verified_command(command, compiled, profile, attestation, policy=policy)
        if (
            digest(script) != source_hash
            or digest(python) != python_hash
            or set(visible.iterdir()) != {script}
        ):
            raise ValueError("client/runtime source changed")
        result.update(
            ok=result["client"]["ok"], confinement_verified=True, source_sha256=source_hash
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    finally:
        write(controller / "result.json", result)
    return result


# Exactly the same authored source and protocol in both arms. A generated child
# exercises the same boundary in a new workspace; it does not control a robot.
MOCK_SOURCE = b"""
import json,sys
from pathlib import Path
def exact(size):
    out=bytearray()
    while len(out)<size:
        chunk=sys.stdin.buffer.read(size-len(out))
        if not chunk: raise RuntimeError("controller pipe ended")
        out.extend(chunk)
    return out
def send(value):
    data=json.dumps(value).encode()
    sys.stdout.buffer.write(len(data).to_bytes(4,"big")+data)
    sys.stdout.buffer.flush()
def call(operation,arguments,**extra):
    send({"kind":"call","request":{"operation":operation,"arguments":arguments,**extra}})
    return json.loads(exact(int.from_bytes(exact(4),"big")))
for operation,arguments in [("journal",{"limit":2}),
                            ("raw",{"endpoint":"ik-trajectory/joint_cmd","limit":2}),
                            ("artifacts",{})]:
    result=call(operation,arguments)
    if result.get("ok") is not True: raise RuntimeError("raw evidence refused")
inventory=result["artifacts"]
artifact=next((row["path"] for row in inventory if row["path"].endswith(".npz")),
              inventory[0]["path"])
if not call("artifact",{"path":artifact,"limit":128})["ok"]:
    raise RuntimeError("artifact bytes refused")
if call("history",{},arm="history")["ok"]:
    raise RuntimeError("arm override allowed")
call("history",{})
Path("worker.py").write_bytes(Path(__file__).read_bytes())
send({"kind":"finished"})
"""


def run_mock(bundle, output, *, blocks=1):
    if type(blocks) is not int or not 1 <= blocks <= 6:
        raise ValueError("blocks must be 1..6")
    bundle, output = Path(bundle).resolve(strict=True), Path(output).absolute()
    if output.resolve() != output or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("mock output must be canonical and separate from evidence")
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir()
    runtime = capture_runtime([Path(sys.base_prefix).resolve()])
    manifest_sha256 = digest(bundle / "manifest.json")
    slots = [
        dict(id=f"mock-{block}-{arm}-{role}", block=block, arm=arm, role=role, status="unstarted")
        for block in range(blocks)
        for arm in ("history", "trace")
        for role in ("participant", "generated-worker")
    ]
    write(output / "slots.json", slots)
    raw = {}
    generated = {}
    for slot in slots:
        slot["status"] = "started"
        write(output / "slots.json", slots)
        controller = private / slot["id"]
        try:
            source = (
                MOCK_SOURCE
                if slot["role"] == "participant"
                else generated[(slot["block"], slot["arm"])]
            )
            workspace = output / slot["id"]
            result = launch_client(
                source=source,
                workspace=workspace,
                controller=controller,
                bundle=bundle,
                manifest_sha256=manifest_sha256,
                arm=slot["arm"],
                session_id=slot["id"],
                runtime=runtime,
                hidden_roots=(Path(__file__).resolve().parents[1], private),
            )
            if not result["ok"]:
                raise ValueError("isolated client failed; see retained controller result")
            events = [
                json.loads(line)
                for line in (controller / "evidence/events.jsonl").read_text().splitlines()
            ]
            responses = [row["response"] for row in events if row["kind"] == "response"]
            if (
                len(responses) != 6
                or not all(row["ok"] for row in responses[:4])
                or responses[4]["ok"]
                or responses[-1]["ok"] != (slot["arm"] == "history")
            ):
                raise ValueError("history grant differs from assignment")
            key = (slot["block"], slot["role"])
            if slot["arm"] == "history":
                raw[key] = responses[:-1]
            elif responses[:-1] != raw[key]:
                raise ValueError("paired raw evidence differs")
            if slot["role"] == "participant":
                generated[(slot["block"], slot["arm"])] = capture_generated_source(
                    workspace / "home"
                )
                if generated[(slot["block"], slot["arm"])] != MOCK_SOURCE:
                    raise ValueError("mock generated source differs")
            slot.update(status="mock_checks_passed", source_sha256=result["source_sha256"])
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            slot.update(status="mock_failure", error=str(exc))
            break
        finally:
            write(output / "slots.json", slots)
    report = dict(
        ok=all(s["status"] == "mock_checks_passed" for s in slots),
        mode="isolated_mock_only",
        slots=slots,
        measured_sessions=0,
        provider_calls=0,
        provider_spend_dollars=0,
        hard_dollar_ceiling=200,
        api_invocation_authorized=False,
        p3_complete=False,
        p4_on_hold=True,
        manifest_sha256=manifest_sha256,
        launcher_sha256=digest(__file__),
        limitation=(
            "Synthetic evidence clients and generated-code boundary only; no robot policy, "
            "qualified action surface, session acceptance, provider or measured treatment effect."
        ),
    )
    write(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = run_mock(args.bundle, args.out, blocks=args.blocks)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        result = dict(ok=False, error=str(exc), provider_calls=0, mode="mock_only")
    print(encoded(result), end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
