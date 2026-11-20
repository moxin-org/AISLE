"""Confined worker and native Dora joined by a private bounded task channel."""

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/unit"))

pytestmark = [
    pytest.mark.accept,
    pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS confinement"),
]

DRIVER = """
import json, os
import pyarrow as pa
from dora import Node
node=Node(); stage=-1; seen=set()
def start(stage):
    meta=dict(env_id=0,seq=stage+1,turn_epoch=7,turn_id=stage,sim_time_ns=stage*10000000)
    if stage==0:
        node.send_output("reset_done",pa.array([1],type=pa.uint32()),dict(meta,seed=300))
        msg=dict(kind="request",med="a",text="a?")
        node.send_output("human_msg",pa.array([json.dumps(msg)]),dict(meta,goal_id="g"))
        names=["human_msg","reset_done"]
    elif stage==1:
        msg=dict(kind="confirm_reply",med="a",text="yes")
        node.send_output("human_msg",pa.array([json.dumps(msg)]),dict(meta,goal_id="g"))
        names=["human_msg"]
    else:
        node.send_output("plan_done",pa.array(['{"ok": true}']),dict(meta,**identity))
        names=["plan_done"]
    node.send_output("turn",pa.array([stage],type=pa.uint64()),
                     dict(meta,target_node="task",expected_inputs=names,
                          expected_counts=[1]*len(names)))
for event in node:
    if event["type"]!="INPUT": continue
    topic=event["id"]
    if topic=="tick" and stage==-1:
        stage=0;start(stage)
    elif topic in ("robot_msg","target_request","turn_done"):
        seen.add(topic)
        if topic=="target_request":
            identity={k:event["metadata"][k] for k in ("goal_id","request_id")}
        expected={"turn_done"}|[{"robot_msg"},{"target_request"},set()][stage]
        if seen==expected:
            if os.environ["FIXTURE_BLOCK"]=="1": continue
            if stage==2: break
            seen=set();stage+=1;start(stage)
"""


@pytest.mark.parametrize("arm,blocked", [("history", False), ("trace", False), ("trace", True)])
def test_confined_worker_native_host_and_blocked_callback_cleanup(tmp_path, arm, blocked):
    """TRT-5/6/7/9/10; TC-2/6/7: actual confined worker remains owned across blocked Dora RPC."""
    if shutil.which("dora") is None:
        pytest.skip("Dora CLI required")
    pytest.importorskip("dora")
    from p3_evidence import digest, export_policy
    from p3_isolated_mock import launch_client
    from p3_task_mock import WORKER_SOURCE
    from test_p3_evidence import run_fixture

    from aisle.harness.matched_runtime import capture_runtime
    from aisle.harness.p3_dora_host import TaskLink, source_hashes
    from aisle.harness.p3_task_boundary import TaskSession
    from aisle.harness.reaper import reap_orphans

    base = tmp_path.resolve()
    bundle = base / "bundle"
    export_policy(run_fixture(base), bundle)
    controller = base / "controller"
    graph_started = []
    with tempfile.TemporaryDirectory(prefix="p3-host-", dir="/tmp") as directory:
        socket_root = Path(directory).resolve()
        socket_path = socket_root / "s"
        probe = (
            f"import socket\ns=socket.socket(socket.AF_UNIX)\n"
            f"try:\n s.connect({str(socket_path)!r})\n"
            "except PermissionError:\n pass\nelse:\n raise RuntimeError('private IPC exposed')\n"
            "finally:\n s.close()\n"
        ).encode()
        source = probe + WORKER_SOURCE

        @contextmanager
        def factory(evidence, output, source_hash):
            # This hook must run after actual confinement admission, before graph startup.
            assert json.loads((controller / "attestation.json").read_text())["capability_pass"]
            seconds = 2 if blocked else 20
            with TaskLink(
                socket_path,
                session_id=evidence.session_id,
                source_sha256=source_hash,
                token="b" * 64,
                timeout_s=seconds,
            ) as link:
                config = dict(
                    version=1,
                    session_id=evidence.session_id,
                    source_sha256=source_hash,
                    token="b" * 64,
                    socket=str(socket_path),
                    timeout_s=seconds,
                    turn_epoch=7,
                    source_hashes=source_hashes(),
                    output=str(controller / "host-result.json"),
                )
                path = controller / "host.json"
                raw = json.dumps(config).encode()
                path.write_bytes(raw)
                driver = controller / "driver.py"
                driver.write_text(DRIVER)

                def q(source):
                    return dict(source=source, queue_size=100)

                graph = controller / "graph.yaml"
                graph.write_text(
                    yaml.safe_dump(
                        dict(
                            nodes=[
                                dict(
                                    id="driver",
                                    path=sys.executable,
                                    args=shlex.join([str(driver)]),
                                    env=dict(FIXTURE_BLOCK="1" if blocked else "0"),
                                    inputs={
                                        "tick": "dora/timer/millis/100",
                                        **{
                                            k: q(f"task/{k}")
                                            for k in ("robot_msg", "target_request", "turn_done")
                                        },
                                    },
                                    outputs=["reset_done", "human_msg", "plan_done", "turn"],
                                ),
                                dict(
                                    id="task",
                                    path=sys.executable,
                                    args=shlex.join(
                                        [
                                            "-m",
                                            "aisle.harness.p3_dora_host",
                                            "--config",
                                            str(path),
                                            "--config-sha256",
                                            hashlib.sha256(raw).hexdigest(),
                                        ]
                                    ),
                                    env=dict(
                                        PYTHONPATH=str(ROOT / "src"),
                                        AISLE_LOCKSTEP="1",
                                        AISLE_TURN_NODE="task",
                                        AISLE_TURN_OUTPUTS="robot_msg,target_request,return_request,turn_done",
                                    ),
                                    inputs={
                                        k: q(f"driver/{k}")
                                        for k in ("reset_done", "human_msg", "plan_done", "turn")
                                    },
                                    outputs=[
                                        "robot_msg",
                                        "target_request",
                                        "return_request",
                                        "turn_done",
                                    ],
                                ),
                            ]
                        )
                    )
                )
                with (controller / "dora.log").open("w") as log:
                    process = subprocess.Popen(
                        ["dora", "run", str(graph)],
                        cwd=controller,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    graph_started.append(process.pid)
                    try:
                        link.accept()
                        with TaskSession(
                            evidence=evidence,
                            transport=link,
                            output=output,
                            source_sha256=source_hash,
                            medicines=["a"],
                        ) as session:
                            yield session
                        assert process.wait(timeout=5) == 0, (controller / "dora.log").read_text()
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                        patterns = (str(driver), "aisle.harness.p3_dora_host")
                        reap_orphans(controller, patterns=patterns)
                        remaining = []
                        for pattern in patterns:
                            found = subprocess.run(
                                ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
                            )
                            for pid in found.stdout.split():
                                cwd = subprocess.run(
                                    ["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                )
                                if f"n{controller}" in cwd.stdout.splitlines():
                                    remaining.append(int(pid))
                        (controller / "graph-cleanup.json").write_text(
                            json.dumps(
                                dict(
                                    pid=process.pid,
                                    returncode=process.returncode,
                                    remaining=remaining,
                                )
                            )
                        )
                        assert not remaining

        result = launch_client(
            source=source,
            workspace=base / "workspace",
            controller=controller,
            bundle=bundle,
            manifest_sha256=digest(bundle / "manifest.json"),
            arm=arm,
            session_id="native-worker",
            runtime=capture_runtime([Path(sys.base_prefix).resolve()]),
            hidden_roots=(ROOT, socket_root),
            timeout_s=30,
            service_factory=factory,
        )
    assert len(graph_started) == 1
    assert result["ok"] is (not blocked), result
    rpc = json.loads((controller / "rpc/result.json").read_text())
    assert rpc["returncode"] is not None
    summary = json.loads((controller / "task/summary.json").read_text())
    assert summary["ok"] is (not blocked)
    assert summary["input_exhausted"] is (not blocked)
    assert result["provider_calls"] == result["measured_sessions"] == 0
    if blocked:
        assert rpc["elapsed_s"] < 8 and rpc["error"]
    else:
        assert result["confinement_verified"]
        assert summary["motion_requests"] == 1
        assert json.loads((controller / "host-result.json").read_text())["ok"]
