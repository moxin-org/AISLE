"""Actual IK executor, guard, and dora transport with synthetic observations; no physics."""

import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from aisle.harness.execution_capture import read_capture

pytestmark = pytest.mark.accept
ROOT = Path(__file__).resolve().parents[2]


def test_executor_emission_and_guard_forwarding_keep_identity(tmp_path, dataflow):
    """HAR-4; BG-1; TC-2/7: actual executor sends and guarded sends retain source identity."""
    if shutil.which("dora") is None:
        pytest.skip("dora CLI required")
    pytest.importorskip("imageio")
    driver = tmp_path / "driver.py"
    driver.write_text("""
import numpy as np
import pyarrow as pa
from dora import Node
from aisle.nodes.grasp_topdown import topdown_quat
from aisle.scenes.pharmacy import load_physics
node = Node()
home = load_physics()["embodiment"]["franka"]["home_qpos"]
for index, event in enumerate(node):
    if event["type"] != "INPUT":
        continue
    meta = dict(sim_time_ns=index * 10_000_000, env_id=0, seq=index)
    if index == 0:
        node.send_output("reset_done", pa.array([1]), meta)
    elif index == 1:
        pose = np.array([0.46, -0.1, 0.50, *topdown_quat(0.)], dtype=np.float32)
        node.send_output("grasp_pose", pa.array(pose), {**meta, "goal_id":"recovery"})
    else:
        node.send_output("joint_state", pa.array(np.array(home, dtype=np.float32)), meta)
    if index >= 20:
        break
""")
    traces = tmp_path / "traces"
    traces.mkdir()
    capture = tmp_path / "executor.jsonl"
    inputs = {
        "reset_done": "driver/reset_done",
        "grasp_pose_2": "driver/grasp_pose",
        "joint_state": "driver/joint_state",
    }
    config = dict(
        path=str(capture),
        capture_id="acceptance",
        node_id="ik",
        inputs=inputs,
        skill=dict(name="ik-trajectory", version="fixture-source"),
    )
    outputs = {
        "driver": ["reset_done", "grasp_pose", "joint_state"],
        "ik": ["joint_cmd", "gripper_cmd"],
        "guard": ["joint_cmd_safe", "gripper_cmd_safe", "violation"],
    }
    common_env = {"PYTHONPATH": str(ROOT / "src"), "AISLE_LOCKSTEP": "0"}
    nodes = [
        dict(
            id="driver",
            path=sys.executable,
            args=str(driver),
            env=common_env,
            inputs={"tick": "dora/timer/millis/100"},
            outputs=outputs["driver"],
        ),
        dict(
            id="ik",
            path=sys.executable,
            args=str(ROOT / "src/aisle/nodes/ik_trajectory.py"),
            inputs=inputs,
            outputs=outputs["ik"],
            env={**common_env, "AISLE_EXECUTION_CAPTURE_CONFIG": json.dumps(config)},
        ),
        dict(
            id="guard",
            path=sys.executable,
            args=str(ROOT / "src/aisle/nodes/budget_guard.py"),
            env=common_env,
            inputs={
                "joint_cmd": "ik/joint_cmd",
                "gripper_cmd": "ik/gripper_cmd",
                "joint_state": "driver/joint_state",
                "reset_done": "driver/reset_done",
            },
            outputs=outputs["guard"],
        ),
        dict(
            id="recorder",
            path=sys.executable,
            args=str(ROOT / "src/aisle/harness/trace_recorder.py"),
            env={**common_env, "AISLE_TRACE_DIR": str(traces), "AISLE_FRAME_CAPTURE_PERIOD_S": "0"},
            inputs={
                f"{node}__{topic}": f"{node}/{topic}"
                for node, topics in outputs.items()
                for topic in topics
            },
        ),
    ]
    graph = tmp_path / "dataflow.yaml"
    graph.write_text(yaml.safe_dump(dict(nodes=nodes)))
    completed = dataflow.run(graph, timeout_s=30)
    assert not completed.timed_out, completed.stderr
    assert completed.returncode == 0, completed.stderr
    journal = read_capture(capture)
    commands = [e for e in journal["events"] if e["kind"] == "command_emitted"]
    assert commands and {e["goal_revision"] for e in commands} == {"recovery"}
    identities = [
        json.loads(line) for line in (traces / "identities.jsonl").read_text().splitlines()
    ]
    guarded = [
        r
        for r in identities
        if r["endpoint"] == "guard__joint_cmd_safe" and r["metadata"].get("invocation_id")
    ]
    assert guarded
    emitted = {e["source"]["seq"]: e for e in commands if e["source"]["endpoint"] == "ik/joint_cmd"}
    for row in guarded:
        meta = row["metadata"]
        source = emitted[meta["command_origin_seq"]]
        assert meta["invocation_id"] == source["invocation_id"]
        assert meta["command_origin_endpoint"] == "ik/joint_cmd"
    from aisle.harness.trace_recorder import trace_row_digest

    for endpoint in {"ik__joint_cmd", "guard__joint_cmd_safe"}:
        with pa.memory_map(str(traces / f"{endpoint}.arrow"), "r") as source:
            rows = pa.ipc.open_stream(source).read_all().to_pylist()
        for evidence in (r for r in identities if r["endpoint"] == endpoint):
            assert evidence["trace_row_sha256"] == trace_row_digest(rows[evidence["row"]])
    assert all(e.get("delivery_completion", "unknown") == "unknown" for e in journal["events"])
    from aisle.harness.execution_capture import verify_capture

    verified = verify_capture(
        capture, traces, config, guard_endpoints={"guard/joint_cmd_safe", "guard/gripper_cmd_safe"}
    )
    assert verified["forwarded"]
