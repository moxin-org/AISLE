"""Real dora T4/executor transport with synthetic one-stage plans; no physics."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.accept
ROOT = Path(__file__).resolve().parents[2]


def test_correlated_completion_serializes_three_motion_requests(tmp_path, dataflow):
    """TC-2/7; HAR-3: actual node loops order initial, return and redelivery."""
    if shutil.which("dora") is None:
        pytest.skip("dora CLI required")
    result = tmp_path / "completed.json"
    executor = tmp_path / "executor.py"
    executor.write_text("""
from types import SimpleNamespace
import numpy as np
from aisle.nodes import ik_trajectory as ik
from aisle.scenes.pharmacy import load_physics
home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
ik.StagedPlan = lambda *a, **k: SimpleNamespace(
    ok=True, stages=[ik.Stage("synthetic-transport-fixture", (home[:7],), 0., .4)])
ik.main()
""")
    driver = tmp_path / "driver.py"
    driver.write_text("""
import json, os
import numpy as np
import pyarrow as pa
from dora import Node
from aisle.scenes.pharmacy import load_physics
node = Node()
home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
pose = np.array([.4, 0., .3, 0., 0., 0., 1.], dtype=np.float32)
tick = 0
completed = []
recovery_due = False
def send(topic, value, **extra):
    metadata = dict(sim_time_ns=tick*10_000_000, env_id=0, seq=tick, **extra)
    node.send_output(topic, pa.array([json.dumps(value)]), metadata)
for event in node:
    if event["type"] != "INPUT":
        continue
    metadata = event.get("metadata") or {}
    topic = event["id"]
    if topic == "tick":
        tick += 1
        if tick == 1:
            send("human_msg", dict(kind="request", med="metformin"), goal_id="initial")
        if recovery_due:
            send("human_msg", dict(kind="request", med="amoxicillin", return_med="metformin"),
                 goal_id="recovery")
            recovery_due = False
        node.send_output("joint_state", pa.array(home), dict(
            sim_time_ns=tick*10_000_000, env_id=0, seq=tick,
            goal_id="sensor", request_id="noise"))
        if tick >= 500:
            break
    elif topic == "robot_msg":
        msg = json.loads(event["value"][0].as_py())
        send("human_msg", dict(kind="confirm_reply", med=msg["med"]), goal_id=metadata["goal_id"])
    elif topic in ("target_request", "return_request"):
        port = "grasp_pose" if topic == "target_request" else "return_pose"
        node.send_output(port, pa.array(pose), dict(metadata))
        if metadata["goal_id"] == "initial":
            send("episode_result", dict(status="success", failure=None, t_end=0., seed=0,
                                        goal_id="initial", verifier="oracle"), goal_id="initial")
            recovery_due = True
    elif topic == "plan_done":
        keys = ("goal_id", "request_id", "sim_time_ns")
        completed.append({key:metadata.get(key) for key in keys})
        if (metadata.get("goal_id") == "recovery"
                and "/target_request/" in metadata.get("request_id", "")):
            break
with open(os.environ["FIXTURE_RESULT"], "w") as stream:
    json.dump(completed, stream)
""")
    env = {"PYTHONPATH": str(ROOT / "src"), "AISLE_LOCKSTEP": "0"}
    graph = tmp_path / "graph.yaml"
    graph.write_text(
        yaml.safe_dump(
            dict(
                nodes=[
                    dict(
                        id="driver",
                        path=sys.executable,
                        args=str(driver),
                        env={**env, "FIXTURE_RESULT": str(result)},
                        inputs={
                            "tick": "dora/timer/millis/10",
                            "robot_msg": "tsm/robot_msg",
                            "target_request": "tsm/target_request",
                            "return_request": "tsm/return_request",
                            "plan_done": "ik/plan_done",
                        },
                        outputs=[
                            "human_msg",
                            "episode_result",
                            "joint_state",
                            "grasp_pose",
                            "return_pose",
                        ],
                    ),
                    dict(
                        id="tsm",
                        path=sys.executable,
                        args=str(ROOT / "src/aisle/nodes/task_state_machine.py"),
                        env={**env, "AISLE_TASK_TIER": "T4"},
                        inputs={
                            "human_msg": "driver/human_msg",
                            "episode_result": "driver/episode_result",
                            "plan_done": "ik/plan_done",
                        },
                        outputs=[
                            "robot_msg",
                            "target_request",
                            "return_request",
                            "episode_feedback",
                        ],
                    ),
                    dict(
                        id="ik",
                        path=sys.executable,
                        args=str(executor),
                        env={**env, "AISLE_TURN_OUTPUTS": "joint_cmd,gripper_cmd,plan_done"},
                        inputs={
                            "grasp_pose": "driver/grasp_pose",
                            "grasp_pose_2": "driver/return_pose",
                            "joint_state": "driver/joint_state",
                        },
                        outputs=["joint_cmd", "gripper_cmd", "plan_done"],
                    ),
                ]
            )
        )
    )
    run = dataflow.run(graph, timeout_s=30)
    assert not run.timed_out and run.returncode == 0, run.stderr
    completed = json.loads(result.read_text())
    assert [(e["goal_id"], e["request_id"]) for e in completed] == [
        ("initial", "initial/target_request/1"),
        ("recovery", "recovery/return_request/1"),
        ("recovery", "recovery/target_request/2"),
    ]
    assert all(
        a["sim_time_ns"] < b["sim_time_ns"] for a, b in zip(completed, completed[1:], strict=False)
    )
