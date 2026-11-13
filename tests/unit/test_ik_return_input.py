"""ADR-32 recovery plans must reach the same guarded executor as normal grasps."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from aisle.nodes import ik_trajectory as ik
from aisle.scenes.pharmacy import load_physics

pytestmark = pytest.mark.unit


class Node:
    def __init__(self, events):
        self.events = events
        self.sent = []

    def __iter__(self):
        return iter(self.events)

    def send_output(self, topic, value, metadata):
        self.sent.append((topic, value.to_pylist(), metadata))


def event(port, stamp, values, **metadata):
    return dict(
        type="INPUT",
        id=port,
        value=pa.array(values),
        metadata=dict(sim_time_ns=stamp, env_id=0, seq=stamp, **metadata),
    )


@pytest.mark.parametrize("port", ["grasp_pose", "grasp_pose_2", "grasp_pose_10"])
def test_indexed_grasp_input_emits_motion_and_completion(monkeypatch, port):
    """TC-2; HAR-3; ADR-32: indexed recovery inputs execute, retaining destination redirect."""
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    node = Node(
        [
            event("reset_done", 0, [1]),
            event(port, 1, [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], place_xy=[0.25, 0.1]),
            event("joint_state", 2, home),
        ]
    )
    calls = []

    def plan(grasp, destination, *args, **kwargs):
        calls.append(destination)
        return SimpleNamespace(ok=True, stages=[ik.Stage("fixture", (home[:7],), 0.0, 0.0)])

    monkeypatch.setattr("aisle.turn_node.Node", lambda: node)
    monkeypatch.setattr(ik, "StagedPlan", plan)
    monkeypatch.setenv("AISLE_TURN_OUTPUTS", "joint_cmd,gripper_cmd,plan_done,turn_done")
    ik.main()
    assert calls == [(0.25, 0.1)]
    assert [topic for topic, _, _ in node.sent].count("joint_cmd") == 1
    assert [topic for topic, _, _ in node.sent].count("plan_done") == 1
    assert all(meta["sim_time_ns"] == 2 for _, _, meta in node.sent)


def test_return_input_does_not_preempt_an_active_grasp(monkeypatch):
    """ADR-32; HAR-3: adding the return port preserves one-plan-at-a-time execution."""
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    pose = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    node = Node(
        [
            event("grasp_pose", 1, pose),
            event("grasp_pose_2", 2, pose),
            event("joint_state", 3, home),
        ]
    )
    calls = []

    def plan(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(ok=True, stages=[ik.Stage("fixture", (home[:7],), 0.0, 0.0)])

    monkeypatch.setattr("aisle.turn_node.Node", lambda: node)
    monkeypatch.setattr(ik, "StagedPlan", plan)
    ik.main()
    assert len(calls) == 1


@pytest.mark.parametrize("reset", [False, True])
def test_plan_done_echoes_accepted_request_with_current_stamp(monkeypatch, reset):
    """TC-2/7: rejected or reset requests cannot supply completion identity."""
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    pose = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    events = [
        event("grasp_pose", 1, pose, goal_id="accepted", request_id="delivery"),
        event("grasp_pose_2", 2, pose, goal_id="rejected", request_id="return"),
    ]
    if reset:
        events.extend(
            [
                event("reset_done", 3, [1]),
                event("grasp_pose", 4, pose, goal_id="next", request_id="new-delivery"),
            ]
        )
    events.append(event("joint_state", 5, home, goal_id="sensor", request_id="unrelated"))
    node = Node(events)
    monkeypatch.setattr("aisle.turn_node.Node", lambda: node)
    monkeypatch.setattr(
        ik,
        "StagedPlan",
        lambda *a, **k: SimpleNamespace(
            ok=True, stages=[ik.Stage("fixture", (home[:7],), 0.0, 0.0)]
        ),
    )
    monkeypatch.setenv("AISLE_TURN_OUTPUTS", "joint_cmd,gripper_cmd,plan_done,turn_done")
    ik.main()
    completions = [metadata for topic, _, metadata in node.sent if topic == "plan_done"]
    assert len(completions) == 1
    assert completions[0]["sim_time_ns"] == 5
    assert completions[0]["goal_id"] == ("next" if reset else "accepted")
    assert completions[0]["request_id"] == ("new-delivery" if reset else "delivery")


@pytest.mark.parametrize("port", ["grasp_pose", "grasp_pose_2"])
def test_real_executor_capture_records_proposal_send_and_return(tmp_path, monkeypatch, port):
    """HAR-4; TC-2/7: executor records actual emitted commands and their causal request."""
    import json

    from aisle.harness.execution_capture import read_capture

    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    node = Node(
        [
            event("reset_done", 0, [1]),
            event(port, 1, [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], goal_id="recovery"),
            event("joint_state", 2, home),
        ]
    )
    import sys

    monkeypatch.setitem(sys.modules, "dora", SimpleNamespace(Node=lambda: node))
    monkeypatch.setattr(
        ik,
        "StagedPlan",
        lambda *a, **k: SimpleNamespace(
            ok=True, stages=[ik.Stage("fixture", (home[:7],), 0.0, 0.0)]
        ),
    )
    capture_path = tmp_path / "execution.jsonl"
    monkeypatch.setenv(
        "AISLE_EXECUTION_CAPTURE_CONFIG",
        json.dumps(
            dict(
                path=str(capture_path),
                capture_id="run",
                node_id="ik",
                skill={"name": "fixture", "version": "sha256:fixture"},
                inputs={
                    "reset_done": "reset/reset_done",
                    port: "planner/grasp_pose",
                    "joint_state": "bridge/joint_state",
                },
            )
        ),
    )
    monkeypatch.setenv("AISLE_TURN_OUTPUTS", "joint_cmd,gripper_cmd,plan_done,turn_done")
    monkeypatch.delenv("AISLE_LOCKSTEP", raising=False)
    ik.main()
    capture = read_capture(capture_path)
    kinds = [e["kind"] for e in capture["events"]]
    assert kinds[:4] == ["reset", "proposed", "accepted", "started"]
    assert kinds[-1] == "returned"
    assert capture["open_invocation"] is None
    command = next(e for e in capture["events"] if e["kind"] == "command_emitted")
    assert command["goal_revision"] == "recovery"
    assert command["observation"]["endpoint"] == "bridge/joint_state"
    raw = next(m for t, v, m in node.sent if t == "joint_cmd")
    assert raw["invocation_id"] == command["invocation_id"]
    assert raw["seq"] == command["source"]["seq"]
