"""TC-7: L1 estimates retain their accepted request identity."""

import json

import pyarrow as pa
import pytest

from aisle.nodes import segmented_pose as pose

pytestmark = pytest.mark.unit


def test_l1_node_keeps_accepted_goal_clears_unknown_and_reset(monkeypatch):
    """TC-2/6/7: observed request identity survives frames; no stale ID on later requests."""

    class Session:
        def __init__(self, **kwargs):
            self.pending = False

        def on_target_request(self, request):
            if request["target_med"] == "bad":
                return False
            self.pending = True
            return True

        def on_reset_done(self):
            self.pending = False

        def on_depth(self, stamp, frame):
            if not self.pending:
                return None
            self.pending = False
            return dict(
                pos=[0.0, 0.0, 0.0],
                target_med="med",
                neighbours=[],
                neighbours_refused=0,
                mask_pixels=300,
            )

    def event(port, value, seq, **extra):
        return dict(
            type="INPUT",
            id=port,
            value=pa.array(value),
            metadata=dict(sim_time_ns=seq, seq=seq, env_id=0, **extra),
        )

    events = [
        event("target_request", [json.dumps({"target_med": "med"})], 1, goal_id="g1"),
        event("target_request", [json.dumps({"target_med": "bad"})], 2, goal_id="refused"),
        event("depth_overhead", [1.0], 3, h=1, w=1, goal_id="frame-stale"),
        event("target_request", [json.dumps({"target_med": "med"})], 4),
        event("depth_overhead", [1.0], 5, h=1, w=1, goal_id="frame-stale"),
        event("target_request", [json.dumps({"target_med": "med"})], 6, goal_id="pre-reset"),
        event("reset_done", [1], 7),
        event("target_request", [json.dumps({"target_med": "med"})], 8),
        event("depth_overhead", [1.0], 9, h=1, w=1),
    ]
    sent = []

    class Node:
        def __iter__(self):
            return iter(events)

        def send_output(self, topic, value, metadata):
            sent.append((topic, metadata))

    monkeypatch.setattr("aisle.turn_node.Node", Node)
    monkeypatch.setattr(pose, "L1Session", Session)
    pose.main()
    assert [m.get("goal_id") for _, m in sent] == ["g1", None, None]
    assert [m["sim_time_ns"] for _, m in sent] == [3, 5, 9]
