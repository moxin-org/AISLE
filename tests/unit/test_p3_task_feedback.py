"""Controller-owned request feedback; independent oracle scoring remains private."""

import json

import pytest
from test_p3_task_boundary import Transport, act, confirm, meta, opened, session

from aisle.harness.p3_task_boundary import TaskRefusal

pytestmark = pytest.mark.unit


def returned(boundary, command):
    boundary.observe(
        "plan_done",
        {"ok": True},
        meta(command["metadata"]["goal_id"], command["metadata"]["request_id"]),
    )


def test_dialogue_corrections_and_request_retries_are_separate():
    """TC-7; HAR-3: corrections never consume a retry and executor return is not success."""
    boundary = opened()
    assert boundary.feedback()["payload"]["phase"] == "confirming"
    correction = meta()
    boundary.observe("human_msg", dict(kind="correction", med="b", text="actually b"), correction)
    assert (
        boundary.observe(
            "human_msg", dict(kind="correction", med="b", text="actually b"), correction
        )
        is None
    )
    confirm(boundary, med="b")
    first = act(boundary, "deliver", "b")
    feedback = boundary.feedback()
    assert feedback["topic"] == "episode_feedback"
    assert feedback["metadata"] == {"goal_id": "initial"}
    assert feedback["payload"]["retries"] == 0
    assert feedback["payload"]["dialogue_corrections"] == 1
    assert feedback["payload"]["phase"] == "delivering"
    returned(boundary, first)
    assert boundary.feedback()["payload"]["phase"] == "awaiting_motion"
    act(boundary, "deliver", "a")
    assert boundary.feedback()["payload"]["retries"] == 1
    assert "success" not in json.dumps(boundary.feedback())


def test_recovery_return_and_redelivery_are_not_retries():
    """TC-7; HAR-3: a new public goal owns new counters while previous motion stays active."""
    boundary = opened()
    confirm(boundary)
    first = act(boundary, "deliver")
    boundary.observe(
        "human_msg",
        dict(kind="request", med="b", return_med="a", text="return a, bring b"),
        meta("recovery"),
    )
    confirm(boundary, "recovery", "b")
    assert boundary.feedback()["payload"]["phase"] == "waiting_previous"
    returned(boundary, first)
    back = act(boundary, "return")
    returned(boundary, back)
    act(boundary, "deliver", "b")
    assert boundary.feedback()["payload"]["retries"] == 0
    assert boundary.feedback()["payload"]["dialogue_corrections"] == 0


def test_retry_cap_is_shared_by_repeated_motion_kinds():
    """HAR-3: at most eight repeated requests per goal; changing medicine cannot reset the cap."""
    boundary = opened()
    boundary.max_motion_requests = 100
    confirm(boundary)
    for kind in ("deliver", "return"):
        command = act(boundary, kind)
        returned(boundary, command)
    for index in range(8):
        command = act(boundary, "deliver" if index % 2 else "return", "a" if index % 2 else "b")
        returned(boundary, command)
    assert boundary.feedback()["payload"]["retries"] == 8
    for kind in ("return", "deliver"):
        with pytest.raises(TaskRefusal, match="retry"):
            act(boundary, kind)
    assert boundary.motion_requests == 10


def test_feedback_clock_and_reset_are_public_lifecycle_based():
    """CON-5; TC-6/7: elapsed feedback follows accepted simulation stamps and reset clears goals."""
    boundary = opened()
    start = boundary.feedback()["payload"]["t"]
    assert start == 0
    boundary.observe("tick", None, {"env_id": 0, "seq": 1, "sim_time_ns": 2_000_000_000})
    assert 1.9 < boundary.feedback()["payload"]["t"] <= 2
    boundary.observe("reset_done", [1], dict(env_id=0, seq=100, sim_time_ns=3_000_000_000))
    assert boundary.feedback() is None


def test_session_publishes_feedback_without_worker_authority(tmp_path):
    """TC-7; TRT-5/10: retained feedback is controller-owned; authored feedback is refused."""
    wire = Transport(
        [
            ("reset_done", [1], meta(seq=0)),
            ("human_msg", dict(kind="request", med="a", text="bring a"), meta()),
            ("tick", None, dict(env_id=0, seq=1000, sim_time_ns=1_000_000_000)),
        ]
    )
    with session(tmp_path, wire) as service:
        for _ in range(3):
            service.call(dict(operation="next", arguments={}))
        refusal = service.call(
            dict(
                operation="act",
                arguments=dict(
                    revision=service.boundary.revision, kind="episode_feedback", med="a"
                ),
            )
        )
        assert not refusal["ok"]
        service.call(dict(operation="next", arguments={}))
    assert [r[0] for r in wire.sent] == ["episode_feedback", "episode_feedback"]
    events = [
        json.loads(line) for line in (tmp_path / "task/events.jsonl").read_text().splitlines()
    ]
    assert len([r for r in events if r["kind"] == "feedback_dispatched"]) == 2


def test_failed_feedback_dispatch_closes_task_service(tmp_path):
    """HAR-4; TRT-9/10: missing feedback is a retained transport failure, never silently omitted."""

    class Broken(Transport):
        def send(self, *args):
            raise OSError("feedback transport failed")

    wire = Broken(
        [
            ("reset_done", [1], meta(seq=0)),
            ("human_msg", dict(kind="request", med="a", text="a"), meta()),
        ]
    )
    with pytest.raises(OSError, match="feedback"):
        with session(tmp_path, wire) as service:
            service.call(dict(operation="next", arguments={}))
            service.call(dict(operation="next", arguments={}))
    assert not json.loads((tmp_path / "task/summary.json").read_text())["ok"]
