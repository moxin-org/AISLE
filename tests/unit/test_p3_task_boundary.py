"""P3 T4 public dialogue and fixed-library authority, without simulation."""

import json
from itertools import count

import pytest

from aisle.harness.p3_task_boundary import TaskBoundary, TaskRefusal, TaskSession

pytestmark = pytest.mark.unit


SEQUENCES = count(1)
TIMES = count(1)


def meta(goal="initial", request=None, seq=None):
    seq = next(SEQUENCES) if seq is None else seq
    result = dict(goal_id=goal, env_id=0, sim_time_ns=next(TIMES), seq=seq, seed="HIDDEN")
    if request is not None:
        result["request_id"] = request
    return result


def opened():
    boundary = TaskBoundary(binding_id="unit-worker", medicines=["a", "b"], max_motion_requests=4)
    boundary.observe("reset_done", [1], meta(seq=0))
    boundary.observe(
        "human_msg", dict(kind="request", med="a", text="bring a", seed="HIDDEN"), meta()
    )
    return boundary


def act(boundary, kind, med="a", **extra):
    return boundary.action(
        dict(revision=boundary.snapshot()["revision"], kind=kind, med=med, **extra)
    )


def confirm(boundary, goal="initial", med="a"):
    act(boundary, "confirm", med)
    boundary.observe("human_msg", dict(kind="confirm_reply", med=med, text="yes"), meta(goal))


def test_public_projection_and_no_oracle_input():
    """VAL-2/6; TC-8: private goals/verdicts and extension fields never become worker data."""
    boundary = opened()
    assert "HIDDEN" not in json.dumps(boundary.snapshot())
    assert "goal_id" not in boundary.snapshot()
    for topic in ["episode_goal", "episode_result", "oracle_state", "episode_meta"]:
        with pytest.raises(ValueError, match="undeclared"):
            boundary.observe(topic, {"secret": "HIDDEN"}, meta())


def test_recovery_waits_for_matching_previous_motion_without_oracle_reset():
    """TC-6/7/8: a new dialogue can arrive during old motion without leaking its verdict."""
    boundary = opened()
    confirm(boundary)
    initial = act(boundary, "deliver")
    old_revision = boundary.snapshot()["revision"]
    boundary.observe(
        "human_msg",
        dict(kind="request", med="b", return_med="a", text="return a, bring b"),
        meta("recovery"),
    )
    confirm(boundary, "recovery", "b")
    with pytest.raises(TaskRefusal, match="busy"):
        act(boundary, "return")
    with pytest.raises(TaskRefusal, match="revision"):
        boundary.action(dict(revision=old_revision, kind="deliver", med="a"))
    assert boundary.observe("plan_done", {"ok": True}, meta("initial", "wrong")) is None
    assert boundary.snapshot()["active"] is not None
    event = boundary.observe(
        "plan_done", {"ok": True}, meta("initial", initial["metadata"]["request_id"])
    )
    assert event["physical_completion"] == "unknown"
    returned = act(boundary, "return")
    assert returned["topic"] == "return_request"
    assert returned["payload"] == {"return_med": "a"}
    assert returned["metadata"]["goal_id"] == "recovery"
    assert "sim_time_ns" not in returned["metadata"]
    assert (
        boundary.observe(
            "plan_done", {"ok": True}, meta("initial", initial["metadata"]["request_id"])
        )
        is None
    )
    with pytest.raises(TaskRefusal, match="busy"):
        act(boundary, "deliver", "b")
    boundary.observe(
        "plan_done", {"ok": True}, meta("recovery", returned["metadata"]["request_id"])
    )
    delivered = act(boundary, "deliver", "b")
    assert delivered["topic"] == "target_request"
    assert delivered["payload"] == {"target_med": "b"}


def test_reset_fences_old_completion_and_repeated_reset():
    """CON-5; TC-6: reset generations prevent a late completion from releasing new motion."""
    boundary = opened()
    confirm(boundary)
    old = act(boundary, "deliver")
    boundary.observe("reset_done", [1], meta(seq=10))
    boundary.observe("human_msg", dict(kind="request", med="a", text="again"), meta())
    confirm(boundary)
    new = act(boundary, "deliver")
    assert old["metadata"]["request_id"] != new["metadata"]["request_id"]
    assert boundary.observe("reset_done", [1], meta(seq=10)) is None
    assert (
        boundary.observe("plan_done", {"ok": True}, meta("initial", old["metadata"]["request_id"]))
        is None
    )
    assert boundary.snapshot()["active"] is not None


@pytest.mark.parametrize(
    "extra",
    [{"topic": "joint_cmd"}, {"goal_id": "private"}, {"request_id": "chosen"}, {"seed": 12}],
)
def test_worker_cannot_choose_transport_authority(extra):
    """VAL-2/6; TC-6: authored requests cannot add endpoints, private goals or correlation IDs."""
    boundary = opened()
    confirm(boundary)
    with pytest.raises(TaskRefusal):
        act(boundary, "deliver", **extra)
    assert boundary.motion_requests == 0


def test_confirmation_and_lifetime_invocation_budget():
    """TC-7; CON-5: motion needs public confirmation; reset does not replenish its budget."""
    boundary = opened()
    boundary.max_motion_requests = 1
    with pytest.raises(TaskRefusal, match="confirmed"):
        act(boundary, "deliver")
    confirm(boundary)
    motion = act(boundary, "deliver")
    boundary.observe("plan_done", {"ok": True}, meta(request=motion["metadata"]["request_id"]))
    with pytest.raises(TaskRefusal, match="budget"):
        act(boundary, "deliver")
    boundary.observe("reset_done", [1], meta(seq=20))
    assert boundary.motion_requests == 1


def test_old_dialogue_cannot_reactivate_superseded_revision():
    """TC-6/7: stale request/confirmation traffic cannot replace the active public task."""
    boundary = opened()
    boundary.observe("human_msg", dict(kind="request", med="b", text="bring b"), meta("new"))
    revision = boundary.snapshot()["revision"]
    assert (
        boundary.observe("human_msg", dict(kind="request", med="a", text="bring a"), meta()) is None
    )
    assert (
        boundary.observe("human_msg", dict(kind="confirm_reply", med="a", text="yes"), meta())
        is None
    )
    assert boundary.snapshot()["revision"] == revision
    assert boundary.snapshot()["confirmed"] is False


class Evidence:
    arm = "trace"
    session_id = "fixture"

    def _verify_bundle(self):
        pass

    def call(self, request):
        return {"ok": False, "error": "evidence_request_refused"}


class Transport:
    def __init__(self, events=()):
        self.events = iter(events)
        self.sent = []

    def next_event(self):
        return next(self.events)

    def send(self, topic, payload, metadata):
        self.sent.append((topic, payload, metadata))


def session(tmp_path, transport=None, calls=32):
    return TaskSession(
        evidence=Evidence(),
        transport=transport or Transport(),
        output=tmp_path / "task",
        source_sha256="a" * 64,
        medicines=["a", "b"],
        max_calls=calls,
        max_motion_requests=4,
    )


def test_early_worker_finish_retains_incomplete_task_input(tmp_path):
    """TRT-9/10: a successful child exit cannot hide unconsumed task inputs."""
    with pytest.raises(ValueError, match="exhaust"):
        with session(tmp_path):
            pass
    result = json.loads((tmp_path / "task/summary.json").read_text())
    assert result["ok"] is False and result["measured_sessions"] == 0


def test_task_requests_retain_source_identity_and_refusals(tmp_path):
    """CON-5; HAR-4; TRT-10: task requests share a bound source and retain refused authority."""
    with session(tmp_path) as service:
        response = service.call(dict(operation="act", arguments=dict(topic="joint_cmd")))
        assert response == {"ok": False, "error": "task_request_refused"}
        assert service.call(dict(operation="next", arguments={})) == {"ok": True, "event": None}
    summary = json.loads((tmp_path / "task/summary.json").read_text())
    assert summary["ok"] is True and summary["source_sha256"] == "a" * 64
    assert summary["calls"] == 2 and summary["refused_calls"] == 1
    assert summary["guard_traversal_verified"] is False


def test_failed_transport_closes_authority_and_retains_reason(tmp_path):
    """TRT-9/10: a failed host dispatch cannot be retried invisibly by authored code."""

    class Broken(Transport):
        def next_event(self):
            raise OSError("private transport error")

    with pytest.raises(OSError):
        with session(tmp_path, Broken()) as service:
            service.call(dict(operation="next", arguments={}))
    summary = json.loads((tmp_path / "task/summary.json").read_text())
    assert summary["ok"] is False
    assert "private transport error" in summary["error"]


def test_caught_transport_failure_cannot_reopen_authority(tmp_path):
    """TRT-9: even a caller that catches a backend error cannot continue the same session."""

    class Broken(Transport):
        def next_event(self):
            raise OSError("transport failed")

    with pytest.raises(ValueError, match="transport failed"):
        with session(tmp_path, Broken()) as service:
            with pytest.raises(OSError):
                service.call(dict(operation="next", arguments={}))
            with pytest.raises(ValueError, match="closed"):
                service.call(dict(operation="next", arguments={}))


def test_malformed_matching_completion_cannot_release_motion():
    """TC-6/7: the fixed executor's completion payload, as well as identity, must match."""
    boundary = opened()
    confirm(boundary)
    motion = act(boundary, "deliver")
    with pytest.raises(ValueError, match="completion"):
        boundary.observe("plan_done", {"ok": False}, meta(request=motion["metadata"]["request_id"]))
    assert boundary.snapshot()["active"] is not None


def test_pre_reset_dialogue_is_not_a_new_task():
    """TC-6/7: delayed pre-reset public traffic cannot recreate an obsolete task."""
    boundary = opened()
    old = meta("late-old-goal")
    boundary.observe("reset_done", [1], meta(seq=100))
    assert boundary.observe("human_msg", dict(kind="request", med="b", text="old"), old) is None
    assert boundary.snapshot()["revision"] is None


def test_medicine_choice_remains_the_workers_decision():
    """TC-8: the boundary cannot replace an authored mistake with the scorer's target."""
    boundary = opened()
    confirm(boundary)
    command = act(boundary, "deliver", "b")
    assert command["payload"] == {"target_med": "b"}


def test_task_call_budget_counts_refusals(tmp_path):
    """CON-5; TRT-10: invalid requests cannot extend the fixed call allowance."""
    with pytest.raises(ValueError, match="budget"):
        with session(tmp_path, calls=1) as service:
            assert service.call({"operation": "act", "arguments": {}})["ok"] is False
            service.call({"operation": "next", "arguments": {}})
    assert json.loads((tmp_path / "task/summary.json").read_text())["calls"] == 1
