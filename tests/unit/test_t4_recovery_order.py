"""Regression for the independently retained P3 return-request rejection."""

import pytest

from aisle.nodes.task_state_machine import TaskStateMachine

pytestmark = pytest.mark.unit


def request(machine, goal, med, return_med=None):
    payload = dict(kind="request", med=med)
    if return_med:
        payload["return_med"] = return_med
    machine.on_human_msg(payload, goal)
    return machine.on_human_msg(dict(kind="confirm_reply", med=med), goal)


def done(machine, emission):
    return machine.on_plan_done(**emission[2])


def test_recovery_waits_for_previous_plan_then_return_before_delivery():
    """TC-7; HAR-3: a successful goal can close before its motion returns."""
    machine = TaskStateMachine(tier="T4", max_retries=2)
    initial = request(machine, "initial", "metformin")[0]
    machine.on_result()
    assert request(machine, "recovery", "amoxicillin", "metformin") == []
    returned = done(machine, initial)
    assert [(topic, value) for topic, value, _ in returned] == [
        ("return_request", {"return_med": "metformin"})
    ]
    delivery = done(machine, returned[0])
    assert [(topic, value) for topic, value, _ in delivery] == [
        ("target_request", {"target_med": "amoxicillin"})
    ]
    assert (
        len({initial[2]["request_id"], returned[0][2]["request_id"], delivery[0][2]["request_id"]})
        == 3
    )


def test_recovery_after_initial_motion_return_starts_only_return():
    """TC-7: an already idle executor still receives serialized recovery."""
    machine = TaskStateMachine(tier="T4", max_retries=2)
    initial = request(machine, "initial", "metformin")[0]
    assert done(machine, initial) == []
    machine.on_result()
    returned = request(machine, "recovery", "amoxicillin", "metformin")
    assert [e[0] for e in returned] == ["return_request"]
    assert [e[0] for e in done(machine, returned[0])] == ["target_request"]


def test_stale_duplicate_and_unattributed_done_do_not_release_delivery():
    """TC-7: obsolete completion cannot advance a newer motion boundary."""
    machine = TaskStateMachine(tier="T4", max_retries=2)
    returned = request(machine, "recovery", "amoxicillin", "metformin")[0]
    assert machine.on_plan_done() == []
    assert machine.on_plan_done(goal_id="old", request_id=returned[2]["request_id"]) == []
    assert machine.on_plan_done(goal_id="recovery", request_id="other") == []
    assert machine.retry_due_tick is None
    delivery = done(machine, returned)[0]
    assert done(machine, returned) == []
    assert machine.retry_due_tick is None
    assert done(machine, delivery) == []
    for _ in range(3):
        emissions = machine.on_tick()
    retries = [e for e in emissions if e[0] == "target_request"]
    assert len(retries) == 1
    assert retries[0][2]["request_id"] != delivery[2]["request_id"]
    assert done(machine, delivery) == []
    assert machine.retry_due_tick is None


def test_prior_motion_can_finish_during_confirmation_without_early_action():
    """TC-7: late initial completion is valid but cannot bypass confirmation."""
    machine = TaskStateMachine(tier="T4")
    initial = request(machine, "initial", "metformin")[0]
    machine.on_result()
    machine.on_human_msg(
        dict(kind="request", med="amoxicillin", return_med="metformin"), "recovery"
    )
    assert done(machine, initial) == []
    assert machine.violations == {}
    emitted = machine.on_human_msg(dict(kind="confirm_reply", med="amoxicillin"), "recovery")
    assert [e[0] for e in emitted] == ["return_request"]


def test_result_closes_pending_recovery_and_new_episode_rejects_late_done():
    """TC-7: closed or superseded intent cannot revive through late completion."""
    machine = TaskStateMachine(tier="T4", max_retries=2)
    initial = request(machine, "initial", "metformin")[0]
    machine.on_result()
    assert request(machine, "recovery", "amoxicillin", "metformin") == []
    machine.on_result()
    assert done(machine, initial) == []
    current = request(machine, "next-episode", "ibuprofen")[0]
    assert done(machine, initial) == []
    assert machine.retry_due_tick is None
    assert done(machine, current) == []
    assert machine.retry_due_tick is not None


def test_stale_goal_result_does_not_close_recovery():
    """TC-7: an obsolete result cannot clear a newer dialogue action."""
    machine = TaskStateMachine(tier="T4")
    returned = request(machine, "recovery", "amoxicillin", "metformin")[0]
    machine.on_result(goal_id="old")
    assert machine.goal_id == "recovery" and machine.goal is not None
    assert [e[0] for e in done(machine, returned)] == ["target_request"]


def test_result_during_return_prevents_late_redelivery():
    """TC-7: completion of a closed return cannot release obsolete delivery."""
    machine = TaskStateMachine(tier="T4", max_retries=2)
    returned = request(machine, "recovery", "amoxicillin", "metformin")[0]
    machine.on_result(goal_id="recovery")
    assert done(machine, returned) == []
    assert machine.on_tick() == []
