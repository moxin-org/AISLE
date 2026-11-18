"""Runtime lifecycle and guarded action boundaries for ADR-74."""

import numpy as np
import pyarrow as pa
import pytest

from aisle.nodes.budget_guard import clamp_joint_cmd, load_limits
from aisle.nodes.pear_backend import ActionDelta
from aisle.nodes.pear_policy import Controller, Motion, plan_motion

pytestmark = pytest.mark.unit


def ready_controller(stamp=100_000_000):
    control = Controller()
    control.reset()
    control.set_goal("pick a cube", "goal-1")
    q = np.array(load_limits("franka").fallback_qpos, dtype=np.float32)
    for name in ("overhead", "wrist"):
        control.observe_frame(
            name,
            pa.array(np.arange(18, dtype=np.uint8)),
            {"h": 2, "w": 3, "enc": "rgb8", "sim_time_ns": stamp, "env_id": 0},
        )
    control.observe_joints(q, stamp, 0)
    return control, q


def test_reset_and_new_goal_invalidate_pending_inference():
    """TC-6/7: a late response cannot cross an episode or instruction boundary."""
    c, _ = ready_controller()
    ticket = c.prepare(100_000_000)
    assert ticket is not None and c.prepare(100_000_000) is None
    assert ticket.goal_id == "goal-1"
    c.reset()
    assert not c.accept(ticket, ActionDelta((0, 0, 0), (0, 0, 0), 0), "reply", 100_000_000)
    assert not c.frames and c.joints is None and not c.history
    c, _ = ready_controller()
    ticket = c.prepare(100_000_000)
    c.set_goal("place elsewhere", "goal-2")
    assert not c.accept(ticket, ActionDelta((0, 0, 0), (0, 0, 0), 0), "reply", 100_000_000)


def test_stale_and_mismatched_frames_do_not_act():
    """TC-2/9, CON-5: freshness uses sim time; cameras must share a stamp and env."""
    c, _ = ready_controller()
    ticket = c.prepare(100_000_000)
    assert not c.accept(ticket, ActionDelta((0, 0, 0), (0, 0, 0), 0), "reply", 700_000_000)
    assert c.motion is None and c.feedback == "stale_observation"
    c, _ = ready_controller()
    c.observe_frame(
        "wrist",
        pa.array(np.arange(18, dtype=np.uint8)),
        {"h": 2, "w": 3, "enc": "rgb8", "sim_time_ns": 110_000_000, "env_id": 0},
    )
    assert c.prepare(100_000_000) is None


def test_changed_robot_rejects_an_otherwise_fresh_reply():
    """TC-2, BG-3: freshness alone cannot authorize a pose based on displaced joints."""
    c, q = ready_controller()
    ticket = c.prepare(100_000_000)
    changed = q.copy()
    changed[0] += 0.02
    c.observe_joints(changed, 110_000_000, 0)
    assert not c.accept(ticket, ActionDelta((0, 0, 0.2), (0, 0, 0), 0), "reply", 110_000_000)
    assert c.motion is None and c.feedback == "robot_moved_during_inference"


def test_cm_delta_uses_measured_fk_and_existing_ik():
    """TC-1, BG-2: model centimeters become meters; solve only from measured joints."""
    from aisle.nodes.ik_trajectory import fk_tcp

    limits = load_limits("franka")
    q = np.array(limits.fallback_qpos, dtype=np.float32)
    motion = plan_motion(q, ActionDelta((0, 0, 0.2), (0, 0, 0), 0), 0)
    assert np.linalg.norm(fk_tcp(motion.path[-1]) - fk_tcp(q[:7]) - [0, 0, 0.002]) < 0.0006
    assert np.all(motion.path[-1] >= limits.q_min_arr[:7])
    assert np.all(motion.path[-1] <= limits.q_max_arr[:7])


def test_motion_tracks_before_finishing_and_passes_the_guard():
    """BG-1/2/3: measured tracking, bounded streaming, and the existing guard remain separate."""
    limits = load_limits("franka")
    q = np.array(limits.fallback_qpos, dtype=np.float32)
    target = q[:7].copy()
    target[0] += 0.01
    motion = Motion([target], 0.0, 0)
    previous = q.copy()
    for tick in range(100):
        command, grip = motion.step(q, tick * 10_000_000)
        if command is not None:
            safe, violations = clamp_joint_cmd(command, previous, limits, False)
            assert not violations
            assert np.allclose(command, safe)
            previous = safe
            q = safe.copy()
        if motion.done:
            break
    assert motion.done and motion.feedback == "reached"
    assert not np.allclose(q, limits.fallback_qpos)


def test_tracking_failure_returns_feedback_not_success():
    """TC-8, BG-3: a blocked movement ends with feedback, never a task-success verdict."""
    q = np.array(load_limits("franka").fallback_qpos, dtype=np.float32)
    target = q[:7].copy()
    target[0] += 0.1
    motion = Motion([target], 0.0, 0)
    motion.step(q, 0)
    command, _ = motion.step(q, 10_000_000_001)
    assert command is None and motion.done and motion.feedback == "motion_timeout"


def test_missing_cameras_and_completed_episode_prevent_requests():
    """TC-6/7/9: no blank-camera substitution and no commands after the verdict."""
    c, _ = ready_controller()
    c.frames.pop("wrist")
    assert c.prepare(100_000_000) is None
    c, _ = ready_controller()
    c.finish()
    assert c.prepare(100_000_000) is None


def test_malformed_proprioception_is_rejected():
    """TC-1/2, BG-3: wrong dimensions, NaN and cross-env state cannot reach FK."""
    c, q = ready_controller()
    for bad in ([0], [float("nan")] * 9, [True] * 9):
        with pytest.raises(ValueError):
            c.observe_joints(bad, 100_000_000, 0)
    with pytest.raises(ValueError):
        c.observe_joints(q, 100_000_000, 1)


def test_missing_fresh_joint_feedback_aborts_motion():
    """TC-2, BG-3: do not keep streaming a trajectory against expired feedback."""
    c, q = ready_controller()
    c.motion = Motion([q[:7]], 0.0, 100_000_000)
    assert c.advance(300_000_000) == (None, None)
    assert c.motion is None and c.feedback == "stale_joint_state"


def test_failed_inference_can_retry_only_on_a_new_snapshot():
    """CON-5, TC-2: a network/model refusal cannot spin on the same observation."""
    c, _ = ready_controller()
    ticket = c.prepare(100_000_000)
    c.reject(ticket, "network_timeout")
    assert c.pending is None
    assert c.prepare(100_000_000) is None


def test_trace_metadata_projects_only_serializable_contract_fields():
    """TC-2, CON-4: Dora's native datetime metadata is not a model/JSON trace field."""
    import datetime
    import json

    from aisle.nodes.pear_policy import trace_metadata

    value = trace_metadata(
        {
            "sim_time_ns": 1,
            "env_id": 0,
            "seq": 2,
            "goal_id": "g",
            "timestamp": datetime.datetime(2026, 1, 1),
            "oracle_state": [7],
        }
    )
    assert json.loads(json.dumps(value)) == {
        "sim_time_ns": 1,
        "env_id": 0,
        "seq": 2,
        "goal_id": "g",
    }


def test_malformed_answer_is_refused_with_its_reason_as_feedback():
    """TC-5: a reply that is not a valid action executes nothing and tells the model why."""
    c, _ = ready_controller()
    ticket = c.prepare(100_000_000)
    assert c.answer(ticket, 'done\n```json\n{"arm": {}}\n```', 100_000_000) is False
    assert c.motion is None and c.pending is None
    assert c.feedback.startswith("answer_refused: ") and "arm" in c.feedback


def test_valid_answer_is_planned_like_a_parsed_action():
    """TC-5: a well-formed reply is parsed and accepted through the same checks."""
    c, _ = ready_controller()
    ticket = c.prepare(100_000_000)
    text = (
        '```json\n{"arm":{"x":0,"y":0,"z":1,"roll":0,"pitch":0,"yaw":0},"gripper":{"width":0}}\n```'
    )
    assert c.answer(ticket, text, 100_000_000) is True
    assert c.motion is not None and c.feedback == "executing"
