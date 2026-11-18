"""Nano embodiment boundary: real pinned URDF, no simulator required."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow as pa
import pytest

from aisle.nodes.pear_backend import Frame, Observation
from aisle.nodes.pear_nano import MODEL, NanoAdapter, NanoGuard
from aisle.nodes.pear_policy import Controller

pytestmark = pytest.mark.unit


def reply(side="left", z=0.2, angle=0):
    obj = {}
    for name in ("left", "right"):
        obj[name + "_arm"] = dict.fromkeys(("x", "y", "z", "pitch", "roll"), 0)
        obj[name + "_gripper"] = {"angle": angle if name == side else 0}
    obj[side + "_arm"]["z"] = z
    return "```json\n" + json.dumps(obj) + "\n```"


def test_real_nano_has_two_five_axis_arms_and_angular_grippers():
    """TC-1/5, BG-2: derive Nano dimensions and limits from its actual URDF."""
    robot = NanoAdapter()
    assert len(robot.names) == 17
    assert len(robot.arms["left"]) == len(robot.arms["right"]) == 5
    assert robot.sha256 == "342da414bca32361ad6be12e76ac769289dc56e382ad44e811361809c5e48529"
    assert len(robot.held) == 5
    assert np.allclose(robot.lower[list(robot.grippers.values())], -0.3)
    assert np.allclose(robot.upper[list(robot.grippers.values())], 0.8771)
    with pytest.raises(ValueError):
        robot.joint_vector(np.zeros(9))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lower", "nan"),
        ("upper", "inf"),
        ("lower", "-inf"),
        ("effort", "nan"),
        ("effort", "inf"),
        ("effort", "-1"),
        ("effort", "0"),
        ("velocity", None),
        ("effort", None),
        ("lower", None),
    ],
)
def test_nano_rejects_invalid_urdf_actuator_limits(tmp_path, field, value):
    """BG-2/3, CON-8: malformed URDF limits cannot weaken guards or escape CLI errors."""
    tree = ET.parse(MODEL)
    limit = tree.getroot().find("joint[@name='left_j1_yaw']/limit")
    if value is None:
        del limit.attrib[field]
    else:
        limit.set(field, value)
    path = tmp_path / "invalid.urdf"
    tree.write(path)
    with pytest.raises(ValueError, match="limit"):
        NanoAdapter(path)


@pytest.mark.parametrize("side", ["left", "right"])
def test_each_arm_ik_moves_only_its_own_urdf_chain(side):
    """BG-2, TC-1: Cartesian centimeters reach the chosen Nano tool without moving the base."""
    robot = NanoAdapter()
    q = np.zeros(17)
    action = robot.parse_action(reply(side))
    motion = robot.plan_motion(q, action, 0)
    end = motion.path[-1]
    assert (
        np.linalg.norm(robot.pose(end, side)[0] - robot.pose(q, side)[0] - [0, 0, 0.002]) < 0.0005
    )
    held = [i for i in range(17) if i not in robot.arms[side]]
    np.testing.assert_array_equal(end[held], q[held])


def test_nano_prompt_has_three_rgb_views_and_measured_state_only():
    """TC-3/9: use the Nano cameras, explicit angular grippers, and no scene truth."""
    robot = NanoAdapter()
    frames = tuple(
        Frame.from_arrow(
            name,
            pa.array(np.arange(18, dtype=np.uint8)),
            {"h": 2, "w": 3, "enc": "rgb8", "env_id": 0, "sim_time_ns": 10},
        )
        for name in robot.cameras
    )
    messages = robot.build_messages("move the cube", Observation(frames, np.zeros(17), 10, 0), [])
    text = json.dumps(messages)
    assert "Franka" not in text and "finger_positions_m" not in text
    assert "left_gripper" in text and "position_rad" in text
    assert len([p for p in messages[-1]["content"] if p["type"] == "image_url"]) == 3
    assert "oracle_state" not in text and "object_positions" not in text


@pytest.mark.parametrize(
    "bad",
    [
        reply(z=6),
        reply(angle=0.2),
        reply().replace('"pitch": 0', '"yaw": 0'),
        reply().replace('"z": 0.2', '"z": true'),
    ],
)
def test_nano_rejects_wrong_dof_units_or_unbounded_actions(bad):
    """BG-2/3: Nano does not accept an invented sixth axis or a linear gripper."""
    with pytest.raises(ValueError):
        NanoAdapter().parse_action(bad)


def test_nano_guard_clamps_velocity_and_locks_non_arm_joints():
    """BG-1/2/3: all 17-channel commands traverse independent URDF-derived bounds."""
    robot = NanoAdapter()
    q = np.zeros(17)
    guard = NanoGuard(robot)
    guard.observe(q, 0)
    requested = np.full(17, 100.0)
    command, violations = guard.apply(requested, 0)
    assert violations
    np.testing.assert_array_equal(command[robot.held], q[robot.held])
    assert np.max(np.abs(command)) <= 0.003001
    assert np.all(command >= robot.lower) and np.all(command <= robot.upper)
    with pytest.raises(ValueError):
        guard.observe([float("nan")] * 17, 1)


def test_nano_guard_cannot_spend_velocity_budget_twice_in_one_tick():
    """BG-2/3: repeated or sub-period commands cannot multiply the velocity allowance."""
    robot = NanoAdapter()
    guard = NanoGuard(robot)
    q = np.zeros(17)
    guard.observe(q, 10_000_000)
    target = q.copy()
    target[robot.arms["left"][0]] = 0.1
    first, _ = guard.apply(target, 10_000_000)
    repeated, reasons = guard.apply(target, 10_000_000)
    np.testing.assert_array_equal(first, repeated)
    assert "duplicate_or_out_of_order_command" in reasons
    next_command, _ = guard.apply(target, 11_000_000)
    assert np.max(np.abs(next_command - first)) <= 0.000301


def test_nano_observation_requires_the_matching_model_and_joint_order():
    """TC-5/9: a 17-vector alone cannot identify the Nano geometry or joint order."""
    robot = NanoAdapter()
    valid = {"robot_model_sha256": robot.sha256, "joint_names": list(robot.names)}
    robot.validate_metadata("joint_state", valid)
    for invalid in (
        {**valid, "robot_model_sha256": "wrong"},
        {**valid, "joint_names": list(reversed(robot.names))},
    ):
        with pytest.raises(ValueError):
            robot.validate_metadata("joint_state", invalid)


def test_controller_uses_nano_profile_and_invalidates_reset_reply():
    """TC-6/7: use the common lifecycle with Nano rather than a separate agent loop."""
    robot = NanoAdapter()
    control = Controller(adapter=robot)
    control.set_goal("raise left tool", "nano-1")
    for name in robot.cameras:
        control.observe_frame(
            name,
            pa.array(np.arange(18, dtype=np.uint8)),
            {"h": 2, "w": 3, "enc": "rgb8", "env_id": 0, "sim_time_ns": 10},
        )
    control.observe_joints(np.zeros(17), 10, 0)
    ticket = control.prepare(10)
    assert ticket is not None
    assert control.accept(ticket, robot.parse_action(reply()), reply(), 10)
    control.reset()
    assert not control.accept(ticket, robot.parse_action(reply()), reply(), 10)


def test_nano_motion_does_not_chase_passive_wheel_or_head_error():
    """BG-2/3: arm completion and output must not command base/head correction."""
    from aisle.nodes.pear_nano import NanoMotion

    robot = NanoAdapter()
    q = np.zeros(17)
    motion = NanoMotion(robot, [q.copy()], 0)
    motion.step(q, 0)
    q[robot.held[0]] = 0.1
    for tick in range(1, 5):
        command, _ = motion.step(q, tick * 10_000_000)
        if command is not None:
            assert command[robot.held[0]] == 0
        if motion.done:
            break
    assert motion.done and motion.feedback == "reached"


def test_nano_motion_tracks_through_small_motor_bias():
    """BG-2/3: use measured tracking feedback when the motor has a static load error."""
    from aisle.nodes.pear_nano import NanoMotion

    robot = NanoAdapter()
    q = np.zeros(17)
    index = robot.arms["left"][1]
    target = q.copy()
    target[index] = 0.01
    motion = NanoMotion(robot, [target], 0)
    for tick in range(200):
        command, _ = motion.step(q, tick * 10_000_000)
        if command is not None:
            q = command.astype(float)
            q[index] -= 0.006
        if motion.done:
            break
    assert motion.done and abs(q[index] - target[index]) < 0.001


def test_nano_holds_one_base_head_reference_until_reset():
    """TC-6, BG-2: later arm requests cannot retarget held head/wheel coordinates."""
    robot = NanoAdapter()
    q = np.zeros(17)
    robot.plan_motion(q, robot.parse_action(reply()), 0)
    q[robot.held[-1]] = 0.01
    second = robot.plan_motion(q, robot.parse_action(reply()), 10)
    assert second.path[0][robot.held[-1]] == 0
    robot.reset()
    third = robot.plan_motion(q, robot.parse_action(reply()), 20)
    assert third.path[0][robot.held[-1]] == 0.01


def test_tracking_displacement_cannot_override_nano_velocity_limit():
    """BG-2/3: incompatible tracking and velocity intervals must hold, never jump."""
    robot = NanoAdapter()
    guard = NanoGuard(robot)
    q = np.zeros(17)
    guard.observe(q, 0)
    first, _ = guard.apply(q, 0)
    q[robot.arms["left"][0]] = 0.1
    guard.observe(q, 10_000_000)
    safe, reasons = guard.apply(q, 10_000_000)
    assert np.max(np.abs(safe - first)) <= 0.003001
    assert "tracking_interval_conflict" in reasons


def test_nano_guard_recovers_after_a_blocked_joint_relaxes():
    """BG-2/3: a tracking conflict is left at the velocity limit, not held for the whole run."""
    robot = NanoAdapter()
    guard = NanoGuard(robot)
    joint = robot.arms["left"][0]
    q = np.zeros(17)
    target = q.copy()
    target[joint] = 1.0
    guard.observe(q, 0)
    for tick in range(1, 20):  # blocked joint: the command runs ahead to the tracking bound
        guard.observe(q, tick * 10_000_000)
        ahead, _ = guard.apply(target, tick * 10_000_000)
    assert ahead[joint] == pytest.approx(0.03, abs=1e-6)
    q[joint] = -0.005  # the joint relaxes away from the command
    previous = ahead
    for tick in range(20, 40):
        guard.observe(q, tick * 10_000_000)
        safe, reasons = guard.apply(q, tick * 10_000_000)
        assert abs(safe[joint] - previous[joint]) <= 0.003001
        previous = safe
    assert reasons == []
    assert safe[joint] == pytest.approx(-0.005, abs=1e-6)


def test_nano_guard_anchors_on_a_float32_joint_limit():
    """BG-1: a measured joint resting on its limit in float32 is anchored inside the limit."""
    robot = NanoAdapter()
    guard = NanoGuard(robot)
    joint = robot.grippers["left"]
    q = np.zeros(17, dtype=np.float32)
    q[joint] = np.nextafter(np.float32(robot.lower[joint]), np.float32(-np.inf))
    guard.observe(q, 0)
    safe, _ = guard.apply(q, 0)
    assert robot.lower[joint] <= guard.last[joint] <= robot.upper[joint]
    assert safe.shape == (17,)
    q[joint] = robot.lower[joint] - 0.01
    with pytest.raises(ValueError):
        NanoGuard(robot).observe(q, 0)


@pytest.mark.parametrize(
    "bad", [np.full(17, np.nan), np.zeros(16), np.array(["x"] * 17), np.full(17, np.inf)]
)
def test_nano_guard_holds_on_a_malformed_command(bad):
    """BG-1/2: non-finite, wrong-length or non-numeric commands hold the last safe command."""
    robot = NanoAdapter()
    guard = NanoGuard(robot)
    guard.observe(np.zeros(17), 0)
    first, _ = guard.apply(np.zeros(17), 0)
    safe, reasons = guard.apply(bad, 10_000_000)
    np.testing.assert_array_equal(safe, first)
    assert reasons == ["malformed_command"]


def test_nano_guard_holds_on_stale_joint_state():
    """BG-2: a command without a fresh measurement holds the last safe command."""
    robot = NanoAdapter()
    guard = NanoGuard(robot)
    guard.observe(np.zeros(17), 0)
    first, _ = guard.apply(np.zeros(17), 0)
    target = np.zeros(17)
    target[robot.arms["left"][0]] = 0.1
    safe, reasons = guard.apply(target, 200_000_000)
    np.testing.assert_array_equal(safe, first)
    assert reasons == ["stale_joint_state"]


def test_nano_action_with_an_oversized_integer_is_a_value_error():
    """TC-5, BG-1: an integer too large for a float is refused as a malformed action."""
    answer = (
        '```json\n{"left_arm":{"x":1' + "0" * 400 + ',"y":0,"z":0,"pitch":0,"roll":0},'
        '"right_arm":{"x":0,"y":0,"z":0,"pitch":0,"roll":0},'
        '"left_gripper":{"angle":0},"right_gripper":{"angle":0}}\n```'
    )
    with pytest.raises(ValueError):
        NanoAdapter.parse_action(answer)
