"""G1 right-arm motion stack: URDF chain, guard limits, guard FK (G1-F1;
BG-2, BG-3, CON-5, CON-7, SCN-4; ADR-g1-embodiment scope A).

The chain is parsed from the vendored official g1_29dof_mode_15 URDF with a
fixed world<-torso base (pelvis at the FK standing height, waist at held
home) and the a7dd234 TCP offset — the guard and the planner share one
model, exactly as SO-101 does through its official URDF."""

from __future__ import annotations

import numpy as np
import pytest

from aisle.embodiment import G1_RIGHT_ARM_JOINTS
from aisle.kinematics import g1_right_chain, so101_chain
from aisle.nodes.budget_guard import (
    clamp_joint_cmd,
    fingers_to_gripper,
    fk_ee_pose,
    gripper_to_fingers,
    load_limits,
)

pytestmark = pytest.mark.unit

#: URDF right-arm position limits (g1_29dof_mode_15_with_dex1_1.urdf) plus
#: the two Dex1-1 prismatic fingers.
G1_LOWER = (
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.614429558,
    -1.614429558,
    -0.02,
    -0.02,
)
G1_UPPER = (
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.972222054,
    1.614429558,
    1.614429558,
    0.0245,
    0.0245,
)
#: URDF-declared motor velocities — the raw actuator envelope, NOT what the
#: guard enforces (see the conservative qdot note below).
G1_URDF_QDOT = (37.0, 37.0, 37.0, 37.0, 37.0, 27.0, 27.0, 0.2, 0.2)
#: G1-H-r1 scene home (wire order), reused as the guard fallback: hand
#: off the shelf, above box tops (outer-loop #6 ruling).
G1_HOME = (-1.2, -0.1, 0.0, 0.3, 0.0, 1.5, 0.0, 0.0245, 0.0245)


def test_g1_chain_and_guard_limits_come_from_official_urdf():
    """BG-2, TC-5, SCN-4, CON-7: the chain parses the vendored official
    mode-15 URDF in G1_RIGHT_ARM_JOINTS order with exact position limits;
    env/limits.toml carries the same 7+2 stops with conservative velocities
    (arm min(URDF, 2.0) rad/s, fingers 0.2 m/s per ADR-g1-embodiment)."""
    chain = g1_right_chain()
    limits = load_limits("g1_right")
    assert chain.joint_names == G1_RIGHT_ARM_JOINTS
    assert chain.q_min == G1_LOWER[:7]
    assert chain.q_max == G1_UPPER[:7]
    assert chain.qdot_max == G1_URDF_QDOT[:7]
    assert limits.n_arm_dof == 7
    assert limits.q_min == G1_LOWER
    assert limits.q_max == G1_UPPER
    assert limits.qdot_max == (2.0,) * 7 + (0.2,) * 2
    assert limits.cmd_dt_s == 0.01
    assert limits.fallback_qpos == G1_HOME
    assert limits.gripper_min == 0.0 and limits.gripper_max == 1.0
    assert limits.gripper_rate_max == pytest.approx(0.2 / 0.0445, rel=1e-9)
    assert limits.gripper_dt_s == 0.0333


def test_g1_normalized_gripper_maps_both_official_endpoints():
    """BG-1, BG-2, TC-5: CONTRACT gripper semantics (0.0 = open … 1.0 =
    closed) map exactly onto the Dex1-1 stops — +q separates the fingers
    (open = upper 0.0245 m), -q closes them (lower -0.02 m)."""
    limits = load_limits("g1_right")
    assert gripper_to_fingers(0.0, limits) == pytest.approx([0.0245, 0.0245])
    assert gripper_to_fingers(1.0, limits) == pytest.approx([-0.02, -0.02])
    for value in (0.0, 0.25, 1.0):
        command = np.concatenate(
            [np.asarray(limits.fallback_qpos[:7]), gripper_to_fingers(value, limits)]
        )
        assert fingers_to_gripper(command, limits) == pytest.approx(value, abs=1e-6)


def test_g1_fk_and_guard_are_deterministic_and_contained():
    """BG-2, BG-3, CON-5: world-frame FK (fixed pelvis base + held waist)
    is deterministic, the home TCP sits inside the workspace AABB, a legal
    command produces zero violations, and a wild one is clamped legal."""
    limits = load_limits("g1_right")
    home = np.asarray(limits.fallback_qpos, dtype=np.float32)
    pos_a, rot_a = fk_ee_pose(home[:7], embodiment="g1_right")
    pos_b, rot_b = fk_ee_pose(home[:7], embodiment="g1_right")
    assert np.array_equal(pos_a, pos_b)
    assert np.array_equal(rot_a, rot_b)
    assert pos_a == pytest.approx(np.array([0.43559, -0.08916, 1.01284]), abs=5e-4), pos_a
    assert all(limits.workspace_min[i] <= pos_a[i] <= limits.workspace_max[i] for i in range(3))
    cmd = home.copy()
    cmd[0] += 0.01
    safe, violations = clamp_joint_cmd(cmd, home, limits, timed_out=False)
    assert safe.shape == (9,)
    assert violations == []
    wild = home.copy()
    wild[1] += 3.0  # far beyond the 2.0 rad/s * cmd_dt envelope
    wild[7] = -0.02  # snap the fingers closed in one tick
    safe, violations = clamp_joint_cmd(wild, home, limits, timed_out=False)
    assert violations, "a wild command must be clamped, not passed"
    assert np.all(safe >= np.asarray(limits.q_min, dtype=np.float32) - 1e-6)
    assert np.all(safe <= np.asarray(limits.q_max, dtype=np.float32) + 1e-6)
    tcp, _ = fk_ee_pose(safe[:7], embodiment="g1_right")
    assert all(limits.workspace_min[i] <= tcp[i] <= limits.workspace_max[i] for i in range(3))


def test_g1_workspace_covers_scene_targets_and_clears_torso_and_floor():
    """BG-2, SCN-4: the AABB admits the a7dd234 shelf band, tray top,
    staging heights and the home TCP, while excluding the torso column
    (x < 0.05), the sub-waist region (z < 0.70) and beyond-reach extents."""
    limits = load_limits("g1_right")
    lo = np.asarray(limits.workspace_min)
    hi = np.asarray(limits.workspace_max)
    covered = [
        (0.26, -0.15, 0.92),  # shelf L0 slot tops (0.86 + 0.06)
        (0.36, 0.05, 1.02),  # slot + pregrasp height 0.10
        (0.07, -0.43, 0.98),  # tray near-top corner (0.95 + 0.03)
        (0.19, -0.27, 0.98),  # tray far-top corner
        (0.30, -0.15, 1.00),  # staging / transfer z band
        (0.43559, -0.08916, 1.01284),  # home TCP (G1-H-r1 off-shelf)
    ]
    for point in covered:
        assert np.all(lo <= point) and np.all(point <= hi), point
    excluded = [
        (0.0, 0.0, 1.0),  # torso column
        (0.30, -0.15, 0.50),  # below the waist band (legs / fixture bases)
        (0.60, -0.15, 1.0),  # beyond the 0.42 m shoulder chain + margin
    ]
    for point in excluded:
        assert not (np.all(lo <= point) and np.all(point <= hi)), point


def test_so101_and_franka_guard_paths_do_not_regress():
    """BG-2 regression: opening the G1 branch changes nothing for so101
    (URDF chain + official limits) or franka (modified-DH flange FK)."""
    so_limits = load_limits("so101")
    assert so101_chain().joint_names != g1_right_chain().joint_names
    home = np.asarray(so_limits.fallback_qpos, dtype=np.float32)
    pos, rot = fk_ee_pose(home[:5], embodiment="so101")
    assert pos.shape == (3,) and rot.shape == (3, 3)
    cmd = home.copy()
    cmd[0] += 0.01
    safe, violations = clamp_joint_cmd(cmd, home, so_limits, timed_out=False)
    assert violations == [] and safe.shape == (6,)
    flange = fk_ee_pose(np.zeros(7), embodiment="franka")
    assert flange[0].shape == (3,)
