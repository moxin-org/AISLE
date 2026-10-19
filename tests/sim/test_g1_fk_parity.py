"""Genesis cross-validation of the g1_right FK chain (G1-F1; BG-2, SCN-4,
CON-5, CON-7).

The guard's world-frame FK — pelvis fixed at the FK standing height, waist
at the all-zeros held home, TCP at the Dex1 grasp center — is checked
against Genesis itself, the only thing that can prove the base transform
and the TCP offset are right. Five deterministic right-arm configurations;
tolerances 1e-4 m / 1e-3 rad per the board entry.

Marker `sim`: imports genesis, runs headless (CON-12). Self-contained
minimal scene — lane1's scene code is not on this branch, so the URDF is
loaded directly with the same fixed-pelvis mounting lane1 pins."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None,
        reason="genesis not installed (uv sync --extra sim)",
    ),
]

_G1_URDF = "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf"
_PELVIS_Z = 0.7872
#: TCP in the right_dex1_base_link frame: grasp center at the finger-AABB
#: mid-x, palm +x -> tool +z (a7dd234). In the right_wrist_yaw_link frame
#: add the fixed right_base_joint 0.0415 => 0.1393 total along +x.
_TCP_XYZ_IN_DEX1_BASE = (0.0978, 0.0, 0.0)
_TCP_RPY = (0.0, math.pi / 2, 0.0)


def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _quat_wxyz_matrix(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_error(r_expected: np.ndarray, r_actual: np.ndarray) -> float:
    value = np.clip((np.trace(r_expected.T @ r_actual) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(value))


def test_g1_chain_fk_matches_genesis_tcp_over_random_configurations():
    """BG-2, SCN-4, CON-5, CON-7: for five deterministic random right-arm
    configurations (held DOFs at standing home, fingers at the open stop),
    the g1_right_chain() world-frame TCP pose equals Genesis's ee_link pose
    composed with the TCP offset — position within 1e-4 m, orientation
    within 1e-3 rad. This is the hard evidence that the fixed base
    transform (pelvis height, waist origins at held home) and the TCP
    offset (right_base_joint 0.0415 + grasp center 0.0978, rpy [0, pi/2,
    0]) are right."""
    import genesis as gs

    from aisle.embodiment import G1_RIGHT_ARM_JOINTS, G1_RIGHT_GRIPPER_JOINTS
    from aisle.kinematics import g1_right_chain
    from aisle.scenes.pharmacy import to_numpy

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning", seed=0)
    scene = gs.Scene()
    robot = scene.add_entity(gs.morphs.URDF(file=_G1_URDF, fixed=True, pos=(0.0, 0.0, _PELVIS_Z)))
    scene.build()

    # Direct per-joint DOF lookup (profile_dof_indices demands the full
    # wire∪held partition; the parity probe only needs the 9 wire indices).
    arm_indices = tuple(robot.get_joint(name).dofs_idx_local[0] for name in G1_RIGHT_ARM_JOINTS)
    finger_indices = tuple(
        robot.get_joint(name).dofs_idx_local[0] for name in G1_RIGHT_GRIPPER_JOINTS
    )
    # Genesis drops the massless right_dex1_base_link on load, so the TCP is
    # anchored at right_wrist_yaw_link with the FULL offset: fixed
    # right_base_joint 0.0415 + grasp center 0.0978 = 0.1393 along +x —
    # exactly the tip g1_right_chain() composes.
    ee_link = robot.get_link("right_wrist_yaw_link")
    tcp_offset = np.asarray((0.1393, 0.0, 0.0), dtype=float)
    tcp_rot = _rpy_matrix(_TCP_RPY)

    chain = g1_right_chain()
    rng = np.random.default_rng(7)
    worst_pos, worst_rot = 0.0, 0.0
    for _ in range(5):
        q_arm = rng.uniform(-0.6, 0.6, size=7)  # comfortably inside limits
        native = np.zeros(robot.n_dofs, dtype=np.float64)
        native[list(arm_indices)] = q_arm
        native[list(finger_indices)] = 0.0245  # fingers at the open stop
        robot.set_qpos(
            native
        )  # link poses follow qpos directly (rapier parity pattern); no dynamics step
        ee_pos = to_numpy(ee_link.get_pos()).reshape(-1)[:3]
        ee_rot = _quat_wxyz_matrix(to_numpy(ee_link.get_quat()).reshape(-1)[:4])
        genesis_tcp = ee_pos + ee_rot @ tcp_offset
        genesis_rot = ee_rot @ tcp_rot
        expected_pos, expected_rot = chain.forward(q_arm)
        worst_pos = max(worst_pos, float(np.abs(genesis_tcp - expected_pos).max()))
        worst_rot = max(worst_rot, _rotation_error(expected_rot, genesis_rot))
    assert worst_pos < 1e-4, f"FK position parity broke: {worst_pos:.3e} m"
    assert worst_rot < 1e-3, f"FK rotation parity broke: {worst_rot:.3e} rad"
    print(f"g1 FK parity: worst pos {worst_pos:.3e} m, worst rot {worst_rot:.3e} rad")
