"""Stable embodiment-level topic contracts (SPEC 010, TC-5)."""

import numpy as np

SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_GRIPPER_JOINTS = ("gripper",)
SO101_JOINTS = SO101_ARM_JOINTS + SO101_GRIPPER_JOINTS

# Unitree G1 right-arm profile `g1_right` (ADR-g1-embodiment, PROPOSED): the
# wire covers the right arm and the two prismatic Dex1-1 fingers in the
# official g1_29dof_mode_15_with_dex1_1 URDF order; every other G1 DOF is held.
G1_RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_RIGHT_GRIPPER_JOINTS = ("right_dex1_finger_joint_1", "right_dex1_finger_joint_2")
G1_RIGHT_JOINTS = G1_RIGHT_ARM_JOINTS + G1_RIGHT_GRIPPER_JOINTS


def profile_joint_names(profile: dict) -> tuple[str, ...] | None:
    """Configured TC-5 wire order, or None for legacy/native profiles."""
    arm = profile.get("arm_joint_names")
    gripper = profile.get("gripper_joint_names")
    if arm is None and gripper is None:
        return None
    if arm is None or gripper is None:
        raise ValueError("arm_joint_names and gripper_joint_names must be configured together")
    return tuple(arm) + tuple(gripper)


def profile_dof_indices(robot, profile: dict) -> tuple[int, ...] | None:
    """Genesis-local DOF indices in the profile's TC-5 wire order."""
    names = profile_joint_names(profile)
    if names is None:
        return None
    indices = []
    for name in names:
        joint = robot.get_joint(name)
        if joint.n_dofs != 1:
            raise ValueError(f"profile joint {name!r} has {joint.n_dofs} DOFs; expected one")
        indices.extend(joint.dofs_idx_local)
    if len(indices) != robot.n_dofs or len(set(indices)) != len(indices):
        raise ValueError(
            f"profile joint order covers DOFs {indices}, but robot has {robot.n_dofs} DOFs"
        )
    return tuple(indices)


def from_wire_joint_order(values: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Map a TC-5 command from wire order into simulator-native DOF order."""
    values = np.asarray(values, dtype=np.float32)
    native = np.empty_like(values)
    native[..., list(indices)] = values
    return native


def to_wire_joint_order(values: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Map simulator-native joint state into the TC-5 wire order."""
    return np.asarray(values)[..., list(indices)]
