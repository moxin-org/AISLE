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


def profile_held_joint_names(profile: dict) -> tuple[str, ...]:
    """TC-5: joints held at home because the wire does not carry them.

    Empty tuple for every profile that does not declare held joints —
    including legacy full-coverage wires (franka/so101/mobile), which the
    ADR-g1-embodiment partition rules leave untouched.
    """
    return tuple(profile.get("held_joint_names", ()))


def _resolve_dof_indices(robot, names: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for name in names:
        joint = robot.get_joint(name)
        if joint.n_dofs != 1:
            raise ValueError(f"profile joint {name!r} has {joint.n_dofs} DOFs; expected one")
        indices.extend(joint.dofs_idx_local)
    return indices


def profile_dof_indices(robot, profile: dict) -> tuple[int, ...] | None:
    """Genesis-local DOF indices in the profile's TC-5 wire order.

    Profiles without held_joint_names keep the legacy contract: the wire
    alone must cover every robot DOF. Profiles with held_joint_names must
    partition the DOF set instead — wire and held are disjoint and their
    union is the whole robot; the return value stays wire-indices-only.
    """
    names = profile_joint_names(profile)
    held_names = profile_held_joint_names(profile)
    if names is None:
        if held_names:
            raise ValueError("held_joint_names requires arm_joint_names and gripper_joint_names")
        return None
    if not held_names:
        indices = _resolve_dof_indices(robot, names)
        if len(indices) != robot.n_dofs or len(set(indices)) != len(indices):
            raise ValueError(
                f"profile joint order covers DOFs {indices}, but robot has {robot.n_dofs} DOFs"
            )
        return tuple(indices)
    wire = _resolve_dof_indices(robot, names)
    held = _resolve_dof_indices(robot, held_names)
    overlap = sorted(set(wire) & set(held))
    if overlap:
        raise ValueError(f"wire and held_joint_names overlap on DOFs {overlap}")
    if len(set(wire) | set(held)) != robot.n_dofs:
        raise ValueError(
            f"wire ∪ held_joint_names covers {len(set(wire) | set(held))} DOFs, "
            f"but robot has {robot.n_dofs} DOFs"
        )
    return tuple(wire)


def held_dof_indices(robot, profile: dict) -> tuple[int, ...]:
    """Genesis-local DOF indices of the profile's held joints, in name order.

    () for profiles that declare none. Partition validity (disjoint from the
    wire, union == whole robot) is enforced by profile_dof_indices.
    """
    held_names = profile_held_joint_names(profile)
    if not held_names:
        return ()
    return tuple(_resolve_dof_indices(robot, held_names))


def from_wire_joint_order(
    values: np.ndarray, indices: tuple[int, ...], base: np.ndarray | None = None
) -> np.ndarray:
    """Map a TC-5 command from wire order into simulator-native DOF order.

    base=None keeps the legacy behavior: the native vector has the wire's
    shape and only the wire indices are defined. With base (a full-length
    native vector, held joints already at home) the result is a copy of
    base with only the wire positions overwritten.
    """
    values = np.asarray(values, dtype=np.float32)
    if base is None:
        native = np.empty_like(values)
        native[..., list(indices)] = values
        return native
    base = np.asarray(base, dtype=np.float32)
    native = base.copy()
    native[..., list(indices)] = values
    return native


def to_wire_joint_order(values: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Map simulator-native joint state into the TC-5 wire order."""
    return np.asarray(values)[..., list(indices)]
