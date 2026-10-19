"""Deterministic kinematics derived from pinned robot descriptions.

SO-101 has no parallel hand-maintained DH table here: origins, axes, limits,
and the fixed TCP frame are parsed from the owner-approved vendored URDF
(ADR-27).  This keeps the planner and safety guard on the same official model
used by Genesis.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from aisle.embodiment import G1_RIGHT_ARM_JOINTS, SO101_ARM_JOINTS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SO101_URDF = _REPO_ROOT / "assets" / "so101" / "so101.urdf"
_G1_URDF = _REPO_ROOT / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf"
#: G1 fixed-pelvis standing height: FK sole probe on the vendored URDF at
#: the all-zeros held home puts both soles at z = -0.7872, so pinning the
#: pelvis here stands the robot on the ground (a7dd234 physics profile).
_G1_PELVIS_HEIGHT_M = 0.7872
#: G1 TCP offset in the right_dex1_base_link frame (a7dd234): the grasp
#: center sits at the finger-AABB mid-x. Derivation: the fixed
#: right_base_joint translates 0.0415 m from right_wrist_yaw_link to the
#: dex1 base, and the between-fingers grasp center adds 0.0978 m
#: (0.0415 + 0.0978 = 0.1393 m along the palm +x); rpy [0, pi/2, 0]
#: turns palm +x into tool +z for top-down grasps.
_G1_TCP_XYZ = (0.0978, 0.0, 0.0)
_G1_TCP_RPY = (0.0, math.pi / 2, 0.0)


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()] if text else default, dtype=float)


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if origin is None:
        return transform
    transform[:3, 3] = _vector(origin.get("xyz"), (0.0, 0.0, 0.0))
    transform[:3, :3] = _rpy_rotation(_vector(origin.get("rpy"), (0.0, 0.0, 0.0)))
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("actuated URDF joint has a zero axis")
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    velocity: float | None


@dataclass(frozen=True)
class UrdfChain:
    """One serial URDF chain from a base link to an end-effector link.

    G1-F1: an optional fixed ``base_transform`` (world <- base link) and
    ``tip_transform`` (ee link <- tool frame) compose into ``forward`` so a
    humanoid arm anchored at a fixed pelvis reports WORLD-frame poses. Both
    default to identity, so every existing chain (so101) is unchanged
    bit-for-bit.
    """

    joints: tuple[UrdfJoint, ...]
    base_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    tip_transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    @property
    def actuated(self) -> tuple[UrdfJoint, ...]:
        return tuple(joint for joint in self.joints if joint.kind != "fixed")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.actuated)

    @property
    def q_min(self) -> tuple[float, ...]:
        return tuple(float(joint.lower) for joint in self.actuated)

    @property
    def q_max(self) -> tuple[float, ...]:
        return tuple(float(joint.upper) for joint in self.actuated)

    @property
    def qdot_max(self) -> tuple[float, ...]:
        return tuple(float(joint.velocity) for joint in self.actuated)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.shape != (len(self.actuated),):
            raise ValueError(f"expected {len(self.actuated)} joints, got {values.shape}")
        transform = self.base_transform.copy()
        value_index = 0
        for joint in self.joints:
            transform = transform @ joint.origin
            if joint.kind != "fixed":
                transform = transform @ _axis_rotation(joint.axis, float(values[value_index]))
                value_index += 1
        transform = transform @ self.tip_transform
        return transform[:3, 3].copy(), transform[:3, :3].copy()


def load_urdf_chain(path: Path, base_link: str, target_link: str) -> UrdfChain:
    root = ET.parse(path).getroot()
    by_child: dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or "link" not in child.attrib:
            raise ValueError(f"URDF joint {joint.get('name')!r} has no child link")
        by_child[child.attrib["link"]] = joint

    elements: list[ET.Element] = []
    link = target_link
    seen = set()
    while link != base_link:
        if link in seen or link not in by_child:
            raise ValueError(f"no serial URDF chain from {base_link!r} to {target_link!r}")
        seen.add(link)
        joint = by_child[link]
        elements.append(joint)
        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            raise ValueError(f"URDF joint {joint.get('name')!r} has no parent link")
        link = parent.attrib["link"]

    joints = []
    for element in reversed(elements):
        kind = element.attrib["type"]
        limit = element.find("limit")
        if kind != "fixed" and limit is None:
            raise ValueError(f"actuated URDF joint {element.attrib['name']!r} has no limits")
        joints.append(
            UrdfJoint(
                name=element.attrib["name"],
                kind=kind,
                origin=_origin_transform(element.find("origin")),
                axis=_vector(
                    element.find("axis").get("xyz") if element.find("axis") is not None else None,
                    (1.0, 0.0, 0.0),
                ),
                lower=float(limit.attrib["lower"]) if limit is not None else None,
                upper=float(limit.attrib["upper"]) if limit is not None else None,
                velocity=float(limit.attrib["velocity"]) if limit is not None else None,
            )
        )
    return UrdfChain(tuple(joints))


@cache
def so101_chain() -> UrdfChain:
    """Official five-axis arm through the fixed ``gripper_frame_link`` TCP."""
    chain = load_urdf_chain(_SO101_URDF, "base_link", "gripper_frame_link")
    if chain.joint_names != SO101_ARM_JOINTS:
        raise ValueError(
            f"official SO-101 chain order {chain.joint_names!r} does not match TC-5 "
            f"{SO101_ARM_JOINTS!r}"
        )
    return chain


@cache
def g1_right_chain() -> UrdfChain:
    """G1 right arm through the Dex1 grasp-center TCP, in WORLD frame.

    Base: ``torso_link`` with the pelvis fixed at the FK standing height
    (0.7872 m) and the three waist joints at the all-zeros held home — the
    a7dd234 scene's fixed-pelvis mounting, so guard FK, planner FK, and
    Genesis all report the same world frame. Tip: the fixed
    right_base_joint (0.0415 m) plus the 0.0978 m between-fingers grasp
    center with rpy [0, pi/2, 0] (see _G1_TCP_XYZ above for the full
    derivation). The seven revolute joints MUST match G1_RIGHT_ARM_JOINTS
    (TC-5 wire order = URDF document order for the right arm).
    """
    root = ET.parse(_G1_URDF).getroot()
    by_child = {
        joint.find("child").attrib["link"]: joint
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    base = np.eye(4)
    base[2, 3] = _G1_PELVIS_HEIGHT_M
    waist = []
    link = "torso_link"
    while link != "pelvis":  # waist at held home (all zeros): origins only
        if link not in by_child:
            raise ValueError(f"no URDF chain from pelvis to {link!r}")
        joint = by_child[link]
        waist.append(joint)
        link = joint.find("parent").attrib["link"]
    for joint in reversed(waist):  # pelvis -> torso_link order
        base = base @ _origin_transform(joint.find("origin"))

    tip = np.eye(4)
    tip[:3, 3] = _vector(
        by_child["right_dex1_base_link"].find("origin").get("xyz"), (0.0, 0.0, 0.0)
    ) + np.asarray(_G1_TCP_XYZ, dtype=float)
    tip[:3, :3] = _rpy_rotation(np.asarray(_G1_TCP_RPY, dtype=float))

    chain = load_urdf_chain(_G1_URDF, "torso_link", "right_wrist_yaw_link")
    if chain.joint_names != G1_RIGHT_ARM_JOINTS:
        raise ValueError(
            f"G1 right-arm chain order {chain.joint_names!r} does not match "
            f"G1_RIGHT_ARM_JOINTS {G1_RIGHT_ARM_JOINTS!r}"
        )
    return UrdfChain(chain.joints, base_transform=base, tip_transform=tip)
