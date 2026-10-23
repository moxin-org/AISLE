import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from aisle.embodiment import (
    G1_RIGHT_ARM_JOINTS,
    G1_RIGHT_GRIPPER_JOINTS,
    G1_RIGHT_JOINTS,
    from_wire_joint_order,
    held_dof_indices,
    profile_dof_indices,
    profile_joint_names,
    to_wire_joint_order,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "assets" / "g1"
URDF = ASSET_ROOT / "g1_29dof_mode_15_with_dex1_1.urdf"
OFFICIAL_COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
OFFICIAL_REPO = "https://github.com/unitreerobotics/unitree_ros"
OFFICIAL_PATH = "robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf"
#: The mode-15 machine drives 29 body DOFs; the two Dex1-1 fingers per hand
#: add 4 prismatic DOFs, so the actuated tree is 33 DOFs (robot name says 29).
TOTAL_DOF = 33
DEX_FINGER_LIMIT = (-0.02, 0.0245)


def _urdf_joint_order(root):
    """Actuated joint names in URDF document order (fixed/floating excluded)."""
    return [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib["type"] not in ("fixed", "floating")
    ]


def _urdf_joints(root):
    return {joint.attrib["name"]: joint for joint in root.findall("joint")}


def test_tc5_g1_right_wire_contract():
    """TC-5: the g1_right profile wires the right arm's seven joints followed
    by the two prismatic Dex1-1 fingers, in official URDF order."""
    assert len(G1_RIGHT_ARM_JOINTS) == 7
    assert len(G1_RIGHT_GRIPPER_JOINTS) == 2
    assert G1_RIGHT_JOINTS == G1_RIGHT_ARM_JOINTS + G1_RIGHT_GRIPPER_JOINTS


def test_scn4_g1_asset_provenance_and_mesh_closure():
    """SCN-4, CON-5: the G1 asset is pinned to the owner-approved official
    unitree_ros commit and every relative mesh reference is vendored."""
    with open(ASSET_ROOT / "PROVENANCE.toml", "rb") as f:
        provenance = tomllib.load(f)
    assert provenance["source"] == {
        "repository": OFFICIAL_REPO,
        "commit": OFFICIAL_COMMIT,
        "path": OFFICIAL_PATH,
        "license": "BSD-3-Clause",
    }
    assert provenance["retrieval"]["date"] == "2026-10-19"
    assert (ASSET_ROOT / "LICENSE").is_file()
    root = ET.parse(URDF).getroot()
    meshes = {node.attrib["filename"] for node in root.findall(".//mesh")}
    assert meshes
    assert all(not Path(mesh).is_absolute() and (ASSET_ROOT / mesh).is_file() for mesh in meshes)


def test_scn4_g1_urdf_right_arm_order_and_fingers():
    """TC-5, SCN-4: the vendored URDF declares the right-arm joints in the
    same order as aisle.embodiment.G1_RIGHT_ARM_JOINTS, and the two right
    Dex1-1 fingers are prismatic with the official stroke limits."""
    root = ET.parse(URDF).getroot()
    joints = _urdf_joints(root)
    document_order = [joint.attrib["name"] for joint in root.findall("joint")]
    urdf_right_arm = [name for name in document_order if name in set(G1_RIGHT_ARM_JOINTS)]
    assert tuple(urdf_right_arm) == G1_RIGHT_ARM_JOINTS
    for name in G1_RIGHT_GRIPPER_JOINTS:
        joint = joints[name]
        assert joint.attrib["type"] == "prismatic"
        limit = joint.find("limit")
        assert limit is not None
        assert (float(limit.attrib["lower"]), float(limit.attrib["upper"])) == DEX_FINGER_LIMIT


ACTUATED_URDF_ORDER = tuple(_urdf_joint_order(ET.parse(URDF).getroot()))
G1_HELD_JOINTS = tuple(name for name in ACTUATED_URDF_ORDER if name not in set(G1_RIGHT_JOINTS))


class _FakeJoint:
    n_dofs = 1

    def __init__(self, index):
        self.dofs_idx_local = [index]


class _FakeG1:
    """Fake Robot in the test_so101_contract.py:49 style: a parser-order→DOF
    map the wire resolution must not trust."""

    def __init__(self, parser_order):
        self.n_dofs = len(parser_order)
        self.by_name = {name: _FakeJoint(index) for index, name in enumerate(parser_order)}

    def get_joint(self, name):
        return self.by_name[name]


def _g1_right_profile():
    return {
        "arm_joint_names": list(G1_RIGHT_ARM_JOINTS),
        "gripper_joint_names": list(G1_RIGHT_GRIPPER_JOINTS),
        "held_joint_names": list(G1_HELD_JOINTS),
    }


def test_tc5_g1_right_resolves_wire_through_shuffled_parser():
    """TC-5: g1_right resolves wire DOF indices by joint name even when the
    simulator's parser order is reversed; held joints take the rest."""
    robot = _FakeG1(tuple(reversed(ACTUATED_URDF_ORDER)))
    profile = _g1_right_profile()
    wire = profile_dof_indices(robot, profile)
    assert wire == tuple(robot.by_name[name].dofs_idx_local[0] for name in G1_RIGHT_JOINTS)
    held = held_dof_indices(robot, profile)
    assert held == tuple(robot.by_name[name].dofs_idx_local[0] for name in G1_HELD_JOINTS)
    assert not set(wire) & set(held)
    assert set(wire) | set(held) == set(range(robot.n_dofs))


def test_tc5_held_partition_rejects_overlap_and_gaps():
    """TC-5: a declared held set must partition the DOFs — overlapping or
    under-covering wire∪held raises instead of silently mis-driving DOFs."""
    robot = _FakeG1(ACTUATED_URDF_ORDER)
    overlapping = _g1_right_profile()
    overlapping["held_joint_names"] = list(G1_HELD_JOINTS[:1]) + [G1_RIGHT_JOINTS[0]]
    with pytest.raises(ValueError, match="overlap"):
        profile_dof_indices(robot, overlapping)
    gappy = _g1_right_profile()
    gappy["held_joint_names"] = list(G1_HELD_JOINTS[:-1])
    with pytest.raises(ValueError, match="DOFs"):
        profile_dof_indices(robot, gappy)
    with pytest.raises(ValueError, match="arm_joint_names"):
        profile_dof_indices(robot, {"held_joint_names": list(G1_HELD_JOINTS)})


def test_tc5_from_wire_overrides_only_wire_positions():
    """TC-5: with a home base the wire command overwrites only the wire
    positions; held joints keep the base value, and the wire round-trips."""
    robot = _FakeG1(ACTUATED_URDF_ORDER)
    wire_indices = profile_dof_indices(robot, _g1_right_profile())
    held_indices = held_dof_indices(robot, _g1_right_profile())
    wire_values = np.linspace(-0.5, 0.5, 9, dtype=np.float32)
    home = np.zeros(robot.n_dofs, dtype=np.float32)
    home[list(held_indices)] = 0.25
    native = from_wire_joint_order(wire_values, wire_indices, base=home)
    assert native[list(held_indices)].tolist() == [0.25] * len(held_indices)
    assert to_wire_joint_order(native, wire_indices).tolist() == wire_values.tolist()
    # Legacy branch is bit-identical for full-coverage wires (the only case
    # the old signature defined): a reversed permutation stays reversed.
    full_perm = tuple(reversed(range(robot.n_dofs)))
    full_values = np.arange(robot.n_dofs, dtype=np.float32)
    legacy = from_wire_joint_order(full_values, full_perm)
    assert legacy.tolist() == full_values[::-1].tolist()


def test_tc5_legacy_profiles_keep_exact_contract():
    """TC-5: profiles without held_joint_names keep the legacy behavior —
    full-coverage wires still resolve, bare profiles still return None and
    an empty held set."""
    robot = _FakeG1(ACTUATED_URDF_ORDER)
    full = {
        "arm_joint_names": list(ACTUATED_URDF_ORDER[:-2]),
        "gripper_joint_names": list(ACTUATED_URDF_ORDER[-2:]),
    }
    assert profile_dof_indices(robot, full) == tuple(range(robot.n_dofs))
    assert profile_joint_names({}) is None
    assert profile_dof_indices(robot, {}) is None
    assert held_dof_indices(robot, full) == ()


def test_scn4_g1_urdf_total_dof_and_mode():
    """SCN-4: the official g1_29dof_mode_15_with_dex1_1 URDF has 33 actuated
    DOFs (29 body + 4 Dex1-1 fingers; the floating base is not a DOF) and
    robot name g1_29dof_mode_15 under mode_machine 15."""
    root = ET.parse(URDF).getroot()
    assert root.attrib["name"] == "g1_29dof_mode_15"
    actuated = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib["type"] not in ("fixed", "floating")
    ]
    assert len(actuated) == TOTAL_DOF
    readme = (ASSET_ROOT / "UPSTREAM_README.md").read_text()
    row = next(line for line in readme.splitlines() if "g1_29dof_mode_15_with_dex1_1" in line)
    assert "`g1_29dof_mode_15_with_dex1_1`" in row
    mode_column = row.split("|")[2].strip()
    assert mode_column == "15"
