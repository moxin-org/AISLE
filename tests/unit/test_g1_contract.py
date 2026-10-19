import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from aisle.embodiment import (
    G1_RIGHT_ARM_JOINTS,
    G1_RIGHT_GRIPPER_JOINTS,
    G1_RIGHT_JOINTS,
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
