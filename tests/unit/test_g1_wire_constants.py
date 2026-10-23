"""G1 right-arm wire constants (TC-5 draft, ADR-g1-embodiment)."""

import pytest

from aisle.embodiment import G1_RIGHT_ARM_JOINTS, G1_RIGHT_GRIPPER_JOINTS, G1_RIGHT_JOINTS

pytestmark = pytest.mark.unit


def test_g1_right_wire_order_is_arm_then_fingers():
    """TC-5: the g1_right wire is 7 right-arm joints followed by 2 Dex1-1 fingers."""
    assert len(G1_RIGHT_ARM_JOINTS) == 7
    assert len(G1_RIGHT_GRIPPER_JOINTS) == 2
    assert G1_RIGHT_JOINTS == G1_RIGHT_ARM_JOINTS + G1_RIGHT_GRIPPER_JOINTS
    assert len(set(G1_RIGHT_JOINTS)) == 9
    assert all(name.startswith("right_") for name in G1_RIGHT_JOINTS)
