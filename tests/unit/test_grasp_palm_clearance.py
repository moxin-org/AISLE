"""Palm clearance in the top-down grasp plan (CAP-5, SCN-2, ADR-72).

The fingertip check treats the gripper as two points. These tests pin the
hand-body rule added after M0-1 on Nexus (episodes 4 and 29): a taller
same-level neighbour under the hand's footprint raises the grasp so the palm
clears it, within a minimum finger engagement.
"""

import numpy as np
import pytest

from aisle.nodes.grasp_topdown import (
    GRIP_ENGAGEMENT,
    PLACE_DROP_GAP,
    palm_lift,
    plan_grasp,
)
from aisle.scenes.pharmacy import load_physics

PALM = {
    "half_length_m": 0.102,
    "half_width_m": 0.032,
    "above_tcp_m": 0.037,
    "clear_m": 0.005,
    "min_grip_m": 0.020,
}
# metformin on the desk, the only grippable axis is y (x half-width 0.035
# exceeds finger_open - finger_clear)
METFORMIN = (0.070, 0.035, 0.095)
TARGET = np.array([0.394, -0.225, 0.1075, 0, 0, 0, 1], dtype=np.float32)
BOTTOM = 0.1075 - 0.095 / 2
# cetirizine (0.065 x 0.030 x 0.110) 9 cm away along the grip axis
TALL_NEIGHBOUR = [0.395, -0.136, 0.0325, 0.015, 0.110]


def plan(neighbours, palm=PALM):
    return plan_grasp(
        TARGET,
        METFORMIN,
        tray_top_z=0.04,
        neighbours=neighbours,
        finger_open=0.04,
        finger_clear=0.008,
        palm=palm,
    )


def test_franka_profile_declares_the_palm_geometry():
    """SCN-2: the hand-body geometry is scene configuration, not a planner
    constant, and the franka profile carries the values the planner reads."""
    assert load_physics()["embodiment"]["franka"]["palm"] == PALM


def test_taller_neighbour_under_the_hand_raises_the_grasp():
    """CAP-5: with the palm 5 mm below the neighbour's top at the default
    engagement, the grasp rises; here the need (1.8 cm) exceeds what the
    minimum engagement allows, so the lift is the 1.5 cm cap."""
    baseline, _, _ = plan([TALL_NEIGHBOUR], palm=None)
    lifted, _, _ = plan([TALL_NEIGHBOUR])
    assert baseline[2] == pytest.approx(0.1075 + 0.0475 - GRIP_ENGAGEMENT, abs=1e-6)
    assert lifted[2] - baseline[2] == pytest.approx(GRIP_ENGAGEMENT - PALM["min_grip_m"], abs=1e-6)
    assert lifted[:2] == pytest.approx(baseline[:2], abs=1e-6)
    assert lifted[3:] == pytest.approx(baseline[3:], abs=1e-6)


def test_release_height_follows_the_shallower_grip():
    """CAP-5: a raised grasp holds the box lower in the hand, so the release
    TCP rises by the same amount and the box still hovers the drop gap."""
    _, _, place_z = plan([TALL_NEIGHBOUR])
    assert place_z == pytest.approx(0.04 + (0.095 - PALM["min_grip_m"]) + PLACE_DROP_GAP, abs=1e-6)


def test_lift_is_exact_when_the_cap_does_not_bind():
    """CAP-5: a neighbour 4 mm taller than the palm's underside needs
    4 mm + clear_m, which fits inside the engagement budget."""
    tcp_z = 0.12
    palm_bottom = tcp_z + PALM["above_tcp_m"]
    height = palm_bottom + 0.004 - BOTTOM
    lift = palm_lift(
        (0.394, -0.225),
        BOTTOM,
        tcp_z,
        0.0,
        [[0.395, -0.136, 0.0325, 0.015, height]],
        PALM,
        GRIP_ENGAGEMENT,
    )
    assert lift == pytest.approx(0.004 + PALM["clear_m"], abs=1e-9)


@pytest.mark.parametrize(
    "neighbour",
    [
        pytest.param([0.395, -0.136, 0.0325, 0.015, 0.080], id="shorter-than-palm"),
        pytest.param([0.394, -0.090, 0.0325, 0.015, 0.110], id="beyond-hand-length"),
        pytest.param([0.474, -0.136, 0.0325, 0.015, 0.110], id="beside-hand-width"),
        pytest.param([0.395, -0.136, 0.0325, 0.015], id="legacy-row-without-height"),
    ],
)
def test_neighbours_outside_the_palm_volume_leave_the_grasp_alone(neighbour):
    """CAP-5: only a neighbour that is both under the footprint and taller
    than the palm's underside changes the plan; a four-value row (no height)
    is tolerated and ignored."""
    baseline, _, base_place = plan([neighbour], palm=None)
    planned, _, place = plan([neighbour])
    assert planned == pytest.approx(baseline, abs=1e-9)
    assert place == pytest.approx(base_place, abs=1e-9)


def test_profiles_without_palm_geometry_keep_the_previous_plan():
    """CAP-5: the rule is opt-in per embodiment; without a palm block the
    planner behaves exactly as before."""
    with_rows, _, _ = plan([TALL_NEIGHBOUR], palm=None)
    assert with_rows[2] == pytest.approx(0.1075 + 0.0475 - GRIP_ENGAGEMENT, abs=1e-6)
