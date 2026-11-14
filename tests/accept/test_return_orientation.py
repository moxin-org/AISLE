"""Return-planner to staged-IK acceptance; no physics or scored task trial."""

import numpy as np
import pytest

pytestmark = pytest.mark.accept


@pytest.mark.parametrize("use_estimate", [False, True])
def test_return_grasp_is_accepted_by_the_actual_ik_planner(use_estimate):
    """CAP-5; TC-1/2: measured and fallback tray grasps form executable staged IK plans."""
    from aisle.nodes.ik_trajectory import StagedPlan
    from aisle.nodes.return_planner import return_grasp_and_slot, return_grasp_from_estimate
    from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout

    physics, meds = load_physics(), load_meds()
    layout = resolve_layout(physics, "franka")
    home = np.asarray(physics["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    pose, slot = return_grasp_and_slot("amoxicillin", meds, layout)
    if use_estimate:
        pose, slot = return_grasp_from_estimate({"pos": pose[:3]}, "amoxicillin", meds, layout)
    plan = StagedPlan(np.asarray(pose), tuple(slot), 0.15, home)
    assert plan.ok, plan.error
    assert len(plan.stages) > 0
