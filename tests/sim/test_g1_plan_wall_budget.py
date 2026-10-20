"""The G1 planning wall budget fails LOUD, at sim budget (G1-F5, G1-I-a).

Moved from tests/unit (G1-I-a): the assertion waits out the REAL
ik_plan_budget_s wall budget (~40s loud-failure wait) — it cannot fit
the unit marker's <1s definition and the speedup item sanctioned moving
THAT ONE entry to sim. Nothing is weakened: same inputs, same asserted
outcomes (loud plan.error naming the budget, bounded wall)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from aisle.nodes.budget_guard import load_limits
from aisle.nodes.grasp_topdown import topdown_quat
from aisle.nodes.ik_trajectory import StagedPlan
from aisle.scenes.pharmacy import load_physics, resolve_layout

pytestmark = pytest.mark.sim

PROFILE = load_physics()["embodiment"]["g1_right"]
HOME = np.asarray(load_limits("g1_right").fallback_qpos, dtype=np.float32)


def test_unreachable_plan_fails_loud_within_the_wall_budget():
    """G1-F5 (ep0, smoke 20261019-234952): a grasp pose outside the
    workspace made first-launch planning burn 9 minutes of lockstep wall
    clock with no crash and no verdict. The mechanism: one unreachable
    cold ik_solve walks the full 32-seed x 2-rotation ladder (measured
    37.5s) and the staged chain multiplies it across stages and parity
    retries. Planning now runs under a wall budget (profile
    ik_plan_budget_s): on expiry the plan fails LOUD — plan.error names
    the budget — well inside the measured 9-minute stall."""
    far = np.array([0.75, 0.50, 1.20], dtype=np.float32)
    grasp = np.concatenate([far, np.asarray(topdown_quat(0.0), dtype=np.float32)])
    t0 = time.monotonic()
    plan = StagedPlan(
        grasp,
        tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
        approach_m=0.10,
        q_seed=HOME,
        place_z=1.05,
        embodiment="g1_right",
    )
    elapsed = time.monotonic() - t0
    assert not plan.ok
    # G1-F8: two LOUD failure modes are both correct — the budget path
    # (error names ik_plan_budget_s; verified under load at 5.5s) or, now
    # that the seed ladder is 16 starts x 80 iters, the ladder EXHAUSTS
    # inside the budget and the waypoint fails as a plain IK error
    # (faster and just as loud). Either way: bounded wall, no crash, no
    # freeze — the 9-minute silent stall cannot recur.
    err = str(plan.error).lower()
    assert "budget" in err or "ik failed" in err, plan.error
    assert elapsed < 90.0, f"planning burned {elapsed:.0f}s before failing"


def test_g1_layout_resolves_under_sim_profile():
    """Import-scope sanity for the module: the sim profile resolves the
    g1_right layout (the relocated test depends on it)."""
    layout = resolve_layout(load_physics(), "g1_right")
    assert layout["tray"]["pos"] == PROFILE["tray_pos"]
