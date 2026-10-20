"""The G1 planning lattice, at sim budget (G1-H-r1 systematic tests,
relocated G1-I-a: every (slot, med) plan solves in ~5.5s of real IK —
25 combinations cannot fit the unit marker's <1s definition, so the
whole lattice lives here. NOTHING is weakened: same plan_grasp entry,
same StagedPlan assertions, same executor-interpolation guard stream and
duration model, verbatim from the unit original (SCN-4, BG-3)."""

from __future__ import annotations

import numpy as np
import pytest

from aisle.nodes.budget_guard import clamp_joint_cmd, fk_ee_pose, load_limits
from aisle.nodes.grasp_topdown import GRIP_ENGAGEMENT, plan_grasp
from aisle.nodes.ik_trajectory import StagedPlan, interpolate_step
from aisle.scenes.pharmacy import (
    MED_NAMES,
    load_meds,
    load_physics,
    resolve_layout,
)

pytestmark = pytest.mark.sim

PHYSICS = load_physics()
PROFILE = PHYSICS["embodiment"]["g1_right"]
LAYOUT = resolve_layout(PHYSICS, "g1_right")
MEDS = load_meds()
LIMITS = load_limits("g1_right")
HOME = np.asarray(LIMITS.fallback_qpos, dtype=np.float32)
TRAY_TOP = PROFILE["tray_pos"][2] + PROFILE["tray_size"][2] / 2
JITTER = float(PROFILE["placement_global_jitter_m"])


def _box_aabb(slot_xy: tuple[float, float], med: str) -> tuple[np.ndarray, np.ndarray]:
    """Worst-case (jitter-inclusive) world AABB of `med` standing on `slot`."""
    shelf = LAYOUT["shelf"]
    size = MEDS[med]["size"]
    z_lo = shelf["pos"][2] + shelf["level_heights"][0] + shelf["board_thickness"] / 2
    lo = np.array([slot_xy[0] - size[0] / 2 - JITTER, slot_xy[1] - size[1] / 2 - JITTER, z_lo])
    hi = np.array(
        [slot_xy[0] + size[0] / 2 + JITTER, slot_xy[1] + size[1] / 2 + JITTER, z_lo + size[2]]
    )
    return lo, hi


def _grasp_for_slot_med(slot_xy: tuple[float, float], med: str):
    """The same plan_grasp entry the expert graph uses, aimed at `med` on
    `slot`; neighbours = the other slots at their worst-case footprint."""
    grasp_z = _box_aabb(slot_xy, med)[0][2] + MEDS[med]["size"][2] / 2
    pose = np.array([slot_xy[0], slot_xy[1], grasp_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    widest = max(MED_NAMES, key=lambda n: MEDS[n]["size"][0])
    neighbours = [
        [float(s[0]), float(s[1]), MEDS[widest]["size"][0] / 2, MEDS[widest]["size"][1] / 2]
        for s in PROFILE["placement_slots_xy"]
        if (float(s[0]), float(s[1])) != slot_xy
    ]
    return plan_grasp(
        pose,
        MEDS[med]["size"],
        grip=GRIP_ENGAGEMENT,
        shelf_front_x=LAYOUT["shelf"]["pos"][0] - LAYOUT["shelf"]["level_size"][0] / 2,
        tray_top_z=TRAY_TOP,
        neighbours=neighbours,
        finger_open=float(PROFILE["gripper_open_m"]),
        finger_clear=float(PROFILE["gripper_finger_clear_m"]),
        topdown_approach=float(PROFILE["pregrasp_height_m"]),
    )


_PLAN_CACHE: dict = {}


def _cached_plan(cache, slot_xy, med):
    key = (slot_xy, med)
    if key not in cache:
        grasp, approach, place_z = _grasp_for_slot_med(slot_xy, med)
        plan = StagedPlan(
            grasp,
            tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
            approach_m=approach,
            q_seed=HOME,
            place_z=place_z,
            embodiment="g1_right",
        )
        assert plan.ok, (slot_xy, med, plan.error)
        cache[key] = plan
    return cache[key]


#: G1-F5 execution-budget model: the MJCF actuators sag ~0.08 rad under
#: gravity (the StageStreamer integral comment), so the duration model
#: feeds the streamer a sagging qpos (cmd - 0.08 on every joint) — the
#: ideal-actuator model measured 26.9s worst while the sagged one runs
#: 32.4s with contact bails (ep1's 60s overrun shape).
_SAG_RAD = 0.08
#: Grasp stabilization margin: the episode also spends ~2 read-move park
#: cycles (2 x (2.2s move + 0.4s settle)) plus perception/reset/verdict
#: settle before and after the plan — 20s total, deducted from the tier's
#: episode timeout for the execution assertion.
_MARGIN_S = 20.0


def _execution_seconds(plan, home: np.ndarray, vel_max: float = 1.0) -> float:
    """Plan execution duration under the real StageStreamer cadence with a
    sagging actuator (G1-F5): ticks at the TC-4 100 Hz joint_state cadence
    until the streamer reports done — the same completion gates (track,
    settle, 4s bail) the executor runs in sim."""
    from aisle.nodes.ik_trajectory import StageStreamer

    streamer = StageStreamer(list(plan.stages), home, 0.01, vel_max, embodiment="g1_right")
    sag = np.full(7, _SAG_RAD, dtype=np.float32)
    qpos = np.concatenate([home[:7], [0.0245, 0.0245]]).astype(np.float32)
    seconds = 0.0
    while not streamer.done and seconds < 120.0:
        full_cmd, _, _ = streamer.step(qpos)
        if full_cmd is not None:
            qpos = (full_cmd[:7] - sag).astype(np.float32)
        seconds += 0.01
    return seconds


def test_every_slot_execution_fits_the_episode_budget():
    """SCN-4, BG-3 (G1-F5): for every slot x med, the plan's EXECUTION
    duration under the real streamer cadence and a sagging actuator fits
    the tier's episode timeout (read from the rollout tier budgets, not a
    hardcoded 60) minus the 20s grasp-stabilization margin — ep1 of the
    20261019-234952 smoke ran the full 60s budget without finishing."""
    from aisle.harness.rollout import tier_budgets

    episode_s, _ = tier_budgets("T0")
    budget = episode_s - _MARGIN_S
    cache: dict = _PLAN_CACHE
    worst = 0.0
    worst_case = None
    for slot in PROFILE["placement_slots_xy"]:
        slot_xy = (float(slot[0]), float(slot[1]))
        for med in MED_NAMES:
            plan = _cached_plan(cache, slot_xy, med)
            seconds = _execution_seconds(plan, HOME)
            if seconds > worst:
                worst, worst_case = seconds, (slot_xy, med)
    assert worst <= budget, f"worst {worst:.1f}s at {worst_case} vs budget {budget:.0f}s"


#: G1-F3 (lane2): the whole lattice plans — the same-side far slot
#: [0.28, -0.24] needed three planner fixes (tray-roll and lower-stage
#: tool-z parity retries, and a rise solve chosen from the seed ladder
#: by executor-arc containment); the xfail markers the G1-H-r1 probe
#: left here are gone, so any slot regression now fails loud.
@pytest.mark.parametrize("slot_xy", [tuple(map(float, x)) for x in PROFILE["placement_slots_xy"]])
@pytest.mark.parametrize("med", MED_NAMES)
def test_every_slot_med_plans_and_passes_the_guard(slot_xy, med):
    """SCN-4, BG-3: for every slot x every med (25 combinations — the whole
    lattice, not a lucky seed), the StagedPlan entry solves home -> grasp ->
    tray -> home with no plan.error, every waypoint TCP stays inside the
    workspace AABB, and the executor's per-tick interpolated stream is
    guard-clean — the cross-body lesson made systematic."""
    plan = _cached_plan(_PLAN_CACHE, slot_xy, med)
    assert plan.ok, plan.error
    lo = np.asarray(LIMITS.workspace_min)
    hi = np.asarray(LIMITS.workspace_max)
    fingers_open = float(PROFILE["gripper_open_qpos"])
    prev = np.concatenate([HOME[:7], [fingers_open, fingers_open]]).astype(np.float32)
    current = HOME[:7].copy()
    for stage in plan.stages:
        for q in stage.path:
            target_q = np.asarray(q, dtype=np.float32)
            tcp = fk_ee_pose(target_q, embodiment="g1_right")[0]
            assert np.all(lo <= tcp) and np.all(tcp <= hi), (slot_xy, med, stage.name, tcp)
            while not np.allclose(current, target_q, atol=1e-6):
                current = interpolate_step(current, target_q, 1.0, 0.01)
                full = np.concatenate([current, [fingers_open, fingers_open]]).astype(np.float32)
                _, violations = clamp_joint_cmd(full, prev, LIMITS, timed_out=False)
                prev = full
                assert violations == [], (slot_xy, med, stage.name, violations)
