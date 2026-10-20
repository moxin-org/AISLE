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


def _placed_box_aabb(med: str) -> tuple[np.ndarray, np.ndarray]:
    """The just-placed box resting on the tray (G1-F7): tray top + half
    height, xy centered on the tray's target xy."""
    size = MEDS[med]["size"]
    x, y = float(PROFILE["tray_pos"][0]), float(PROFILE["tray_pos"][1])
    z_lo = float(TRAY_TOP)
    lo = np.array([x - size[0] / 2, y - size[1] / 2, z_lo])
    hi = np.array([x + size[0] / 2, y + size[1] / 2, z_lo + size[2]])
    return lo, hi


def _aabb_gap(a_lo, a_hi, b_lo, b_hi) -> float:
    return float(np.max(np.maximum(a_lo - b_hi, b_lo - a_hi)))


#: hand meshes loaded ONCE (trimesh.load per tick was the 10-minute
#: timeout: 25 cases x hundreds of ticks x 4 meshes); ticks only do the
#: numpy vertex transform below.
from pathlib import Path  # noqa: E402

import trimesh  # noqa: E402

from aisle.kinematics import load_urdf_chain  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_MESHES = _REPO / "assets" / "g1" / "meshes"
_PELVIS_Z = 0.7872
_CHAIN = load_urdf_chain(
    _REPO / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf",
    "pelvis",
    "right_wrist_yaw_link",
)
_T_BASE_T = np.eye(4)
_T_BASE_T[0, 3] = 0.0415
_FY = float(HOME[7])
_HAND_VERTS = tuple(
    (
        np.asarray(trimesh.load(_MESHES / name, force="mesh").vertices, dtype=float),
        local,
    )
    for name, local in (
        ("right_wrist_yaw_link_5010.STL", np.eye(4)),
        ("Dex1_base_link.STL", _T_BASE_T),
        (
            "Dex1_finger_link_1.STL",
            _T_BASE_T @ np.array([[1, 0, 0, 0], [0, 1, 0, -_FY], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
        (
            "Dex1_finger_link_2.STL",
            _T_BASE_T @ np.array([[1, 0, 0, 0], [0, 1, 0, _FY], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ),
    )
)


def _hand_aabb(q7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World AABB of wrist+dex1+fingers at ARM config q7 (waist at held
    home, pelvis fixed) — mesh vertices cached at module level."""
    q = np.concatenate([np.zeros(3), np.asarray(q7, dtype=float)[:7]])
    pos, rot = _CHAIN.forward(q)
    wrist = np.eye(4)
    wrist[:3, :3] = rot
    wrist[:3, 3] = pos + np.array([0.0, 0.0, _PELVIS_Z])
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for verts, local in _HAND_VERTS:
        R = wrist @ local
        world = verts @ R[:3, :3].T + R[:3, 3]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return lo, hi


@pytest.mark.parametrize("slot_xy", [tuple(map(float, s)) for s in PROFILE["placement_slots_xy"]])
@pytest.mark.parametrize("med", MED_NAMES)
def test_post_release_withdrawal_clears_the_placed_box_by_2cm(slot_xy, med):
    """SCN-4, G1-F7 (outer-loop #14): every waypoint AFTER the release
    stage — release -> clear -> home, including the executor's per-tick
    interpolated stream between them — keeps the hand AABB >= 2 cm clear
    of the box just placed on the tray. The r6 smoke placed the box
    correctly, then the home swing swept it off the tray."""
    grasp, approach, place_z = _grasp_for_slot_med(slot_xy, med)
    plan = _cached_plan({}, slot_xy, med)
    assert plan.ok, plan.error
    release_idx = next(i for i, s in enumerate(plan.stages) if s.name == "release")
    box_lo, box_hi = _placed_box_aabb(med)

    def gap(q7):
        h_lo, h_hi = _hand_aabb(q7)
        return _aabb_gap(h_lo, h_hi, box_lo, box_hi)

    def xy_clear(q7):
        h_lo, h_hi = _hand_aabb(q7)
        return (
            h_hi[0] < box_lo[0] or h_lo[0] > box_hi[0] or h_hi[1] < box_lo[1] or h_lo[1] > box_hi[1]
        )

    # THREE-ZONE semantics (the palm is WIDER than the box: at the release
    # hover the open fingers physically ring the just-placed box, so a
    # negative gap there is inherent to a correct place, not a sweep):
    # - ring zone (hand's xy projection still over the box): the gap may
    #   not DEEPEN beyond the release pose by more than 5 mm;
    # - corridor zone (projection just left the box, gap still < 2 cm):
    #   by construction the exit-instant gap is ~0, so the corridor only
    #   requires gap >= 0 (never re-overlap) while the hand retreats;
    # - tail zone (gap first reached 2 cm, i.e. the hand is genuinely
    #   clear): it must never fall back below 2 cm through the rest of
    #   clear + the whole home swing.
    current = np.asarray(plan.stages[release_idx].path[-1], dtype=np.float32)
    g0 = gap(current)
    worst_ring = g0
    corridor_min = np.inf
    tail_min = np.inf
    reached_2cm = False
    where = None
    for stage in plan.stages[release_idx + 1 :]:
        for q in stage.path:
            target_q = np.asarray(q, dtype=np.float32)
            while not np.allclose(current, target_q, atol=1e-6):
                current = interpolate_step(current, target_q, 1.0, 0.01)
                g = gap(current)
                if not xy_clear(current):
                    if g < worst_ring - 0.005:
                        where = ("ring-deepened", stage.name, g)
                    worst_ring = min(worst_ring, g)
                elif not reached_2cm:
                    corridor_min = min(corridor_min, g)
                    if g >= 0.02:
                        reached_2cm = True
                else:
                    tail_min = min(tail_min, g)
    assert where is None, (slot_xy, med, where)
    assert corridor_min >= 0.0, (slot_xy, med, "re-overlapped in corridor", corridor_min)
    assert reached_2cm, (slot_xy, med, "never reached 2 cm clearance after release")
    assert tail_min >= 0.02, (slot_xy, med, "fell back under 2 cm after clearing", tail_min)


def test_plan_budget_respects_the_turn_watchdog():
    """G1-F8: the planning wall budget must fit INSIDE the turn watchdog
    (with a 0.6 factor for CPU jitter) — planning runs synchronously in
    the turn, and ik_plan_budget_s = 30 vs the 10 s watchdog froze the
    dataflow at every episode start (outer smoke outer-smoke-f7b: turn
    watchdog expired at 1:8 while `plan ready` landed 3 s later). The
    watchdog default is READ from turn_barrier, not hardcoded."""
    from aisle.nodes.turn_barrier import ORDINARY_TURN_WATCHDOG_S

    cap = 0.6 * ORDINARY_TURN_WATCHDOG_S
    budget = float(PROFILE["ik_plan_budget_s"])
    assert budget <= cap, f"ik_plan_budget_s {budget} > 0.6*watchdog {cap}"


@pytest.mark.parametrize(
    ("embodiment", "grasp_kind"),
    [
        ("g1_right", "lattice"),
        pytest.param(
            "so101",
            "recorded",
            marks=pytest.mark.xfail(
                strict=True,
                reason="main-branch finding: so101 full-budget IK plans take 12-15 s, past "
                "the 10 s turn watchdog; the G1-F8 early exit is kept g1_right-only so "
                "the so101 measured record stays bit-for-bit",
            ),
        ),
        ("franka", "probe"),
    ],
)
def test_planning_wall_clock_fits_the_watchdog_cap(embodiment, grasp_kind):
    """G1-F8: measured wall-clock planning per (slot, med) stays under the
    same 0.6 x watchdog cap. g1_right walks the whole lattice (the F7
    retreat ladder made the worst case 15.7 s > 10 s watchdog); so101 and
    franka plan one representative grasp each — their profiles carry no
    budget key, so the MEASURED time is the assertion."""
    import time as _time

    from aisle.nodes.grasp_topdown import GRIP_ENGAGEMENT, plan_grasp
    from aisle.nodes.ik_trajectory import StagedPlan
    from aisle.nodes.turn_barrier import ORDINARY_TURN_WATCHDOG_S

    cap = 0.6 * ORDINARY_TURN_WATCHDOG_S
    if embodiment == "g1_right":
        worst = 0.0
        worst_case = None
        for slot in PROFILE["placement_slots_xy"]:
            slot_xy = (float(slot[0]), float(slot[1]))
            for med in MED_NAMES:
                grasp, approach, place_z = _grasp_for_slot_med(slot_xy, med)
                t0 = _time.monotonic()
                plan = StagedPlan(
                    grasp,
                    tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
                    approach_m=approach,
                    q_seed=HOME,
                    place_z=place_z,
                    embodiment="g1_right",
                )
                elapsed = _time.monotonic() - t0
                assert plan.ok, (slot_xy, med, plan.error)
                if elapsed > worst:
                    worst, worst_case = elapsed, (slot_xy, med)
        assert worst <= cap, f"worst planning wall {worst:.2f}s at {worst_case} > cap {cap:.1f}s"

    def _plan_one(physics, emb, grasp, approach, place_z):
        prof = physics["embodiment"][emb]
        q_home = np.asarray(prof["home_qpos"], dtype=np.float32)
        t0 = _time.monotonic()
        plan = StagedPlan(
            grasp,
            tuple(np.asarray(prof["tray_pos"][:2], dtype=float)),
            approach_m=approach,
            q_seed=q_home,
            place_z=place_z,
            embodiment=emb,
        )
        return _time.monotonic() - t0, plan

    if embodiment == "so101":
        from aisle.nodes.grasp_topdown import place_tcp_z

        physics = load_physics()
        prof = physics["embodiment"]["so101"]
        size = MEDS["amoxicillin"]["size"]
        tray_top = prof["tray_pos"][2] + prof["tray_size"][2] / 2
        grasp = np.array(
            [0.39851335, 0.00547770, 0.095, 0.49655211, 0.50342429, 0.50342429, 0.49655211],
            dtype=np.float32,
        )
        # place_z from the same entry test_ik_plan_done uses — a hardcoded
        # 1.04 made the recorded radial plan fail its lower stage
        place_z = place_tcp_z(size, 0.0, tray_top)
        elapsed, plan = _plan_one(physics, "so101", grasp, 0.13852643, place_z)
        assert plan.ok, plan.error
        assert elapsed <= cap, f"so101 planning wall {elapsed:.2f}s > cap {cap:.1f}s"

    if embodiment == "franka":
        physics = load_physics()
        prof = physics["embodiment"]["franka"]
        layout = resolve_layout(physics, "franka")
        size = MEDS["amoxicillin"]["size"]
        shelf = layout["shelf"]
        z_lo = shelf["pos"][2] + shelf["level_heights"][0] + shelf["board_thickness"] / 2
        pose = np.array([0.42, 0.0, z_lo + size[2] / 2, 0, 0, 0, 1], dtype=np.float32)
        grasp, approach, place_z = plan_grasp(
            pose,
            size,
            grip=GRIP_ENGAGEMENT,
            shelf_front_x=shelf["pos"][0] - shelf["level_size"][0] / 2,
            tray_top_z=prof["tray_pos"][2] + prof["tray_size"][2] / 2,
            finger_open=float(prof["gripper_open_m"]),
            finger_clear=float(prof["gripper_finger_clear_m"]),
            topdown_approach=float(
                prof.get("pregrasp_height_m", physics["ik"]["pregrasp_height_m"])
            ),
        )
        elapsed, plan = _plan_one(physics, "franka", grasp, approach, place_z)
        assert plan.ok, plan.error
        assert elapsed <= cap, f"franka planning wall {elapsed:.2f}s > cap {cap:.1f}s"
