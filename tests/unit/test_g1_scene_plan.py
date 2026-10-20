"""G1-H-r1 systematic scene-plan tests (SCN-4, BG-3).

The M0-6 gate lesson: every single-case test was green while the gate
failed 50/50 on STRUCTURAL geometry — the home hand parked over the
shelf boxes (opening collisions) and an unplannable cross-body slot.
These two checks pin the WHOLE lattice, not lucky instances, pure
kinematics (no scene build):

- home clearance: the home right wrist + Dex1-1 AABB clears the tallest
  possible box on EVERY slot by >= 3 cm (outer-loop #6 constraint (a));
- all-slot plannability: for every slot x every med, the SAME StagedPlan
  entry the expert graph uses plans home -> grasp -> tray -> home with
  no plan.error, and the executor's interpolated stream is guard-clean.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from aisle.kinematics import load_urdf_chain
from aisle.nodes.budget_guard import clamp_joint_cmd, fk_ee_pose, load_limits
from aisle.nodes.grasp_topdown import GRIP_ENGAGEMENT, plan_grasp
from aisle.nodes.ik_trajectory import StagedPlan, interpolate_step
from aisle.scenes.pharmacy import (
    MED_NAMES,
    load_meds,
    load_physics,
    resolve_layout,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MESHES = REPO_ROOT / "assets" / "g1" / "meshes"
PELVIS_Z = 0.7872  # FK standing height (ADR-g1-embodiment)

PHYSICS = load_physics()
PROFILE = PHYSICS["embodiment"]["g1_right"]
LAYOUT = resolve_layout(PHYSICS, "g1_right")
MEDS = load_meds()
LIMITS = load_limits("g1_right")
HOME = np.asarray(LIMITS.fallback_qpos, dtype=np.float32)
TRAY_TOP = PROFILE["tray_pos"][2] + PROFILE["tray_size"][2] / 2
CLEARANCE_M = 0.03
JITTER = float(PROFILE["placement_global_jitter_m"])

#: hand geometry as (mesh file, transform from the right_wrist_yaw frame):
#: wrist mesh, the fixed Dex1 base (0.0415 m along +x), and both fingers
#: at the home open stop — the finger joints are PRISMATIC along
#: (0,-1,0)/(0,+1,0), so q=+0.0245 is a TRANSLATION, not a rotation.
_TY = float(HOME[7])
_T_BASE = np.eye(4)
_T_BASE[0, 3] = 0.0415
_T_F1 = np.eye(4)
_T_F1[1, 3] = -_TY
_T_F2 = np.eye(4)
_T_F2[1, 3] = _TY
_HAND = (
    ("right_wrist_yaw_link_5010.STL", np.eye(4)),
    ("Dex1_base_link.STL", _T_BASE),
    ("Dex1_finger_link_1.STL", _T_BASE @ _T_F1),
    ("Dex1_finger_link_2.STL", _T_BASE @ _T_F2),
)


def _hand_aabb() -> tuple[np.ndarray, np.ndarray]:
    """World AABB of wrist+dex1+fingers at HOME, from the vendored URDF."""
    chain = load_urdf_chain(
        REPO_ROOT / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf",
        "pelvis",
        "right_wrist_yaw_link",
    )
    assert len(chain.actuated) == 10  # 3 held waist + 7 arm
    # held waist zeros + the ARM at home — NOT all-zeros (that is the
    # a7dd234 ready pose whose hand parked over the shelf).
    q = np.concatenate([np.zeros(3), HOME[:7]])
    pos, rot = chain.forward(q)
    wrist = np.eye(4)
    wrist[:3, :3] = rot
    wrist[:3, 3] = pos + np.array([0.0, 0.0, PELVIS_Z])  # fixed pelvis
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name, local in _HAND:
        mesh = trimesh.load(MESHES / name, force="mesh")
        world = trimesh.transformations.transform_points(
            np.asarray(mesh.vertices, dtype=float), wrist @ local
        )
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return lo, hi


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


def test_home_hand_clears_every_slot_by_three_centimeters():
    """SCN-4 (outer-loop #6 constraint (a)): at home the right wrist + Dex1
    AABB keeps >= 3 cm of separation from the AABB of EVERY med standing on
    EVERY slot (worst-case jitter included) — the opening-collision root
    cause of the first M0-6 gate run, pinned structurally."""
    hand_lo, hand_hi = _hand_aabb()
    worst = np.inf
    for slot in PROFILE["placement_slots_xy"]:
        for med in MED_NAMES:
            box_lo, box_hi = _box_aabb((float(slot[0]), float(slot[1])), med)
            gaps = np.maximum(hand_lo - box_hi, box_lo - hand_hi)
            worst = min(worst, float(np.max(gaps)))
            assert float(np.max(gaps)) >= CLEARANCE_M, (slot, med, gaps)
    assert worst >= CLEARANCE_M  # sanity: the min over the whole lattice


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
    worst = 0.0
    worst_case = None
    for slot in PROFILE["placement_slots_xy"]:
        slot_xy = (float(slot[0]), float(slot[1]))
        for med in MED_NAMES:
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
            seconds = _execution_seconds(plan, HOME)
            if seconds > worst:
                worst, worst_case = seconds, (slot_xy, med)
    assert worst <= budget, f"worst {worst:.1f}s at {worst_case} vs budget {budget:.0f}s"


def test_unreachable_plan_fails_loud_within_the_wall_budget():
    """G1-F5 (ep0, smoke 20261019-234952): a grasp pose outside the
    workspace made first-launch planning burn 9 minutes of lockstep wall
    clock with no crash and no verdict. The mechanism: one unreachable
    cold ik_solve walks the full 32-seed x 2-rotation ladder (measured
    37.5s) and the staged chain multiplies it across stages and parity
    retries. Planning now runs under a wall budget (profile
    ik_plan_budget_s): on expiry the plan fails LOUD — plan.error names
    the budget — well inside the measured 9-minute stall."""
    import time as _time

    far = np.array([0.75, 0.50, 1.20, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    grasp = np.concatenate([far[:3], _quat_topdown_xyzw()]).astype(np.float32)
    t0 = _time.monotonic()
    plan = StagedPlan(
        grasp,
        tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
        approach_m=0.10,
        q_seed=HOME,
        place_z=1.05,
        embodiment="g1_right",
    )
    elapsed = _time.monotonic() - t0
    assert not plan.ok
    assert "budget" in str(plan.error).lower(), plan.error
    assert elapsed < 90.0, f"planning burned {elapsed:.0f}s before failing"


def _quat_topdown_xyzw() -> np.ndarray:
    """Top-down quaternion matching the planner's convention (yaw 0)."""
    from aisle.nodes.grasp_topdown import topdown_quat

    return np.asarray(topdown_quat(0.0), dtype=np.float32)


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
    grasp, approach, place_z = _grasp_for_slot_med(slot_xy, med)
    plan = StagedPlan(
        grasp,
        tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
        approach_m=approach,
        q_seed=HOME,
        place_z=place_z,
        embodiment="g1_right",
    )
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
