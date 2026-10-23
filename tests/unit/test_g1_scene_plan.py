"""G1-H-r1 structural home-clearance test (SCN-4; fast, planning-free).

The planning lattice (every slot x med plans and passes the guard, the
execution-duration budget) moved to tests/sim/test_g1_scene_plan.py in
G1-I-a: 25 real IK plans at ~5.5s each cannot fit the unit marker's <1s
definition. Assertions are unchanged there. This file keeps the
mesh-AABB home-clearance check, which needs no solving."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from aisle.kinematics import load_urdf_chain
from aisle.nodes.budget_guard import load_limits
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
