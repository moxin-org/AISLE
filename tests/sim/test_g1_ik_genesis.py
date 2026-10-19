"""Genesis consistency for G1 IK solutions (G1-F2; SCN-4, TC-5, CON-5).

Three deterministic scene seeds: the top-down grasp poses planned for the
C1 scene are solved by aisle IK, the solutions are written into Genesis's
wire joints (held DOFs at standing home, fingers at the open stop), and
the realized Genesis TCP must sit within 1 cm of the IK target.

Marker `sim`: imports genesis, runs headless (CON-12). Self-contained
minimal scene — the URDF is loaded directly with lane1's fixed-pelvis
mounting; link poses are read straight after set_qpos (the rapier parity
pattern)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None,
        reason="genesis not installed (uv sync --extra sim)",
    ),
]

_G1_URDF = "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf"
_PELVIS_Z = 0.7872
_TCP_OFFSET = np.asarray((0.1393, 0.0, 0.0), dtype=float)


@pytest.mark.parametrize("seed", [0, 2, 7])
def test_g1_ik_solution_realizes_in_genesis_within_one_cm(seed):
    """SCN-4, TC-5, CON-5: for each seed's planned top-down grasp target,
    ik_solve on the G1-F1 chain — written into Genesis's right-arm wire
    joints — puts the Genesis TCP within 1 cm of the target position."""
    import genesis as gs

    from aisle.embodiment import G1_RIGHT_ARM_JOINTS, G1_RIGHT_GRIPPER_JOINTS
    from aisle.nodes.budget_guard import load_limits
    from aisle.nodes.grasp_topdown import plan_grasp
    from aisle.nodes.ik_trajectory import ik_solve, topdown_rotation
    from aisle.scenes.pharmacy import (
        MED_NAMES,
        load_meds,
        load_physics,
        resolve_layout,
        sample_placements,
        to_numpy,
    )

    physics = load_physics()
    profile = physics["embodiment"]["g1_right"]
    layout = resolve_layout(physics, "g1_right")
    meds = load_meds()
    placements = sample_placements(seed, MED_NAMES, layout)
    target_placement = next(p for p in placements if p.name == "cetirizine")
    neighbours = [
        [p.x, p.y, meds[p.name]["size"][0] / 2, meds[p.name]["size"][1] / 2]
        for p in placements
        if p.name != target_placement.name
    ]
    pose = np.array(
        [target_placement.x, target_placement.y, target_placement.z, 0, 0, 0, 1],
        dtype=np.float32,
    )
    grasp, _, _ = plan_grasp(
        pose,
        meds["cetirizine"]["size"],
        shelf_front_x=layout["shelf"]["pos"][0] - layout["shelf"]["level_size"][0] / 2,
        tray_top_z=profile["tray_pos"][2] + profile["tray_size"][2] / 2,
        neighbours=neighbours,
        finger_open=float(profile["gripper_open_m"]),
        finger_clear=float(profile["gripper_finger_clear_m"]),
        topdown_approach=float(profile["pregrasp_height_m"]),
    )
    target = grasp[:3].astype(np.float64)
    home = np.asarray(load_limits("g1_right").fallback_qpos, dtype=np.float32)
    q = ik_solve(target, topdown_rotation(0.0), home[:7], embodiment="g1_right")
    assert q is not None, f"seed {seed}: IK failed"

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu, logging_level="warning", seed=0)
    scene = gs.Scene()
    robot = scene.add_entity(
        gs.morphs.URDF(file=_G1_URDF, fixed=True, pos=(0.0, 0.0, _PELVIS_Z))
    )
    scene.build()
    arm_indices = [robot.get_joint(name).dofs_idx_local[0] for name in G1_RIGHT_ARM_JOINTS]
    finger_indices = [
        robot.get_joint(name).dofs_idx_local[0] for name in G1_RIGHT_GRIPPER_JOINTS
    ]
    native = np.zeros(robot.n_dofs, dtype=np.float64)
    native[arm_indices] = q
    native[finger_indices] = 0.0245  # open stop
    robot.set_qpos(native)
    wrist = robot.get_link("right_wrist_yaw_link")
    pos = to_numpy(wrist.get_pos()).reshape(-1)[:3]
    quat_wxyz = to_numpy(wrist.get_quat()).reshape(-1)[:4]
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    tcp = pos + rotation @ _TCP_OFFSET
    error = float(np.linalg.norm(tcp - target))
    assert error < 0.01, f"seed {seed}: Genesis TCP off target by {error * 1000:.1f} mm"
    print(f"seed {seed}: genesis TCP error {error * 1000:.2f} mm")
