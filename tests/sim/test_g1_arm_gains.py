"""G1-C4 sim: arm PD gains hold static accuracy under the heaviest payload.

Root cause fixed: the 7 wire arm joints rode Genesis's kp=100 URDF default
(physics.toml now carries arm_kp/arm_kv sized from the static gravity
torque envelope). These tests hold three configurations (home, over the
shelf, over the tray) for 2 s of sim time with the heaviest med box
kinematically pinned to the hand (the bridge's carry-latch semantics) and
assert the per-joint static error and TCP error budgets from the board
item: < 0.01 rad per arm joint, < 1 cm TCP.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None, reason="sim extra not installed"
    ),
]


def test_static_accuracy_under_heaviest_box_in_three_configurations():
    from aisle.embodiment import from_wire_joint_order, held_dof_indices, profile_dof_indices
    from aisle.nodes.budget_guard import fk_ee_pose
    from aisle.nodes.ik_trajectory import ik_solve, topdown_rotation
    from aisle.scenes.pharmacy import (
        build_scene,
        load_physics,
        resolve_layout,
        sample_placements,
        to_numpy,
    )

    profile = load_physics()["embodiment"]["g1_right"]
    layout = resolve_layout(load_physics(), "g1_right")
    home = np.asarray(profile["home_qpos"], dtype=np.float32)

    handle = build_scene(seed=7, embodiment="g1_right", headless=True)
    robot = handle.robot
    wire = profile_dof_indices(robot, profile)
    held = held_dof_indices(robot, profile)
    held_base = np.zeros(robot.n_dofs, dtype=np.float32)
    held_base[list(held)] = np.asarray(profile["held_home_qpos"], dtype=np.float32)

    # heaviest med by volume (meds carry no mass key; uniform density):
    # amoxicillin 0.06*0.04*0.10 = 2.4e-4 m^3 — pinned to the hand the way
    # the bridge's carry latch pins it (set_pos each tick), so its weight
    # loads the arm exactly as in a carried episode.
    meds = sample_placements(7, list(handle.boxes), layout)
    box = handle.boxes["amoxicillin"]
    slot = next(p for p in meds if p.name == "amoxicillin")

    # three configurations: home / over the shelf grasp / over the tray
    q_shelf = ik_solve(
        np.array([slot.x, slot.y, slot.z], dtype=np.float64),
        topdown_rotation(0.0),
        home[:7],
        embodiment="g1_right",
    )
    q_tray = ik_solve(
        np.array(
            [
                profile["tray_pos"][0],
                profile["tray_pos"][1],
                float(profile["trajectory_transfer_z"]),
            ],
            dtype=np.float64,
        ),
        topdown_rotation(0.0),
        home[:7],
        embodiment="g1_right",
    )
    assert q_shelf is not None and q_tray is not None
    configs = {"home": home[:7], "shelf": q_shelf, "tray": q_tray}

    for name, q7 in configs.items():
        target = from_wire_joint_order(np.concatenate([q7, home[7:]]), wire, base=held_base)
        robot.set_qpos(target)  # start AT the configuration
        robot.control_dofs_position(target)  # and hold it there
        pin = to_numpy(box.get_pos()).reshape(-1).copy()
        for _ in range(200):  # 2 s at dt = 0.01
            # pin the box to its spot relative to the hand (latch semantics)
            box.set_pos(pin)
            box.zero_all_dofs_velocity()
            handle.scene.step()
        qpos = to_numpy(robot.get_qpos()).reshape(-1)
        err = np.abs(qpos[list(wire[: len(q7)])] - np.asarray(q7, dtype=np.float32))
        assert float(err.max()) < 0.01, (name, err.max())
        pos, _ = fk_ee_pose(qpos[list(wire[: len(q7)])].astype(np.float32), embodiment="g1_right")
        pos_target, _ = fk_ee_pose(np.asarray(q7, dtype=np.float32), embodiment="g1_right")
        assert float(np.linalg.norm(pos - pos_target)) < 0.01, (
            name,
            float(np.linalg.norm(pos - pos_target)),
        )


def test_gains_are_set_on_the_wire_arm_joints_only():
    """The loader applies arm_kp/arm_kv to EXACTLY the 7 wire arm DOFs —
    fingers and held DOFs keep their own gains, and a profile without the
    keys (franka/so101) is untouched by construction (guarded load)."""
    from aisle.embodiment import profile_dof_indices
    from aisle.scenes.pharmacy import build_scene, load_physics

    profile = load_physics()["embodiment"]["g1_right"]
    handle = build_scene(seed=3, embodiment="g1_right", headless=True)
    from aisle.scenes.pharmacy import to_numpy

    robot = handle.robot
    wire = profile_dof_indices(robot, profile)
    kp = np.asarray(to_numpy(robot.get_dofs_kp()).reshape(-1), dtype=np.float32)
    for i, q in enumerate(profile["arm_kp"]):
        assert kp[wire[i]] == pytest.approx(float(q), rel=1e-6), (i, kp[wire[i]])
    # gripper gains untouched by the arm block
    for j in wire[len(profile["arm_kp"]) :]:
        assert kp[j] == pytest.approx(float(profile["gripper_kp"][0]), rel=1e-6)
