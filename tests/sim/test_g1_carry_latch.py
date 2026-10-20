"""G1-H-r4 sim: the carry latch fires under the bridge's real close path.

Widest and narrowest grasp-axis meds (omeprazole 0.045 m, cetirizine
0.030 m) are grasped top-down; the arm and fingers are driven the way
the bridge drives them (control_dofs_position on the full-length wire
target, held DOFs at held home), the bridge's exact grip normalization
feeds a real CarryLatch EVERY tick, and the latch's set_pos/set_quat
output is applied the way the bridge applies it — then the arm lifts
0.1 m with the fingers kept at the squeeze command, and the box must
still ride the hand.
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


@pytest.mark.parametrize("med", ["omeprazole", "cetirizine"])
def test_latch_fires_and_holds_through_a_0p1m_lift(med):
    from aisle.embodiment import from_wire_joint_order, held_dof_indices, profile_dof_indices
    from aisle.nodes.dora_genesis import CarryLatch, frame_point_wxyz
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
    open_q = float(profile["gripper_open_qpos"])
    close_q = float(profile["gripper_close_qpos"])
    cmd = float(profile["gripper_grasp_cmd"])
    latch = CarryLatch(
        close_threshold=float(profile["carry_latch_close"]),
        release_threshold=float(profile["carry_latch_release"]),
        max_distance_m=float(profile["carry_latch_max_distance_m"]),
    )

    handle = build_scene(seed=7, embodiment="g1_right", headless=True)
    robot = handle.robot
    wire = profile_dof_indices(robot, profile)
    held = held_dof_indices(robot, profile)
    gripper_dofs = int(profile.get("gripper_dofs", 2))
    finger_idx = list(wire[-gripper_dofs:])
    held_base = np.zeros(robot.n_dofs, dtype=np.float32)
    held_base[list(held)] = np.asarray(profile["held_home_qpos"], dtype=np.float32)

    wrist = robot.get_link(profile["ee_link"])
    tcp_offset = np.asarray(profile["ee_frame_offset_xyz"], dtype=np.float64)

    def command(q7: np.ndarray, finger_q: float) -> None:
        """The bridge's joint_cmd/gripper_cmd application, verbatim:
        control_dofs_position on the full-length target (held DOFs at
        their held home), gripper dofs at the mapped finger qpos."""
        finger_cmd = np.full(gripper_dofs, finger_q, dtype=np.float32)
        robot.control_dofs_position(
            from_wire_joint_order(np.concatenate([q7, finger_cmd]), wire, base=held_base)
        )

    def latch_tick() -> float:
        """One bridge tick's latch path: grip from the finger qpos, update
        against every box, apply the returned pin. Returns the grip."""
        finger = float(to_numpy(robot.get_qpos()).reshape(-1)[finger_idx[0]])
        grip = float(np.clip((open_q - finger) / (open_q - close_q), 0.0, 1.0))
        hand_pos = to_numpy(wrist.get_pos()).reshape(-1)[:3]
        hand_quat = to_numpy(wrist.get_quat()).reshape(-1)[:4]
        candidates = {
            name: (to_numpy(e.get_pos()).reshape(-1)[:3], to_numpy(e.get_quat()).reshape(-1)[:4])
            for name, e in handle.boxes.items()
        }
        attached = latch.update(
            grip, frame_point_wxyz(hand_pos, hand_quat, tcp_offset), hand_quat, candidates
        )
        if attached is not None:
            name, pos, quat = attached
            handle.boxes[name].set_pos(pos)
            handle.boxes[name].set_quat(quat)
            handle.boxes[name].zero_all_dofs_velocity()
        return grip

    target = next(p for p in sample_placements(7, list(handle.boxes), layout) if p.name == med)
    grasp_z = target.z
    q_grasp = ik_solve(
        np.array([target.x, target.y, grasp_z], dtype=np.float64),
        topdown_rotation(0.0),
        home[:7],
        embodiment="g1_right",
    )
    assert q_grasp is not None, med

    # teleport the arm AT the grasp (test setup only), then hold it there
    # the bridge way while the fingers close onto the box
    robot.set_qpos(from_wire_joint_order(np.concatenate([q_grasp, home[7:]]), wire, base=held_base))
    command(q_grasp, float(home[7]))
    for _ in range(20):
        handle.scene.step()

    finger_q = open_q - cmd * (open_q - close_q)  # the bridge's gripper_cmd mapping
    command(q_grasp, finger_q)
    latched_at_grip = None
    for _ in range(400):
        handle.scene.step()
        grip = latch_tick()
        if latch.held_name == med and latched_at_grip is None:
            latched_at_grip = grip

    assert latch.held_name == med, (med, latch.held_name)
    assert latched_at_grip is not None
    assert latched_at_grip >= float(profile["carry_latch_close"])

    z_before = float(to_numpy(handle.boxes[med].get_pos()).reshape(-1)[2])
    tcp_z_before = float(
        frame_point_wxyz(
            to_numpy(wrist.get_pos()).reshape(-1)[:3],
            to_numpy(wrist.get_quat()).reshape(-1)[:4],
            tcp_offset,
        )[2]
    )

    # lift 0.1 m with the fingers HELD at the squeeze command (reopening
    # would drop the grip to the release threshold); the latch keeps
    # pinning the box every tick, exactly as in a bridge transit
    q_lift = ik_solve(
        np.array([target.x, target.y, grasp_z + 0.1], dtype=np.float64),
        topdown_rotation(0.0),
        q_grasp,
        embodiment="g1_right",
    )
    assert q_lift is not None, med
    command(q_lift, finger_q)
    for _ in range(200):
        handle.scene.step()
        latch_tick()

    z_after = float(to_numpy(handle.boxes[med].get_pos()).reshape(-1)[2])
    tcp_z_after = float(
        frame_point_wxyz(
            to_numpy(wrist.get_pos()).reshape(-1)[:3],
            to_numpy(wrist.get_quat()).reshape(-1)[:4],
            tcp_offset,
        )[2]
    )
    # The lift invariant is "the box rides the hand": box rise tracks the
    # TCP's ACTUAL rise (the arm's gravity steady-state error — ~25% sag
    # on Genesis default gains, G1-F deferral — means a commanded 0.1 m
    # settles near 0.075 m; the latch must move the box with whatever the
    # hand actually did).
    box_rise = z_after - z_before
    tcp_rise = tcp_z_after - tcp_z_before
    assert tcp_rise > 0.05, (med, tcp_rise)
    assert box_rise >= 0.9 * tcp_rise, (med, box_rise, tcp_rise)
    assert latch.held_name == med
