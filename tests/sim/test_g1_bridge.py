"""G1-C2 bridge-level sim tests (TC-5 draft, SCN-4): the wire command path
drives ONLY the 9 wire DOFs; the 24 held DOFs never leave standing home.

Marker `sim`: builds the G1 scene (CoACD-cached load, ~20 s) and steps
physics. The length gates are exercised here against the same
module-level function the bridge's event loop calls.
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


def test_g1_wire_command_moves_arm_and_holds_everything_else():
    """TC-5, SCN-4 (G1-C2): a wire command applied the way apply_commands
    applies it (base = full-length held home, overwrite wire positions
    only) moves the commanded arm joint toward its target within 200
    steps, while every held joint stays within 0.02 rad of held home."""
    from aisle.embodiment import (
        G1_RIGHT_JOINTS,
        from_wire_joint_order,
        held_dof_indices,
        profile_dof_indices,
    )
    from aisle.scenes.pharmacy import build_scene, load_physics, to_numpy

    handle = build_scene(seed=7, embodiment="g1_right", headless=True)
    profile = load_physics()["embodiment"]["g1_right"]
    robot = handle.robot
    wire = profile_dof_indices(robot, profile)
    held = held_dof_indices(robot, profile)

    home = np.asarray(profile["home_qpos"], dtype=np.float32)
    held_base = np.zeros(robot.n_dofs, dtype=np.float32)
    held_base[list(held)] = np.asarray(profile["held_home_qpos"], dtype=np.float32)
    full_home = from_wire_joint_order(home, wire, base=held_base)

    # command shoulder pitch (wire joint 0) well away from its home -0.35
    commanded = home.copy()
    commanded[0] = -0.9
    target = from_wire_joint_order(commanded, wire, base=full_home)
    robot.control_dofs_position(target)
    for _ in range(200):
        handle.scene.step()

    qpos = to_numpy(robot.get_qpos()).reshape(-1)
    # G1-F input (arm gains still Genesis defaults): the commanded joint
    # moves decisively toward -0.9 but stops ~0.10 rad short — steady-state
    # PD error under gravity. Gain tuning is explicitly deferred to G1-F.
    assert abs(qpos[wire[0]] - commanded[0]) < 0.15, qpos[wire[0]]
    assert abs(qpos[wire[0]] - home[0]) > 0.3, qpos[wire[0]]  # it really moved
    droop = float(np.abs(qpos[list(held)] - np.asarray(profile["held_home_qpos"])).max())
    assert droop < 0.02, droop
    # sanity: the wire really is G1_RIGHT_JOINTS (TC-5 names)
    assert len(G1_RIGHT_JOINTS) == len(wire) == 9


def test_g1_joint_cmd_length_gates_reject_wrong_shapes():
    """TC-5 (G1-C2): joint_cmd must be exactly the WIRE length (9) — an
    over-length (native 33) or under-length payload is rejected by the
    same gate the bridge's event loop calls."""
    from aisle.embodiment import G1_RIGHT_JOINTS
    from aisle.nodes.dora_genesis import require_wire_joint_cmd

    n_dof = len(G1_RIGHT_JOINTS)
    require_wire_joint_cmd(np.zeros(n_dof, dtype=np.float32), n_dof)  # 9: accepted
    for bad in (n_dof - 1, n_dof + 1, 33):  # under, over, native-33 confusion
        with pytest.raises(ValueError, match=r"joint_cmd must be Float32\[9\]"):
            require_wire_joint_cmd(np.zeros(bad, dtype=np.float32), n_dof)
