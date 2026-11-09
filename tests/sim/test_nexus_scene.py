"""Sim acceptance tests for the Nexus engine backend (ADR-67): the frozen
pharmacy scene (SPEC 020 SCN-1, SCN-3..5, SCN-7) and the bridge's object
surface (SPEC 030), realized on Nexus.

Marker `sim`: imports nexus3d (and genesis for the Franka asset), runs
headless. Skipped when the nexus3d wheel is not installed.

Scene order: only the newest scene can render (a later build frees the
previous scene's cameras), so tests that only read a built scene share one
module-scoped build, each group that steps, teleports or renders gets exactly
one build, and the groups that render come last.
"""

import importlib.util

import numpy as np
import pytest

from aisle.scenes.pharmacy import MED_NAMES, load_physics, oracle_state, to_numpy

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("nexus3d") is None or importlib.util.find_spec("genesis") is None,
        reason="nexus3d wheel or the sim extra not installed",
    ),
]


# Function-scoped on purpose: the shared Nexus viewer renders the LATEST
# scene only (a later build supersedes the previous one's cameras), so a test
# that renders or mutates gets a fresh scene (about a second each).
@pytest.fixture
def handle():
    from aisle.sim import build_scene

    return build_scene("nexus", seed=7, embodiment="franka", n_envs=1, headless=True)


# Module-scoped: these tests only read the built scene (no step, no teleport,
# no render), so one build serves them all.
@pytest.fixture(scope="module")
def pristine():
    from aisle.sim import build_scene

    return build_scene("nexus", seed=7, embodiment="franka", n_envs=1, headless=True)


@pytest.fixture(scope="module")
def so101_handle():
    from aisle.sim import build_scene

    return build_scene("nexus", seed=3, embodiment="so101", n_envs=1, headless=True)


def test_build_determinism(pristine):
    """SCN-1, SCN-7 on Nexus: the initial oracle_state is a pure function of
    the seed (bitwise) and differs across seeds."""
    from aisle.sim import build_scene

    first = oracle_state(pristine)
    again = oracle_state(build_scene("nexus", seed=7, embodiment="franka", headless=True))
    other = oracle_state(build_scene("nexus", seed=11, embodiment="franka", headless=True))
    assert first.dtype == np.float32
    assert first.shape == (len(MED_NAMES) * 7,)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    # a superseded scene keeps its physics readbacks but refuses to render
    assert np.array_equal(oracle_state(pristine), first)
    with pytest.raises(RuntimeError, match="superseded"):
        pristine.cams["overhead"].render()


def test_placements_match_frozen_sampler(pristine):
    """ADR-67: the Nexus builder places the boxes exactly where the frozen
    sampler says (same pure function, same seed), before any step."""
    from aisle.scenes.pharmacy import resolve_layout, sample_placements

    placements = {
        p.name: p for p in sample_placements(7, MED_NAMES, resolve_layout(load_physics(), "franka"))
    }
    for name, entity in pristine.boxes.items():
        pos = to_numpy(entity.get_pos()).reshape(-1)[:3]
        expected = placements[name]
        assert np.allclose(pos, [expected.x, expected.y, expected.z], atol=1e-5), name


def test_reachability_and_home(pristine):
    """SCN-3/SCN-4 (Nexus IK) and SCN-1: every placement admits an IK
    solution and the robot rests at its home pose."""
    assert pristine.reachability_errors == []
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    actual = to_numpy(pristine.robot.get_qpos()).reshape(-1)[: home.shape[0]]
    assert np.allclose(actual, home, atol=1e-4)


def test_realized_calibration_matches_nominal(pristine):
    """BRG-8 / VER-8: the realized overhead transform is the Genesis look-at
    convention, so the published v1 block passes stage 0."""
    from aisle.nodes.dora_genesis import realized_calibration
    from aisle.verifier.calibration import build_calibration_v1

    physics = load_physics()
    realized = realized_calibration(pristine, physics, is_store=False)
    from aisle.scenes.pharmacy import wrist_mount_transform

    wrist = wrist_mount_transform(physics["cameras"], physics["embodiment"]["franka"])
    nominal = build_calibration_v1(
        overhead_pos=physics["cameras"]["overhead_pos"],
        overhead_lookat=physics["cameras"]["overhead_lookat"],
        overhead_resolution=(640, 480),
        overhead_fov_deg=55,
        wrist_offset_m=wrist[:3, 3].tolist(),
        wrist_resolution=(320, 240),
        wrist_fov_deg=70,
        wrist_mount_rotation_gl=wrist[:3, :3],
    )
    assert realized["overhead"]["intrinsics"] == nominal["overhead"]["intrinsics"]
    assert np.allclose(
        realized["overhead"]["cam_to_base"]["pos"],
        nominal["overhead"]["cam_to_base"]["pos"],
        atol=1e-5,
    )
    assert np.allclose(
        realized["overhead"]["cam_to_base"]["quat_xyzw"],
        nominal["overhead"]["cam_to_base"]["quat_xyzw"],
        atol=1e-4,
    )


def test_step_teleport_and_control(handle):
    """BRG-2/BRG-4: stepping advances physics, a teleport reset lands a box
    exactly where asked with zero velocity, and position control holds the
    home pose under gravity."""
    robot = handle.robot
    home = to_numpy(robot.get_qpos()).reshape(-1).copy()
    robot.control_dofs_position(home)
    box = next(iter(handle.boxes.values()))
    robot_before = to_numpy(robot.get_link("hand").get_pos()).reshape(-1)
    for _ in range(50):
        handle.scene.step()
    after = to_numpy(robot.get_qpos()).reshape(-1)
    assert np.allclose(after[:7], home[:7], atol=0.02)
    assert np.allclose(
        to_numpy(robot.get_link("hand").get_pos()).reshape(-1), robot_before, atol=0.02
    )
    box.set_pos(np.array([0.3, -0.45, 0.3], dtype=np.float32))
    box.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    box.zero_all_dofs_velocity()
    assert np.allclose(to_numpy(box.get_pos()).reshape(-1), [0.3, -0.45, 0.3], atol=1e-5)
    handle.scene.step()
    pos = to_numpy(box.get_pos()).reshape(-1)
    assert pos[2] < 0.3 and pos[2] > 0.29  # falling under gravity, one step later


def test_resting_boxes_hold_still(handle):
    """SCN-3 on Nexus: boxes at rest on the boards must not creep past the
    oracle's knock threshold over an episode. The substep count in
    nexus_physics.toml is chosen for this; one second of settling must stay
    well under a millimeter per box."""
    robot = handle.robot
    robot.control_dofs_position(to_numpy(robot.get_qpos()).reshape(-1))
    start = oracle_state(handle).reshape(-1, 7)[:, :3]
    for _ in range(100):
        handle.scene.step()
    drift = np.linalg.norm(oracle_state(handle).reshape(-1, 7)[:, :3] - start, axis=1)
    assert drift.max() < 1.0e-3, drift


def _scripted_run(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Builds a scene, drives the arm off its home pose and drops every box
    into one tilted pile, then returns the final joint positions and oracle
    state. The pile matters: without the deterministic mode, the first run
    of a process settled it differently from the later ones."""
    from aisle.sim import build_scene

    handle = build_scene("nexus", seed=seed, embodiment="franka", n_envs=1, headless=True)
    assert handle.scene.state.deterministic()
    robot = handle.robot
    target = to_numpy(robot.get_qpos()).reshape(-1).copy()
    target[:7] += np.float32(0.2)
    robot.control_dofs_position(target)
    for i, box in enumerate(handle.boxes.values()):
        box.set_pos(np.array([0.3 + 0.01 * (i % 3), -0.45, 0.25 + 0.06 * i], dtype=np.float32))
        tilt = [0.38268, 0.0] if i % 2 else [0.0, 0.38268]
        box.set_quat(np.array([0.92388, *tilt, 0.0], dtype=np.float32))
        box.zero_all_dofs_velocity()
    for _ in range(300):
        handle.scene.step()
    return to_numpy(robot.get_qpos()).reshape(-1).copy(), oracle_state(handle).copy()


def test_stepping_is_reproducible():
    """CON-5 on Nexus: with the engine's deterministic mode on, two runs of
    the same seed and commands end bit-identical, contacts included."""
    first_qpos, first_oracle = _scripted_run(7)
    again_qpos, again_oracle = _scripted_run(7)
    assert np.array_equal(first_qpos, again_qpos)
    assert np.array_equal(first_oracle, again_oracle)


def test_so101_urdf_matches_frozen_chain(so101_handle):
    """ADR-67: the imported SO-101 kinematics agree with the frozen URDF
    chain (`aisle.kinematics`) at several configurations, so grasp planning
    and the guard see the same arm the physics does."""
    from aisle.kinematics import so101_chain

    chain = so101_chain()
    robot = so101_handle.robot
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, size=5)
        expected, _ = chain.forward(q)
        pos, _ = robot.scene.state.robot_forward_kinematics(
            robot.robots[0], [float(v) for v in q] + [0.0], "gripper_frame_link"
        )
        assert np.allclose(pos, expected, atol=2e-4), (q, pos, expected)


def test_so101_home_and_gripper_gains(so101_handle):
    """SCN-1 / TC-5: the SO-101 rests at its profile home in wire order and
    carries the profile's gripper gains; batched envs share the layout."""
    from aisle.embodiment import profile_dof_indices

    profile = load_physics()["embodiment"]["so101"]
    robot = so101_handle.robot
    wire = profile_dof_indices(robot, profile)
    qpos = to_numpy(robot.get_qpos()).reshape(-1)
    assert np.allclose(qpos[list(wire)], profile["home_qpos"], atol=1e-4)
    gripper = robot.robots[0]
    assert np.isclose(gripper.kp[wire[-1]], profile["gripper_kp"][0])
    assert np.isclose(gripper.kv[wire[-1]], profile["gripper_kv"][0])
    assert so101_handle.reachability_errors == []


def test_batched_build_oracle_covers_all_envs():
    """SCN-1 on Nexus: a batched build reports every env and identical
    initial placements."""
    from aisle.sim import build_scene

    batched = build_scene("nexus", seed=7, embodiment="franka", n_envs=2, headless=True)
    state = oracle_state(batched)
    assert state.shape == (2, len(MED_NAMES) * 7)
    assert np.array_equal(state[0], state[1])
    qpos = to_numpy(batched.robot.get_qpos())
    assert qpos.shape == (2, batched.robot.n_dofs)


def test_no_interpenetration(pristine):
    """SCN-3 on Nexus: the initial placements the frozen sampler produced are
    realized free of interpenetration (pairwise AABBs of the meds)."""
    aabbs = []
    for name, entity in pristine.boxes.items():
        pos = to_numpy(entity.get_pos()).reshape(-1)[:3]
        half = np.asarray(pristine.med_sizes[name]) / 2.0
        aabbs.append((name, pos - half, pos + half))
    for i, (name_a, lo_a, hi_a) in enumerate(aabbs):
        for name_b, lo_b, hi_b in aabbs[i + 1 :]:
            overlap = np.all(lo_a < hi_b) and np.all(lo_b < hi_a)
            assert not overlap, (name_a, name_b)


def test_oracle_quaternions_are_xyzw(pristine):
    """TC-1 on Nexus: oracle_state quaternions are (x, y, z, w) wire order.
    The Nexus readback is w-first like Genesis's, so a missing roll here
    would publish the same wrong wire order the Genesis test pins."""
    state = oracle_state(pristine)
    for i in range(len(MED_NAMES)):
        quat = state[i * 7 + 3 : i * 7 + 7]
        assert abs(quat[3]) > 0.99, quat  # w last
        assert np.all(np.abs(quat[:3]) < 0.1), quat


def test_boxes_follow_oracle_order(pristine):
    """SCN-1 on Nexus: the boxes dict is in the fixed meds.toml order, which
    is the oracle_state layout (TC table)."""
    assert list(pristine.boxes) == MED_NAMES


@pytest.fixture(scope="module")
def mobile_scene():
    """One mobile-profile build for the re-basing tests (they move the base,
    never the arm, so they compose on one scene)."""
    from aisle.sim import build_scene

    return build_scene("nexus", seed=0, embodiment="mobile", n_envs=1, headless=True)


def test_mobile_builds_franka_arm_and_rebases(mobile_scene):
    """MOB-4 and ADR-13 on Nexus: the mobile profile builds the same franka
    arm (it rests at the frozen franka home, as the fixed-base build does),
    and re-basing through `NexusRobot.set_pos`/`set_quat` moves the whole
    arm's world mount while the arm stays base-relative (joint FK unchanged).

    Nexus roots the arm at a fixed body whose pose the GPU re-reads every
    step, so the links must ride the base rather than the root sliding out
    from under them."""
    import math

    from aisle.mobility.base import integrate_base_pose
    from aisle.nodes.ik_trajectory import fk_tcp

    mobile = mobile_scene
    robot = mobile.robot
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    qpos = to_numpy(robot.get_qpos()).reshape(-1)
    assert np.allclose(qpos[: home.shape[0]], home, atol=1e-4)  # MOB-4: the same arm
    tcp_before = fk_tcp(qpos[:7])

    hand_before = to_numpy(robot.get_link("hand").get_pos()).reshape(-1)[:3]

    pose = [0.0, 0.0, 0.0]
    for _ in range(10):  # 1 m forward at 1 m/s over 0.1 s ticks
        pose = integrate_base_pose(pose, [1.0, 0.0], dt=0.1)
    robot.set_pos(np.array([pose[0], pose[1], 0.0], dtype=np.float32))
    mobile.scene.step()

    assert to_numpy(robot.get_pos()).reshape(-1)[:2] == pytest.approx([1.0, 0.0], abs=1e-3)
    # the links rode the base: the hand moved by the same world translation
    hand_after = to_numpy(robot.get_link("hand").get_pos()).reshape(-1)[:3]
    assert hand_after - hand_before == pytest.approx([1.0, 0.0, 0.0], abs=1e-2)
    # and the arm-frame FK is unchanged: the arm rode the base, it did not move
    assert fk_tcp(to_numpy(robot.get_qpos()).reshape(-1)[:7]) == pytest.approx(tcp_before, abs=1e-2)

    # a yaw re-base rotates the arm about the base origin, which stays put
    robot.set_quat(
        np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], dtype=np.float32)
    )
    mobile.scene.step()
    root_now = to_numpy(robot.get_pos()).reshape(-1)[:3]
    assert root_now[:2] == pytest.approx([1.0, 0.0], abs=1e-3)
    hand_yawed = to_numpy(robot.get_link("hand").get_pos()).reshape(-1)[:3]
    offset = hand_after - np.asarray([1.0, 0.0, 0.0])  # base-frame hand offset
    expected = root_now + np.asarray([-offset[1], offset[0], offset[2]])  # +90 deg about z
    assert hand_yawed == pytest.approx(expected, abs=1e-2), (hand_yawed, expected)


def test_mobile_rebase_survives_set_pos_then_set_quat(mobile_scene):
    """ADR-13/MOB-4 on Nexus: the bridge re-bases with `set_pos` immediately
    followed by `set_quat`, with no step between, so the pair must compose.

    Regression: `set_quat` used to read the root position back from
    `robot_state`, which still reports the pre-step root, so it wrote the old
    position back and cancelled the `set_pos` before it. On Nexus a mobile base
    then only ever turned in place, its translation silently dropped. The root
    write is now remembered until the next step consumes it."""
    import math

    mobile = mobile_scene
    robot = mobile.robot
    yaw = math.pi / 3
    robot.set_pos(np.array([2.0, 0.5, 0.0], dtype=np.float32))
    robot.set_quat(np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float32))
    mobile.scene.step()
    assert to_numpy(robot.get_pos()).reshape(-1)[:2] == pytest.approx([2.0, 0.5], abs=1e-3)


# --- the store scene on Nexus (SPEC 200), ported from test_store_scene.py ---


def _xyz(entity) -> list[float]:
    return [float(v) for v in to_numpy(entity.get_pos()).reshape(-1)[:3]]


def _assert_yaw(entity, yaw: float, label) -> None:
    """The entity's physical orientation matches the composed world yaw:
    quaternion equal up to sign, and the rotated +x (front face) points where
    the slot faces."""
    import math

    from aisle.scenes.store import yaw_quat_wxyz

    got = [float(v) for v in to_numpy(entity.get_quat()).reshape(-1)[:4]]
    want = list(yaw_quat_wxyz(yaw))
    if got[0] * want[0] + got[3] * want[3] < 0:  # q and -q are the same rotation
        want = [-c for c in want]
    assert got == pytest.approx(want, abs=1e-5), (label, got, want)
    w, _, _, z = got
    front = (1 - 2 * z * z, 2 * w * z)
    assert front[0] == pytest.approx(math.cos(yaw), abs=1e-5), label
    assert front[1] == pytest.approx(math.sin(yaw), abs=1e-5), label


@pytest.fixture(scope="module")
def store_handle():
    """One store build, shared in file order: the planogram check reads it,
    then the oracle/teleport check disturbs and resets it."""
    from aisle.sim import build_store

    return build_store("nexus", seed=1, scenario="S1")


def test_store_build_realizes_the_planogram(store_handle):
    """RS-1/RS-2 on Nexus (ADR-67, T16/ADR-19): the Nexus store is generated
    from the same planogram.toml as the Genesis one: the full stock spawns
    at its slots' world template poses with the composed unit yaw, the shelf
    boards carry that yaw too, the bin holds one item per category, counter
    and bin sit where the planogram puts them, and the mobile-profile robot
    starts at the store-frame origin."""
    import math

    from aisle.scenes.store import full_stock, slot_world_pose

    handle = store_handle
    plano = handle.planogram

    assert list(handle.items) == [item.item_id for item in full_stock(plano)]
    for slot_id, slot in plano["slots"].items():
        item_id = f"{slot_id}#0"
        assert item_id in handle.items, f"slot {slot_id} not stocked"
        world, yaw = slot_world_pose(plano, slot_id)
        size_z = handle.med_sizes[slot["category"]][2]
        pos = _xyz(handle.items[item_id])
        assert pos[0] == pytest.approx(world[0], abs=1e-4), slot_id
        assert pos[1] == pytest.approx(world[1], abs=1e-4), slot_id
        assert pos[2] == pytest.approx(world[2] + size_z / 2, abs=1e-4), slot_id
        _assert_yaw(handle.items[item_id], yaw, slot_id)

    # the shelf boards the items rest on carry the unit's yaw as well: the
    # Nexus builder composes it into the board's spawn quaternion (RS-2)
    boards = {entity.name: entity for entity in handle.scene.entities}
    for unit_id, unit in plano["units"].items():
        for level in range(len(plano["store"]["unit_geometry"]["level_heights"])):
            _assert_yaw(boards[f"{unit_id}_level_{level}"], unit["yaw"], unit_id)

    store = plano["store"]
    bin_top = store["bin_pos"][2] + store["bin_size"][2] / 2
    bin_items = [i for i in handle.items if i.startswith("bin#")]
    assert len(bin_items) == len(handle.med_sizes)
    for item_id in bin_items:
        pos = _xyz(handle.items[item_id])
        size_z = handle.med_sizes[handle.categories[item_id]][2]
        assert pos[2] == pytest.approx(bin_top + size_z / 2, abs=1e-4), item_id

    assert _xyz(handle.counter)[:2] == pytest.approx(store["counter_pos"][:2], abs=1e-4)
    assert _xyz(handle.bin)[:2] == pytest.approx(store["bin_pos"][:2], abs=1e-4)

    assert len(plano["units"]) >= 3
    assert len({u["zone"] for u in plano["units"].values()}) >= 2

    base = _xyz(handle.robot)
    assert math.hypot(base[0], base[1]) < 0.05
    assert handle.embodiment == "mobile"


def test_store_oracle_and_teleport_reset(store_handle):
    """T15 Stage A / T16 (ADR-18, ADR-19) on Nexus: store_oracle_state is
    n_items*7 in stock order with TC-1 (x, y, z, w) quats matching the spawn
    poses, and `teleport_store_reset` restores a disturbed item exactly (the
    reset path the store bridge runs on this engine)."""
    import math

    from aisle.scenes.store import (
        full_stock,
        generate_episode,
        spawn_pose,
        store_oracle_state,
        teleport_store_reset,
    )

    handle = store_handle
    stock = full_stock(handle.planogram)
    state = store_oracle_state(handle)
    assert state.dtype == np.float32
    assert state.shape == (len(stock) * 7,)
    x, _y, z, yaw = spawn_pose(handle.planogram, stock[0])
    assert state[0] == pytest.approx(x, abs=1e-5)
    assert state[2] == pytest.approx(z, abs=1e-5)
    assert abs(float(state[5])) == pytest.approx(abs(math.sin(yaw / 2)), abs=1e-5)  # qz
    assert abs(float(state[6])) == pytest.approx(abs(math.cos(yaw / 2)), abs=1e-5)  # w last

    item = handle.items[stock[0].item_id]
    item.set_pos(np.array([0.0, 0.0, 0.5], dtype=np.float32))
    assert store_oracle_state(handle)[0] != pytest.approx(x, abs=1e-3)
    teleport_store_reset(handle, handle.episode)
    assert np.allclose(store_oracle_state(handle), state, atol=1e-5)

    # T16: the one built scene realizes another scenario by teleport alone:
    # S2's emptied slots go to the stash line, the entity set never changes
    from aisle.scenes.store import STASH_Y

    s2 = generate_episode(7, "S2")
    teleport_store_reset(handle, s2)
    emptied = [entry["slot"] for entry in s2["restock"]]
    for item_spec in stock:
        stashed = _xyz(handle.items[item_spec.item_id])[1] == pytest.approx(STASH_Y, abs=1e-4)
        assert stashed == any(item_spec.item_id.startswith(f"{s}#") for s in emptied), item_spec


@pytest.mark.xfail(
    reason="issue #109 again, in the store path and on both engines: the store "
    "builder attaches the wrist camera with a translation-only offset while "
    "realized_calibration publishes a cam_to_ee built from wrist_mount_transform, "
    "whose wrist_rotation_xyzw is a half turn about x. Measured on nexus: the "
    "attached camera looks along the EE link's -Z (back up the arm) while the "
    "published block says +Z, elementwise difference 2.0. aisle.scenes.store "
    "builds the same identity offset, so Genesis publishes the same wrong block.",
    strict=True,
)
def test_store_wrist_cam_to_ee_matches_the_attached_camera(store_handle):
    """VER-8/SCN-5 for the store scene on Nexus: the published `cam_to_ee`
    must equal the mount the engine realized, exactly as the desk scene's
    twin requires. Nothing checks this for a store run, and an L1 or L2
    policy reading the store wrist frame is looking the other way."""
    from aisle.nodes.dora_genesis import realized_calibration
    from aisle.verifier.calibration import GL_TO_CV, rotation_from_quat_xyzw

    handle = store_handle
    handle.scene.step()
    handle.cams["wrist"].render(rgb=True)  # an attached camera syncs on render

    link = handle.robot.get_link("hand")
    link_rot = rotation_from_quat_xyzw(np.roll(to_numpy(link.get_quat()).reshape(-1)[:4], -1))
    cam_gl = np.asarray(to_numpy(handle.cams["wrist"].transform)).reshape(4, 4)
    realized_cv = (link_rot.T @ cam_gl[:3, :3]) @ GL_TO_CV
    published = rotation_from_quat_xyzw(
        realized_calibration(handle, load_physics(), is_store=True)["wrist"]["cam_to_ee"][
            "quat_xyzw"
        ]
    )
    assert np.allclose(published, realized_cv, atol=1e-5), (
        "published store cam_to_ee disagrees with the attached camera "
        f"(maxdiff {np.abs(published - realized_cv).max():.3e})"
    )


def test_lighting_dr_renders_the_per_channel_ambient_it_records():
    """SCN-6 on Nexus: the lighting toggle draws one ambient value per channel
    and `dr_applied` records the triple, so the camera must render the triple,
    not its mean. Re-rendering with the mean shifts each channel toward it."""
    from aisle.scenes.pharmacy import DRToggle, SceneCfg
    from aisle.sim import build_scene

    lit = build_scene(
        "nexus",
        seed=7,
        embodiment="franka",
        n_envs=1,
        headless=True,
        cfg=SceneCfg(lighting=DRToggle(enabled=True, seed=5)),
    )
    ambient = np.asarray(lit.dr_applied["ambient"], dtype=np.float64)
    assert np.ptp(ambient) > 0.05 and lit.scene.ambient == [float(c) for c in ambient]
    camera = lit.cams["overhead"]
    drawn = camera.render()[0].reshape(-1, 3).mean(axis=0)
    lit.scene.engine.viewer.set_sensor_camera_ambient_color(camera.cam_id, [ambient.mean()] * 3)
    lit.scene.sync(force=True)
    averaged = camera.render()[0].reshape(-1, 3).mean(axis=0)
    brighter = ambient > ambient.mean()
    assert np.all((drawn - averaged)[brighter] > 0) and np.all((drawn - averaged)[~brighter] < 0)


def test_rebuilding_frees_the_superseded_scene_cameras():
    """A superseded scene's cameras cannot render (one live scene per
    process), so the next build frees them: each holds render targets and a
    shadow atlas, and a wgpu panic after about ten builds was the result of
    keeping them. The superseded camera still refuses with a Python error."""
    from aisle.sim import build_scene

    first = build_scene("nexus", seed=0, embodiment="franka", n_envs=1, headless=True)
    for seed in range(1, 16):
        latest = build_scene("nexus", seed=seed, embodiment="franka", n_envs=1, headless=True)
        assert latest.scene.engine.viewer.num_sensor_cameras() == len(latest.cams)
    assert latest.cams["overhead"].render()[0].shape == (480, 640, 3)
    with pytest.raises(RuntimeError, match="superseded"):
        first.cams["overhead"].render()


# --- the live scene: the last build in the file, so nothing supersedes it ---


@pytest.fixture(scope="module")
def live():
    """One franka build for the tests that render or step, in file order:
    L1 first (it wants the settled build), then the wrist conformance, the
    debug camera, and finally the reset-to-rest check, which is the only one
    that leaves the arm somewhere else."""
    from aisle.sim import build_scene

    return build_scene("nexus", seed=3, embodiment="franka", n_envs=1, headless=True)


def test_cameras_render_all_passes(live):
    """SCN-5 / TC-9: overhead 640x480 fov 55, wrist 320x240 fov 70 attached
    to the EE link; one pass yields rgb (uint8), metric depth (float32) and
    a segmentation map whose ids are the scene's own map (background -1)."""
    handle = live
    overhead, wrist = handle.cams["overhead"], handle.cams["wrist"]
    assert tuple(overhead.res) == (640, 480) and overhead.fov == 55
    assert tuple(wrist.res) == (320, 240) and wrist.fov == 70
    assert wrist._attached_link is not None and wrist._attached_link.name == "hand"
    rgb, depth, seg, _ = overhead.render(rgb=True, depth=True, segmentation=True)
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert depth.shape == (480, 640) and depth.dtype == np.float32
    assert seg.shape == (480, 640) and seg.dtype == np.int64
    zfar = float(handle.scene.camera_planes[1])
    hit = depth[depth < zfar]
    assert hit.size > 0 and 0.3 < hit.min() < hit.max() < 2.0  # meters, camera at z=1.2
    # Genesis reads its cleared depth buffer, so a pixel that hit nothing comes
    # back at the far plane. Nexus must agree: L2 back-projects a bounding box,
    # and a 0.0 there would sit above every real surface (TC-9).
    sky = seg == -1
    if sky.any():
        assert np.all(depth[sky] >= zfar)
    ids = {int(i) for i in np.unique(seg)} - {-1}
    idx_dict = handle.scene.segmentation_idx_dict
    assert ids and ids <= set(idx_dict)
    # every box is visible from the overhead camera under its own id
    entity_idx = {name: entity.idx for name, entity in handle.boxes.items()}
    box_ids = {sid for sid, ref in idx_dict.items() if ref != -1 and ref[0] in entity_idx.values()}
    assert box_ids <= ids
    wrist_rgb = wrist.render()[0]
    assert wrist_rgb.shape == (240, 320, 3)


def test_l1_estimate_matches_nexus_ground_truth(live):
    """TC-9 on Nexus: seg + depth from one render pass, ids from the map the
    bridge publishes, pose ESTIMATED with the same estimator and the same
    10 mm gate the Genesis twin uses (tests/sim/test_l1_perception.py). If
    the estimate does not land on the ground truth L0 would have given away,
    an L1 number on this engine measures the estimator, not the policy."""
    from aisle.nodes.dora_genesis import realized_calibration, segmentation_id_map
    from aisle.nodes.segmented_pose import estimate_pose
    from aisle.scenes.pharmacy import load_meds
    from aisle.verifier.stages import backproject_overhead

    # the Genesis gate (tests/sim/test_l1_perception.py MAX_XY_ERROR_M): a
    # regression bound with room for settling wobble, not an accuracy claim
    max_xy_error_m = 0.010

    handle = live
    meds = load_meds()
    calibration = realized_calibration(handle, load_physics(), is_store=False)

    _rgb, depth, seg, _ = handle.cams["overhead"].render(rgb=True, depth=True, segmentation=True)
    seg = np.asarray(seg, dtype=np.int32)  # TC-1: the wire type is the contract
    depth = np.asarray(depth, dtype=np.float32)
    id_map = segmentation_id_map(
        handle.scene.segmentation_idx_dict,
        {name: entity.idx for name, entity in handle.boxes.items()},
    )
    # the ids are the scene's own segmentation numbering, not entity indices
    assert set(id_map) == set(handle.boxes)
    assert all(
        np.iinfo(np.int32).min < i < np.iinfo(np.int32).max for ids in id_map.values() for i in ids
    )

    def project(d, px):
        return backproject_overhead(d, calibration, px)

    errors = {}
    for name, entity in handle.boxes.items():
        size = meds[name]["size"]
        estimate = estimate_pose(
            seg, depth, id_map[name], float(size[2]), project, footprint_m=tuple(size[:2])
        )
        truth = np.asarray(to_numpy(entity.get_pos())).reshape(-1)[:3]
        errors[name] = float(np.linalg.norm(np.asarray(estimate["pos"][:2]) - truth[:2]))
        assert abs(estimate["pos"][2] - truth[2]) < max_xy_error_m, (name, estimate, truth)

    assert errors, "no meds in the scene to estimate"
    assert max(errors.values()) < max_xy_error_m, errors


def test_wrist_cam_to_ee_matches_the_attached_nexus_camera(live):
    """VER-8/SCN-5 on Nexus, the issue-#109 regression ported: the published
    `cam_to_ee` must equal the mount the engine actually realized, in OpenCV.

    The overhead-only check cannot see a wrist camera attached to the wrong
    body or composed with the offset in the wrong order: both halves stay
    self-consistent while the camera points back up the arm, which is how no
    recorded wrist frame ever contained the workspace."""
    from aisle.nodes.dora_genesis import realized_calibration
    from aisle.scenes.pharmacy import FRANKA_EE_LINK
    from aisle.verifier.calibration import GL_TO_CV, rotation_from_quat_xyzw

    handle = live
    handle.scene.step()
    # an attached camera's transform is stale until the scene syncs, which
    # rendering does (the Genesis twin hit the same trap while diagnosing #109)
    handle.cams["wrist"].render(rgb=True)

    link = handle.robot.get_link(FRANKA_EE_LINK)
    link_quat_wxyz = to_numpy(link.get_quat()).reshape(-1)[:4]
    link_rot = rotation_from_quat_xyzw(np.roll(link_quat_wxyz, -1))
    cam_gl = np.asarray(to_numpy(handle.cams["wrist"].transform)).reshape(4, 4)

    realized_cv = (link_rot.T @ cam_gl[:3, :3]) @ GL_TO_CV
    calibration = realized_calibration(handle, load_physics(), is_store=False)
    published = rotation_from_quat_xyzw(calibration["wrist"]["cam_to_ee"]["quat_xyzw"])
    assert np.allclose(published, realized_cv, atol=1e-5), (
        "published cam_to_ee disagrees with the attached camera "
        f"(maxdiff {np.abs(published - realized_cv).max():.3e})"
    )

    # the realized translation is the published offset, in the link frame
    link_pos = to_numpy(link.get_pos()).reshape(-1)[:3]
    realized_offset = link_rot.T @ (cam_gl[:3, 3] - link_pos)
    assert np.allclose(realized_offset, calibration["wrist"]["cam_to_ee"]["pos"], atol=1e-5), (
        realized_offset,
        calibration["wrist"]["cam_to_ee"]["pos"],
    )

    # and the camera must look where the gripper goes: the CV optical axis
    # (+Z) is the link's approach axis, not its reverse
    optical_axis_in_link = realized_cv[:, 2]
    assert optical_axis_in_link[2] == pytest.approx(1.0, abs=1e-5), (
        f"wrist camera looks along {optical_axis_in_link} of the EE link, not its +Z approach axis"
    )


def test_debug_camera_attaches_after_build(live):
    """ADR-67 operator tooling: the Nexus branch of `add_debug_camera`
    (dora_genesis.py) joins a built scene, since Nexus accepts cameras at any
    time, renders at the debug resolution and leaves the frozen wire cameras
    untouched (SCN-5). The Genesis twin passes the literal "genesis", so this
    branch had no coverage at all."""
    from aisle.nodes.dora_genesis import DEBUG_CAMERA_RES, add_debug_camera, parse_debug_view
    from aisle.scenes.pharmacy import resolve_layout

    handle = live
    eye, lookat = parse_debug_view(
        {"AISLE_DEBUG_VIEW": "side"}, resolve_layout(load_physics(), "franka")
    )
    camera = add_debug_camera(handle.scene, "nexus", eye, lookat)
    frame = np.asarray(camera.render(rgb=True)[0])
    assert frame.shape == (DEBUG_CAMERA_RES[1], DEBUG_CAMERA_RES[0], 3)
    assert frame.std() > 0
    assert set(handle.cams) == {"overhead", "wrist"}
    assert np.asarray(handle.cams["overhead"].render(rgb=True)[0]).shape == (480, 640, 3)


def test_zeroing_velocities_brings_the_arm_to_rest(live):
    """BRG-4/TC-6 on Nexus: the bridge calls `zero_all_dofs_velocity` on
    every reset, so it must leave the arm actually at rest (every link's
    linear and angular velocity zero) without moving it. The implementation
    brings the robot to rest by writing its own qpos back, so a coordinate
    drift here would teleport the arm on each reset.

    Runs last in the file: it is the one test that leaves the arm sagging."""
    handle = live
    robot = handle.robot
    for _ in range(20):  # no position targets: the arm sags under gravity
        handle.scene.step()
    moving = max(float(np.abs(np.asarray(robot._state(0)[k])).max()) for k in (3, 4))
    assert moving > 1e-4, moving  # there is motion to undo

    # the generalized velocities say the same thing as the link velocities,
    # and they are read back rather than reported as zeros
    assert np.abs(to_numpy(robot.get_dofs_velocity())).max() > 1e-4

    before = to_numpy(robot.get_qpos()).reshape(-1).copy()
    robot.zero_all_dofs_velocity()
    lin, ang = np.asarray(robot._state(0)[3]), np.asarray(robot._state(0)[4])
    assert lin.shape[0] == ang.shape[0] == len(robot.links)
    assert np.abs(lin).max() == 0.0, lin
    assert np.abs(ang).max() == 0.0, ang
    qvel = to_numpy(robot.get_dofs_velocity()).reshape(-1)
    assert qvel.shape == (robot.n_dofs,) and np.abs(qvel).max() == 0.0, qvel
    assert np.allclose(to_numpy(robot.get_qpos()).reshape(-1), before, atol=1e-6)
