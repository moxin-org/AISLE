"""Sim acceptance tests for the pharmacy scene (SPEC 020 SCN-1, SCN-3..5, SCN-7).

Marker `sim`: imports genesis, runs headless (CON-12). Run via
`uv sync --extra sim && uv run pytest -m sim`.
"""

import importlib.util

import numpy as np
import pytest

from aisle.scenes.pharmacy import (
    MED_NAMES,
    DRToggle,
    SceneCfg,
    build_scene,
    load_physics,
    oracle_state,
    resolve_layout,
    to_numpy,
)

# find_spec keeps collection sim-free: genesis is only executed inside tests
pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None, reason="sim extra not installed"
    ),
]


@pytest.fixture(scope="module")
def handle():
    return build_scene(seed=7, embodiment="franka", n_envs=1, headless=True)


def test_build_determinism(handle):
    """SCN-1, SCN-7: build_scene is a pure function of its arguments — the
    same (seed, cfg, platform) yields a bitwise-identical initial
    oracle_state; a different seed yields a different one."""
    first = oracle_state(handle)
    again = oracle_state(build_scene(seed=7, embodiment="franka", n_envs=1, headless=True))
    other = oracle_state(build_scene(seed=11, embodiment="franka", n_envs=1, headless=True))
    assert first.dtype == np.float32
    assert first.shape == (len(MED_NAMES) * 7,)
    assert np.array_equal(first, again)  # bitwise
    assert not np.array_equal(first, other)


def test_reachability(handle):
    """SCN-3, SCN-4: every sampled box placement admits a deterministic IK
    solution (asserted inside build_scene, position AND rotation), and the
    tray and shelf sit inside the franka layout profile's reach."""
    layout = resolve_layout(load_physics(), "franka")
    base = to_numpy(handle.robot.get_pos()).reshape(-1)[:3]
    for target in (layout["tray"]["pos"], layout["shelf"]["pos"]):
        assert np.linalg.norm(np.asarray(target) - base) <= layout["reach_m"]
    assert handle.reachability_errors == []


def test_no_interpenetration(handle):
    """SCN-3: initial box placements are rejection-sampled free of
    interpenetration — pairwise AABBs of the five boxes do not overlap."""
    aabbs = []
    for name, entity in handle.boxes.items():
        pos = to_numpy(entity.get_pos()).reshape(-1)[:3]
        half = np.asarray(handle.med_sizes[name]) / 2.0
        aabbs.append((name, pos - half, pos + half))
    for i, (name_a, lo_a, hi_a) in enumerate(aabbs):
        for name_b, lo_b, hi_b in aabbs[i + 1 :]:
            overlap = np.all(lo_a < hi_b) and np.all(lo_b < hi_a)
            assert not overlap, (name_a, name_b)


def test_cameras(handle):
    """SCN-5: overhead 640x480 fov 55 fixed; wrist 320x240 fov 70 ATTACHED
    to the EE link; the scene renders on the rasterizer path."""
    assert set(handle.cams) == {"overhead", "wrist"}
    overhead, wrist = handle.cams["overhead"], handle.cams["wrist"]
    assert tuple(overhead.res) == (640, 480) and overhead.fov == 55
    assert tuple(wrist.res) == (320, 240) and wrist.fov == 70
    assert getattr(wrist, "_attached_link", None) is not None
    assert type(handle.scene.visualizer.renderer).__name__ == "Rasterizer"


def test_oracle_quaternions_are_xyzw(handle):
    """TC-1: oracle_state quaternions are (x, y, z, w) wire order — boxes
    rest axis-aligned, so each quaternion block must be ~identity with w
    LAST, not first (genesis's native order)."""
    state = oracle_state(handle)
    for i in range(len(MED_NAMES)):
        quat = state[i * 7 + 3 : i * 7 + 7]
        assert abs(quat[3]) > 0.99, quat  # w last
        assert np.all(np.abs(quat[:3]) < 0.1), quat


def test_robot_starts_at_home_qpos(handle):
    """SCN-1: the built robot rests AT its configured home pose — the
    all-zeros qpos0 violates franka joint limits and self-collides, which
    T05 control must never inherit."""
    home = np.asarray(load_physics()["embodiment"]["franka"]["home_qpos"], dtype=np.float32)
    actual = to_numpy(handle.robot.get_qpos()).reshape(-1)[: home.shape[0]]
    assert np.allclose(actual, home, atol=1e-4)


def test_batched_build_oracle_covers_all_envs():
    """SCN-1: a batched build returns oracle_state of shape (n_envs,
    n_obj*7) — no environment is silently discarded — and reachability is
    still asserted at build time."""
    batched = build_scene(seed=7, embodiment="franka", n_envs=2, headless=True)
    state = oracle_state(batched)
    assert state.shape == (2, len(MED_NAMES) * 7)
    assert np.array_equal(state[0], state[1])  # identical initial placements


def test_boxes_follow_oracle_order(handle):
    """SCN-1: boxes dict insertion order is the fixed meds.toml order, which
    is the oracle_state layout (TC table)."""
    assert list(handle.boxes) == MED_NAMES


def test_dr_toggles_are_effective_seeded_and_isolated(handle):
    """SCN-6: an enabled toggle actually changes its axis, the same toggle
    seed reproduces the same values, a different seed differs — and box
    placements (oracle_state) stay untouched throughout."""
    cfg_a = SceneCfg(friction_jitter=DRToggle(enabled=True, seed=3))
    jittered_a = build_scene(seed=7, embodiment="franka", headless=True, cfg=cfg_a)
    jittered_b = build_scene(seed=7, embodiment="franka", headless=True, cfg=cfg_a)
    jittered_c = build_scene(
        seed=7,
        embodiment="franka",
        headless=True,
        cfg=SceneCfg(friction_jitter=DRToggle(enabled=True, seed=4)),
    )
    base_frictions = handle.dr_applied["frictions"]
    assert jittered_a.dr_applied["frictions"] != base_frictions  # effective
    assert jittered_a.dr_applied["frictions"] == jittered_b.dr_applied["frictions"]  # seeded
    assert jittered_a.dr_applied["frictions"] != jittered_c.dr_applied["frictions"]  # per-seed
    assert np.array_equal(oracle_state(handle), oracle_state(jittered_a))  # isolated

    lit = build_scene(
        seed=7,
        embodiment="franka",
        headless=True,
        cfg=SceneCfg(lighting=DRToggle(enabled=True, seed=5)),
    )
    assert lit.dr_applied["ambient"] != handle.dr_applied["ambient"]
    textured = build_scene(
        seed=7,
        embodiment="franka",
        headless=True,
        cfg=SceneCfg(textures=DRToggle(enabled=True, seed=9)),
    )
    assert textured.dr_applied["colors"] != handle.dr_applied["colors"]
    assert np.array_equal(oracle_state(handle), oracle_state(textured))
    shaken = build_scene(
        seed=7,
        embodiment="franka",
        headless=True,
        cfg=SceneCfg(camera_jitter=DRToggle(enabled=True, seed=6)),
    )
    assert shaken.dr_applied["overhead_pos"] != handle.dr_applied["overhead_pos"]


def test_so101_official_profile_builds():
    """SCN-1, SCN-3, SCN-4, SCN-5, TC-5: the official SO-101 asset builds
    as five arm plus one gripper DOF, starts at its configured home, keeps
    every placement reachable, and composes the official fixed TCP frame into
    its explicit end-effector camera mount."""
    from aisle.embodiment import SO101_JOINTS, profile_dof_indices

    handle = build_scene(seed=7, embodiment="so101", headless=True)
    profile = load_physics()["embodiment"]["so101"]
    indices = profile_dof_indices(handle.robot, profile)
    names = tuple(profile["arm_joint_names"] + profile["gripper_joint_names"])
    assert handle.robot.n_dofs == len(SO101_JOINTS)
    assert names == SO101_JOINTS
    actual = to_numpy(handle.robot.get_qpos()).reshape(-1)[list(indices)]
    assert np.allclose(actual, profile["home_qpos"])
    assert handle.cams["wrist"]._attached_link.name == profile["ee_link"]
    assert handle.reachability_errors == []
    assert list(handle.boxes) == MED_NAMES


def test_g1_right_profile_builds():
    """SCN-4, TC-5: the G1 g1_right profile builds as 33 native DOFs (9
    wire + 24 held), starts every wire joint at home_qpos and every held
    joint at held_home_qpos, keeps all placements IK-reachable, and the
    held-DOF PD holds standing home against gravity (200 steps, < 0.02 rad
    droop)."""
    from aisle.embodiment import G1_RIGHT_JOINTS, held_dof_indices, profile_dof_indices

    handle = build_scene(seed=7, embodiment="g1_right", headless=True)
    profile = load_physics()["embodiment"]["g1_right"]
    wire = profile_dof_indices(handle.robot, profile)
    held = held_dof_indices(handle.robot, profile)
    names = tuple(profile["arm_joint_names"] + profile["gripper_joint_names"])
    assert handle.robot.n_dofs == 33
    assert names == G1_RIGHT_JOINTS
    qpos = to_numpy(handle.robot.get_qpos()).reshape(-1)
    assert np.allclose(qpos[list(wire)], profile["home_qpos"], atol=1e-5)
    assert np.allclose(qpos[list(held)], profile["held_home_qpos"], atol=1e-5)
    assert handle.reachability_errors == []
    assert list(handle.boxes) == MED_NAMES
    for _ in range(200):
        handle.scene.step()
    after = to_numpy(handle.robot.get_qpos()).reshape(-1)
    droop = float(np.abs(after[list(held)] - np.asarray(profile["held_home_qpos"])).max())
    assert droop < 0.02, f"held joints drooped {droop:.4f} rad from standing home"


def test_physics_stability_at_substeps_one():
    """SCN-2/ADR-7: with substeps=1 (the bridge tick-budget setting),
    contact-rich stacking stays stable — 200 steps leave every box resting
    on its shelf level, not exploded or fallen through."""
    handle = build_scene(seed=7, embodiment="franka", headless=True)
    initial = {n: to_numpy(e.get_pos()).reshape(-1)[2] for n, e in handle.boxes.items()}
    for _ in range(200):
        handle.scene.step()
    for name, entity in handle.boxes.items():
        z = float(to_numpy(entity.get_pos()).reshape(-1)[2])
        assert abs(z - initial[name]) < 0.02, (name, initial[name], z)
        vel = to_numpy(entity.get_dofs_velocity()).reshape(-1)
        assert float(abs(vel).max()) < 0.5, (name, vel)


def test_debug_camera_attaches_after_build(handle):
    """ADR-67 operator tooling: AISLE_DEBUG_VIEW's camera joins a built
    Genesis scene through the visualizer's debug path (the frozen builder
    fixes the wire cameras at build, SCN-5), renders at the debug resolution
    and leaves the overhead camera untouched."""
    from aisle.nodes.dora_genesis import DEBUG_CAMERA_RES, add_debug_camera, parse_debug_view

    eye, lookat = parse_debug_view(
        {"AISLE_DEBUG_VIEW": "side"}, resolve_layout(load_physics(), "franka")
    )
    camera = add_debug_camera(handle.scene, "genesis", eye, lookat)
    frame = np.asarray(camera.render(rgb=True)[0])
    assert frame.shape == (DEBUG_CAMERA_RES[1], DEBUG_CAMERA_RES[0], 3)
    assert frame.std() > 0
    assert set(handle.cams) == {"overhead", "wrist"}
    assert np.asarray(handle.cams["overhead"].render(rgb=True)[0]).shape == (480, 640, 3)
