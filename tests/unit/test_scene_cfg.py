"""Unit tests for the pharmacy scene configuration (SPEC 020 SCN-2, SCN-6).

Config parsing and the pure placement sampler only — no sim, no genesis
import (CON-12 unit marker).
"""

import dataclasses
import itertools
import tomllib

import pytest
from cli_helpers import REPO_ROOT, run_cli

pytestmark = pytest.mark.unit

SCENES = REPO_ROOT / "src" / "aisle" / "scenes"
MED_NAMES = ["amoxicillin", "ibuprofen", "cetirizine", "omeprazole", "metformin"]


def load_meds() -> dict:
    with open(SCENES / "meds.toml", "rb") as f:
        return tomllib.load(f)


def load_physics() -> dict:
    with open(SCENES / "physics.toml", "rb") as f:
        return tomllib.load(f)


def test_med_config():
    """SCN-2: exactly the five fixed medicine names, in oracle_state order,
    each with a 3-vector size in meters and an RGBA color."""
    meds = load_meds()
    assert list(meds) == MED_NAMES
    for name, spec in meds.items():
        assert len(spec["size"]) == 3 and all(0 < s < 0.3 for s in spec["size"]), name
        assert len(spec["color"]) == 4 and all(0 <= c <= 1 for c in spec["color"]), name


def test_physics_config():
    """SCN-2, SCN-4: physics and layout constants live in physics.toml —
    materials, sim step, shared shelf geometry, and a per-embodiment layout
    profile whose shelf and tray are inside that embodiment's reach."""
    physics = load_physics()
    assert physics["sim"]["dt"] > 0
    for material in ("box", "shelf", "tray"):
        assert physics["materials"][material]["friction"] > 0
    for embodiment in ("franka", "so101", "g1_right"):
        profile = physics["embodiment"][embodiment]
        assert len(profile["shelf_level_heights"]) == 2  # M0 env-change: two levels
        assert profile["reach_m"] > 0
        for key in ("shelf_pos", "tray_pos"):
            distance = sum(c * c for c in profile[key]) ** 0.5
            assert distance <= profile["reach_m"], (embodiment, key)


def test_so101_profile_matches_official_joint_space():
    """TC-5, SCN-2, SCN-4: the SO-101 profile uses the official 5+1
    joint order and URDF gripper range, with a complete legal home pose."""
    from aisle.embodiment import SO101_ARM_JOINTS, SO101_GRIPPER_JOINTS

    profile = load_physics()["embodiment"]["so101"]
    assert tuple(profile["arm_joint_names"]) == SO101_ARM_JOINTS
    assert tuple(profile["gripper_joint_names"]) == SO101_GRIPPER_JOINTS
    assert profile["ee_link"] == "gripper_link"
    assert profile["ee_frame_offset_xyz"] == pytest.approx([-0.0079, -0.000218121, -0.0981274])
    assert profile["ee_frame_offset_rpy"] == pytest.approx([0.0, 3.14159, 0.0])
    assert len(profile["home_qpos"]) == 6
    assert profile["gripper_open_qpos"] == pytest.approx(1.74533)
    assert profile["gripper_close_qpos"] == pytest.approx(-0.174533)


def test_so101_collision_meshes_preserve_the_official_finger_gap():
    """M0-5, SCN-2, SCN-4: the official non-convex SO-101 collision
    meshes are decomposed instead of collapsed to one convex hull, which
    would fill the gripper gap and make a physical pinch impossible."""
    from aisle.scenes.pharmacy import so101_urdf_options

    profile = load_physics()["embodiment"]["so101"]
    options = so101_urdf_options(profile)
    assert profile["collision_decompose_error_threshold"] == pytest.approx(0.0)
    assert options == {
        "fixed": True,
        "convexify": True,
        "decompose_robot_error_threshold": 0.0,
    }


def test_so101_profile_preserves_canonical_medicine_dimensions():
    """M0-5, SCN-2, SCN-4: an embodiment profile may move the shelf and
    tray, but the five fixed T0 medicine dimensions remain those in meds.toml."""
    from aisle.scenes.pharmacy import resolve_layout

    physics = load_physics()
    meds = load_meds()
    layout = resolve_layout(physics, "so101")
    profile = physics["embodiment"]["so101"]
    assert "med_scale" not in profile
    assert "med_scale" not in layout
    assert profile["kinematic_carry_latch"] is True
    assert profile["carry_latch_close"] > profile["carry_latch_release"]
    assert profile["carry_latch_max_distance_m"] == pytest.approx(0.20)
    tray_top = profile["tray_pos"][2] + profile["tray_size"][2] / 2
    assert tray_top + min(spec["size"][2] / 2 for spec in meds.values()) > 0.07
    max_box_half_extent = max(max(spec["size"][:2]) for spec in meds.values()) / 2
    capture_offset = (
        profile["front_tcp_overshoot_m"] ** 2 + profile["front_jaw_center_offset_m"] ** 2
    ) ** 0.5
    assert min(profile["tray_size"][:2]) / 2 > max_box_half_extent + capture_offset
    assert meds["amoxicillin"]["size"] == [0.060, 0.040, 0.100]


def test_dr_toggles_default_off():
    """SCN-6: every domain-randomization toggle defaults OFF and each is
    independently seedable; DR distribution constants live in physics.toml,
    not code."""
    from aisle.scenes.pharmacy import SceneCfg

    cfg = SceneCfg()
    toggles = [
        f.name for f in dataclasses.fields(cfg) if type(getattr(cfg, f.name)).__name__ == "DRToggle"
    ]
    assert set(toggles) == {"lighting", "textures", "friction_jitter", "camera_jitter"}
    for name in toggles:
        toggle = getattr(cfg, name)
        assert toggle.enabled is False
        assert isinstance(toggle.seed, int)  # independently seedable
    dr = load_physics()["domain_randomization"]
    assert {"friction_jitter_frac", "camera_jitter_m", "ambient_min", "texture_scale_min"} <= set(
        dr
    )


def test_placement_sampler_deterministic():
    """SCN-1, SCN-3 (CON-5): the placement sampler is a pure function of its
    seed — identical placements for identical seeds, different for
    different seeds, across the shelf levels."""
    from aisle.scenes.pharmacy import load_physics as load_p
    from aisle.scenes.pharmacy import resolve_layout, sample_placements

    layout = resolve_layout(load_p(), "franka")
    a = sample_placements(seed=7, med_names=MED_NAMES, layout=layout)
    b = sample_placements(seed=7, med_names=MED_NAMES, layout=layout)
    c = sample_placements(seed=8, med_names=MED_NAMES, layout=layout)
    assert a == b
    assert a != c
    assert [p.name for p in a] == MED_NAMES
    for p in a:
        assert 0 <= p.level < 3


@pytest.fixture(scope="module")
def placements_200(request):
    """One generation pass of 200 seeds x both embodiments, shared by the
    sweep tests below (each sample_placements call reparses meds.toml, so
    regenerating per-test doubles the suite's sampler work)."""
    from aisle.scenes.pharmacy import load_physics as load_p
    from aisle.scenes.pharmacy import resolve_layout, sample_placements

    physics = load_p()
    out = {}
    for embodiment in ("franka", "so101", "g1_right"):
        layout = resolve_layout(physics, embodiment)
        out[embodiment] = (
            layout,
            [sample_placements(seed, MED_NAMES, layout) for seed in range(200)],
        )
    return out


@pytest.mark.parametrize("embodiment", ["franka", "so101", "g1_right"])
def test_placements_never_interpenetrate_or_exceed_reach(embodiment, placements_200):
    """SCN-3: across 200 seeds (including the seed-99 regression the review
    found), no two same-level boxes overlap on BOTH axes, and every
    pre-grasp target respects the reach pre-filter."""
    layout, per_seed = placements_200[embodiment]
    meds = load_meds()
    ik = layout["ik"]
    max_target = layout["reach_m"] * ik["reach_margin_frac"]
    for seed, placements in enumerate(per_seed):
        for p in placements:
            target = (p.x**2 + p.y**2 + (p.z + ik["pregrasp_height_m"]) ** 2) ** 0.5
            assert target <= max_target, (seed, p)
        shelf = layout["shelf"]
        for p in placements:
            if p.level + 1 < len(shelf["level_heights"]):
                board_bottom = (
                    shelf["pos"][2]
                    + shelf["level_heights"][p.level + 1]
                    - shelf["board_thickness"] / 2
                )
                box_top = p.z + meds[p.name]["size"][2] / 2
                assert box_top < board_bottom, (seed, p.name, "intersects board above")
        for a, b in itertools.combinations(placements, 2):
            if a.level != b.level:
                continue
            half_x = (meds[a.name]["size"][0] + meds[b.name]["size"][0]) / 2
            half_y = (meds[a.name]["size"][1] + meds[b.name]["size"][1]) / 2
            overlap = abs(a.x - b.x) < half_x and abs(a.y - b.y) < half_y
            assert not overlap, (seed, a.name, b.name)


def test_so101_radial_front_placements_clear_observed_forearm_envelope(placements_200):
    """M0-5, SCN-3, SCN-4: every SO-101 pair clears the radial-front
    center-distance envelope measured from the complete failing campaign."""
    layout, per_seed = placements_200["so101"]
    minimum = layout["center_separation_m"]
    assert minimum >= 0.18
    for seed, placements in enumerate(per_seed):
        for a, b in itertools.combinations(placements, 2):
            distance = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
            assert distance >= minimum, (seed, a.name, b.name, distance)


@pytest.mark.parametrize("embodiment", ["franka", "so101", "g1_right"])
def test_shelf_levels_clear_tallest_box(embodiment):
    """SCN-3: every embodiment's board-to-board clearance fits the tallest
    medicine plus board thickness and separation margin — a box standing on
    any level can never intersect the board above (the so101 regression the
    review found)."""
    physics = load_physics()
    profile = physics["embodiment"][embodiment]
    shelf = physics["shelf"]
    tallest = max(spec["size"][2] for spec in load_meds().values())
    heights = profile["shelf_level_heights"]
    for below, above in zip(heights, heights[1:], strict=False):
        clearance = (above - shelf["board_thickness"] / 2) - (below + shelf["board_thickness"] / 2)
        assert clearance >= tallest + shelf["min_separation"], (embodiment, below, above)


def test_g1_right_profile_keys_lengths_and_urdf_consistency():
    """TC-5, SCN-4: the g1_right profile keys the held-DOF contract — wire
    9 + held 24 = 33 actuated URDF joints, partition disjoint, held list in
    URDF document order, home lengths match, gripper stops from the URDF,
    and the pelvis height comes from the FK sole probe."""
    import xml.etree.ElementTree as ET

    from aisle.embodiment import G1_RIGHT_ARM_JOINTS, G1_RIGHT_GRIPPER_JOINTS

    physics = load_physics()
    profile = physics["embodiment"]["g1_right"]
    wire = tuple(profile["arm_joint_names"]) + tuple(profile["gripper_joint_names"])
    assert wire == G1_RIGHT_ARM_JOINTS + G1_RIGHT_GRIPPER_JOINTS
    held = profile["held_joint_names"]
    assert len(profile["home_qpos"]) == 9
    assert len(held) == 24 and len(profile["held_home_qpos"]) == 24
    urdf = ET.parse(REPO_ROOT / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf")
    doc_order = [
        j.attrib["name"]
        for j in urdf.getroot().findall("joint")
        if j.attrib["type"] not in ("fixed", "floating")
    ]
    assert len(doc_order) == 33
    assert held == [name for name in doc_order if name in set(held)]
    assert set(wire) | set(held) == set(doc_order)
    assert not (set(wire) & set(held))
    # Dex1-1 stops (URDF limits [-0.02, 0.0245]; +q separates => open)
    assert profile["gripper_open_qpos"] == pytest.approx(0.0245)
    assert profile["gripper_close_qpos"] == pytest.approx(-0.02)
    assert profile["gripper_dofs"] == 2
    # FK sole probe: soles flat on the ground at zero-pose standing home
    assert profile["pelvis_height_m"] == pytest.approx(0.7872)
    assert all(q == 0.0 for q in profile["held_home_qpos"])
    # held-DOF PD must exist and be positive (gravity droop, SCN-4)
    assert profile["held_kp"] > 0 and profile["held_kv"] > 0


def test_g1_urdf_options_pin_the_fk_pelvis_height():
    """SCN-4: the G1 loads fixed at the FK-derived pelvis height with full
    CoACD decomposition (addendum-2 ruling: threshold 0.0)."""
    from aisle.scenes.pharmacy import g1_urdf_options

    options = g1_urdf_options(load_physics()["embodiment"]["g1_right"])
    assert options == {
        "fixed": True,
        "pos": (0.0, 0.0, 0.7872),
        "convexify": True,
        "decompose_robot_error_threshold": 0.0,
    }


def test_unknown_embodiment_load_rejected_without_sim():
    """SCN-4: robot_morph raises for an unknown embodiment BEFORE importing
    genesis — a typo cannot silently build the SO-101 model (the hole this
    closes), and unit tests stay sim-free (CON-12)."""
    from aisle.scenes.pharmacy import load_physics as load_p
    from aisle.scenes.pharmacy import robot_morph

    with pytest.raises(ValueError, match="unknown embodiment 'humanoid'"):
        robot_morph("humanoid", load_p())


def test_unknown_embodiment_rejected():
    """SCN-4: an embodiment without a layout profile is an explicit error,
    not a KeyError. (`mobile` is a real profile since T11; use a name with
    no [embodiment.*] table.)"""
    from aisle.scenes.pharmacy import load_physics as load_p
    from aisle.scenes.pharmacy import resolve_layout

    with pytest.raises(ValueError, match="humanoid"):
        resolve_layout(load_p(), "humanoid")


def test_module_import_stays_sim_free():
    """CON-12: importing aisle.scenes.pharmacy must not import genesis, so
    unit tests and the validator run without sim dependencies."""
    probe = (
        "import sys; import aisle.scenes.pharmacy; "
        "assert 'genesis' not in sys.modules, 'genesis imported at module level'"
    )
    proc = run_cli(["-c", probe])
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("sim_extra", "platform_name", "cuda_available", "expected"),
    [
        ("sim", "Darwin", False, "metal"),
        ("sim", "Darwin", True, "metal"),
        ("sim", "Linux", True, "cpu"),
        ("sim", "Linux", False, "cpu"),
        ("sim", "Windows", True, "cpu"),
        ("cuda", "Linux", True, "cuda"),
    ],
)
def test_select_genesis_backend(sim_extra, platform_name, cuda_available, expected):
    """SCN-7, CON-5: the attested extra, not ambient hardware, selects the
    backend; the portable sim extra stays CPU on Linux even on a GPU host."""
    from aisle.scenes.pharmacy import select_genesis_backend

    assert select_genesis_backend(sim_extra, platform_name, cuda_available) == expected


@pytest.mark.parametrize(
    ("sim_extra", "platform_name", "cuda_available", "message"),
    [
        ("cuda", "Linux", False, "CUDA device"),
        ("cuda", "Darwin", True, "Linux"),
        ("bogus", "Linux", True, "simulation extra"),
    ],
)
def test_select_genesis_backend_fails_closed(sim_extra, platform_name, cuda_available, message):
    """SCN-7, CON-5: an explicit CUDA identity must never fall back to CPU
    or run on a platform for which the locked CUDA selection is undefined."""
    from aisle.scenes.pharmacy import select_genesis_backend

    with pytest.raises(ValueError, match=message):
        select_genesis_backend(sim_extra, platform_name, cuda_available)


def test_sampled_boxes_always_have_open_sky(placements_200):
    """SCN-3 / ADR-12: the staggered sampler's open bands and the
    planner's needs_front safety net agree — across 200 seeds and both
    embodiments, NO sampled placement triggers front-mode. The proven
    top-down grasp works on every sampled box; a regression in the band
    math (overhang, HAND_CLEARANCE_M, band-fit guard) surfaces here, not
    in a 4-hour acceptance run."""
    from aisle.nodes.grasp_topdown import needs_front

    for embodiment, (layout, per_seed) in placements_200.items():
        shelf = layout["shelf"]
        for seed, placements in enumerate(per_seed):
            for p in placements:
                assert not needs_front(p.x, p.z, shelf), (embodiment, seed, p)


def test_needs_front_covers_board_span_and_clearance_strip():
    """ADR-12: out-of-band poses under a higher board — including the
    hand_clearance_m strip in front of its span, where the T10 physics
    replay showed the hand landing on the board edge — trigger the
    front-mode safety net; open-sky poses do not."""
    from aisle.nodes.grasp_topdown import needs_front
    from aisle.scenes.pharmacy import load_physics as load_p
    from aisle.scenes.pharmacy import resolve_layout

    shelf = resolve_layout(load_p(), "franka")["shelf"]
    hand_clearance = shelf["hand_clearance_m"]  # SCN-2: sourced from physics.toml
    rear_x = shelf["pos"][0] + shelf["level_size"][0] / 2
    upper_front = rear_x - shelf["level_depths"][1]
    lower_box_z = shelf["pos"][2] + shelf["level_heights"][0] + shelf["board_thickness"] / 2 + 0.05
    upper_box_z = shelf["pos"][2] + shelf["level_heights"][1] + shelf["board_thickness"] / 2 + 0.05
    # directly under the upper board
    assert needs_front(rear_x - 0.01, lower_box_z, shelf)
    # in the reserved clearance strip just in front of the board span
    assert needs_front(upper_front - hand_clearance / 2, lower_box_z, shelf)
    # in the open band, clear of the strip
    assert not needs_front(upper_front - hand_clearance - 0.02, lower_box_z, shelf)
    # on the top level there is no board above
    assert not needs_front(rear_x - 0.01, upper_box_z, shelf)


def test_placements_never_start_inside_tray_footprint(placements_200):
    """SCN-3 / VER-3: no sampled box may START inside the tray's
    wrong_object entry footprint — the verifier fails an episode at t=0
    the moment ANY non-target box is in the tray region (M0-1 run
    m0-1-e634e4: the widened shelf overlapped the old tray corner and
    seeds 1/5/7/9 were instant wrong_object)."""
    for embodiment, (layout, per_seed) in placements_200.items():
        tray = layout["tray"]
        x_min = tray["pos"][0] - tray["size"][0] / 2
        x_max = tray["pos"][0] + tray["size"][0] / 2
        y_min = tray["pos"][1] - tray["size"][1] / 2
        y_max = tray["pos"][1] + tray["size"][1] / 2
        for seed, placements in enumerate(per_seed):
            for p in placements:
                inside = x_min <= p.x <= x_max and y_min <= p.y <= y_max
                assert not inside, (embodiment, seed, p)


class TestLabelTextures:
    """T2's rendered med labels (design doc section 7/8.3; scene sign-off
    2026-08-10): deterministic textures from PIL's EMBEDDED font only, so
    the same seed still builds the same scene byte-for-byte (CON-5)."""

    def test_texture_is_deterministic_and_label_distinct(self):
        import numpy as np

        from aisle.scenes.pharmacy import label_texture_image

        a = label_texture_image("AMOXICILLIN", [0.85, 0.2, 0.2, 1.0])
        b = label_texture_image("AMOXICILLIN", [0.85, 0.2, 0.2, 1.0])
        c = label_texture_image("METFORMIN", [0.85, 0.2, 0.2, 1.0])
        assert a.dtype == np.uint8 and a.shape == (256, 256, 3)
        assert (a == b).all()
        assert not (a == c).all()

    def test_ink_contrast_follows_background_luminance(self):
        """Black text on light boxes, white on dark — the doc's predicted
        legibility pass measures against this renderer, so the contrast
        rule is part of the measured surface, not styling."""
        from aisle.scenes.pharmacy import label_texture_image

        light = label_texture_image("X", [0.9, 0.9, 0.2, 1.0])
        dark = label_texture_image("X", [0.1, 0.1, 0.4, 1.0])
        # ink pixels are the minority; check the extreme pixel present
        assert (light == 0).any() and not (dark == 0).all()
        assert (dark == 255).any()

    def test_every_med_declares_a_label(self):
        from aisle.scenes.pharmacy import load_meds

        for name, spec in load_meds().items():
            assert spec.get("label"), f"{name} has no label (T2, SCN-2)"

    def test_labels_default_off_everywhere(self):
        """Pre-T2 scenes must stay byte-identical: SceneCfg and the bridge
        config both default labels OFF, and the ambient env cannot flip it
        (rollout scrubs AISLE_LABELS — the graph owns the pixels it
        attests)."""
        from aisle.harness.rollout import SCRUBBED_ENV, scrub_bringup_env
        from aisle.nodes.dora_genesis import parse_bridge_config
        from aisle.scenes.pharmacy import SceneCfg

        assert SceneCfg().labels is False
        assert parse_bridge_config({}).labels is False
        assert parse_bridge_config({"AISLE_LABELS": "1"}).labels is True
        assert "AISLE_LABELS" in SCRUBBED_ENV
        assert "AISLE_LABELS" not in scrub_bringup_env({"AISLE_LABELS": "1", "K": "v"})

    def test_target_meds_cannot_leak_from_the_ambient_shell(self):
        """PR #178 review: the rollout runner never sets AISLE_TARGET_MEDS
        and no graph declares it, so an ambient developer-shell value was
        the only way it could reach a measured run — silently re-targeting
        every episode while the attestation stayed clean. Scrubbed, the
        client falls back to its deterministic seed-derived default."""
        from aisle.harness.rollout import SCRUBBED_ENV, scrub_bringup_env

        assert "AISLE_TARGET_MEDS" in SCRUBBED_ENV
        kept = scrub_bringup_env({"AISLE_TARGET_MEDS": "ibuprofen", "AISLE_SEEDS": "0"})
        assert "AISLE_TARGET_MEDS" not in kept
        assert kept["AISLE_SEEDS"] == "0"  # unrelated config survives

    def test_junk_label_toggle_is_refused_not_guessed(self):
        import pytest

        from aisle.nodes.dora_genesis import parse_bridge_config

        with pytest.raises(ValueError, match="AISLE_LABELS"):
            parse_bridge_config({"AISLE_LABELS": "maybe"})

    def test_occlusion_layout_geometry_and_parity(self):
        """T3 (design doc §3): the occlusion post-pass parks the blocker
        directly in front of the seed-designated target — same row, same
        level, 1.5 cm face gap (inside min_separation: occlusion IS the
        tier) — deterministically; placements are untouched when off."""
        from aisle.scenes.pharmacy import (
            apply_occlusion,
            load_meds,
            load_physics,
            occluded_target,
            resolve_layout,
            sample_placements,
        )

        meds = load_meds()
        names = list(meds)
        layout = resolve_layout(load_physics(), "franka")
        for seed in (0, 3, 7):
            base = sample_placements(seed, names, layout)
            occ = apply_occlusion(base, seed, names, layout)
            assert occ == apply_occlusion(base, seed, names, layout)  # CON-5
            target = occluded_target(seed, names)
            assert target == names[seed % len(names)]
            blocker = names[(seed + 1) % len(names)]
            t = {p.name: p for p in occ}[target]
            b = {p.name: p for p in occ}[blocker]
            gap = t.x - meds[target]["size"][0] / 2 - (b.x + meds[blocker]["size"][0] / 2)
            assert gap == pytest.approx(0.015, abs=1e-9)
            assert t.y == b.y and t.level == b.level
            assert b.x < t.x  # blocker in FRONT (smaller x = shelf front)
            untouched = {p.name for p in base} - {target, blocker}
            for p in occ:
                if p.name in untouched:
                    assert p == {q.name: q for q in base}[p.name]

    def test_occlusion_toggle_default_off_scrubbed_and_junk_refused(self):
        """Same attestation contract as labels/shuffle: default off,
        graph-declared only (ambient scrubbed), junk refused."""
        import pytest

        from aisle.harness.rollout import SCRUBBED_ENV, scrub_bringup_env
        from aisle.nodes.dora_genesis import parse_bridge_config
        from aisle.scenes.pharmacy import SceneCfg

        assert SceneCfg().occlusion is False
        assert parse_bridge_config({}).occlusion is False
        assert parse_bridge_config({"AISLE_OCCLUSION": "1"}).occlusion is True
        assert "AISLE_OCCLUSION" in SCRUBBED_ENV
        assert scrub_bringup_env({"AISLE_OCCLUSION": "1"}) == {}
        with pytest.raises(ValueError, match="AISLE_OCCLUSION"):
            parse_bridge_config({"AISLE_OCCLUSION": "maybe"})

    def test_shuffle_colors_default_off_scrubbed_and_junk_refused(self):
        """T2 no-color-prior toggle (AISLE_SHUFFLE_COLORS): same contract
        as AISLE_LABELS — default off (pre-T2 scenes byte-identical),
        graph-declared only (ambient scrubbed), junk refused."""
        import pytest

        from aisle.harness.rollout import SCRUBBED_ENV, scrub_bringup_env
        from aisle.nodes.dora_genesis import parse_bridge_config
        from aisle.scenes.pharmacy import SceneCfg

        assert SceneCfg().shuffle_colors is False
        assert parse_bridge_config({}).shuffle_colors is False
        assert parse_bridge_config({"AISLE_SHUFFLE_COLORS": "1"}).shuffle_colors is True
        assert "AISLE_SHUFFLE_COLORS" in SCRUBBED_ENV
        assert "AISLE_TASK_TIER" in SCRUBBED_ENV
        scrubbed = scrub_bringup_env({"AISLE_SHUFFLE_COLORS": "1", "AISLE_TASK_TIER": "T2"})
        assert scrubbed == {}
        with pytest.raises(ValueError, match="AISLE_SHUFFLE_COLORS"):
            parse_bridge_config({"AISLE_SHUFFLE_COLORS": "maybe"})
