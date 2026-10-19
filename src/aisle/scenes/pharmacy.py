"""Pharmacy scene builder (SPEC 020).

`build_scene` is a pure function of its arguments (SCN-1, CON-5): all
randomness flows from the injected seed through explicit `random.Random`
instances (genesis's own RNG is pinned and never relied on), every physical
constant lives in meds.toml / physics.toml (SCN-2), and genesis is imported
lazily so unit tests and the validator never pay for sim dependencies.
An embodiment is a scene+driver profile swap (M0-5): shelf/tray placement
and scale come from the per-embodiment layout sections in physics.toml.
"""

from __future__ import annotations

import math
import platform
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aisle.embodiment import held_dof_indices, profile_dof_indices

_SCENES_DIR = Path(__file__).parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
SO101_URDF = _REPO_ROOT / "assets" / "so101" / "so101.urdf"
G1_URDF = _REPO_ROOT / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf"
FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"
FRANKA_EE_LINK = "hand"

_MAX_PLACEMENT_TRIES = 1000
_MAX_LAYOUT_RESTARTS = 64


def select_genesis_backend(sim_extra: str, platform_name: str, cuda_available: bool = False) -> str:
    """Resolve an explicit, attested dependency selection to one backend.

    The portable ``sim`` environment never changes physics merely because a
    GPU happens to be visible. The Linux-only ``cuda`` environment is an
    explicit opt-in and fails closed instead of silently falling back.
    """
    if sim_extra == "sim":
        return "metal" if platform_name == "Darwin" else "cpu"
    if sim_extra != "cuda":
        raise ValueError(f"unknown simulation extra {sim_extra!r}; expected 'sim' or 'cuda'")
    if platform_name != "Linux":
        raise ValueError("the locked CUDA simulation extra is supported only on Linux")
    if not cuda_available:
        raise ValueError("the CUDA simulation extra requires an available CUDA device")
    return "cuda"


def load_meds() -> dict:
    with open(_SCENES_DIR / "meds.toml", "rb") as f:
        return tomllib.load(f)


def load_physics() -> dict:
    with open(_SCENES_DIR / "physics.toml", "rb") as f:
        return tomllib.load(f)


def so101_urdf_options(profile: dict) -> dict[str, Any]:
    """Genesis import options that retain the official gripper geometry."""
    return {
        "fixed": True,
        "convexify": True,
        "decompose_robot_error_threshold": float(profile["collision_decompose_error_threshold"]),
    }


def g1_urdf_options(profile: dict) -> dict[str, Any]:
    """Genesis import options for the G1 (ADR-g1-embodiment).

    The pelvis is FIXED at the FK-derived standing height (soles on the
    ground at held home, zero pose); CoACD full decomposition
    (addendum-2 ruling) keeps the official non-convex collision meshes.
    """
    return {
        "fixed": True,
        "pos": (0.0, 0.0, float(profile["pelvis_height_m"])),
        "convexify": True,
        "decompose_robot_error_threshold": float(profile["collision_decompose_error_threshold"]),
    }


def robot_morph(embodiment: str, physics: dict) -> Any:
    """Resolve an embodiment to a Genesis morph; unknown names fail loud.

    SCN-4: the former else-branch silently loaded the SO-101 model for any
    name that resolved in physics.toml but was not franka — the explicit
    ValueError here keeps a typo from building the wrong robot. It fires
    BEFORE the genesis import so unit tests reject unknown names without a
    sim environment.
    """
    if embodiment not in ("franka", "mobile", "so101", "g1_right"):
        raise ValueError(f"unknown embodiment {embodiment!r}")
    import genesis as gs

    profile = physics["embodiment"][embodiment]
    if embodiment in ("franka", "mobile"):
        # mobile reuses the franka arm on a kinematic base (ADR-13); the
        # bridge re-bases it each tick
        return gs.morphs.MJCF(file=FRANKA_MJCF)
    if embodiment == "so101":
        if not SO101_URDF.exists():
            raise FileNotFoundError(
                f"so101 asset missing: {SO101_URDF} (acquisition pending, ADR-6)"
            )
        return gs.morphs.URDF(file=str(SO101_URDF), **so101_urdf_options(profile))
    if embodiment == "g1_right":
        if not G1_URDF.exists():
            raise FileNotFoundError(f"g1 asset missing: {G1_URDF} (G1-A vendoring incomplete)")
        return gs.morphs.URDF(file=str(G1_URDF), **g1_urdf_options(profile))
    raise ValueError(f"unknown embodiment {embodiment!r}")


def native_home_qpos(robot, profile: dict, wire_dof_indices) -> np.ndarray | None:
    """Full-length native home, or None when the profile has no home_qpos.

    Wire joints take profile home_qpos (TC-5 order). Profiles with
    held_joint_names fill every held DOF with held_home_qpos first and
    overwrite only the wire positions (from_wire_joint_order base=);
    legacy full-coverage wires keep the exact old np.empty semantics —
    including its garbage-filled tail — so behavior is bit-identical.
    """
    if "home_qpos" not in profile:
        return None
    home = np.asarray(profile["home_qpos"], dtype=np.float32)
    if wire_dof_indices is None:
        return home
    if "held_joint_names" in profile:
        base = np.zeros(robot.n_dofs, dtype=np.float32)
        base[list(held_dof_indices(robot, profile))] = np.asarray(
            profile["held_home_qpos"], dtype=np.float32
        )
        from aisle.embodiment import from_wire_joint_order

        return from_wire_joint_order(home, wire_dof_indices, base=base)
    native_home = np.empty(robot.n_dofs, dtype=np.float32)
    native_home[list(wire_dof_indices)] = home
    return native_home


def resolve_layout(physics: dict, embodiment: str) -> dict:
    """Merge shared geometry with the embodiment's layout profile: shelf
    position/levels/size, tray position/size, reach, and the ik section."""
    profiles = physics["embodiment"]
    if embodiment not in profiles:
        raise ValueError(
            f"unknown embodiment {embodiment!r}; add [embodiment.{embodiment}] to physics.toml"
        )
    profile = profiles[embodiment]
    return {
        "shelf": {
            **physics["shelf"],
            "pos": profile["shelf_pos"],
            "level_heights": profile["shelf_level_heights"],
            "level_depths": profile["shelf_level_depths"],
            "level_size": profile["shelf_level_size"],
            # per-embodiment override (finger-sweep clearance scales with
            # the gripper): same pattern as pregrasp_height_m
            **(
                {"min_separation": profile["min_separation"]} if "min_separation" in profile else {}
            ),
            **(
                {"hand_clearance_m": profile["shelf_hand_clearance_m"]}
                if "shelf_hand_clearance_m" in profile
                else {}
            ),
            **(
                {"edge_margin": profile["shelf_edge_margin_m"]}
                if "shelf_edge_margin_m" in profile
                else {}
            ),
        },
        "tray": {
            **physics["tray"],
            "pos": profile["tray_pos"],
            "size": profile["tray_size"],
        },
        "reach_m": profile["reach_m"],
        "placement_radius_m": float(profile.get("placement_radius_m", profile["reach_m"])),
        "center_separation_m": float(profile.get("center_separation_m", 0.0)),
        "placement_slots_xy": profile.get("placement_slots_xy"),
        "placement_global_jitter_m": float(profile.get("placement_global_jitter_m", 0.0)),
        "ik": {
            **physics["ik"],
            **(
                {"pregrasp_height_m": profile["pregrasp_height_m"]}
                if "pregrasp_height_m" in profile
                else {}
            ),
            **({"max_starts": profile["ik_max_starts"]} if "ik_max_starts" in profile else {}),
            **(
                {"max_solver_iters": profile["ik_max_solver_iters"]}
                if "ik_max_solver_iters" in profile
                else {}
            ),
            **(
                {"full_range_starts": profile["ik_full_range_starts"]}
                if "ik_full_range_starts" in profile
                else {}
            ),
            **({"pos_tol_m": profile["ik_pos_tol_m"]} if "ik_pos_tol_m" in profile else {}),
        },
    }


MED_NAMES = list(load_meds())


@dataclass(frozen=True)
class DRToggle:
    """One domain-randomization axis: off by default, independently seeded
    (SCN-6)."""

    enabled: bool = False
    seed: int = 0


@dataclass(frozen=True)
class SceneCfg:
    # SCN-3's build-time reachability assert is unconditional by spec —
    # deliberately NOT a toggle here
    lighting: DRToggle = field(default_factory=DRToggle)
    textures: DRToggle = field(default_factory=DRToggle)
    friction_jitter: DRToggle = field(default_factory=DRToggle)
    camera_jitter: DRToggle = field(default_factory=DRToggle)
    # T2 (design doc section 7/8.3): printed med labels rendered as box
    # textures. OFF by default so every pre-T2 scene stays byte-identical;
    # a T2 graph declares it on the bridge (AISLE_LABELS), where the graph
    # hash attests it -- the same pattern as the perception rung (TC-9).
    labels: bool = False
    # T2's "no color prior" rule: a seeded permutation reassigns the box
    # COLORS across meds (labels, sizes and physics stay with the med),
    # so color no longer predicts identity and only the printed word
    # does. OFF by default; the gated draw touches no existing RNG
    # stream, so pre-T2 scenes stay byte-identical.
    shuffle_colors: bool = False
    # T3 (design doc section 3): the tier's occlusion layout — the
    # designated target (med index seed % n, the same rule the T3
    # rollout client uses to pick the episode target) gets a BLOCKER
    # box relocated directly in front of it on the same board. OFF by
    # default; scenes stay byte-identical when off, and the transform is
    # a pure post-pass over the sampled placements (scene = f(seed)).
    occlusion: bool = False


@dataclass(frozen=True)
class Placement:
    name: str
    level: int
    x: float
    y: float
    z: float


@dataclass
class SceneHandle:
    scene: Any
    robot: Any
    boxes: dict[str, Any]
    tray: Any
    cams: dict[str, Any]
    embodiment: str
    seed: int
    med_sizes: dict[str, list[float]]
    dr_applied: dict[str, Any] = field(default_factory=dict)
    reachability_errors: list[str] = field(default_factory=list)


def level_x_span(shelf: dict, level: int) -> tuple[float, float]:
    """A level board's x-span. Boards are REAR-ALIGNED within the shelf
    footprint (staggered shelving, ADR-12) — the single source of truth
    for that convention, shared by the sampler, the scene builder, and
    the grasp planner's needs_front safety net."""
    rear_x = shelf["pos"][0] + shelf["level_size"][0] / 2
    return rear_x - shelf["level_depths"][level], rear_x


def open_band(shelf: dict, level: int) -> tuple[float, float]:
    """The level's x-band with OPEN SKY: its board span, ending
    hand_clearance_m before any higher (shallower, rear-aligned) board's
    front plane — top-down grasps need the hand column clear (ADR-12)."""
    x_min, x_max = level_x_span(shelf, level)
    for higher in range(level + 1, len(shelf["level_depths"])):
        x_max = min(x_max, level_x_span(shelf, higher)[0] - shelf["hand_clearance_m"])
    return x_min, x_max


def occluded_target(seed: int, med_names: list[str]) -> str:
    """T3's designated target for a seed — the SAME rule the rollout
    client uses to pick the episode target, so the occluded box and the
    requested med always coincide (scene stays a pure f(seed))."""
    return med_names[seed % len(med_names)]


def apply_occlusion(
    placements: list[Placement], seed: int, med_names: list[str], layout: dict
) -> list[Placement]:
    """T3 layout post-pass (CON-5 pure): move a BLOCKER directly in
    front of the designated target on the same board, 1.5 cm of face
    gap — inside min_separation on purpose, occlusion IS the tier. The
    target is first shifted to the rear of its level's open band so both
    boxes fit the board; every coordinate is a deterministic function of
    the sampled layout."""
    meds = load_meds()
    shelf = layout["shelf"]
    target_name = occluded_target(seed, med_names)
    blocker_name = med_names[(seed + 1) % len(med_names)]
    by_name = {p.name: p for p in placements}
    target = by_name[target_name]
    # the pair ALWAYS lives on level 0: its deep board fits both boxes
    # (the level-1 board is too shallow — the first layout draft parked
    # the target hanging past its rear edge and the box fell on its own
    # at t=0.36 s, a `dropped` verdict with no robot contact), and level
    # 0's covered rear band (under the upper board) is the under-board
    # front-only zone the tier's rearrangement premise needs
    level = 0
    band_min, band_max = level_x_span(shelf, level)
    open_min, open_max = open_band(shelf, level)
    target_half = meds[target_name]["size"][0] / 2
    blocker_half = meds[blocker_name]["size"][0] / 2
    gap = 0.015
    # rear-align the target within the board, the COVERED band (past the
    # open-sky front — occlusion under the board is the tier), and the
    # reach envelope (the sampler's pregrasp filter restated: this
    # post-pass must not park the target where SCN-3's reachability
    # assert aborts the build)
    ik = layout["ik"]
    max_target = layout["reach_m"] * ik["reach_margin_frac"]
    z_target = (
        shelf["pos"][2]
        + shelf["level_heights"][level]
        + shelf["board_thickness"] / 2
        + meds[target_name]["size"][2] / 2
    )
    target_x = band_max - shelf["edge_margin"] - target_half
    while (
        math.hypot(target_x, target.y, z_target + ik["pregrasp_height_m"]) > max_target
        and target_x > open_max
    ):
        target_x -= 0.01
    blocker_x = target_x - target_half - gap - blocker_half
    if blocker_x - blocker_half < band_min + shelf["edge_margin"]:
        blocker_x = band_min + shelf["edge_margin"] + blocker_half
        target_x = blocker_x + blocker_half + gap + target_half
    z_blocker = (
        shelf["pos"][2]
        + shelf["level_heights"][level]
        + shelf["board_thickness"] / 2
        + meds[blocker_name]["size"][2] / 2
    )
    moved = {
        target_name: Placement(name=target_name, level=level, x=target_x, y=target.y, z=z_target),
        blocker_name: Placement(
            name=blocker_name, level=level, x=blocker_x, y=target.y, z=z_blocker
        ),
    }
    # any OTHER box overlapping the relocated pair swaps to a vacated
    # original slot (deterministic, at most two vacancies — enough: the
    # sampler placed at most that many boxes in the pair's footprint)
    vacated = [p for name, p in by_name.items() if name in moved]
    out: list[Placement] = []
    for p in placements:
        if p.name in moved:
            out.append(moved[p.name])
            continue
        half_x, half_y = meds[p.name]["size"][0] / 2, meds[p.name]["size"][1] / 2
        clash = p.level == level and any(
            abs(p.x - m.x) < half_x + meds[m.name]["size"][0] / 2 + shelf["min_separation"]
            and abs(p.y - m.y) < half_y + meds[m.name]["size"][1] / 2 + shelf["min_separation"]
            for m in moved.values()
        )
        if clash and vacated:
            slot = vacated.pop(0)
            slot_z = (
                shelf["pos"][2]
                + shelf["level_heights"][slot.level]
                + shelf["board_thickness"] / 2
                + meds[p.name]["size"][2] / 2
            )
            out.append(Placement(name=p.name, level=slot.level, x=slot.x, y=slot.y, z=slot_z))
        else:
            out.append(p)
    return out


def sample_placements(seed: int, med_names: list[str], layout: dict) -> list[Placement]:
    """Rejection-sample per-seed box placements on the shelf levels
    (SCN-3): inside the level bounds minus edge margins, per-axis AABB
    separation of min_separation, and a geometric reach pre-filter so the
    IK backstop cannot abort the build on corner placements. Pure function
    of the seed."""
    rng = random.Random(seed)
    shelf = layout["shelf"]
    ik = layout["ik"]
    max_target = layout["reach_m"] * ik["reach_margin_frac"]
    meds = load_meds()
    width = shelf["level_size"][1]

    # levels whose nearest-point candidates can never pass the reach filter
    # (e.g. so101's top level) are excluded up front, not burned as tries
    tallest = max(spec["size"][2] for spec in meds.values())
    usable_levels = [
        lvl
        for lvl, height in enumerate(shelf["level_heights"])
        if math.hypot(
            abs(open_band(shelf, lvl)[0]) + shelf["edge_margin"],
            0.0,
            shelf["pos"][2]
            + height
            + shelf["board_thickness"] / 2
            + tallest / 2
            + ik["pregrasp_height_m"],
        )
        <= max_target
    ]
    if not usable_levels:
        raise AssertionError("no shelf level is inside the reach envelope (check layout profile)")
    if layout["placement_slots_xy"]:
        return _sample_profile_slots(
            rng,
            med_names,
            layout,
            meds,
            usable_levels[0],
            max_target,
        )
    # Sequential rejection can paint itself into a corner even when a valid
    # layout exists (the larger official SO-101 jaw corridor exposes this at
    # seed 22). Restart the whole layout from the same injected RNG stream;
    # the bound and stream remain deterministic (CON-5).
    for _restart in range(_MAX_LAYOUT_RESTARTS):
        placed: list[Placement] = []
        for name in med_names:
            size = meds[name]["size"]
            half_x, half_y = size[0] / 2, size[1] / 2
            for _ in range(_MAX_PLACEMENT_TRIES):
                level = usable_levels[rng.randrange(len(usable_levels))]
                band_min, band_max = open_band(shelf, level)
                x_lo = band_min - shelf["pos"][0] + shelf["edge_margin"] + half_x
                x_hi = band_max - shelf["pos"][0] - shelf["edge_margin"] - half_x
                if x_hi < x_lo:
                    continue  # this med cannot fit the level's open band
                local_x = rng.uniform(x_lo, x_hi)
                local_y = rng.uniform(
                    -width / 2 + shelf["edge_margin"] + half_y,
                    width / 2 - shelf["edge_margin"] - half_y,
                )
                candidate = Placement(
                    name=name,
                    level=level,
                    x=shelf["pos"][0] + local_x,
                    y=shelf["pos"][1] + local_y,
                    z=shelf["pos"][2]
                    + shelf["level_heights"][level]
                    + shelf["board_thickness"] / 2
                    + size[2] / 2,
                )
                pregrasp_distance = math.hypot(
                    candidate.x, candidate.y, candidate.z + ik["pregrasp_height_m"]
                )
                if (
                    pregrasp_distance > max_target
                    or math.hypot(candidate.x, candidate.y) > layout["placement_radius_m"]
                ):
                    continue
                if _separated(
                    candidate,
                    half_x,
                    half_y,
                    placed,
                    meds,
                    shelf["min_separation"],
                    layout["center_separation_m"],
                ):
                    placed.append(candidate)
                    break
            else:
                break
        else:
            return placed
    raise AssertionError(
        f"could not place all medicines after {_MAX_LAYOUT_RESTARTS} deterministic restarts"
    )


def _sample_profile_slots(
    rng: random.Random,
    med_names: list[str],
    layout: dict,
    meds: dict,
    level: int,
    max_target: float,
) -> list[Placement]:
    """Randomize medicines over a measured collision-free slot lattice.

    The slot coordinates and jitter are physics configuration, not inline
    scene constants (SCN-2). Every shuffled/jittered candidate is still
    rejection-checked for board bounds, reach, radial envelope, and pairwise
    separation (SCN-3).
    """
    shelf = layout["shelf"]
    slots = [tuple(map(float, slot)) for slot in layout["placement_slots_xy"]]
    if len(slots) != len(med_names):
        raise AssertionError("placement_slots_xy must have one slot per medicine")
    jitter = layout["placement_global_jitter_m"]
    band_min, band_max = open_band(shelf, level)
    for _restart in range(_MAX_LAYOUT_RESTARTS):
        shuffled = slots.copy()
        rng.shuffle(shuffled)
        dx, dy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
        placed: list[Placement] = []
        for name, (slot_x, slot_y) in zip(med_names, shuffled, strict=True):
            size = meds[name]["size"]
            half_x, half_y = size[0] / 2, size[1] / 2
            candidate = Placement(
                name=name,
                level=level,
                x=slot_x + dx,
                y=slot_y + dy,
                z=shelf["pos"][2]
                + shelf["level_heights"][level]
                + shelf["board_thickness"] / 2
                + size[2] / 2,
            )
            pregrasp_distance = math.hypot(
                candidate.x,
                candidate.y,
                candidate.z + layout["ik"]["pregrasp_height_m"],
            )
            in_bounds = (
                band_min + shelf["edge_margin"] + half_x
                <= candidate.x
                <= band_max - shelf["edge_margin"] - half_x
                and -shelf["level_size"][1] / 2 + shelf["edge_margin"] + half_y
                <= candidate.y
                <= shelf["level_size"][1] / 2 - shelf["edge_margin"] - half_y
            )
            if (
                not in_bounds
                or pregrasp_distance > max_target
                or math.hypot(candidate.x, candidate.y) > layout["placement_radius_m"]
                or not _separated(
                    candidate,
                    half_x,
                    half_y,
                    placed,
                    meds,
                    shelf["min_separation"],
                    layout["center_separation_m"],
                )
            ):
                break
            placed.append(candidate)
        else:
            return placed
    raise AssertionError(
        f"could not assign collision-free profile slots after {_MAX_LAYOUT_RESTARTS} attempts"
    )


def _separated(
    candidate: Placement,
    half_x: float,
    half_y: float,
    placed: list[Placement],
    meds: dict,
    min_separation: float,
    center_separation_m: float = 0.0,
) -> bool:
    """AABBs overlap iff BOTH axis gaps are below their half-extent sums, so
    separation requires at least one axis to clear its sum plus margin."""
    for other in placed:
        if other.level != candidate.level:
            continue
        if math.hypot(candidate.x - other.x, candidate.y - other.y) < center_separation_m:
            return False
        required_x = half_x + meds[other.name]["size"][0] / 2 + min_separation
        required_y = half_y + meds[other.name]["size"][1] / 2 + min_separation
        clear_x = abs(candidate.x - other.x) >= required_x
        clear_y = abs(candidate.y - other.y) >= required_y
        if not (clear_x or clear_y):
            return False
    return True


def _ensure_genesis(backend_name: str | None = None):
    import genesis as gs

    system = platform.system()
    backend_name = backend_name or select_genesis_backend("sim", system)
    cuda_available = backend_name == "cuda"
    if cuda_available:
        import torch

        cuda_available = torch.cuda.is_available()
    selected_extra = "cuda" if backend_name == "cuda" else "sim"
    resolved = select_genesis_backend(selected_extra, system, cuda_available)
    if resolved != backend_name:
        raise ValueError(
            f"simulation backend {backend_name!r} is incompatible with {system}; "
            f"expected {resolved!r}"
        )
    expected = {"metal": gs.metal, "cuda": gs.cuda, "cpu": gs.cpu}[backend_name]
    if not getattr(gs, "_initialized", False):
        # fixed seed: genesis's internal RNG must never be an input to build
        # outcomes (CON-5); reachability IK is additionally made
        # deterministic via explicit init_qpos and max_samples=1
        # performance_mode is deliberately OFF: it recompiles kernels for
        # minutes in every fresh process (measured >5 min), wrecking test
        # and node startup; substeps=1 alone keeps the step budget (ADR-7)
        gs.init(backend=expected, logging_level="warning", seed=0)
    elif gs.backend != expected:
        # a foreign pre-initialization would silently change build results
        # for identical arguments (CON-5) — refuse loudly instead
        raise RuntimeError(
            f"genesis already initialized with backend {gs.backend}; "
            f"build_scene requires {expected}"
        )
    return gs


def to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return np.asarray(tensor, dtype=np.float32)


def wrist_mount_rotation(cam_cfg: dict) -> np.ndarray:
    """The wrist camera's GL-convention mount rotation (SCN-5), from
    `wrist_rotation_xyzw` in physics.toml. The verifier's calibration
    derives `cam_to_ee` from the SAME value, so the published block and
    the built scene cannot disagree (VER-8)."""
    x, y, z, w = (float(v) for v in cam_cfg["wrist_rotation_xyzw"])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        # float64: this rotation reaches the PUBLISHED calibration block,
        # whose wrist check is EXACT against the float64 nominal (VER-8 e).
        # float32 here broke every realistic-verifier episode after #122
        # (0.05 -> 0.05000000074, stage-0 refusal) -- found by the first
        # A7 run, where the realistic verdict became load-bearing.
        dtype=np.float64,
    )


def ee_frame_transform(profile: dict | None) -> np.ndarray:
    """EE-link-to-official-TCP transform; identity when no fixed frame is
    configured. float64: composes into the published calibration (VER-8 e
    exact wrist check) -- see wrist_mount_rotation."""
    fixed = np.eye(4, dtype=np.float64)
    if not profile or "ee_frame_offset_xyz" not in profile:
        return fixed
    roll, pitch, yaw = (float(v) for v in profile["ee_frame_offset_rpy"])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    fixed[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    fixed[:3, 3] = profile["ee_frame_offset_xyz"]
    return fixed


def wrist_mount_transform(cam_cfg: dict, profile: dict | None = None) -> np.ndarray:
    """EE-link-to-camera transform, including an official fixed TCP frame
    when the simulator collapses that massless URDF link (SCN-5, ADR-27)."""
    # float64 end to end: the position published from this transform must
    # equal the config value EXACTLY (VER-8 e) -- see wrist_mount_rotation
    camera = np.eye(4, dtype=np.float64)
    camera[:3, :3] = wrist_mount_rotation(cam_cfg)
    camera[:3, 3] = cam_cfg["wrist_offset_m"]
    fixed = ee_frame_transform(profile)
    return fixed @ camera


def _rotation_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Branch-stable rotation-matrix to Genesis wxyz quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [s / 4.0, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s],
            dtype=np.float32,
        )
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1e-12)) * 2.0
    xyz = [0.0, 0.0, 0.0]
    xyz[i] = s / 4.0
    xyz[j] = (r[j, i] + r[i, j]) / s
    xyz[k] = (r[k, i] + r[i, k]) / s
    w = (r[k, j] - r[j, k]) / s
    return np.array([w, *xyz], dtype=np.float32)


def label_texture_image(label: str, color: list, px: int = 256) -> np.ndarray:
    """Deterministic label texture for a med box (T2, SCN-2): the med's
    (possibly DR-jittered) color as background, the label text in the
    contrast-picked ink (black over light colors, white over dark).

    PIL's EMBEDDED bitmap font only -- a system font would make the texture
    bytes platform-dependent and break CON-5's same-seed-same-scene
    contract. The text FILLS the face (single word, auto-scaled to ~95%
    width): the doc's predicted legibility pass (section 7) MEASURED the
    alternatives -- a 2-band px//8 layout identified 2/5 even on clean
    96 px crops, while face-filling text reads 5/5 down to 16 px under
    blur-tolerant normalized correlation, which is what the overhead
    camera's 20-40 px med crops actually offer."""
    from PIL import Image, ImageDraw, ImageFont

    rgb = tuple(int(round(255 * float(c))) for c in color[:3])
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    ink = (0, 0, 0) if luminance > 127 else (255, 255, 255)
    image = Image.new("RGB", (px, px), rgb)
    draw = ImageDraw.Draw(image)
    size = px
    while size > 8:
        font = ImageFont.load_default(size=size)
        if draw.textlength(label, font=font) <= px * 0.95:
            break
        size = int(size * 0.9)
    draw.text((px / 2, px / 2), label, fill=ink, font=font, anchor="mm")
    return np.asarray(image, dtype=np.uint8)


def build_scene(
    seed: int,
    embodiment: str = "franka",
    n_envs: int = 1,
    headless: bool = True,
    cfg: SceneCfg | None = None,
    sim_backend: str | None = None,
) -> SceneHandle:
    cfg = cfg or SceneCfg()
    gs = _ensure_genesis(sim_backend)
    physics = load_physics()
    layout = resolve_layout(physics, embodiment)
    meds = load_meds()
    shelf, tray_cfg = layout["shelf"], layout["tray"]
    dr_cfg = physics["domain_randomization"]

    for label, target in (("tray", tray_cfg["pos"]), ("shelf", shelf["pos"])):
        distance = math.hypot(*target)
        assert distance <= layout["reach_m"], (
            f"{label} at {target} outside {embodiment} workspace (SCN-4)"
        )

    lighting_rng = random.Random(cfg.lighting.seed)
    textures_rng = random.Random(cfg.textures.seed)
    friction_rng = random.Random(cfg.friction_jitter.seed)
    camera_rng = random.Random(cfg.camera_jitter.seed)

    ambient = (dr_cfg["ambient_default"],) * 3
    if cfg.lighting.enabled:
        ambient = tuple(
            min(1.0, dr_cfg["ambient_min"] + lighting_rng.random() * dr_cfg["ambient_range"])
            for _ in range(3)
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=physics["sim"]["dt"],
            substeps=physics["sim"]["substeps"],
            gravity=tuple(physics["sim"]["gravity"]),
        ),
        # keep ALL self-collision pairs: genesis filters pairs that collide
        # at the (invalid, all-zeros) neutral pose during build, which would
        # permanently disable those checks; we move to home_qpos before any
        # step, so the transient neutral contacts never simulate
        rigid_options=gs.options.RigidOptions(enable_neutral_collision=True),
        vis_options=gs.options.VisOptions(ambient_light=ambient),
        renderer=gs.renderers.Rasterizer(),  # SCN-5: Metal-safe default path
        show_viewer=not headless,
    )
    scene.add_entity(gs.morphs.Plane())

    shelf_material = gs.materials.Rigid(friction=physics["materials"]["shelf"]["friction"])
    width = shelf["level_size"][1]
    for level, (level_height, level_depth) in enumerate(
        zip(shelf["level_heights"], shelf["level_depths"], strict=True)
    ):
        # boards are REAR-ALIGNED (level_x_span): upper (shallower) boards
        # leave the lower level's front band open to the sky
        x_min, x_max = level_x_span(shelf, level)
        scene.add_entity(
            gs.morphs.Box(
                size=(level_depth, width, shelf["board_thickness"]),
                pos=((x_min + x_max) / 2, shelf["pos"][1], shelf["pos"][2] + level_height),
                fixed=True,
            ),
            material=shelf_material,
        )

    tray_material = gs.materials.Rigid(friction=physics["materials"]["tray"]["friction"])
    tray = scene.add_entity(
        gs.morphs.Box(size=tuple(tray_cfg["size"]), pos=tuple(tray_cfg["pos"]), fixed=True),
        material=tray_material,
    )

    # SCN-4: the loader is explicit per embodiment (robot_morph raises on
    # unknown names); franka/mobile share the MJCF arm (ADR-13).
    robot = scene.add_entity(robot_morph(embodiment, physics))

    box_physics = physics["materials"]["box"]
    applied_frictions: dict[str, float] = {}
    applied_colors: dict[str, list[float]] = {}
    boxes: dict[str, Any] = {}
    # T2 no-color-prior: a DEDICATED seeded stream permutes which med
    # wears which color -- independent of the jitter streams so layouts
    # and DR draws match the unshuffled scene at the same seed
    color_by_med = {name: meds[name]["color"] for name in meds}
    if cfg.shuffle_colors:
        names = list(meds)
        shuffled = list(np.random.default_rng(seed ^ 0x5EED).permutation(names))
        color_by_med = {
            name: meds[donor]["color"] for name, donor in zip(names, shuffled, strict=True)
        }
    placements = sample_placements(seed, list(meds), layout)
    if cfg.occlusion:
        placements = apply_occlusion(placements, seed, list(meds), layout)
    for placement in placements:
        friction = box_physics["friction"]
        if cfg.friction_jitter.enabled:
            friction *= 1.0 + (friction_rng.random() - 0.5) * dr_cfg["friction_jitter_frac"]
        applied_frictions[placement.name] = friction
        color = list(color_by_med[placement.name])
        if cfg.textures.enabled:
            scale_min, scale_range = dr_cfg["texture_scale_min"], dr_cfg["texture_scale_range"]
            color = [
                min(1.0, c * (scale_min + textures_rng.random() * scale_range)) for c in color[:3]
            ] + [color[3]]
        applied_colors[placement.name] = color
        if cfg.labels:
            # T2: labels need IMAGE textures, and Genesis Box primitives
            # carry no UVs ("Texture given but asset missing uv info", the
            # doc's predicted legibility pass) -- labeled meds use the
            # committed UV-mapped unit-cube MESH instead, per-axis scaled
            # to the med size; convexify of a cube is the same box, so
            # collision geometry matches the primitive path. The texture
            # COMPOSES with texture DR: the jittered color is the
            # background, so randomization perturbs appearance without
            # erasing the text.
            morph = gs.morphs.Mesh(
                file=str(_SCENES_DIR / "assets" / "unit_cube_uv.obj"),
                scale=tuple(meds[placement.name]["size"]),
                pos=(placement.x, placement.y, placement.z),
            )
            surface = gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=label_texture_image(
                        str(meds[placement.name].get("label", placement.name.upper())), color
                    )
                )
            )
        else:
            morph = gs.morphs.Box(
                size=tuple(meds[placement.name]["size"]),
                pos=(placement.x, placement.y, placement.z),
            )
            surface = gs.surfaces.Default(color=tuple(color))
        boxes[placement.name] = scene.add_entity(
            morph,
            material=gs.materials.Rigid(friction=friction, rho=box_physics["density_kg_m3"]),
            surface=surface,
        )

    cam_cfg = physics["cameras"]
    overhead_pos = list(cam_cfg["overhead_pos"])
    if cfg.camera_jitter.enabled:
        jitter = dr_cfg["camera_jitter_m"]
        overhead_pos = [p + (camera_rng.random() - 0.5) * jitter for p in overhead_pos]
    cams = {
        "overhead": scene.add_camera(
            res=(640, 480),
            pos=tuple(overhead_pos),
            lookat=tuple(cam_cfg["overhead_lookat"]),
            fov=55,
            GUI=False,
        ),
        "wrist": scene.add_camera(res=(320, 240), fov=70, GUI=False),
    }

    if n_envs == 1:
        scene.build()
    else:
        scene.build(n_envs=n_envs)

    # Start the robot AT its home pose. Configured profiles are expressed in
    # TC-5 wire order; map by official joint name instead of assuming the
    # URDF parser preserves XML order.
    profile = physics["embodiment"][embodiment]
    wire_dof_indices = profile_dof_indices(robot, profile)
    if "home_qpos" in profile:
        home = native_home_qpos(robot, profile, wire_dof_indices)
        robot.set_qpos(home if n_envs == 1 else np.tile(home, (n_envs, 1)))
    # finger-dof gains: without these the tendon-approximated gripper
    # actuator ignores position control and the fingers fall closed
    if "gripper_dofs" in profile and "gripper_kp" in profile:
        if wire_dof_indices is None:
            count = int(profile["gripper_dofs"])
            finger_dofs = list(range(robot.n_dofs - count, robot.n_dofs))
        else:
            count = len(profile["gripper_joint_names"])
            finger_dofs = list(wire_dof_indices[-count:])
        robot.set_dofs_kp(
            np.asarray(profile["gripper_kp"], dtype=np.float32), dofs_idx_local=finger_dofs
        )
        robot.set_dofs_kv(
            np.asarray(profile["gripper_kv"], dtype=np.float32), dofs_idx_local=finger_dofs
        )
    # Held-DOF PD (g1_right): the 24 non-wire joints stay at held home —
    # gravity must not droop them (SCN-4). The PD target is always the
    # held home, set once here; the bridge never commands these DOFs.
    if "held_joint_names" in profile:
        held_dofs = list(held_dof_indices(robot, profile))
        robot.set_dofs_kp(
            np.full(len(held_dofs), float(profile["held_kp"]), dtype=np.float32),
            dofs_idx_local=held_dofs,
        )
        robot.set_dofs_kv(
            np.full(len(held_dofs), float(profile["held_kv"]), dtype=np.float32),
            dofs_idx_local=held_dofs,
        )
        # ...and the target IS that home: gains alone do not hold a DOF.
        # Full-length target (wire positions equal their home; the bridge
        # overwrites those every tick anyway). Batched like set_qpos above.
        robot.control_dofs_position(home if n_envs == 1 else np.tile(home, (n_envs, 1)))

    ee_link = robot.get_link(profile.get("ee_link", FRANKA_EE_LINK))
    # SCN-5: orientation as well as position. SO-101 composes the official
    # fixed gripper-frame transform that Genesis collapses during import.
    offset = wrist_mount_transform(cam_cfg, profile)
    cams["wrist"].attach(ee_link, offset_T=offset)

    handle = SceneHandle(
        scene=scene,
        robot=robot,
        boxes=boxes,
        tray=tray,
        cams=cams,
        embodiment=embodiment,
        seed=seed,
        med_sizes={name: list(meds[name]["size"]) for name in meds},
        dr_applied={
            "ambient": ambient,
            "overhead_pos": overhead_pos,
            "frictions": applied_frictions,
            "colors": applied_colors,
        },
    )

    # SCN-3: asserted at build time, unconditionally; placements are seed-
    # identical across batched envs, so env 0 witnesses reachability for all
    _assert_reachable(handle, ee_link, layout, n_envs)
    return handle


def _assert_reachable(handle: SceneHandle, ee_link, layout: dict, n_envs: int = 1) -> None:
    """SCN-3: every box placement must admit an IK solution to its pre-grasp
    pose. Deterministic multi-start (CON-5): explicit seeded init_qpos
    perturbations with max_samples=1, so genesis's global RNG never
    influences the outcome; position AND rotation error are both checked."""
    ik_cfg = layout["ik"]
    rng = random.Random(handle.seed)
    profile = load_physics()["embodiment"][handle.embodiment]
    wire_dof_indices = profile_dof_indices(handle.robot, profile)
    if "home_qpos" in profile:
        home = native_home_qpos(handle.robot, profile, wire_dof_indices)
    else:
        home = to_numpy(handle.robot.get_qpos()).reshape(-1)[: handle.robot.n_dofs]

    if handle.embodiment == "so101":
        # The official chain cannot realize the provisional vertical
        # top-down pose at this compact shelf. Validate the SAME native
        # radial-front pregrasp/insertion geometry the production planner
        # uses (ADR-27), through the pure URDF-derived IK used at runtime.
        from aisle.nodes.grasp_topdown import plan_grasp
        from aisle.nodes.ik_trajectory import (
            ik_continuation,
            ik_solve,
            quat_to_rotation,
        )

        shelf = layout["shelf"]
        shelf_front_x = shelf["pos"][0] - shelf["level_size"][0] / 2
        front_clearance = float(profile["front_clearance_m"])
        front_overshoot = float(profile["front_tcp_overshoot_m"])
        jaw_center_offset = float(profile["front_jaw_center_offset_m"])
        vertical_offset = float(profile["front_vertical_offset_m"])
        arm_home = home[: len(profile["arm_joint_names"])]
        failures = []
        centres = {
            name: to_numpy(entity.get_pos()).reshape(-1)[:3]
            for name, entity in handle.boxes.items()
        }
        for name in handle.boxes:
            centre = centres[name]
            target_pose = np.array([*centre, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            neighbours = [
                [
                    other_pos[0],
                    other_pos[1],
                    handle.med_sizes[other][0] / 2,
                    handle.med_sizes[other][1] / 2,
                ]
                for other, other_pos in centres.items()
                if other != name
            ]
            grasp_pose, approach, _ = plan_grasp(
                target_pose,
                handle.med_sizes[name],
                front=True,
                shelf_front_x=shelf_front_x,
                tray_top_z=0.0,
                radial_front=True,
                neighbours=neighbours,
                front_clearance=front_clearance,
                front_tcp_overshoot=front_overshoot,
                front_jaw_center_offset=jaw_center_offset,
                front_vertical_offset=vertical_offset,
            )
            grasp_pos = grasp_pose[:3].astype(np.float64)
            rotation = quat_to_rotation(grasp_pose[3:])
            pregrasp = grasp_pos - rotation[:, 2] * approach
            q_pre = ik_solve(pregrasp, rotation, arm_home, embodiment="so101")
            path = (
                ik_continuation(
                    pregrasp,
                    grasp_pos,
                    rotation,
                    q_pre,
                    embodiment="so101",
                )
                if q_pre is not None
                else None
            )
            if path is None:
                failures.append(f"{name}: radial-front pregrasp/insertion IK failed")
        handle.reachability_errors = failures
        assert not failures, f"unreachable placements (SCN-3): {failures}"
        return

    frame = ee_frame_transform(profile)
    local_point = frame[:3, 3] if "ee_frame_offset_xyz" in profile else None
    downward = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    link_quat = _rotation_to_quat_wxyz(downward @ frame[:3, :3].T)
    # Genesis supports one-axis alignment for underactuated arms. Aligning
    # the tool Z axis fixes the top-down approach direction while leaving
    # rotation about that axis free (five task constraints for five joints).
    rot_mask = [False, False, True] if profile.get("ik_free_yaw", False) else [True] * 3
    if wire_dof_indices is None:
        arm_dof_indices = list(range(handle.robot.n_dofs))
    else:
        arm_dof_indices = list(wire_dof_indices[: len(profile["arm_joint_names"])])
    lower, upper = handle.robot.get_dofs_limit()
    dof_limits = np.column_stack((to_numpy(lower), to_numpy(upper)))
    starts = [home]
    if ik_cfg.get("full_range_starts", False):
        for _ in range(1, ik_cfg["max_starts"]):
            init_qpos = home.copy()
            for dof in arm_dof_indices:
                lo, hi = dof_limits[dof]
                init_qpos[dof] = rng.uniform(float(lo), float(hi))
            starts.append(init_qpos)
    failures: list[str] = []
    for name, entity in handle.boxes.items():
        target = to_numpy(entity.get_pos()).reshape(-1)[:3] + np.array(
            [0.0, 0.0, ik_cfg["pregrasp_height_m"]], dtype=np.float32
        )
        best = None
        for attempt in range(ik_cfg["max_starts"]):
            if ik_cfg.get("full_range_starts", False):
                init_qpos = starts[attempt]
            elif attempt == 0:
                init_qpos = home
            else:
                perturbation = np.array(
                    [
                        (rng.random() - 0.5) * 2 * ik_cfg["init_perturbation_rad"]
                        for _ in range(home.shape[0])
                    ],
                    dtype=np.float32,
                )
                init_qpos = home + perturbation
            if n_envs > 1:  # genesis requires batch-shaped inputs
                pos_arg = np.tile(target, (n_envs, 1))
                quat_arg = np.tile(link_quat, (n_envs, 1))
                init_arg = np.tile(init_qpos, (n_envs, 1))
            else:
                pos_arg, quat_arg, init_arg = target, link_quat, init_qpos
            _, error = handle.robot.inverse_kinematics(
                link=ee_link,
                pos=pos_arg,
                quat=quat_arg,
                local_point=local_point,
                init_qpos=init_arg,
                max_samples=1,
                max_solver_iters=ik_cfg["max_solver_iters"],
                rot_mask=rot_mask,
                dofs_idx_local=arm_dof_indices,
                return_error=True,
            )
            # env 0 witnesses all envs: placements are seed-identical
            error = to_numpy(error).reshape(-1)[:6]
            pos_error = float(np.linalg.norm(error[:3]))
            rot_error = float(np.linalg.norm(error[3:6][rot_mask]))
            best = min(best or (pos_error, rot_error), (pos_error, rot_error))
            if pos_error <= ik_cfg["pos_tol_m"] and rot_error <= ik_cfg["rot_tol_rad"]:
                break
        else:
            failures.append(f"{name}: best ik error pos {best[0]:.4f} m rot {best[1]:.4f} rad")
    handle.reachability_errors = failures
    assert not failures, f"unreachable placements (SCN-3): {failures}"


def oracle_state(handle: SceneHandle) -> np.ndarray:
    """Ground-truth state: per box (meds.toml order) position (3) then
    quaternion in TC-1 (x, y, z, w) order — genesis returns (w, x, y, z),
    reordered here so the wire format matches the topic contract. Shape
    (n_obj*7,) for a single env, (n_envs, n_obj*7) for batched builds."""
    parts = []
    for entity in handle.boxes.values():
        pos = np.atleast_2d(to_numpy(entity.get_pos()))
        quat_wxyz = np.atleast_2d(to_numpy(entity.get_quat()))
        quat_xyzw = np.roll(quat_wxyz, -1, axis=-1)
        parts.extend((pos, quat_xyzw))
    state = np.concatenate(parts, axis=-1).astype(np.float32)
    return state[0] if state.shape[0] == 1 else state


def desk_scan_obstacles(physics: dict, embodiment: str) -> list[tuple[float, float, float, float]]:
    """AABBs (cx, cy, hx, hy) the base_scan raycast sees: the shelf boards
    and the tray, in the store frame (SPEC 210 MOB-1, ADR-13).

    Lives here, not in the bridge, because the budget guard reads it to
    build its MOB-3 keep-out (issue #189): geometry the guard refuses
    motion against is a safety verdict input and belongs inside the CON-7
    fence. Its store-scene twin, `store_scan_obstacles`, was already
    frozen in scenes/store.py -- this one sitting in dora_genesis.py was
    the accident, not the rule."""
    layout = resolve_layout(physics, embodiment)
    shelf, tray = layout["shelf"], layout["tray"]
    width = shelf["level_size"][1]
    obstacles = [
        ((x0 + x1) / 2, shelf["pos"][1], (x1 - x0) / 2, width / 2)
        for level in range(len(shelf["level_heights"]))
        for x0, x1 in [level_x_span(shelf, level)]
    ]
    obstacles.append((tray["pos"][0], tray["pos"][1], tray["size"][0] / 2, tray["size"][1] / 2))
    return obstacles
