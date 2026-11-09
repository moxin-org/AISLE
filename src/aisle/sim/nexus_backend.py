"""Nexus engine backend (ADR-67): the frozen scenes, rebuilt on Nexus.

`build_scene` / `build_store` here are the Nexus counterparts of the frozen
`aisle.scenes.pharmacy.build_scene` / `aisle.scenes.store.build_store`. They
call the SAME frozen pure functions for everything that decides the scene
(layout resolution, seeded placements, occlusion, colors, label textures,
camera mounts) and only replace the Genesis object construction. The
objects they return (scene, robot, entities, links, cameras) implement the
duck-typed surface the bridge and the frozen helpers already use on Genesis
objects: same method names, same argument shapes, quaternions in (w, x, y,
z) order, numpy arrays where Genesis returns tensors (`to_numpy` passes
them through).

Nexus renders through its viewer: sensor cameras are offscreen surfaces on
the viewer's scene graph, so every camera concern (pose, attachment,
depth, segmentation) stays out of the physics engine.

`nexus3d` is imported lazily (CON-12): importing this module is sim-free.
"""

from __future__ import annotations

import io
import math
import platform
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aisle.embodiment import profile_dof_indices
from aisle.scenes.pharmacy import (
    FRANKA_EE_LINK,
    FRANKA_MJCF,
    SO101_URDF,
    SceneCfg,
    SceneHandle,
    _assert_reachable,
    _rotation_to_quat_wxyz,
    apply_occlusion,
    label_texture_image,
    level_x_span,
    load_meds,
    load_physics,
    resolve_layout,
    sample_placements,
    wrist_mount_transform,
)
from aisle.sim import select_nexus_backend

_SIM_DIR = Path(__file__).parent
_SCENES_ASSETS = _SIM_DIR.parent / "scenes" / "assets"
_ENGINE: NexusEngine | None = None
_BACKGROUND_SEG_ID = -1


def load_nexus_physics() -> dict:
    with open(_SIM_DIR / "nexus_physics.toml", "rb") as f:
        return tomllib.load(f)


def franka_mjcf_path() -> Path:
    """The Genesis-bundled Franka MJCF (SCN-4), located without importing
    genesis: the same file both engines load."""
    import importlib.util

    spec = importlib.util.find_spec("genesis")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("genesis is not installed; the franka MJCF ships with it (SCN-4)")
    path = Path(next(iter(spec.submodule_search_locations))) / "assets" / FRANKA_MJCF
    if not path.exists():
        raise FileNotFoundError(f"franka MJCF missing from the genesis package: {path}")
    return path


# --- math helpers (numpy, float64; wire conversions happen in the callers) --


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(r) -> np.ndarray:
    """Rotation matrix to (w, x, y, z): the frozen scene module's own
    branch-stable conversion, so both engines agree bit for bit."""
    return _rotation_to_quat_wxyz(np.asarray(r, dtype=np.float64))


def pos_lookat_up_to_transform(pos, lookat, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Genesis's look-at camera transform (OpenGL frame: -Z forward, +Y up):
    `z = pos - lookat`, `x = cross(up, z)`, `y = cross(z, x)`. Mirrors the
    pinned Genesis `_np_z_up_to_R`, including its degenerate branch, so the
    realized calibration matches what VER-8 re-derives."""
    pos = np.asarray(pos, dtype=np.float64)
    z = pos - np.asarray(lookat, dtype=np.float64)
    norm = float(np.linalg.norm(z))
    if norm > 1e-6:
        z = z / norm
    else:
        z = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.5 else np.array([0.0, 0.0, 1.0])
    x = np.cross(np.asarray(up, dtype=np.float64), z)
    xn = float(np.linalg.norm(x))
    if xn > np.finfo(np.float32).eps:
        x = x / xn
        y = np.cross(z, x)
    else:
        # colinear z and up: Genesis falls back to the identity GL frame
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0], transform[:3, 1], transform[:3, 2], transform[:3, 3] = x, y, z, pos
    return transform


def _per_dof(values, dofs: list, name: str) -> np.ndarray:
    """Genesis gain-shape rule: one value for every addressed DoF, or one per
    DoF. Any other length is a malformed gains profile, refused rather than
    silently broadcasting its first entry."""
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or values.size not in (1, len(dofs)):
        raise ValueError(
            f"{name} has {values.size} entries for {len(dofs)} DoFs; give one or one per DoF"
        )
    return values


def _broadcast_rows(values, n_envs: int, envs_idx) -> tuple[np.ndarray, list[int]]:
    """Genesis call-shape rule: `(k,)` or `(1, k)` broadcasts to every
    addressed env, `(n, k)` gives one row per addressed env."""
    rows = np.atleast_2d(np.asarray(values, dtype=np.float64))
    envs = list(range(n_envs)) if envs_idx is None else [int(e) for e in envs_idx]
    if rows.shape[0] == 1 and len(envs) > 1:
        rows = np.repeat(rows, len(envs), axis=0)
    if rows.shape[0] != len(envs):
        raise ValueError(f"{rows.shape[0]} rows for {len(envs)} environments")
    return rows, envs


def _squeeze_envs(rows: np.ndarray, n_envs: int) -> np.ndarray:
    """Genesis return-shape rule: single-env scenes return `(k,)`, batched
    ones `(n_envs, k)`."""
    rows = np.asarray(rows, dtype=np.float32)
    return rows[0] if n_envs == 1 else rows


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def checkerboard_texture(squares: int, color_a, color_b, px_per_square: int = 64) -> np.ndarray:
    """An RGB checkerboard of `squares` x `squares` cells alternating the two
    colors (RGB or RGBA in [0, 1]), for the floor slab's textured cube."""
    squares = max(1, int(squares))
    a = (np.asarray(color_a[:3], dtype=np.float64) * 255).round().astype(np.uint8)
    b = (np.asarray(color_b[:3], dtype=np.float64) * 255).round().astype(np.uint8)
    rows, cols = np.indices((squares, squares))
    cells = ((rows + cols) % 2).astype(bool)
    image = np.where(cells[..., None], b, a).astype(np.uint8)
    return np.repeat(np.repeat(image, px_per_square, axis=0), px_per_square, axis=1)


def load_uv_cube() -> tuple[list[list[float]], list[list[int]], list[list[float]]]:
    """The committed UV-mapped unit cube (T2 labels) as (vertices, faces,
    uvs) with one vertex per (position, uv) pair, as the renderer wants."""
    positions: list[list[float]] = []
    uvs: list[list[float]] = []
    verts: dict[tuple[int, int], int] = {}
    out_pos: list[list[float]] = []
    out_uv: list[list[float]] = []
    faces: list[list[int]] = []
    for line in (_SCENES_ASSETS / "unit_cube_uv.obj").read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            positions.append([float(v) for v in parts[1:4]])
        elif parts[0] == "vt":
            uvs.append([float(v) for v in parts[1:3]])
        elif parts[0] == "f":
            face = []
            for token in parts[1:4]:
                vi, ti = (int(t) - 1 for t in token.split("/")[:2])
                key = (vi, ti)
                if key not in verts:
                    verts[key] = len(out_pos)
                    out_pos.append(positions[vi])
                    out_uv.append(uvs[ti])
                face.append(verts[key])
            faces.append(face)
    return out_pos, faces, out_uv


# --- engine (process-wide GPU context) ------------------------------------


class NexusEngine:
    """One headless Nexus viewer (the GPU context and render scene graph) and
    compute pipeline per process, like `_ensure_genesis`'s one `gs.init`.
    Scenes share it; the viewer's scene graph is cleared between builds."""

    def __init__(self, backend_name: str, headless: bool = True):
        import nexus3d

        self.nexus3d = nexus3d
        self.backend_name = backend_name
        self.headless = headless
        viewer = nexus3d.NexusViewer(64, 64, headless=headless)
        # raises ValueError for a backend the installed wheel was built without
        viewer = viewer.with_backend(backend_name)
        viewer.init_backend()
        # one node per body: the depth and segmentation passes only see
        # non-instanced nodes, and per-body ids need them
        viewer.set_sensor_rendering(True)
        nexus_physics = load_nexus_physics()
        camera = nexus_physics["camera"]
        viewer.set_sensor_antialiasing(int(camera["msaa_samples"]))
        viewer.set_sensor_shadow_softness(float(camera["shadow_softness"]))
        viewer.set_sensor_shadow_range(
            float(camera["shadow_first_cascade_m"]), float(camera["shadow_distance_m"])
        )
        viewer.set_sensor_shadow_resolution(
            int(camera["shadow_resolution"]), int(camera["shadow_atlas_layers"])
        )
        viewer.add_directional_light(nexus3d.Vec3(*nexus_physics["light"]["direction"]))
        self.viewer = viewer
        self.pipeline = nexus3d.NexusPipeline()
        self.pipeline.preload_pipelines(viewer)
        self.scene_count = 0
        # the live scene's sensor cameras; each holds its own render targets
        # and shadow atlas on the GPU until it is released
        self.live_cameras: list[NexusCamera] = []

    def release_cameras(self) -> None:
        """Free the live scene's sensor cameras. Called when a new scene
        supersedes it, whose cameras could no longer render anyway, so a
        process can build any number of scenes without exhausting the GPU."""
        for camera in self.live_cameras:
            camera.release()
        self.live_cameras = []


def _ensure_nexus(backend_name: str | None = None, headless: bool = True) -> NexusEngine:
    """Nexus twin of `_ensure_genesis`: create the process-wide engine once,
    refuse a later build asking for another backend (CON-5)."""
    global _ENGINE
    system = platform.system()
    backend_name = backend_name or select_nexus_backend("sim", system)
    if backend_name == "cuda" and system != "Linux":
        raise ValueError("the CUDA backend is supported only on Linux")
    if _ENGINE is None:
        _ENGINE = NexusEngine(backend_name, headless=headless)
    elif _ENGINE.backend_name != backend_name:
        raise RuntimeError(
            f"nexus already initialized with backend {_ENGINE.backend_name!r}; "
            f"build_scene requires {backend_name!r}"
        )
    return _ENGINE


# --- scene objects (the Genesis duck-typed surface) ------------------------


class NexusEntity:
    """A free rigid body per environment (a med box, a board, a tray)."""

    def __init__(self, scene: NexusScene, idx: int, name: str, handles: list, fixed: bool):
        self.scene = scene
        self.idx = idx
        self.name = name
        self.handles = handles  # one RigidBodyHandle per env
        self.fixed = fixed
        self.link_bodies = [[h] for h in handles]  # per env, for segmentation ids

    def _rows(self) -> np.ndarray:
        pos, quat = self.scene.body_poses()
        rows = [self.scene.gpu_index(env, h) for env, h in enumerate(self.handles)]
        return pos[rows], quat[rows]

    def get_pos(self) -> np.ndarray:
        return _squeeze_envs(self._rows()[0], self.scene.n_envs)

    def get_quat(self) -> np.ndarray:
        return _squeeze_envs(self._rows()[1], self.scene.n_envs)

    def get_dofs_velocity(self) -> np.ndarray:
        lin, ang = self.scene.body_velocities()
        rows = [self.scene.gpu_index(env, h) for env, h in enumerate(self.handles)]
        return _squeeze_envs(np.concatenate([lin[rows], ang[rows]], axis=-1), self.scene.n_envs)

    def set_pos(self, pos, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(pos, self.scene.n_envs, envs_idx)
        _, quat = self._rows()
        for row, env in zip(rows, envs, strict=True):
            self.scene.set_body_pose(env, self.handles[env], row[:3], quat[env])

    def set_quat(self, quat, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(quat, self.scene.n_envs, envs_idx)
        pos, _ = self._rows()
        for row, env in zip(rows, envs, strict=True):
            self.scene.set_body_pose(env, self.handles[env], pos[env], row[:4])

    def zero_all_dofs_velocity(self, envs_idx=None) -> None:
        envs = range(self.scene.n_envs) if envs_idx is None else envs_idx
        for env in envs:
            self.scene.set_body_velocity(int(env), self.handles[int(env)])


@dataclass
class NexusJoint:
    name: str
    n_dofs: int
    dofs_idx_local: list[int]


class NexusLink:
    """One robot link; poses come from the robot's GPU readback cache."""

    def __init__(self, robot: NexusRobot, index: int, name: str):
        self.robot = robot
        self.index = index
        self.name = name

    def get_pos(self) -> np.ndarray:
        return _squeeze_envs(self.robot._link_poses()[0][:, self.index], self.robot.scene.n_envs)

    def get_quat(self) -> np.ndarray:
        return _squeeze_envs(self.robot._link_poses()[1][:, self.index], self.robot.scene.n_envs)


class NexusRobot:
    """One articulated robot, loaded once per environment."""

    def __init__(self, scene: NexusScene, robots: list, name: str):
        self.scene = scene
        self.robots = robots  # one nexus3d.Robot per env, identical layouts
        self.name = name
        first = robots[0]
        self.n_dofs = int(first.n_dofs)
        self.n_qs = self.n_dofs
        self.links = [NexusLink(self, i, n) for i, n in enumerate(first.link_names)]
        self.joints = [
            NexusJoint(name=n, n_dofs=int(nd), dofs_idx_local=list(range(off, off + nd)))
            for n, nd, off in zip(
                first.joint_names, first.joint_ndofs, first.joint_dof_offsets, strict=True
            )
        ]
        self.link_bodies = [list(r.link_body_handles) for r in robots]  # per env
        self._state_cache: dict[int, tuple] | None = None
        # last root pose written by set_pos/set_quat, per env. The GPU link
        # workspace only picks a written root up on the next step, so the
        # second half of a set_pos + set_quat pair would otherwise read the
        # stale root back and undo the first half (SPEC 210 re-basing).
        self._root_writes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        nexus_defaults = load_nexus_physics()["robot"]
        for robot in robots:
            robot.kp = [k if k > 0 else nexus_defaults["default_kp"] for k in robot.kp]
            robot.kv = [v if v > 0 else nexus_defaults["default_kv"] for v in robot.kv]
        self._targets = None

    # -- structure -------------------------------------------------------

    def get_joint(self, name: str) -> NexusJoint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(f"robot {self.name!r} has no joint {name!r}")

    def get_link(self, name: str) -> NexusLink:
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(f"robot {self.name!r} has no link {name!r}")

    def get_dofs_limit(self) -> tuple[np.ndarray, np.ndarray]:
        first = self.robots[0]
        return (
            np.asarray(first.dof_lower, dtype=np.float32),
            np.asarray(first.dof_upper, dtype=np.float32),
        )

    # -- state -----------------------------------------------------------

    def invalidate(self) -> None:
        self._state_cache = None

    def _state(self, env: int) -> tuple:
        if self._state_cache is None:
            self._state_cache = {}
        if env not in self._state_cache:
            engine = self.scene.engine
            self._state_cache[env] = self.scene.state.robot_state(engine.viewer, self.robots[env])
        return self._state_cache[env]

    def _link_poses(self) -> tuple[np.ndarray, np.ndarray]:
        states = [self._state(env) for env in range(self.scene.n_envs)]
        return (
            np.stack([s[1] for s in states]),  # (n_envs, n_links, 3)
            np.stack([s[2] for s in states]),  # (n_envs, n_links, 4) wxyz
        )

    def get_qpos(self) -> np.ndarray:
        rows = np.stack([self._state(env)[0] for env in range(self.scene.n_envs)])
        return _squeeze_envs(rows, self.scene.n_envs)

    def get_dofs_velocity(self) -> np.ndarray:
        """Generalized velocities, read back from the GPU dof state in the
        same order as `get_qpos`."""
        viewer = self.scene.engine.viewer
        rows = [
            np.asarray(self.scene.state.robot_qvel(viewer, self.robots[env]), dtype=np.float32)
            for env in range(self.scene.n_envs)
        ]
        return _squeeze_envs(np.stack(rows), self.scene.n_envs)

    def set_qpos(self, qpos, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(qpos, self.scene.n_envs, envs_idx)
        viewer = self.scene.engine.viewer
        for row, env in zip(rows, envs, strict=True):
            self.scene.state.set_robot_qpos(viewer, self.robots[env], [float(v) for v in row])
        self.scene.invalidate_poses()

    def zero_all_dofs_velocity(self, envs_idx=None) -> None:
        """Re-send the simulated qpos: `set_robot_qpos` zeroes the generalized
        velocities and the link body velocities on the GPU, so writing the
        current coordinates back is how the robot is brought to rest without
        moving it (the bridge calls this on every reset)."""
        envs = range(self.scene.n_envs) if envs_idx is None else [int(e) for e in envs_idx]
        for env in envs:
            qpos = self._state(env)[0]
            self.scene.state.set_robot_qpos(
                self.scene.engine.viewer, self.robots[env], [float(v) for v in qpos]
            )
        self.scene.invalidate_poses()

    def get_pos(self) -> np.ndarray:
        return self.links[0].get_pos()

    def get_quat(self) -> np.ndarray:
        return self.links[0].get_quat()

    def _root_pose(self, env: int) -> tuple[np.ndarray, np.ndarray]:
        """The root pose to write the other half of a re-basing against: the
        pending write if there is one, else the simulated pose."""
        pending = self._root_writes.get(env)
        if pending is not None:
            return pending
        pos, quat = self._link_poses()
        return np.asarray(pos[env, 0], dtype=np.float64), np.asarray(quat[env, 0], dtype=np.float64)

    def root_write_consumed(self) -> None:
        """A step has read the written root out of the body buffer, so the
        link readback is authoritative again."""
        self._root_writes.clear()

    def _write_root(self, env: int, pos, quat) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
        quat = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
        self.scene.set_body_pose(env, self.link_bodies[env][0], pos, quat)
        self._root_writes[env] = (pos, quat)
        self.invalidate()

    def set_pos(self, pos, envs_idx=None) -> None:
        """Re-base the (fixed-root) robot: the GPU reads a fixed root's pose
        from the body buffer every step (SPEC 210 mobile re-basing)."""
        rows, envs = _broadcast_rows(pos, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            self._write_root(env, row[:3], self._root_pose(env)[1])

    def set_quat(self, quat, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(quat, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            self._write_root(env, self._root_pose(env)[0], row[:4])

    # -- control ---------------------------------------------------------

    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None) -> None:
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        for robot in self.robots:
            robot.set_pd_gains(dofs, kp=[float(v) for v in _per_dof(kp, dofs, "kp")])

    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None) -> None:
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        for robot in self.robots:
            robot.set_pd_gains(dofs, kv=[float(v) for v in _per_dof(kv, dofs, "kv")])

    def control_dofs_position(self, target, dofs_idx_local=None, envs_idx=None) -> None:
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        rows, envs = _broadcast_rows(target, self.scene.n_envs, envs_idx)
        viewer = self.scene.engine.viewer
        for row, env in zip(rows, envs, strict=True):
            self.scene.state.set_robot_targets(
                viewer, self.robots[env], [float(v) for v in row[: len(dofs)]], dofs
            )

    # -- kinematics ------------------------------------------------------

    def inverse_kinematics(
        self,
        link,
        pos,
        quat,
        local_point=None,
        init_qpos=None,
        max_samples: int = 1,
        max_solver_iters: int = 100,
        rot_mask=None,
        dofs_idx_local=None,
        return_error: bool = False,
        **_ignored,
    ):
        """Genesis-compatible IK on environment 0's CPU kinematic model.

        `rot_mask[k]` asks to align the link's k-th axis with the target's;
        Nexus constrains world-frame rotation components instead, so the world
        axis closest to the one free tool axis is left unconstrained. The
        returned rotation error follows Genesis's masked-axis convention:
        component k is the angle between the current and target k-th axes,
        zero where the mask is False."""
        link_name = link.name if isinstance(link, NexusLink) else str(link)
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)[0]
        quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)[0]
        mask = [True, True, True] if rot_mask is None else [bool(m) for m in rot_mask]
        target_rot = quat_wxyz_to_matrix(quat)
        constrained = [True, True, True, True, True, True]
        if not all(mask):
            free = [k for k, m in enumerate(mask) if not m]
            if len(free) == 2:
                aligned = mask.index(True)
                # aligning one tool axis leaves rotation about it free: free
                # the world axis it points along most
                axis = target_rot[:, aligned]
                constrained[3 + int(np.argmax(np.abs(axis)))] = False
            else:
                for k in free:
                    constrained[3 + k] = False
        if init_qpos is None:
            init = [float(v) for v in self._state(0)[0]]
        else:
            init = [
                float(v)
                for v in np.asarray(init_qpos, dtype=np.float64).reshape(-1, self.n_dofs)[0]
            ]
        robot = self.robots[0]
        qpos, error = self.scene.state.robot_inverse_kinematics(
            robot,
            link_name,
            [float(v) for v in pos],
            [float(v) for v in quat],
            init,
            local_point=None if local_point is None else [float(v) for v in local_point],
            constrained_axes=constrained,
            dofs=None if dofs_idx_local is None else [int(d) for d in dofs_idx_local],
            max_iters=int(max_solver_iters),
        )
        qpos = np.asarray(qpos, dtype=np.float32)
        if return_error:
            # Genesis convention: per masked axis, the angle between axes
            fk_pos, fk_quat = self.scene.state.robot_forward_kinematics(
                robot, list(map(float, qpos)), link_name
            )
            current_rot = quat_wxyz_to_matrix(fk_quat)
            point = (
                np.zeros(3) if local_point is None else np.asarray(local_point, dtype=np.float64)
            )
            realized = np.asarray(fk_pos) + current_rot @ point
            lin_err = pos - realized
            rot_err = np.zeros(3)
            for k, on in enumerate(mask):
                if on:
                    cosine = float(np.clip(np.dot(current_rot[:, k], target_rot[:, k]), -1.0, 1.0))
                    rot_err[k] = math.acos(cosine)
            err = np.concatenate([lin_err, rot_err]).astype(np.float32)
            if self.scene.n_envs > 1:
                return np.tile(qpos, (self.scene.n_envs, 1)), np.tile(err, (self.scene.n_envs, 1))
            return qpos, err
        return np.tile(qpos, (self.scene.n_envs, 1)) if self.scene.n_envs > 1 else qpos


class NexusCamera:
    """An offscreen sensor camera on the viewer, in Genesis's camera terms:
    `res` (w, h), vertical `fov` in degrees, a GL-convention `transform`,
    `attach(link, offset_T)` and `render(rgb, depth, segmentation)`."""

    def __init__(self, scene: NexusScene, cam_id: int, res: tuple[int, int], fov: float):
        self.scene = scene
        self.cam_id = cam_id
        self.res = tuple(int(v) for v in res)
        self.fov = fov
        self._attached_link: NexusLink | None = None
        self._attached_offset: np.ndarray | None = None
        self._released_pose: tuple | None = None

    def release(self) -> None:
        """Free the viewer camera, keeping its last pose readable."""
        self._released_pose = self.scene.engine.viewer.sensor_camera_pose(self.cam_id)
        self.scene.engine.viewer.remove_sensor_camera(self.cam_id)

    @property
    def transform(self) -> np.ndarray:
        pos, quat = self._released_pose or self.scene.engine.viewer.sensor_camera_pose(self.cam_id)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quat_wxyz_to_matrix(quat)
        transform[:3, 3] = pos
        return transform

    def set_pose(self, transform=None, pos=None, lookat=None, up=(0.0, 0.0, 1.0)) -> None:
        self.scene.activate()
        if transform is None:
            transform = pos_lookat_up_to_transform(pos, lookat, up)
        transform = np.asarray(transform, dtype=np.float64)
        self.scene.engine.viewer.set_sensor_camera_pose(
            self.cam_id,
            [float(v) for v in transform[:3, 3]],
            [float(v) for v in matrix_to_quat_wxyz(transform[:3, :3])],
        )

    def attach(self, rigid_link: NexusLink, offset_T) -> None:
        self._attached_link = rigid_link
        self._attached_offset = np.asarray(offset_T, dtype=np.float64)
        if self.scene.built:
            self._apply_attachment()

    def _apply_attachment(self) -> None:
        link, offset = self._attached_link, self._attached_offset
        if link is None or offset is None:
            return
        body = link.robot.link_bodies[0][link.index]
        self.scene.activate()
        self.scene.engine.viewer.attach_sensor_camera(
            self.cam_id,
            0,
            body,
            [float(v) for v in offset[:3, 3]],
            [float(v) for v in matrix_to_quat_wxyz(offset[:3, :3])],
            self.scene.state,
        )

    def render(self, rgb=True, depth=False, segmentation=False, **_ignored):
        self.scene.sync()
        rgb_arr, depth_arr, seg_arr = self.scene.engine.viewer.render_sensor_camera(
            self.cam_id, rgb=bool(rgb), depth=bool(depth), segmentation=bool(segmentation)
        )
        if depth_arr is not None:
            # Genesis reads the cleared depth buffer, so a pixel that hit
            # nothing comes back at the FAR plane; the renderer writes 0.0
            # there instead. The difference is not cosmetic: L2 back-projects
            # a bounding-box mask, and a 0.0 sample lands at the camera, which
            # is above every real surface, so the background captured the
            # top-surface quantile and the grasp pose with it.
            depth_arr = np.asarray(depth_arr, dtype=np.float32)
            depth_arr[depth_arr <= 0.0] = np.float32(self.scene.camera_planes[1])
        if seg_arr is not None:
            # Genesis: int64 ids keyed by `segmentation_idx_dict`, -1 for the
            # background; the renderer's background is 0 and no scene id is 0
            seg_arr = np.asarray(seg_arr, dtype=np.int64)
            seg_arr[seg_arr == 0] = _BACKGROUND_SEG_ID
        return rgb_arr, depth_arr, seg_arr, None


class NexusScene:
    """The Nexus counterpart of a Genesis `Scene`: bodies, robots and cameras
    in `n_envs` batched environments, stepped by one compute pipeline."""

    def __init__(
        self,
        engine: NexusEngine,
        dt: float,
        substeps: int,
        gravity,
        n_envs: int,
        ambient,
        background_rgba,
        camera_planes: tuple[float, float],
    ):
        nx = engine.nexus3d
        self.engine = engine
        self.nexus3d = nx
        self.n_envs = int(n_envs)
        self.dt = float(dt)
        self.gravity = [float(g) for g in gravity]
        # per-channel, as Genesis's `ambient_light` takes it, so the lighting
        # DR draw that `dr_applied` records is the one rendered
        self.ambient = [float(c) for c in ambient]
        self.background_rgba = [float(c) for c in background_rgba]
        self.camera_planes = camera_planes
        self.state = nx.NexusState()
        self.friction_combine_rule = str(
            load_nexus_physics()["sim"].get("friction_combine_rule", "average")
        )
        # GPU timestamp queries: harvested by `sync`, read by `perf_stats`
        self.timestamps = nx.GpuTimestamps(engine.viewer, 2048)
        for _ in range(1, self.n_envs):
            self.state.add_environment()
        self.state.set_rbd_timestep(self.dt, int(substeps))
        self.entities: list[NexusEntity] = []
        self.robots: list[NexusRobot] = []
        self.cameras: list[NexusCamera] = []
        self.built = False
        self.segmentation_idx_dict: dict[int, Any] = {}
        self._pose_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._vel_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._synced = False
        engine.scene_count += 1
        # one render generation per scene: a later build supersedes this
        # scene's rendering (its nodes leave the shared graph and its cameras
        # are freed), physics stays
        engine.release_cameras()
        self.generation = engine.viewer.begin_scene()

    # -- construction ----------------------------------------------------

    def add_box(
        self,
        name: str,
        size,
        pos,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fixed: bool = False,
        friction: float = 0.5,
        density: float | None = None,
        color=(0.7, 0.7, 0.7, 1.0),
        label_texture: np.ndarray | None = None,
    ) -> NexusEntity:
        nx = self.nexus3d
        half = [float(s) / 2 for s in size]
        w, x, y, z = (float(v) for v in quat_wxyz)
        pose = nx.Pose.from_parts(nx.Vec3(*(float(p) for p in pos)), nx.Quat.from_xyzw(x, y, z, w))
        handles = []
        for env in range(self.n_envs):
            builder = nx.RigidBodyBuilder.fixed() if fixed else nx.RigidBodyBuilder.dynamic()
            body = builder.pose(pose).build()
            # Genesis takes the larger friction of a touching pair; Nexus
            # defaults to the average, so ask for the same rule
            collider = (
                nx.ColliderBuilder.cuboid(*half)
                .friction(float(friction))
                .friction_combine_rule(self.friction_combine_rule)
            )
            if density is not None:
                collider = collider.density(float(density))
            collider = collider.build()
            handle = self.state.insert_rigid_body_in(env, body, collider, nx.RbdCoupling.NONE)
            handles.append(handle)
            if env == 0:
                self._register_box_visual(handle, collider, size, color, label_texture, name)
        entity = NexusEntity(self, len(self.entities), name, handles, fixed)
        self.entities.append(entity)
        return entity

    def _register_box_visual(self, handle, collider, size, color, label_texture, name) -> None:
        nx = self.nexus3d
        viewer = self.engine.viewer
        rgba = [float(c) for c in color]
        if len(rgba) == 3:
            rgba.append(1.0)
        if label_texture is None:
            viewer.insert_sensor_shape(0, handle, collider.shared_shape(), nx.Pose.IDENTITY, rgba)
            return
        # T2 labels: the committed UV cube scaled to the med size, textured
        # with the label image; collision stays the cuboid
        positions, faces, uvs = load_uv_cube()
        scaled = [[p[0] * size[0], p[1] * size[1], p[2] * size[2]] for p in positions]
        mesh = nx.ColliderBuilder.trimesh(scaled, faces).build().shared_shape()
        viewer.insert_sensor_shape(
            0,
            handle,
            mesh,
            nx.Pose.IDENTITY,
            [1.0, 1.0, 1.0, rgba[3]],
            uvs=uvs,
            texture=_png_bytes(label_texture),
            texture_name=f"label-{name}",
        )

    def add_ground(self, size, friction: float) -> NexusEntity:
        ground_cfg = load_nexus_physics()["ground"]
        ground = self.add_box(
            "ground",
            size,
            (0.0, 0.0, -float(size[2]) / 2),
            fixed=True,
            friction=friction,
            color=tuple(ground_cfg["color"]),
            label_texture=checkerboard_texture(
                int(ground_cfg["checker_squares"]),
                ground_cfg["color"],
                ground_cfg["checker_color"],
            ),
        )
        if not ground_cfg.get("casts_shadows", True):
            self.engine.viewer.set_body_casts_shadows(0, ground.handles[0], False)
        return ground

    def add_urdf_robot(self, path: Path, name: str, convex_hull: bool) -> NexusRobot:
        nx = self.nexus3d
        options = nx.UrdfLoaderOptions(make_roots_fixed=True, convex_hull=convex_hull)
        dynamics = load_nexus_physics()["robot"]
        robots = []
        for env in range(self.n_envs):
            robot = self.state.load_urdf_robot(env, str(path), options)
            # URDF joints declare no armature or damping; without them the
            # GPU solver has no stable PD hold on a light arm (read at build)
            self.state.set_robot_joint_dynamics(
                robot,
                armature=[float(dynamics["default_armature"])] * robot.n_dofs,
                damping=[float(dynamics["default_damping"])] * robot.n_dofs,
            )
            if env == 0:
                for body, shape, local_pose in robot.render_shapes:
                    self.engine.viewer.insert_visual_shape(0, body, shape, local_pose)
            robots.append(robot)
        wrapper = NexusRobot(self, robots, name)
        self.robots.append(wrapper)
        return wrapper

    def add_mjcf_robot(self, path: Path, name: str) -> NexusRobot:
        robots = [
            self.state.load_mjcf_robot(
                self.engine.viewer, env, str(path), register_visuals=env == 0
            )
            for env in range(self.n_envs)
        ]
        wrapper = NexusRobot(self, robots, name)
        self.robots.append(wrapper)
        return wrapper

    def add_camera(self, res, fov: float, pos=None, lookat=None) -> NexusCamera:
        znear, zfar = self.camera_planes
        cam_id = self.engine.viewer.add_sensor_camera(
            int(res[0]), int(res[1]), float(fov), znear, zfar
        )
        self.engine.viewer.set_sensor_camera_ambient(cam_id, 1.0)
        self.engine.viewer.set_sensor_camera_ambient_color(cam_id, self.ambient)
        self.engine.viewer.set_sensor_camera_background(cam_id, self.background_rgba)
        camera = NexusCamera(self, cam_id, res, fov)
        self.engine.live_cameras.append(camera)
        if pos is not None and lookat is not None:
            camera.set_pose(pos=pos, lookat=lookat)
        self.cameras.append(camera)
        return camera

    def build(self, n_envs: int | None = None) -> None:
        if n_envs is not None and int(n_envs) != self.n_envs:
            raise ValueError("n_envs is fixed at scene construction for the nexus backend")
        nx = self.nexus3d
        viewer = self.engine.viewer
        self.state.finalize(viewer)
        self.state.set_rbd_gravity(viewer, nx.Vec3(*self.gravity))
        # link poses are only derived from the joint coordinates by the first
        # step: write them now so readbacks before any step are consistent
        for robot in self.robots:
            for per_env in robot.robots:
                self.state.set_robot_qpos(viewer, per_env, list(self.state.robot_cpu_qpos(per_env)))
        self.built = True
        self._assign_segmentation_ids()
        self.sync(force=True)
        for camera in self.cameras:
            camera._apply_attachment()
        self.sync(force=True)

    def _assign_segmentation_ids(self) -> None:
        """Genesis's default `link` segmentation level: one id per (entity,
        link); `segmentation_idx_dict` maps id -> (entity_idx, link_idx) with
        -1 for the background (TC-9). Ids start at 1: 0 is the renderer's
        background, which `render` maps to -1."""
        viewer = self.engine.viewer
        idx_dict: dict[int, Any] = {_BACKGROUND_SEG_ID: _BACKGROUND_SEG_ID}
        seg_id = 1
        for entity in self.entities:
            viewer.set_body_segmentation_id(0, entity.handles[0], seg_id)
            idx_dict[seg_id] = (entity.idx, 0)
            seg_id += 1
        for robot in self.robots:
            entity_idx = len(self.entities) + self.robots.index(robot)
            for link_idx, body in enumerate(robot.link_bodies[0]):
                viewer.set_body_segmentation_id(0, body, seg_id)
                idx_dict[seg_id] = (entity_idx, link_idx)
                seg_id += 1
        self.segmentation_idx_dict = idx_dict

    # -- stepping and readback -------------------------------------------

    def step(self) -> None:
        self.engine.pipeline.simulate(self.engine.viewer, self.state, self.timestamps)
        self.invalidate_poses()
        for robot in self.robots:
            robot.root_write_consumed()

    def perf_stats(self) -> dict:
        """Engine-side timing of the last step: `gpu_ms` is the summed GPU
        pass time from the timestamp queries (lags a frame or two behind
        the step that produced it) and `gpu_passes` the per-pass split.
        Empty until the first harvest. The bridge writes these into its
        timing sidecar so engines can be compared on the same run."""
        # harvesting happens in the viewer sync, which the render path also
        # needs; a step-only tick would otherwise never see fresh timings
        self.sync()
        stats = self.state.run_stats()
        gpu_ms = float(stats.get("gpu_total_time_ms", 0.0))
        if gpu_ms <= 0.0:
            return {}
        return {
            "gpu_ms": gpu_ms,
            "gpu_passes": {str(k): float(v) for k, v in stats.get("gpu_pass_times", {}).items()},
            "encoding_ms": float(stats.get("encoding_time_ms", 0.0)),
        }

    def activate(self) -> None:
        """Refuse to render a scene a newer build superseded: the shared
        viewer draws one scene at a time (the latest), and its render nodes
        left the graph when the next scene began. Physics readbacks still
        work on a superseded scene; only cameras do not."""
        if self.engine.viewer.active_scene() != self.generation:
            raise RuntimeError(
                "this nexus scene was superseded by a newer build_scene in this process; "
                "its cameras can no longer render (one live scene per process)"
            )

    def sync(self, force: bool = False) -> None:
        """Pull GPU poses into the render scene graph (and the attached
        cameras) once per step, before any render."""
        self.activate()
        if force or not self._synced:
            self.engine.viewer.sync(self.state, self.timestamps)
            self._synced = True

    def invalidate_poses(self) -> None:
        self._pose_cache = None
        self._vel_cache = None
        self._synced = False
        for robot in self.robots:
            robot.invalidate()

    def body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pose_cache is None:
            pos, quat = self.state.read_body_poses(self.engine.viewer)
            self._pose_cache = (
                np.asarray(pos, dtype=np.float32),
                np.asarray(quat, dtype=np.float32),
            )
        return self._pose_cache

    def body_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        if self._vel_cache is None:
            lin, ang = self.state.read_body_velocities(self.engine.viewer)
            self._vel_cache = (np.asarray(lin, dtype=np.float32), np.asarray(ang, dtype=np.float32))
        return self._vel_cache

    def gpu_index(self, env: int, handle) -> int:
        index = self.state.body_gpu_index(env, handle)
        if index is None:
            raise RuntimeError("scene not built (call build() first)")
        return int(index)

    def set_body_pose(self, env: int, handle, pos, quat_wxyz) -> None:
        self.state.set_body_pose(
            self.engine.viewer, env, handle, [float(v) for v in pos], [float(v) for v in quat_wxyz]
        )
        # a teleport writes the pose in two halves (set_pos then set_quat) and
        # each reads back the other, so dropping the cache here cost two GPU
        # readbacks per body on every reset. The written row is known: patch
        # it instead, and keep the velocity cache, which the write leaves alone
        self._synced = False
        if self._pose_cache is not None:
            row = self.gpu_index(env, handle)
            self._pose_cache[0][row] = pos
            self._pose_cache[1][row] = quat_wxyz
        for robot in self.robots:
            robot.invalidate()

    def set_body_velocity(
        self, env: int, handle, linvel=(0.0, 0.0, 0.0), angvel=(0.0, 0.0, 0.0)
    ) -> None:
        self.state.set_body_velocity(
            self.engine.viewer, env, handle, [float(v) for v in linvel], [float(v) for v in angvel]
        )
        self._vel_cache = None


# --- scene builders (mirror the frozen Genesis builders) --------------------


def _new_scene(engine: NexusEngine, physics: dict, n_envs: int, ambient) -> NexusScene:
    nexus_physics = load_nexus_physics()
    scene = NexusScene(
        engine,
        dt=physics["sim"]["dt"],
        # the engine's own substep count: the frozen value is Genesis's
        substeps=nexus_physics["sim"]["substeps"],
        gravity=physics["sim"]["gravity"],
        n_envs=n_envs,
        ambient=ambient,
        background_rgba=nexus_physics["camera"]["background_rgba"],
        camera_planes=(nexus_physics["camera"]["znear"], nexus_physics["camera"]["zfar"]),
    )
    sim = nexus_physics["sim"]
    scene.state.set_rbd_solver_params(
        contact_natural_frequency=float(sim["contact_natural_frequency"]),
        static_contact_natural_frequency=float(sim["static_contact_natural_frequency"]),
        allowed_linear_error=float(sim["allowed_linear_error"]),
        internal_pgs_iterations=int(sim["internal_pgs_iterations"]),
        friction_in_bias_pass=bool(sim.get("friction_in_bias_pass", False)),
    )
    scene.state.set_rbd_implicit_coriolis(engine.viewer, bool(sim.get("implicit_coriolis", True)))
    scene.state.set_rbd_substep_refresh(
        engine.viewer,
        bool(sim.get("substep_refresh", True)),
        bool(sim.get("substep_refresh_light", False)),
    )
    # the viewer reapplies its own setting to the state at every sync
    deterministic = bool(sim.get("deterministic", False))
    engine.viewer.set_deterministic(deterministic)
    scene.state.set_deterministic(engine.viewer, deterministic)
    return scene


def _add_robot(scene: NexusScene, embodiment: str, physics: dict) -> NexusRobot:
    nexus_physics = load_nexus_physics()
    if embodiment in ("franka", "mobile"):
        return scene.add_mjcf_robot(franka_mjcf_path(), "franka")
    if not SO101_URDF.exists():
        raise FileNotFoundError(f"so101 asset missing: {SO101_URDF} (acquisition pending, ADR-6)")
    return scene.add_urdf_robot(SO101_URDF, embodiment, bool(nexus_physics["robot"]["convex_hull"]))


def _apply_home_and_gains(robot: NexusRobot, profile: dict, n_envs: int) -> None:
    """The frozen builder's post-build robot setup, verbatim in behavior:
    home pose in TC-5 wire order mapped by joint name, then the profile's
    gripper gains."""
    wire_dof_indices = profile_dof_indices(robot, profile)
    if "home_qpos" in profile:
        home = np.asarray(profile["home_qpos"], dtype=np.float32)
        if wire_dof_indices is not None:
            native_home = np.empty(robot.n_dofs, dtype=np.float32)
            native_home[list(wire_dof_indices)] = home
            home = native_home
        robot.set_qpos(home if n_envs == 1 else np.tile(home, (n_envs, 1)))
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


def build_scene(
    seed: int,
    embodiment: str = "franka",
    n_envs: int = 1,
    headless: bool = True,
    cfg: SceneCfg | None = None,
    sim_backend: str | None = None,
) -> SceneHandle:
    """Nexus twin of `aisle.scenes.pharmacy.build_scene` (SPEC 020): the same
    seeded layout, DR draws, colors, labels and cameras, realized on Nexus.
    Every scene-deciding value comes from the frozen module's pure functions."""
    cfg = cfg or SceneCfg()
    engine = _ensure_nexus(sim_backend, headless=headless)
    physics = load_physics()
    nexus_physics = load_nexus_physics()
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

    scene = _new_scene(engine, physics, n_envs, ambient)
    ground = nexus_physics["ground"]
    scene.add_ground(ground["size"], ground["friction"])

    shelf_friction = physics["materials"]["shelf"]["friction"]
    width = shelf["level_size"][1]
    for level, (level_height, level_depth) in enumerate(
        zip(shelf["level_heights"], shelf["level_depths"], strict=True)
    ):
        x_min, x_max = level_x_span(shelf, level)
        scene.add_box(
            f"shelf_{level}",
            (level_depth, width, shelf["board_thickness"]),
            ((x_min + x_max) / 2, shelf["pos"][1], shelf["pos"][2] + level_height),
            fixed=True,
            friction=shelf_friction,
        )
    tray = scene.add_box(
        "tray",
        tuple(tray_cfg["size"]),
        tuple(tray_cfg["pos"]),
        fixed=True,
        friction=physics["materials"]["tray"]["friction"],
    )

    robot = _add_robot(scene, embodiment, physics)

    box_physics = physics["materials"]["box"]
    applied_frictions: dict[str, float] = {}
    applied_colors: dict[str, list[float]] = {}
    boxes: dict[str, Any] = {}
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
        label_texture = None
        if cfg.labels:
            label_texture = label_texture_image(
                str(meds[placement.name].get("label", placement.name.upper())), color
            )
        boxes[placement.name] = scene.add_box(
            placement.name,
            tuple(meds[placement.name]["size"]),
            (placement.x, placement.y, placement.z),
            friction=friction,
            density=box_physics["density_kg_m3"],
            color=tuple(color),
            label_texture=label_texture,
        )

    cam_cfg = physics["cameras"]
    overhead_pos = list(cam_cfg["overhead_pos"])
    if cfg.camera_jitter.enabled:
        jitter = dr_cfg["camera_jitter_m"]
        overhead_pos = [p + (camera_rng.random() - 0.5) * jitter for p in overhead_pos]
    cams = {
        "overhead": scene.add_camera(
            (640, 480), 55, pos=tuple(overhead_pos), lookat=tuple(cam_cfg["overhead_lookat"])
        ),
        "wrist": scene.add_camera((320, 240), 70),
    }

    scene.build()

    profile = physics["embodiment"][embodiment]
    _apply_home_and_gains(robot, profile, n_envs)

    ee_link = robot.get_link(profile.get("ee_link", FRANKA_EE_LINK))
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
    _assert_reachable(handle, ee_link, layout, n_envs)
    return handle


def build_store(
    seed: int,
    scenario: str,
    embodiment: str = "mobile",
    n_envs: int = 1,
    headless: bool = True,
    sim_backend: str | None = None,
):
    """Nexus twin of `aisle.scenes.store.build_store` (SPEC 200 RS-2/RS-3)."""
    from aisle.scenes.store import (
        StoreHandle,
        episode_layout,
        full_stock,
        generate_episode,
        load_planogram,
        yaw_quat_wxyz,
    )

    engine = _ensure_nexus(sim_backend, headless=headless)
    meds = load_meds()
    physics = load_physics()
    nexus_physics = load_nexus_physics()
    plano = load_planogram()
    episode = generate_episode(seed, scenario)
    store, geo = plano["store"], plano["store"]["unit_geometry"]
    profile = physics["embodiment"][embodiment]

    scene = _new_scene(
        engine, physics, n_envs, (physics["domain_randomization"]["ambient_default"],) * 3
    )
    ground = nexus_physics["ground"]
    scene.add_ground(ground["size"], ground["friction"])

    shelf_friction = physics["materials"]["shelf"]["friction"]
    for unit_id, unit in plano["units"].items():
        for level, level_height in enumerate(geo["level_heights"]):
            scene.add_box(
                f"{unit_id}_level_{level}",
                (geo["depth"], geo["width"], geo["board_thickness"]),
                (unit["pos"][0], unit["pos"][1], level_height),
                quat_wxyz=yaw_quat_wxyz(unit["yaw"]),
                fixed=True,
                friction=shelf_friction,
            )
    tray_friction = physics["materials"]["tray"]["friction"]
    counter = scene.add_box(
        "counter",
        tuple(store["counter_size"]),
        tuple(store["counter_pos"]),
        fixed=True,
        friction=tray_friction,
    )
    bin_entity = scene.add_box(
        "bin", tuple(store["bin_size"]), tuple(store["bin_pos"]), fixed=True, friction=tray_friction
    )

    robot = scene.add_mjcf_robot(franka_mjcf_path(), "franka")

    box_physics = physics["materials"]["box"]
    items: dict[str, Any] = {}
    categories: dict[str, str] = {}
    layout = episode_layout(plano, episode, meds)
    for item in full_stock(plano):
        x, y, z, yaw = layout[item.item_id]
        categories[item.item_id] = item.category
        items[item.item_id] = scene.add_box(
            item.item_id,
            tuple(meds[item.category]["size"]),
            (x, y, z),
            quat_wxyz=yaw_quat_wxyz(yaw),
            friction=box_physics["friction"],
            density=box_physics["density_kg_m3"],
            color=tuple(meds[item.category]["color"]),
        )

    cam_cfg = physics["cameras"]
    cams = {
        "overhead": scene.add_camera(
            (640, 480),
            70,
            pos=tuple(cam_cfg["store_overhead_pos"]),
            lookat=tuple(cam_cfg["store_overhead_lookat"]),
        ),
        "wrist": scene.add_camera((320, 240), 70),
    }

    scene.build()

    home = np.asarray(profile["home_qpos"], dtype=np.float32)
    robot.set_qpos(home if n_envs == 1 else np.tile(home, (n_envs, 1)))
    count = int(profile["gripper_dofs"])
    finger_dofs = list(range(robot.n_dofs - count, robot.n_dofs))
    robot.set_dofs_kp(
        np.asarray(profile["gripper_kp"], dtype=np.float32), dofs_idx_local=finger_dofs
    )
    robot.set_dofs_kv(
        np.asarray(profile["gripper_kv"], dtype=np.float32), dofs_idx_local=finger_dofs
    )

    ee_link = robot.get_link("hand")
    offset = np.eye(4, dtype=np.float32)
    offset[:3, 3] = cam_cfg["wrist_offset_m"]
    cams["wrist"].attach(ee_link, offset_T=offset)

    return StoreHandle(
        scene=scene,
        robot=robot,
        items=items,
        categories=categories,
        counter=counter,
        bin=bin_entity,
        cams=cams,
        planogram=plano,
        episode=episode,
        embodiment=embodiment,
        seed=seed,
        scenario=scenario,
        med_sizes={name: list(meds[name]["size"]) for name in meds},
    )


__all__ = [
    "NexusCamera",
    "NexusEngine",
    "NexusEntity",
    "NexusJoint",
    "NexusLink",
    "NexusRobot",
    "NexusScene",
    "build_scene",
    "build_store",
    "franka_mjcf_path",
    "load_nexus_physics",
    "load_uv_cube",
    "matrix_to_quat_wxyz",
    "pos_lookat_up_to_transform",
    "quat_wxyz_to_matrix",
]
