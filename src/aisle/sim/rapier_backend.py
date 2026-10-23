"""rapier engine backend (ADR-68): the frozen scenes, stepped by rapier on the CPU.

`build_scene` / `build_store` here are the rapier counterparts of the frozen
`aisle.scenes.pharmacy.build_scene` / `aisle.scenes.store.build_store`, and of
their Nexus twins in `nexus_backend`. They call the SAME frozen pure functions
for everything that decides the scene (layout resolution, seeded placements,
occlusion, colors, label textures, camera mounts) and only replace the object
construction, behind the same duck-typed surface the bridge already uses on
Genesis objects.

The split that makes this engine possible: rapier owns the physics, the Nexus
viewer owns the pixels. Every scene is built twice, into one
`rapier3d.PhysicsWorld` per environment and into one Nexus scene that is built
exactly as the Nexus backend builds it and never stepped. A step advances
rapier; the resulting poses are written into the Nexus state just before the
viewer syncs, so RGB, metric depth and segmentation follow externally computed
poses. Two consequences are worth naming: a rapier run and a Nexus run share
the identical renderer, camera model and segmentation ids, and this backend
needs both wheels.

The render mirror is reused, not forked: `RapierScene` composes a `NexusScene`
and calls its `add_box`, `add_ground`, `add_mjcf_robot`, `add_urdf_robot`,
`add_camera` and `build`, so textures, shadows, cameras and the segmentation
numbering are literally the Nexus backend's.

`rapier3d` and `nexus3d` are imported lazily (CON-12): importing this module is
sim-free.
"""

from __future__ import annotations

import math
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aisle.embodiment import profile_dof_indices
from aisle.scenes.pharmacy import (
    FRANKA_EE_LINK,
    SO101_URDF,
    SceneCfg,
    SceneHandle,
    _assert_reachable,
    apply_occlusion,
    label_texture_image,
    level_x_span,
    load_meds,
    load_physics,
    resolve_layout,
    sample_placements,
    wrist_mount_transform,
)
from aisle.sim.nexus_backend import (
    NexusScene,
    _broadcast_rows,
    _ensure_nexus,
    _per_dof,
    _squeeze_envs,
    franka_mjcf_path,
    quat_wxyz_to_matrix,
)

_SIM_DIR = Path(__file__).parent


def load_rapier_physics() -> dict:
    with open(_SIM_DIR / "rapier_physics.toml", "rb") as f:
        return tomllib.load(f)


# --- small conversions between rapier types and the wire's numpy arrays -----


def _xyz(vec) -> np.ndarray:
    return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)


def _wxyz(rot) -> np.ndarray:
    return np.array([float(rot.w), float(rot.i), float(rot.j), float(rot.k)], dtype=np.float64)


def _quat_xyzw(quat_wxyz) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return (x, y, z, w)


def _combine_rule(name: str):
    """The rapier enum for a friction-combine rule named in the toml."""
    import rapier3d as rp

    try:
        return getattr(rp.CoefficientCombineRule, str(name).upper())
    except AttributeError as exc:
        raise ValueError(f"unknown friction_combine_rule {name!r}") from exc


def _isometry(pos, quat_wxyz) -> tuple:
    """rapier accepts a (translation, rotation) pair wherever it wants a pose."""
    import rapier3d as rp

    return (
        rp.Vec3(float(pos[0]), float(pos[1]), float(pos[2])),
        rp.Quaternion.from_tuple(_quat_xyzw(quat_wxyz)),
    )


# --- the render mirror ------------------------------------------------------


class RapierRenderMirror(NexusScene):
    """The Nexus scene a rapier scene renders through: built by the Nexus
    backend's own constructors, never stepped. `sync` is the one hook the
    rapier side needs, because every render path goes through it."""

    physics: Any = None

    def step(self) -> None:
        raise RuntimeError("the rapier backend never steps its render mirror (ADR-68)")

    def sync(self, force: bool = False) -> None:
        if self.physics is not None:
            self.physics.flush_render()
        super().sync(force=force)


# --- scene objects (the Genesis duck-typed surface) -------------------------


class RapierEntity:
    """A free rigid body per environment (a med box, a board, a tray), with
    its render twin in the Nexus mirror."""

    def __init__(self, scene: RapierScene, name: str, handles: list, mirror, fixed: bool):
        self.scene = scene
        self.name = name
        self.handles = handles  # one rapier RigidBodyHandle per env
        self.mirror = mirror  # the NexusEntity that draws it
        self.fixed = fixed
        self.idx = mirror.idx

    def _body(self, env: int):
        return self.scene.worlds[env].rigid_bodies.get(self.handles[env])

    def get_pos(self) -> np.ndarray:
        rows = [_xyz(self._body(env).translation) for env in range(self.scene.n_envs)]
        return _squeeze_envs(np.stack(rows), self.scene.n_envs)

    def get_quat(self) -> np.ndarray:
        rows = [_wxyz(self._body(env).rotation) for env in range(self.scene.n_envs)]
        return _squeeze_envs(np.stack(rows), self.scene.n_envs)

    def get_dofs_velocity(self) -> np.ndarray:
        rows = []
        for env in range(self.scene.n_envs):
            body = self._body(env)
            rows.append(np.concatenate([_xyz(body.linvel), _xyz(body.angvel)]))
        return _squeeze_envs(np.stack(rows), self.scene.n_envs)

    def _write(self, env: int, pos, quat_wxyz) -> None:
        import rapier3d as rp

        body = self._body(env)
        body.translation = (float(pos[0]), float(pos[1]), float(pos[2]))
        body.rotation = rp.Quaternion.from_tuple(_quat_xyzw(quat_wxyz))
        body.wake_up()
        self.scene.render_dirty()

    def set_pos(self, pos, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(pos, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            self._write(env, row[:3], _wxyz(self._body(env).rotation))

    def set_quat(self, quat, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(quat, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            self._write(env, _xyz(self._body(env).translation), row[:4])

    def zero_all_dofs_velocity(self, envs_idx=None) -> None:
        envs = range(self.scene.n_envs) if envs_idx is None else envs_idx
        for env in envs:
            body = self._body(int(env))
            body.linvel = (0.0, 0.0, 0.0)
            body.angvel = (0.0, 0.0, 0.0)


@dataclass
class RapierJoint:
    name: str
    n_dofs: int
    dofs_idx_local: list[int]


class RapierLink:
    """One robot link; poses come from the link's rapier rigid body."""

    def __init__(self, robot: RapierRobot, index: int, name: str):
        self.robot = robot
        self.index = index
        self.name = name

    def get_pos(self) -> np.ndarray:
        return _squeeze_envs(self.robot._link_poses()[0][:, self.index], self.robot.scene.n_envs)

    def get_quat(self) -> np.ndarray:
        return _squeeze_envs(self.robot._link_poses()[1][:, self.index], self.robot.scene.n_envs)


@dataclass
class RapierArticulation:
    """One environment's copy of a robot: the world it lives in, a handle that
    resolves to its multibody, and its link bodies in link order."""

    world: Any
    joint_handle: Any
    link_bodies: list

    def multibody(self):
        return self.world.multibody_joints.multibody(self.joint_handle)

    def link_joint_handle(self, link_index: int):
        """The multibody-joint handle naming one link, which is what rapier's
        IK addresses. A multibody joint is keyed by its child link's rigid
        body, so the handle is that body's raw parts."""
        import rapier3d as rp

        body = self.link_bodies[link_index]
        return rp.MultibodyJointHandle.from_raw_parts(body.index, body.generation)


class RapierRobot:
    """One articulated robot as a rapier reduced-coordinate multibody, one per
    environment, mirrored by a Nexus robot that only ever draws it."""

    def __init__(
        self,
        scene: RapierScene,
        arts: list[RapierArticulation],
        mirror,
        name: str,
        joint_links: list[tuple[str, int]],
    ):
        self.scene = scene
        self.arts = arts
        self.mirror = mirror
        self.name = name
        first = arts[0].multibody()
        self.n_dofs = int(first.ndofs)
        self.n_qs = self.n_dofs
        # The renderer is driven by qpos, so the two models must agree on the
        # DoF and link layout. They come from the same file, but a loader
        # disagreeing silently would corrupt every frame.
        link_names = list(mirror.robots[0].link_names)
        if len(link_names) != first.num_links:
            raise RuntimeError(
                f"rapier loaded {first.num_links} links for {name!r} but nexus loaded "
                f"{len(link_names)}; the two loaders disagree on the model"
            )
        if int(mirror.n_dofs) != self.n_dofs:
            raise RuntimeError(
                f"rapier loaded {self.n_dofs} DoFs for {name!r} but nexus loaded {mirror.n_dofs}"
            )
        self.links = [RapierLink(self, i, n) for i, n in enumerate(link_names)]
        self.joints = self._build_joints(first, joint_links)
        self.dof_axes = self._build_dof_axes(first)
        # Gains from the model. rapier's MJCF binding reports `kp`/`kv` as None
        # for MuJoCo "general" actuators (the Franka's kind), so they are read
        # off the Nexus loader, which resolves them from the same file: both
        # engines then run the identical servo.
        self.kp = np.asarray(mirror.robots[0].kp, dtype=np.float64).copy()
        self.kv = np.asarray(mirror.robots[0].kv, dtype=np.float64).copy()
        self._motorized = False
        self._targets = np.zeros((scene.n_envs, self.n_dofs), dtype=np.float64)

    # -- structure -------------------------------------------------------

    def _build_joints(self, multibody, joint_links) -> list[RapierJoint]:
        """The model's joints in the order the loader reports them, each
        carrying its DoF indices into the generalized vectors. A multibody
        link's `assembly_id` is that index, so the joint that moves a link
        owns the DoFs at its child link's assembly id."""
        joints = []
        for name, link_id in joint_links:
            link = multibody.get_link(link_id)
            n_dofs = int(link.ndofs)
            offset = int(link.assembly_id)
            joints.append(
                RapierJoint(
                    name=name, n_dofs=n_dofs, dofs_idx_local=list(range(offset, offset + n_dofs))
                )
            )
        return joints

    def _build_dof_axes(self, multibody) -> list[tuple[int, Any]]:
        """Per DoF, the link whose joint owns it and that joint's free axis.

        rapier builds a 1-DoF joint with its free axis first in the local
        frames, so a revolute joint moves along ANG_X and a prismatic one
        along LIN_X. Which of the two a joint is is not exposed by the MJCF
        binding, so it is read off the body jacobian: the column of a
        revolute DoF is a unit angular velocity, a prismatic one's is a unit
        linear velocity.
        """
        import rapier3d as rp

        axes: list[tuple[int, Any]] = [(-1, None)] * self.n_dofs
        for link_id in range(multibody.num_links):
            link = multibody.get_link(link_id)
            if link.ndofs == 0:
                continue
            jacobian = np.asarray(multibody.body_jacobian(link_id), dtype=np.float64)
            for k in range(int(link.ndofs)):
                dof = int(link.assembly_id) + k
                angular = float(np.linalg.norm(jacobian[3:6, dof]))
                axes[dof] = (link_id, rp.JointAxis.ANG_X if angular > 0.5 else rp.JointAxis.LIN_X)
        missing = [d for d, (link_id, _) in enumerate(axes) if link_id < 0]
        if missing:
            raise RuntimeError(f"robot {self.name!r} has DoFs owned by no link: {missing}")
        return axes

    def get_joint(self, name: str) -> RapierJoint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(f"robot {self.name!r} has no joint {name!r}")

    def get_link(self, name: str) -> RapierLink:
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(f"robot {self.name!r} has no link {name!r}")

    def get_dofs_limit(self) -> tuple[np.ndarray, np.ndarray]:
        """rapier exposes no per-DoF limit getter on a multibody link, so the
        limits come from the Nexus mirror's read of the same model."""
        first = self.mirror.robots[0]
        return (
            np.asarray(first.dof_lower, dtype=np.float32),
            np.asarray(first.dof_upper, dtype=np.float32),
        )

    @property
    def link_bodies(self) -> list[list]:
        """The render bodies, for a sensor camera to attach to (SCN-5)."""
        return self.mirror.link_bodies

    # -- state -----------------------------------------------------------

    def _link_poses(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.empty((self.scene.n_envs, len(self.links), 3), dtype=np.float64)
        quat = np.empty((self.scene.n_envs, len(self.links), 4), dtype=np.float64)
        for env, art in enumerate(self.arts):
            bodies = art.world.rigid_bodies
            for index, handle in enumerate(art.link_bodies):
                body = bodies.get(handle)
                pos[env, index] = _xyz(body.translation)
                quat[env, index] = _wxyz(body.rotation)
        return pos, quat

    def get_qpos(self) -> np.ndarray:
        rows = [np.asarray(art.multibody().generalized_position()) for art in self.arts]
        return _squeeze_envs(np.stack(rows), self.scene.n_envs)

    def get_dofs_velocity(self) -> np.ndarray:
        rows = [np.asarray(art.multibody().generalized_velocity()) for art in self.arts]
        return _squeeze_envs(np.stack(rows), self.scene.n_envs)

    def _write_qpos(self, env: int, qpos) -> None:
        """Drive the generalized coordinates to `qpos` and refresh the link
        bodies. `apply_displacements` is the only writer rapier exposes, so a
        coordinate is set by displacing it from where it is."""
        art = self.arts[env]
        multibody = art.multibody()
        current = multibody.generalized_position()
        multibody.apply_displacements([float(q) - c for q, c in zip(qpos, current, strict=True)])
        multibody.forward_kinematics(art.world.rigid_bodies, False)
        multibody.update_rigid_bodies(art.world.rigid_bodies, False)
        self._wake(env)

    def _wake(self, env: int) -> None:
        """A teleported arm must integrate on the next step: one rapier put to
        sleep after settling would otherwise stay at rest (TC-6 reset)."""
        bodies = self.arts[env].world.rigid_bodies
        for handle in self.arts[env].link_bodies:
            bodies.get(handle).wake_up()

    def set_qpos(self, qpos, envs_idx=None) -> None:
        """Genesis's `set_qpos`, with the Nexus backend's rest semantics: the
        write also brings the arm to rest, which is what the bridge's reset
        path (TC-6) expects on every engine."""
        rows, envs = _broadcast_rows(qpos, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            self._write_qpos(env, row[: self.n_dofs])
            self._rest(env)
        self.scene.render_dirty()

    def _rest(self, env: int) -> None:
        art = self.arts[env]
        multibody = art.multibody()
        multibody.set_generalized_velocity([0.0] * self.n_dofs)
        bodies = art.world.rigid_bodies
        for handle in art.link_bodies:
            body = bodies.get(handle)
            body.linvel = (0.0, 0.0, 0.0)
            body.angvel = (0.0, 0.0, 0.0)

    def zero_all_dofs_velocity(self, envs_idx=None) -> None:
        envs = range(self.scene.n_envs) if envs_idx is None else [int(e) for e in envs_idx]
        for env in envs:
            self._rest(env)
        self.scene.render_dirty()

    def get_pos(self) -> np.ndarray:
        return self.links[0].get_pos()

    def get_quat(self) -> np.ndarray:
        return self.links[0].get_quat()

    def _write_root(self, env: int, pos, quat_wxyz) -> None:
        """Re-base the (fixed-root) arm: write the root body's pose, then let
        the multibody re-read it and carry every link along (SPEC 210)."""
        import rapier3d as rp

        art = self.arts[env]
        root = art.world.rigid_bodies.get(art.link_bodies[0])
        root.translation = (float(pos[0]), float(pos[1]), float(pos[2]))
        root.rotation = rp.Quaternion.from_tuple(_quat_xyzw(quat_wxyz))
        multibody = art.multibody()
        multibody.forward_kinematics(art.world.rigid_bodies, True)
        multibody.update_rigid_bodies(art.world.rigid_bodies, False)
        self._wake(env)
        self.scene.render_dirty()

    def set_pos(self, pos, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(pos, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            root = self.arts[env].world.rigid_bodies.get(self.arts[env].link_bodies[0])
            self._write_root(env, row[:3], _wxyz(root.rotation))

    def set_quat(self, quat, envs_idx=None) -> None:
        rows, envs = _broadcast_rows(quat, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            root = self.arts[env].world.rigid_bodies.get(self.arts[env].link_bodies[0])
            self._write_root(env, _xyz(root.translation), row[:4])

    # -- control ---------------------------------------------------------

    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None) -> None:
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        self.kp[dofs] = _per_dof(kp, dofs, "kp")
        if self._motorized:
            self._push_motors(dofs)

    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None) -> None:
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        self.kv[dofs] = _per_dof(kv, dofs, "kv")
        if self._motorized:
            self._push_motors(dofs)

    def control_dofs_position(self, target, dofs_idx_local=None, envs_idx=None) -> None:
        """Genesis position control: one PD servo per commanded DoF. The
        motors are installed on the first command and not before, so an
        uncommanded arm sags under gravity exactly as it does on the other
        two engines."""
        dofs = list(range(self.n_dofs)) if dofs_idx_local is None else list(dofs_idx_local)
        rows, envs = _broadcast_rows(target, self.scene.n_envs, envs_idx)
        for row, env in zip(rows, envs, strict=True):
            self._targets[env, dofs] = row[: len(dofs)]
        self._motorized = True
        self._push_motors(dofs, envs)

    def _push_motors(self, dofs, envs=None) -> None:
        import rapier3d as rp

        envs = range(self.scene.n_envs) if envs is None else envs
        for env in envs:
            multibody = self.arts[int(env)].multibody()
            for dof in dofs:
                link_id, axis = self.dof_axes[dof]
                multibody.set_link_motor(
                    link_id,
                    axis,
                    float(self._targets[int(env), dof]),
                    0.0,
                    float(self.kp[dof]),
                    float(self.kv[dof]),
                )
                multibody.set_link_motor_model(link_id, axis, rp.MotorModel.FORCE_BASED)

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
        """Genesis-compatible IK on environment 0's multibody.

        rapier's `inverse_kinematics_for_link` returns the displacement that
        drives a link to a world pose, over a chosen subset of DoFs and a
        chosen set of constrained axes. Genesis's `rot_mask[k]` instead asks
        to align the link's k-th axis with the target's, so a single aligned
        tool axis is mapped to the world axis it points along most, as the
        Nexus backend does. The solve runs on the live multibody and restores
        the configuration it found afterwards.
        """
        import rapier3d as rp

        link_name = link.name if isinstance(link, RapierLink) else str(link)
        link_index = self.get_link(link_name).index
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)[0]
        quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)[0]
        mask = [True, True, True] if rot_mask is None else [bool(m) for m in rot_mask]
        target_rot = quat_wxyz_to_matrix(quat)
        constrained = [True] * 6
        if not all(mask):
            free = [k for k, m in enumerate(mask) if not m]
            if len(free) == 2:
                aligned = mask.index(True)
                axis = target_rot[:, aligned]
                constrained[3 + int(np.argmax(np.abs(axis)))] = False
            else:
                for k in free:
                    constrained[3 + k] = False
        axes_mask = rp.JointAxesMask.empty()
        for bit, on in zip(
            (
                rp.JointAxesMask.LIN_X,
                rp.JointAxesMask.LIN_Y,
                rp.JointAxesMask.LIN_Z,
                rp.JointAxesMask.ANG_X,
                rp.JointAxesMask.ANG_Y,
                rp.JointAxesMask.ANG_Z,
            ),
            constrained,
            strict=True,
        ):
            if on:
                axes_mask = axes_mask | bit

        point = np.zeros(3) if local_point is None else np.asarray(local_point, dtype=np.float64)
        # rapier drives the link origin; Genesis drives a point in the link
        # frame, so the target is shifted back along the target rotation
        target_pos = pos - target_rot @ point

        art = self.arts[0]
        saved = list(art.multibody().generalized_position())
        if init_qpos is not None:
            init = np.asarray(init_qpos, dtype=np.float64).reshape(-1, self.n_dofs)[0]
            self._write_qpos(0, init)
        joint_can_move = None
        if dofs_idx_local is not None:
            allowed = {int(d) for d in dofs_idx_local}

            def joint_can_move(link) -> bool:
                # rapier frees whole joints, so a link moves only when all its DoFs may
                start = link.assembly_id
                return all(i in allowed for i in range(start, start + link.ndofs))

        ik_cfg = self.scene.cfg["ik"]
        option = rp.InverseKinematicsOption(
            max_iters=int(max_solver_iters),
            constrained_axes=axes_mask,
            damping=float(ik_cfg["damping"]),
            epsilon_linear=float(ik_cfg["epsilon_linear"]),
            epsilon_angular=float(ik_cfg["epsilon_angular"]),
        )
        displacement = art.world.multibody_joints.inverse_kinematics_for_link(
            art.world.rigid_bodies,
            art.link_joint_handle(link_index),
            _isometry(target_pos, quat),
            option,
            joint_can_move=joint_can_move,
        )
        multibody = art.multibody()
        multibody.apply_displacements([float(v) for v in displacement])
        multibody.forward_kinematics(art.world.rigid_bodies, False)
        multibody.update_rigid_bodies(art.world.rigid_bodies, False)
        qpos = np.asarray(multibody.generalized_position(), dtype=np.float32)

        error = None
        if return_error:
            body = art.world.rigid_bodies.get(art.link_bodies[link_index])
            current_rot = quat_wxyz_to_matrix(_wxyz(body.rotation))
            realized = _xyz(body.translation) + current_rot @ point
            rot_error = np.zeros(3)
            for k, on in enumerate(mask):
                if on:
                    cosine = float(np.clip(np.dot(current_rot[:, k], target_rot[:, k]), -1.0, 1.0))
                    rot_error[k] = math.acos(cosine)
            error = np.concatenate([pos - realized, rot_error]).astype(np.float32)

        self._write_qpos(0, saved)
        self.scene.render_dirty()
        batched = self.scene.n_envs > 1
        if return_error:
            if batched:
                return np.tile(qpos, (self.scene.n_envs, 1)), np.tile(error, (self.scene.n_envs, 1))
            return qpos, error
        return np.tile(qpos, (self.scene.n_envs, 1)) if batched else qpos


# --- the scene --------------------------------------------------------------


class RapierScene:
    """The rapier counterpart of a Genesis `Scene`: one `PhysicsWorld` per
    environment, and one Nexus scene that draws environment 0."""

    def __init__(self, engine, physics: dict, n_envs: int, ambient) -> None:
        import rapier3d as rp

        rapier_physics = load_rapier_physics()
        self.rp = rp
        self.n_envs = int(n_envs)
        self.dt = float(physics["sim"]["dt"])
        self.gravity = [float(g) for g in physics["sim"]["gravity"]]
        self.cfg = rapier_physics
        camera = rapier_physics["camera"]
        self.render = RapierRenderMirror(
            engine,
            dt=self.dt,
            substeps=1,
            gravity=self.gravity,
            n_envs=self.n_envs,
            ambient=ambient,
            background_rgba=camera["background_rgba"],
            camera_planes=(camera["znear"], camera["zfar"]),
        )
        self.render.physics = self
        self.worlds = [self._new_world() for _ in range(self.n_envs)]
        self.entities: list[RapierEntity] = []
        self.robots: list[RapierRobot] = []
        self.friction_combine_rule = _combine_rule(rapier_physics["sim"]["friction_combine_rule"])
        self._dynamic_count = 0
        self._render_dirty = True

    def _new_world(self):
        rp, sim = self.rp, self.cfg["sim"]
        world = rp.PhysicsWorld(gravity=tuple(self.gravity))
        # Single-threaded by construction: the island solver's work split is
        # rapier's only run-to-run ordering variation, and CON-5 wants the
        # reproducible answer more than the parallel one.
        world.set_num_threads(int(sim["num_threads"]))
        params = world.integration_parameters
        params.dt = self.dt
        params.length_unit = float(sim["length_unit"])
        params.num_solver_iterations = int(sim["num_solver_iterations"])
        params.num_internal_pgs_iterations = int(sim["num_internal_pgs_iterations"])
        params.warmstart_coefficient = float(sim["warmstart_coefficient"])
        params.normalized_allowed_linear_error = float(sim["allowed_linear_error"]) / float(
            sim["length_unit"]
        )
        params.contact_softness = rp.SpringCoefficients(
            float(sim["contact_natural_frequency"]), float(sim["contact_damping_ratio"])
        )
        params.static_contact_softness = rp.SpringCoefficients(
            float(sim["static_contact_natural_frequency"]), float(sim["contact_damping_ratio"])
        )
        world.physics_pipeline.counters.enable()
        return world

    @property
    def segmentation_idx_dict(self) -> dict:
        return self.render.segmentation_idx_dict

    @property
    def built(self) -> bool:
        return self.render.built

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
    ) -> RapierEntity:
        rp = self.rp
        mirror = self.render.add_box(
            name,
            size,
            pos,
            quat_wxyz=quat_wxyz,
            fixed=fixed,
            friction=friction,
            density=density,
            color=color,
            label_texture=label_texture,
        )
        half = [float(s) / 2 for s in size]
        body_type = rp.RigidBodyType.FIXED if fixed else rp.RigidBodyType.DYNAMIC
        handles = []
        for world in self.worlds:
            collider = (
                rp.Collider.cuboid(*half)
                .friction(float(friction))
                .friction_combine_rule(self.friction_combine_rule)
            )
            if density is not None:
                collider = collider.density(float(density))
            builder = rp.RigidBodyBuilder(body_type).position(_isometry(pos, quat_wxyz))
            handles.append(world.add_body(builder, [collider]))
        if not fixed:
            self._dynamic_count += 1
        entity = RapierEntity(self, name, handles, mirror, fixed)
        self.entities.append(entity)
        return entity

    def add_ground(self, size, friction: float) -> RapierEntity:
        """The floor slab. Its checkerboard texture and shadow flag are the
        Nexus backend's own, so the mirror is built by `NexusScene.add_ground`
        and only the collision slab is added here."""
        mirror = self.render.add_ground(size, friction)
        pos = (0.0, 0.0, -float(size[2]) / 2)
        handles = [self._add_fixed_slab(world, size, pos, friction) for world in self.worlds]
        entity = RapierEntity(self, "ground", handles, mirror, True)
        self.entities.append(entity)
        return entity

    def _add_fixed_slab(self, world, size, pos, friction: float):
        rp = self.rp
        collider = (
            rp.Collider.cuboid(*[float(s) / 2 for s in size])
            .friction(float(friction))
            .friction_combine_rule(self.friction_combine_rule)
        )
        builder = rp.RigidBodyBuilder(rp.RigidBodyType.FIXED).position(
            _isometry(pos, (1.0, 0.0, 0.0, 0.0))
        )
        return world.add_body(builder, [collider])

    def _warm_up(self, world) -> None:
        """A multibody is built with a 6-DoF free root that the engine
        collapses only during the first step, so `ndofs` reads six too many
        until one step has run. Take it under zero gravity, with only fixed
        scenery in the world, so nothing else moves."""
        if self._dynamic_count:
            raise RuntimeError(
                "a rapier robot must be inserted before any dynamic body: its warm-up step "
                f"would advance {self._dynamic_count} free bodies"
            )
        gravity = world.gravity
        world.gravity = (0.0, 0.0, 0.0)
        world.step()
        world.gravity = gravity

    def _articulate(self, world, joint_handles, link_bodies) -> RapierArticulation:
        handle = next(h for h in joint_handles if h is not None)
        return RapierArticulation(world=world, joint_handle=handle, link_bodies=link_bodies)

    def add_mjcf_robot(self, path: Path, name: str) -> RapierRobot:
        from rapier3d.loaders import mjcf

        mirror = self.render.add_mjcf_robot(path, name)
        arts, joint_links = [], None
        for world in self.worlds:
            options = mjcf.MjcfLoaderOptions()
            # the Franka's base is welded to the world in MuJoCo; a dynamic
            # root would insert a 6-DoF free joint ahead of joint1
            options.make_roots_fixed = True
            robot, _model = mjcf.MjcfRobot.from_file(str(path), options)
            joint_names = list(robot.joint_names)
            handles = robot.insert_using_multibody_joints(
                world.rigid_bodies, world.colliders, world.multibody_joints, world.impulse_joints
            )
            self._warm_up(world)
            # body_names[0] is the MJCF worldbody, which is not a link
            link_bodies = [h.body for h in handles.bodies[1:]]
            if joint_links is None:
                joint_links = self._name_links(joint_names, handles.joints, link_bodies)
            arts.append(self._articulate(world, [j.joint for j in handles.joints], link_bodies))
        robot = RapierRobot(self, arts, mirror, name, joint_links)
        self.robots.append(robot)
        return robot

    def add_urdf_robot(self, path: Path, name: str, convex_hull: bool) -> RapierRobot:
        from rapier3d.loaders import urdf

        mirror = self.render.add_urdf_robot(path, name, convex_hull)
        dynamics = self.cfg["robot"]
        arts, joint_links = [], None
        for world in self.worlds:
            options = urdf.UrdfLoaderOptions()
            options.make_roots_fixed = True
            robot, description = urdf.UrdfRobot.from_file(str(path), options)
            joint_names = [j.name for j in description.joints]
            handles = robot.insert_using_multibody_joints(
                world.rigid_bodies, world.colliders, world.multibody_joints
            )
            self._warm_up(world)
            link_bodies = [h.body for h in handles.links]
            if joint_links is None:
                joint_links = self._name_links(joint_names, handles.joints, link_bodies)
            art = self._articulate(world, [j.joint for j in handles.joints], link_bodies)
            # URDF joints declare no armature or damping; a light arm has no
            # stable PD hold without them, as on the Nexus backend
            multibody = art.multibody()
            multibody.set_armature([float(dynamics["default_armature"])] * multibody.ndofs)
            multibody.set_damping([float(dynamics["default_damping"])] * multibody.ndofs)
            arts.append(art)
        robot = RapierRobot(self, arts, mirror, name, joint_links)
        self.robots.append(robot)
        return robot

    @staticmethod
    def _name_links(joint_names, joint_handles, link_bodies) -> list[tuple[str, int]]:
        """Pair every named joint with the multibody link it moves: a joint's
        DoFs live on its child link, which `link2` names by rigid body."""
        index_of = {handle: i for i, handle in enumerate(link_bodies)}
        pairs = []
        for name, handle in zip(joint_names, joint_handles, strict=True):
            if name is None or handle.link2 not in index_of:
                continue
            pairs.append((str(name), index_of[handle.link2]))
        return pairs

    def add_camera(self, res, fov: float, pos=None, lookat=None):
        return self.render.add_camera(res, fov, pos=pos, lookat=lookat)

    def build(self, n_envs: int | None = None) -> None:
        if n_envs is not None and int(n_envs) != self.n_envs:
            raise ValueError("n_envs is fixed at scene construction for the rapier backend")
        self.render.build()

    # -- stepping and readback -------------------------------------------

    def step(self) -> None:
        for world in self.worlds:
            world.step()
        self.render_dirty()

    def render_dirty(self) -> None:
        self._render_dirty = True

    def flush_render(self) -> None:
        """Push rapier's poses into the Nexus state. Called from the render
        mirror's `sync`, so nothing is copied on a step that is never drawn."""
        if not self._render_dirty or not self.render.built:
            return
        self._render_dirty = False
        for entity in self.entities:
            if entity.fixed:
                continue
            for env in range(self.n_envs):
                body = entity._body(env)
                self.render.set_body_pose(
                    env, entity.mirror.handles[env], _xyz(body.translation), _wxyz(body.rotation)
                )
        for robot in self.robots:
            for env, art in enumerate(robot.arts):
                # every link's world pose, written straight into the mirror's
                # body buffer. Pushing the joint coordinates instead would run
                # the mirror's own forward kinematics from ITS root, which the
                # mirror never learns about, so a re-based robot (the mobile
                # store embodiment) rendered frozen at the origin while its
                # boxes moved.
                mirrored = robot.mirror.link_bodies[env]
                for index, body_handle in enumerate(art.link_bodies):
                    body = art.world.rigid_bodies.get(body_handle)
                    self.render.set_body_pose(
                        env, mirrored[index], _xyz(body.translation), _wxyz(body.rotation)
                    )
        self.render.invalidate_poses()
        self.render._synced = False

    def perf_stats(self) -> dict:
        """Engine-side timing of the last step, from rapier's own counters.
        No `gpu_ms`: rapier's step is a synchronous CPU call, so the bridge's
        wall-clock step time already is the engine time, and claiming a GPU
        number here would put a render cost in a physics column."""
        counters = self.worlds[0].physics_pipeline.counters
        if not counters.enabled:
            return {}
        return {
            "cpu_ms": float(counters.step_time_ms),
            "collision_detection_ms": float(counters.stages.collision_detection_time_ms),
            "solver_ms": float(counters.stages.solver_time_ms),
        }


# --- scene builders (mirror the frozen Genesis builders) --------------------


def _resolve_backend(sim_backend: str | None) -> None:
    """rapier steps on the CPU and takes no other backend. The renderer's
    adapter is chosen by the Nexus engine, not by this name (ADR-68)."""
    if sim_backend not in (None, "cpu"):
        raise ValueError(f"the rapier engine is CPU only; got backend {sim_backend!r}")


def _add_robot(scene: RapierScene, embodiment: str) -> RapierRobot:
    if embodiment in ("franka", "mobile"):
        return scene.add_mjcf_robot(franka_mjcf_path(), "franka")
    if embodiment == "g1_right":
        raise NotImplementedError("g1_right: Genesis only (ADR-g1-embodiment)")
    if not SO101_URDF.exists():
        raise FileNotFoundError(f"so101 asset missing: {SO101_URDF} (acquisition pending, ADR-6)")
    return scene.add_urdf_robot(
        SO101_URDF, embodiment, bool(load_rapier_physics()["robot"]["convex_hull"])
    )


def _apply_home_and_gains(robot: RapierRobot, profile: dict, n_envs: int) -> None:
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
    """rapier twin of `aisle.scenes.pharmacy.build_scene` (SPEC 020): the same
    seeded layout, DR draws, colors, labels and cameras, stepped by rapier and
    rendered through the Nexus viewer (ADR-68)."""
    cfg = cfg or SceneCfg()
    _resolve_backend(sim_backend)
    engine = _ensure_nexus(None, headless=headless)
    physics = load_physics()
    rapier_physics = load_rapier_physics()
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

    scene = RapierScene(engine, physics, n_envs, ambient)
    ground = rapier_physics["ground"]
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

    robot = _add_robot(scene, embodiment)

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
    """rapier twin of `aisle.scenes.store.build_store` (SPEC 200 RS-2/RS-3)."""
    from aisle.scenes.store import (
        StoreHandle,
        episode_layout,
        full_stock,
        generate_episode,
        load_planogram,
        yaw_quat_wxyz,
    )

    _resolve_backend(sim_backend)
    engine = _ensure_nexus(None, headless=headless)
    meds = load_meds()
    physics = load_physics()
    rapier_physics = load_rapier_physics()
    plano = load_planogram()
    episode = generate_episode(seed, scenario)
    store, geo = plano["store"], plano["store"]["unit_geometry"]
    profile = physics["embodiment"][embodiment]

    scene = RapierScene(
        engine, physics, n_envs, (physics["domain_randomization"]["ambient_default"],) * 3
    )
    ground = rapier_physics["ground"]
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
    "RapierArticulation",
    "RapierEntity",
    "RapierJoint",
    "RapierLink",
    "RapierRenderMirror",
    "RapierRobot",
    "RapierScene",
    "build_scene",
    "build_store",
    "load_rapier_physics",
]
