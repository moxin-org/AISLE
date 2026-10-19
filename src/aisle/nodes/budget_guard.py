"""Budget-guard node (SPEC 080 BG-1..5).

Interposes on all motion command edges (BG-1; topology enforced by the
validator, VAL-5). The clamping core is pure and unit-tested without dora
or sim (CON-12): limits come exclusively from env/limits.toml (BG-2), and
on violation the command is clamped — never dropped — to the nearest
legal value while a violation JSON is published (BG-3). The guard must
never crash the dataflow (BG-3) and adds <2 ms p99 per command (BG-4,
measured in tests/unit/test_guard_latency.py).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from aisle.topics import stamp

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Panda modified-DH rows (a_{i-1}, d_i, cos(alpha_{i-1}), sin(alpha_{i-1}))
# for joints 1..7 (official Franka kinematics; alphas are 0 or +-pi/2 so
# the trig is exact and precomputed off the per-command path, BG-4)
_FRANKA_DH = (
    (0.0, 0.333, 1.0, 0.0),
    (0.0, 0.0, 0.0, -1.0),
    (0.0, 0.316, 0.0, 1.0),
    (0.0825, 0.0, 0.0, 1.0),
    (-0.0825, 0.384, 0.0, -1.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.088, 0.0, 0.0, 1.0),
)
_FRANKA_FLANGE_D = 0.107

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class GuardLimits:
    """BG-2: every limit the guard enforces, loaded from env/limits.toml."""

    embodiment: str
    n_arm_dof: int
    q_min: tuple[float, ...]
    q_max: tuple[float, ...]
    qdot_max: tuple[float, ...]
    cmd_dt_s: float
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    fallback_qpos: tuple[float, ...]
    gripper_min: float
    gripper_max: float
    gripper_rate_max: float
    gripper_dt_s: float
    wall_timeout_s: float

    # precomputed off the per-command path (BG-4); cached_property writes
    # the instance __dict__ directly, so frozen is preserved
    @cached_property
    def q_min_arr(self) -> np.ndarray:
        return np.asarray(self.q_min, dtype=np.float32)

    @cached_property
    def q_max_arr(self) -> np.ndarray:
        return np.asarray(self.q_max, dtype=np.float32)

    @cached_property
    def max_step_arr(self) -> np.ndarray:
        return np.asarray(self.qdot_max, dtype=np.float32) * self.cmd_dt_s


# mobile reuses the franka arm's limits (ADR-14); its own [embodiment.mobile]
# section carries only the base limits (load_base_limits). Mirrors the
# validator's EMBODIMENT_ARM resolution.
_ARM_EMBODIMENT = {"mobile": "franka"}


def load_limits(embodiment: str) -> GuardLimits:
    with open(_REPO_ROOT / "env" / "limits.toml", "rb") as f:
        raw = tomllib.load(f)
    arm_kind = _ARM_EMBODIMENT.get(embodiment, embodiment)
    if arm_kind not in raw["embodiment"]:
        raise ValueError(
            f"env/limits.toml has no limits section for embodiment {embodiment!r};"
            " the guard refuses to guess (BG-2)"
        )
    emb = raw["embodiment"][arm_kind]
    return GuardLimits(
        embodiment=arm_kind,
        n_arm_dof=emb["n_arm_dof"],
        q_min=tuple(emb["q_min"]),
        q_max=tuple(emb["q_max"]),
        qdot_max=tuple(emb["qdot_max"]),
        cmd_dt_s=emb["cmd_dt_s"],
        workspace_min=tuple(emb["workspace_min"]),
        workspace_max=tuple(emb["workspace_max"]),
        fallback_qpos=tuple(emb["fallback_qpos"]),
        gripper_min=emb["gripper_min"],
        gripper_max=emb["gripper_max"],
        gripper_rate_max=emb["gripper_rate_max"],
        gripper_dt_s=emb["gripper_dt_s"],
        wall_timeout_s=raw["episode"]["wall_timeout_s"],
    )


def fk_flange(q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flange position and rotation matrix (base frame) via modified-DH
    forward kinematics on the commanded arm pose (BG-2; also the shared
    kinematics for ik-trajectory)."""
    T = np.eye(4)
    for (a, d, ca, sa), theta in zip(_FRANKA_DH, q_arm, strict=True):
        ct, st = math.cos(float(theta)), math.sin(float(theta))
        T = T @ np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return T[:3, 3] + T[:3, 2] * _FRANKA_FLANGE_D, T[:3, :3]


def fk_ee_pose(q_arm: np.ndarray, embodiment: str = "franka") -> tuple[np.ndarray, np.ndarray]:
    """Official end-effector pose for an embodiment (BG-2).

    Panda retains its verified modified-DH implementation. SO-101 parses the
    vendored official URDF through its fixed ``gripper_frame_link`` so the
    guard, planner, and simulator share one chain.
    """
    if embodiment == "so101":
        from aisle.kinematics import so101_chain

        return so101_chain().forward(q_arm)
    if embodiment == "g1_right":
        from aisle.kinematics import g1_right_chain

        # world frame (fixed pelvis + held waist) — same frame as the
        # workspace AABB in env/limits.toml (G1-F1)
        return g1_right_chain().forward(q_arm)
    if embodiment in {"franka", "mobile"}:
        return fk_flange(q_arm)
    raise ValueError(f"no forward kinematics for embodiment {embodiment!r}")


def fk_ee_pos(
    q_arm: np.ndarray, limits: GuardLimits | None = None, embodiment: str = "franka"
) -> np.ndarray:
    """End-effector position only (the guard's workspace check)."""
    profile = limits.embodiment if limits is not None else embodiment
    return fk_ee_pose(q_arm, profile)[0]


def gripper_to_fingers(g: float, limits: GuardLimits) -> np.ndarray:
    """Normalized gripper (0 open .. 1 closed) -> physical joint positions.

    The official profiles use q_max as open and q_min as closed. Franka's
    q_min happens to be zero; SO-101's revolute jaw has a non-zero lower
    endpoint, so interpolation must retain both endpoints (ADR-27).
    """
    open_pos = limits.q_max_arr[limits.n_arm_dof :]
    closed_pos = limits.q_min_arr[limits.n_arm_dof :]
    if g <= 0.0:
        return open_pos.copy()
    if g >= 1.0:
        return closed_pos.copy()
    return open_pos + float(g) * (closed_pos - open_pos)


def fingers_to_gripper(q: np.ndarray, limits: GuardLimits) -> float:
    """Inverse of gripper_to_fingers on the finger slice of a command."""
    open_pos = limits.q_max_arr[limits.n_arm_dof :]
    closed_pos = limits.q_min_arr[limits.n_arm_dof :]
    physical = np.asarray(q, np.float32)[limits.n_arm_dof :]
    return float(np.mean((open_pos - physical) / (open_pos - closed_pos)))


def _inside(ee: np.ndarray, limits: GuardLimits) -> bool:
    return all(limits.workspace_min[i] <= ee[i] <= limits.workspace_max[i] for i in range(3))


def _viol(
    reason: str, requested, clamped, joint: int | None = None, axis: str | None = None
) -> dict:
    v = {"reason": reason, "requested": requested, "clamped": clamped}
    if axis is not None:
        v["axis"] = axis
    else:
        v["joint"] = joint
    return v


def clamp_joint_cmd(
    cmd: np.ndarray, last_safe: np.ndarray, limits: GuardLimits, timed_out: bool
) -> tuple[np.ndarray, list[dict]]:
    """BG-3: pure clamp — always returns a legal command, never raises.

    Order: wall timeout (hold) -> malformed screen (hold bad entries) ->
    position -> velocity (vs last safe + contract dt) -> workspace (FK on
    the result, pulled back along the segment from last safe)."""
    last = np.asarray(last_safe, dtype=np.float32)
    n = len(limits.q_min)
    violations: list[dict] = []

    if timed_out:
        return last.copy(), [_viol("wall_timeout", None, None)]

    cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
    if cmd.shape != (n,):
        return last.copy(), [_viol("malformed", None, None)]
    safe = cmd.copy()
    for i in np.flatnonzero(~np.isfinite(safe)):
        violations.append(_viol("malformed", None, float(last[i]), joint=int(i)))
        safe[i] = last[i]

    clipped = np.clip(safe, limits.q_min_arr, limits.q_max_arr)
    for i in np.flatnonzero(clipped != safe):
        violations.append(_viol("position", float(safe[i]), float(clipped[i]), joint=int(i)))
    safe = clipped
    # BG-2: the workspace check applies to the COMMANDED pose — judged here,
    # before the velocity clamp shortens the step, so an out-of-workspace
    # intent is reported even when velocity limiting already contains it;
    # commanded_ee is None iff the commanded pose was inside
    commanded_ee = fk_ee_pos(safe[: limits.n_arm_dof], limits)
    if _inside(commanded_ee, limits):
        commanded_ee = None

    stepped = np.clip(safe, last - limits.max_step_arr, last + limits.max_step_arr)
    for i in np.flatnonzero(stepped != safe):
        violations.append(_viol("velocity", float(safe[i]), float(stepped[i]), joint=int(i)))
    velocity_clamped = stepped is not safe and bool(np.any(stepped != safe))
    safe = stepped

    # containment is an invariant of the OUTPUT regardless of what was
    # reported: FK is nonlinear, so even a velocity-shortened step must be
    # verified and pulled back if needed. When velocity left the command
    # untouched, its FK is the commanded one already computed.
    final_ee = (
        fk_ee_pos(safe[: limits.n_arm_dof], limits)
        if velocity_clamped
        else (commanded_ee if commanded_ee is not None else None)
    )
    if final_ee is not None and not _inside(final_ee, limits):
        if commanded_ee is None:  # velocity-clamped pose strayed on its own
            commanded_ee = final_ee
        if _inside(fk_ee_pos(last[: limits.n_arm_dof], limits), limits):
            # largest t in [0, 1] along last -> safe whose FK stays inside
            good, bad = 0.0, 1.0
            for _ in range(12):  # sub-millimeter resolution on any step
                mid = (good + bad) / 2
                if _inside(
                    fk_ee_pos((last + mid * (safe - last))[: limits.n_arm_dof], limits), limits
                ):
                    good = mid
                else:
                    bad = mid
            safe = (last + good * (safe - last)).astype(np.float32)
        else:  # last safe itself is outside (should not happen): hold home
            safe = np.asarray(limits.fallback_qpos, dtype=np.float32)
        final_ee = fk_ee_pos(safe[: limits.n_arm_dof], limits)
    if commanded_ee is not None:
        axis = next(
            (
                i
                for i in range(3)
                if not limits.workspace_min[i] <= commanded_ee[i] <= limits.workspace_max[i]
            ),
            2,
        )
        violations.append(
            {
                "reason": "workspace",
                "axis": _AXES[axis],
                "requested": float(commanded_ee[axis]),
                "clamped": float(final_ee[axis]),
            }
        )

    return safe, violations


def clamp_gripper_cmd(
    value: float, last_safe: float, limits: GuardLimits, timed_out: bool
) -> tuple[float, list[dict]]:
    """BG-1/BG-3: scalar gripper command under the SAME regime as joints —
    wall timeout holds it, NaN holds it at the last safe value, then range
    and rate (vs last safe + contract dt) clamps (PR review: the gripper
    must not bypass timeout or velocity enforcement)."""
    if timed_out:
        return last_safe, [_viol("wall_timeout", None, None, axis="gripper")]
    if not math.isfinite(value):
        return last_safe, [_viol("malformed", None, last_safe, axis="gripper")]
    violations = []
    clamped = min(max(value, limits.gripper_min), limits.gripper_max)
    if clamped != value:
        violations.append(_viol("position", value, clamped, axis="gripper"))
    max_step = limits.gripper_rate_max * limits.gripper_dt_s
    stepped = min(max(clamped, last_safe - max_step), last_safe + max_step)
    if stepped != clamped:
        violations.append(_viol("velocity", clamped, stepped, axis="gripper"))
    return stepped, violations


class EpisodeTimer:
    """BG-2 wall timer, anchored at the RESET that starts the episode —
    not the first command, or a policy could delay its first command to
    stretch the budget (PR review round 2). Idle pauses do not restart it
    either. Before any reset is seen (bare startup) the first command
    anchors, the only signal available."""

    def __init__(self) -> None:
        self._start: float | None = None

    def on_command(self, now: float) -> float:
        if self._start is None:
            self._start = now
        return now - self._start

    def timed_out(self, now: float, budget_s: float) -> bool:
        """BG-2: has the episode wall budget elapsed? Anchors like
        on_command — the first caller starts the clock."""
        return self.on_command(now) > budget_s

    def on_reset(self, now: float) -> None:
        self._start = now


def violation_payload(violation: dict, seq: int) -> dict:
    """BG-3: the published violation JSON is {reason, joint|axis, requested,
    clamped, seq}."""
    return {**violation, "seq": seq}


def main(clock=None) -> None:
    """Guard node: the clock is injected (CON-5)."""
    import json
    import os
    import sys
    import time

    import pyarrow as pa

    from aisle.mobility.guard import (
        base_blind_drive,
        base_watchdog_reason,
        blind_onset,
        clamp_base_cmd,
        load_base_limits,
        parse_env_id,
        parse_sim_stamp,
        reset_base_watchdog,
        sim_clock_is_blind,
        update_arm_motion_window,
        valid_base_pose,
    )
    from aisle.turn_node import Node

    clock = clock or time.monotonic
    embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
    lockstep = os.environ.get("AISLE_LOCKSTEP", "0").strip().lower() in ("1", "true", "yes")
    limits = load_limits(embodiment)
    fallback = np.asarray(limits.fallback_qpos, dtype=np.float32)
    is_mobile = embodiment == "mobile"
    base_limits = load_base_limits(embodiment) if is_mobile else None
    # MOB-3 keep-out geometry: the shelf AABBs and the base footprint radius
    # the base must not drive into with the arm extended
    shelves: list = []
    footprint_r = 0.0
    if is_mobile:
        from aisle.scenes.pharmacy import load_physics

        physics = load_physics()
        # keep-out geometry follows the SCENE (T15/ADR-18): the store's
        # units + counter + bin, or the desk shelf/tray
        if os.environ.get("AISLE_SCENE", "pharmacy") == "store":
            from aisle.scenes.store import load_planogram, store_scan_obstacles

            shelves = store_scan_obstacles(load_planogram())
        else:
            from aisle.scenes.pharmacy import desk_scan_obstacles

            shelves = desk_scan_obstacles(physics, embodiment)
        footprint_r = float(physics["embodiment"][embodiment]["base_footprint_radius_m"])

    node = Node()
    envs: dict[int, dict] = {}
    seq: dict[str, int] = {}
    counts: dict[str, int] = {}

    def new_state() -> dict:
        return {
            "last_safe": fallback,
            "last_gripper": 0.0,
            # MOB-3 mutex: the base is held at creep until this deadline; a
            # commanded arm-target change pushes it out, silence lets it pass
            "arm_motion_deadline_ns": None,
            "last_arm_cmd_sim_ns": None,
            "arm_motion_stamp_trusted": True,
            "base_pose": None,  # latest base_pose feedback (MOB-3 keep-out)
            # the watchdog's clocks (ADR-29, see run_watchdog): newest pose
            # sim stamp (None = sim clock blind), the pose stamp referenced
            # by the last base_cmd, the last base_cmd's wall time (the wall
            # net's reference), and the last pose's wall arrival (the pose
            # stream's own liveness signal)
            "base_pose_sim_ns": None,
            "last_base_cmd_sim_ns": None,
            "last_base_cmd_wall_t": None,
            "last_pose_wall_t": None,
            # wall time the sim clock most recently WENT blind (None while
            # it is healthy). The blind-drive net measures from here, not
            # from the last command, so a producer that keeps commanding
            # cannot hold the net open forever (issue #182).
            "blind_since_wall": None,
            # blind-drive stop already reported for the CURRENT blind
            # stretch. The stop itself is re-applied to every command (that
            # is what makes it stick), but the violation is an EDGE: without
            # this the guard emitted one violation and one stderr line per
            # pose — ~250 per nav goal — which both floods the recorded
            # violation topic under dora's drop-oldest backpressure and
            # feeds issue #183. Cleared with the onset (issue #182 review).
            "blind_stop_reported": False,
            "last_base_safe": [0.0, 0.0],  # last emitted safe base cmd
            "timer": EpisodeTimer(),
        }

    def next_seq(topic: str) -> int:
        seq[topic] = seq.get(topic, 0) + 1
        return seq[topic]

    def send(topic: str, value, metadata: dict, s: int | None = None) -> None:
        node.send_output(topic, value, stamp(metadata, s if s is not None else next_seq(topic)))

    def publish_violations(violations: list[dict], metadata: dict) -> None:
        for v in violations:
            counts[v["reason"]] = counts.get(v["reason"], 0) + 1
            s = next_seq("violation")
            payload = violation_payload(v, s)
            send("violation", pa.array([json.dumps(payload)]), metadata, s=s)
            print(f"guard violation: {payload}", file=sys.stderr)

    def refresh_blind(state: dict, now: float) -> tuple[bool, bool]:
        # (is the sim clock blind now, has it been blind long enough to stop
        # the base). Both the COMMAND path and the pose-driven watchdog call
        # this so they cannot disagree (issue #182 review). The onset is
        # maintained whether or not the base is currently moving: the
        # quantity is "how long have we had no clock", which does not pause
        # while the base is stopped — so a base commanded after a long blind
        # stretch is stopped at once rather than granted a fresh window.
        blind = sim_clock_is_blind(
            base_pose_sim_ns=state["base_pose_sim_ns"],
            last_pose_wall_t=state["last_pose_wall_t"],
            now_wall=now,
            limits=base_limits,
        )
        state["blind_since_wall"] = blind_onset(
            state["blind_since_wall"], sim_clock_blind=blind, now_wall=now
        )
        if state["blind_since_wall"] is None:
            # the clock came back: re-arm the edge so a LATER blind stretch
            # reports its own violation instead of being silently swallowed
            state["blind_stop_reported"] = False
        return blind, base_blind_drive(
            blind_since_wall=state["blind_since_wall"], now_wall=now, limits=base_limits
        )

    def run_watchdog(env_id: int, state: dict, now: float) -> None:
        # MOB-3 watchdog (ADR-29): stop a latched moving base ONCE when the
        # pure verdict says so. Called from the base_pose handler (the sim
        # clock — alive exactly when the base can move) and swept over all
        # envs on the BG-5 stats tick (the wall net's home: it also covers
        # unstamped sources, a hung sim, and env_ids absent from the pose
        # stream, which the sim clock cannot see).
        #
        # The blind bookkeeping runs BEFORE the moving-base gate: "how long
        # have we had no sim clock" does not pause while the base is
        # stopped, and the tick sweep is the only thing that keeps it
        # current for an env whose producer is silent (issue #182 review).
        blind, _ = refresh_blind(state, now)
        if state["last_base_safe"] == [0.0, 0.0]:
            return
        if state["last_base_cmd_sim_ns"] is None and state["base_pose_sim_ns"] is not None:
            # a command latched before any pose was seen: anchor it at the
            # FIRST pose stamp — a fair fresh reference (0 would falsely
            # stale-stop a guard that joined mid-run; the sim clock is
            # monotonic across episodes) that still lets a dead producer go
            # stale base_staleness_s of sim time later
            state["last_base_cmd_sim_ns"] = state["base_pose_sim_ns"]
        # the wall net only arms when the sim clock is demonstrably blind
        # (PR #156 review): the latest pose carried no usable stamp, no pose
        # ever arrived for this env, or the pose stream itself went silent
        # past the backstop. While valid stamps flow, the sim-time check
        # owns the verdict — a healthy-but-slow sim can never trip the net.
        reason = base_watchdog_reason(
            episode_timed_out=state["timer"].timed_out(now, limits.wall_timeout_s),
            last_cmd_sim_ns=state["last_base_cmd_sim_ns"],
            now_sim_ns=state["base_pose_sim_ns"],
            last_cmd_wall_t=state["last_base_cmd_wall_t"],
            now_wall=now,
            sim_clock_blind=blind,
            blind_since_wall=state["blind_since_wall"],
            limits=base_limits,
        )
        if reason is None:
            return
        # locate the stop in sim time when the clock is known (trace tooling
        # assumes non-decreasing per-topic stamps; PR #156 review)
        meta = {"env_id": env_id}
        if state["base_pose_sim_ns"] is not None:
            meta["sim_time_ns"] = state["base_pose_sim_ns"]
        send("base_cmd_safe", pa.array(np.zeros(2, dtype=np.float32)), meta)
        publish_violations(
            [
                {
                    "reason": reason,
                    "axis": "cmd",
                    "requested": state["last_base_safe"],
                    "clamped": [0.0, 0.0],
                }
            ],
            meta,
        )
        state["last_base_safe"] = [0.0, 0.0]

    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        now = clock()
        if event["id"].startswith(("joint_cmd", "reset_joint_cmd")):
            # reset_joint_cmd: the behavioral reset's motion (RST-2)
            # rides the SAME clamp path — the reset has no private
            # channel to the arm (VAL-5). Fleet mode wires N executors
            # as joint_cmd_0..N-1 inputs (dora input ids are unique per
            # node): the prefix match clamps them all identically, and
            # per-env state is already keyed by metadata env_id
            env_id = parse_env_id(metadata)
            state = envs.setdefault(env_id, new_state())
            timed_out = state["timer"].timed_out(now, limits.wall_timeout_s)
            prev_arm = np.asarray(state["last_safe"], dtype=np.float32)[: limits.n_arm_dof]
            safe, violations = clamp_joint_cmd(
                event["value"].to_numpy(zero_copy_only=False),
                state["last_safe"],
                limits,
                timed_out=timed_out,
            )
            # MOB-3 mutex: a commanded arm-target CHANGE (re)opens a hold
            # window of base_limits.arm_motion_hold_s. The window PERSISTS
            # while the arm travels even if the same target repeats, and
            # EXPIRES on command silence — so a settled arm releases the base
            # and a still-moving arm keeps it clamped. Deterministic (CON-5).
            if is_mobile:
                changed = bool(np.any(safe[: limits.n_arm_dof] != prev_arm))
                # An earlier changed target with an untrusted stamp may still
                # be moving.  The first later trustworthy command therefore
                # opens a full hold window even when it repeats that target.
                changed = changed or not state["arm_motion_stamp_trusted"]
                (
                    state["arm_motion_deadline_ns"],
                    state["last_arm_cmd_sim_ns"],
                    state["arm_motion_stamp_trusted"],
                ) = update_arm_motion_window(
                    state["arm_motion_deadline_ns"],
                    state["last_arm_cmd_sim_ns"],
                    changed,
                    metadata,
                    base_limits.arm_motion_hold_s,
                )
            state["last_safe"] = safe
            # the fingers ARE the gripper: keep the gripper channel's rate
            # reference in sync so alternating channels cannot double the
            # effective finger rate (PR review round 2)
            state["last_gripper"] = fingers_to_gripper(safe, limits)
            send("joint_cmd_safe", pa.array(safe), metadata)
            publish_violations(violations, metadata)
        elif event["id"] == "base_pose" and is_mobile:
            # MOB-3 keep-out feedback: cache the base pose. VALIDATE it first
            # (BG-3): a malformed pose must not crash clamp_base_cmd or bypass
            # keep-out — a bad pose caches None so keep-out fails closed.
            env_id = parse_env_id(metadata)
            state = envs.setdefault(env_id, new_state())
            state["last_pose_wall_t"] = now
            stamp_ns = parse_sim_stamp(metadata)
            if (
                stamp_ns is not None
                and state["last_base_cmd_sim_ns"] is not None
                and stamp_ns < state["last_base_cmd_sim_ns"]
            ):
                # a regressing stamp (bridge restart, misbehaving source)
                # would leave `now - anchor` negative and the sim-stale check
                # silently open forever — re-anchor and say so (PR #156
                # review), so staleness restarts from the new clock
                print(
                    f"guard: base_pose sim stamp regressed ({stamp_ns} < "
                    f"{state['last_base_cmd_sim_ns']}); re-anchoring the watchdog",
                    file=sys.stderr,
                )
                state["last_base_cmd_sim_ns"] = stamp_ns
            state["base_pose_sim_ns"] = stamp_ns
            pose = event["value"].to_numpy(zero_copy_only=False).tolist()
            if valid_base_pose(pose):
                state["base_pose"] = pose
            else:
                state["base_pose"] = None
                publish_violations(
                    [
                        {
                            "reason": "base_pose_malformed",
                            "axis": "pose",
                            "requested": None,
                            "clamped": None,
                        }
                    ],
                    metadata,
                )
            # the bridge latches the last base_cmd_safe and integrates it
            # every tick, so a dead producer or a timed-out episode would
            # drive forever — the pose stream is the watchdog's clock (ADR-29)
            run_watchdog(env_id, state, now)
        elif event["id"] == "base_cmd" and is_mobile:
            # MOB-3: base velocity limits, arm/base mutual exclusion (base
            # clamped to creep while the arm moves), the shelf keep-out (no
            # entry into a shelf zone with the arm reaching), and the BG-2
            # episode wall timeout. Never dropped (BG-3).
            env_id = parse_env_id(metadata)
            state = envs.setdefault(env_id, new_state())
            # commands carry no sim stamp: the staleness reference is the
            # newest pose stamp the guard has seen (ADR-29)
            state["last_base_cmd_sim_ns"] = state["base_pose_sim_ns"]
            state["last_base_cmd_wall_t"] = now
            timed_out = state["timer"].timed_out(now, limits.wall_timeout_s)
            cmd_stamp_ns = parse_sim_stamp(metadata)
            arm_in_motion = not state["arm_motion_stamp_trusted"]
            if state["arm_motion_deadline_ns"] is not None:
                arm_in_motion = (
                    arm_in_motion
                    or cmd_stamp_ns is None
                    or (
                        state["last_arm_cmd_sim_ns"] is not None
                        and cmd_stamp_ns < state["last_arm_cmd_sim_ns"]
                    )
                    or cmd_stamp_ns < state["arm_motion_deadline_ns"]
                )
            arm = np.asarray(state["last_safe"], dtype=np.float32)[: limits.n_arm_dof]
            ee = fk_ee_pos(arm)
            arm_extended = float(np.hypot(ee[0], ee[1])) > base_limits.arm_extended_reach_m
            safe_b, violations = clamp_base_cmd(
                event["value"].to_numpy(zero_copy_only=False),
                arm_in_motion,
                base_limits,
                base_pose=state["base_pose"],
                shelves=shelves,
                arm_extended=arm_extended,
                footprint_radius=footprint_r,
            )
            if timed_out and safe_b != [0.0, 0.0]:
                violations.append(
                    {
                        "reason": "base_timeout",
                        "axis": "cmd",
                        "requested": safe_b,
                        "clamped": [0.0, 0.0],
                    }
                )
                safe_b = [0.0, 0.0]
            # MOB-3 blind-drive stop, applied HERE and not only in the
            # pose-driven watchdog (issue #182 review). The watchdog's zero
            # is emitted from the base_pose handler, but a producer that
            # commands on every pose re-latches a nonzero base_cmd_safe
            # immediately behind it, and the bridge is a last-write-wins
            # latch — so the stop was overwritten before it was ever
            # integrated and the base never slowed down. Zeroing on the
            # COMMAND path is what makes the stop stick: every command is
            # zeroed for as long as the clock stays blind. Same predicate
            # the watchdog uses, so the two cannot disagree.
            _, blind_drive = refresh_blind(state, now)
            if blind_drive and safe_b != [0.0, 0.0]:
                if not state["blind_stop_reported"]:
                    # EDGE-triggered: the stop repeats, the violation does
                    # not (one per pose flooded the topic and stderr)
                    violations.append(
                        {
                            "reason": "base_blind_wall",
                            "axis": "cmd",
                            "requested": safe_b,
                            "clamped": [0.0, 0.0],
                        }
                    )
                    state["blind_stop_reported"] = True
                safe_b = [0.0, 0.0]
            state["last_base_safe"] = safe_b
            send("base_cmd_safe", pa.array(np.asarray(safe_b, dtype=np.float32)), metadata)
            publish_violations(violations, metadata)
        elif event["id"].startswith(("gripper_cmd", "reset_gripper_cmd")):
            env_id = parse_env_id(metadata)
            state = envs.setdefault(env_id, new_state())
            timed_out = state["timer"].timed_out(now, limits.wall_timeout_s)
            raw = event["value"].to_numpy(zero_copy_only=False)
            value = float(raw[0]) if len(raw) else float("nan")
            safe_g, violations = clamp_gripper_cmd(
                value, state["last_gripper"], limits, timed_out=timed_out
            )
            state["last_gripper"] = safe_g
            updated = np.array(state["last_safe"], dtype=np.float32)
            updated[limits.n_arm_dof :] = gripper_to_fingers(safe_g, limits)
            state["last_safe"] = updated
            send("gripper_cmd_safe", pa.array(np.array([safe_g], dtype=np.float32)), metadata)
            publish_violations(violations, metadata)
        elif event["id"] == "reset_done":
            # the authoritative episode boundary: the wall timer anchors
            # HERE (not at the first command), and velocity/hold state is
            # re-referenced to home — the robot IS at home after a
            # teleport reset.
            # Fleet mode (BRG-5): the boundary is PER ENV — sliced by the
            # reply's env_id. The all-envs loop let a NEIGHBOUR's reset
            # snap this env's last_gripper to 0.0 (OPEN) mid-carry: the
            # next clamp opened the fingers and the box dropped at
            # exactly the neighbour's reset moment (fleet probes 5-6,
            # seed-3 'dropped' at t~26). A reply without env_id keeps the
            # legacy whole-guard boundary.
            # No refusal check (ADR-34, issue #195): every reply on this
            # topic is a boundary, because refusals now answer on
            # `reset_refused` and the guard does not subscribe to it. The
            # filter that stood here guarded a real hazard — a refused reset
            # never touched the sim, so the robot is NOT at home, and
            # re-referencing velocity/hold state to the home qpos would
            # clamp the next command against a false origin, permitting a
            # larger real jump than the limit allows. That hazard is now
            # excluded by the graph rather than by this branch, and the
            # exclusion is enforced by
            # tests/unit/test_episode_boundary_wiring.py.
            reset_env = metadata.get("env_id")
            if isinstance(reset_env, int) and not isinstance(reset_env, bool):
                boundary_envs = [(reset_env, envs.setdefault(reset_env, new_state()))]
            else:
                boundary_envs = list(envs.items())
            for env_id, state in boundary_envs:
                if is_mobile and state["last_base_safe"] != [0.0, 0.0]:
                    # a pre-reset nonzero cmd can still be IN FLIGHT to the
                    # bridge; merely clearing our latch mirror would let it
                    # re-latch unwatched (PR #156 review). Emitting an
                    # explicit zero on the SAME channel orders it after any
                    # in-flight command, closing the window structurally.
                    send(
                        "base_cmd_safe",
                        pa.array(np.zeros(2, dtype=np.float32)),
                        {**metadata, "env_id": env_id},
                    )
                state["timer"].on_reset(now)
                state["last_safe"] = fallback
                state["last_gripper"] = 0.0
                state["arm_motion_deadline_ns"] = None
                state["last_arm_cmd_sim_ns"] = None
                state["arm_motion_stamp_trusted"] = True
                # MOB-3: clear the cached pose (keep-out fails closed until a
                # fresh pose arrives) and the watchdog/latched-base state
                # (the sim clock itself keeps running across episodes, so
                # base_pose_sim_ns is NOT reset)
                state["base_pose"] = None
                # every episode-scoped watchdog field, from the one list
                # that defines them (BASE_WATCHDOG_EPISODE_RESET). Inline
                # assignments here are how issue #182's blind onset came to
                # survive the boundary: a new state key was added and this
                # handler was not updated, and nothing could catch it.
                reset_base_watchdog(state)
        elif event["id"] == "tick":
            # the wall net's sweep (ADR-29): every env, fail-closed, on the
            # BG-5 cadence — it can only fire on a pathological run
            if is_mobile and not lockstep:
                for env_id, state in envs.items():
                    run_watchdog(env_id, state, now)
            # BG-5: cumulative violation counts every 5 s, timer-driven —
            # emitted even when no commands flow
            send("guard_stats", pa.array([json.dumps({"violations": counts})]), {})


if __name__ == "__main__":
    main()
