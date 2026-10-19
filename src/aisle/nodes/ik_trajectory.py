"""ik-trajectory node (CAP-5): grasp_pose + joint_state -> joint_cmd/gripper_cmd.

Staged pick-and-place executor: pregrasp along the grasp's approach axis,
descend, close, lift, transfer over the tray, lower, release, home.
Waypoints are solved with deterministic damped-least-squares IK on each
profile's shared kinematics: the verified Panda chain or the pinned official
SO-101 URDF chain. Commands stream at the joint_state cadence,
velocity-bounded per the manifest's max_joint_vel_rad_s. A reset aborts any
active plan (episode boundary — stale plans froze the guard in the first live
run).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aisle.nodes.budget_guard import (
    fk_ee_pose,
    fk_flange,
    gripper_to_fingers,
    load_limits,
)
from aisle.topics import parse_sim_stamp

# Panda hand: flange plate -> TCP between the fingertips
TCP_OFFSET = 0.1034
STAGES = (
    "rise",
    "staging",
    "pregrasp",
    "preclose",
    "advance",
    "close",
    "lift",
    "retract",
    "transfer",
    "lower",
    "release",
    "clear",
    "home",
)
# small vertical lift right after closing, before retracting: front-mode
# boxes must rise off their board but stay under the board above
LIFT_H = 0.015
# stage-completion tracking tolerance (rad) and bounded at-target dwell (s)
TRACK_TOL = 0.10
STAGE_BAIL_S = 4.0
# divergence bail (transit-collision mechanism 2, seed 15): a tracking
# error far beyond gravity-sag scale means the arm is pressed against
# something the command cannot push through — bail in DIVERGE_BAIL_S,
# not STAGE_BAIL_S, so the press lasts ~1 s instead of the full dwell.
# 0.8 rad sits 3x above the worst measured sag (0.27 rad, T15 round 12)
# and 3x below the measured wrist-flip class (2.25-3.07 rad).
DIVERGE_TOL = 0.8
DIVERGE_BAIL_S = 1.0
# wrist-flip filter (transit-collision mechanism 2): the staged read
# entry sits 6-14 cm behind the read pose at near-identical orientation,
# so a wrist joint jumping more than this across the STAGED->READ hop is
# a flipped IK branch, never articulation — the measured flip class is
# 2.25-3.07 rad. Scoped to that hop ONLY (v2): the first filter also
# bounded home-referenced hops and measured 0.5 -> 0.375 on the n=8
# (24 read_move exhaustions; home->read legitimately articulates the
# wrist past this bound). Flipped candidates are dropped at SOLVE time
# so the executor never commands the press.
WRIST_FLIP_TOL = 2.0
WRIST_JOINTS = slice(4, 7)  # franka wrist: joints 5-7


def wrist_hop_flips(q_a: np.ndarray, q_b: np.ndarray) -> bool:
    """True when a joint hop crosses a wrist-flip branch boundary."""
    delta = np.abs(np.asarray(q_b, dtype=np.float64) - np.asarray(q_a, dtype=np.float64))
    return bool(delta[WRIST_JOINTS].max() > WRIST_FLIP_TOL)


# gripper ramp per 100 Hz tick and message cadence: emission is
# 100/GRIP_SEND_EVERY = 25 Hz (the gripper_cmd contract is <=30 Hz;
# every-3rd-tick's 33.3 Hz was illegal) and the per-message step
# (GRIP_SEND_EVERY * GRIP_STEP_PER_TICK = 0.04) stays <= the guard's
# gripper_rate_max * gripper_dt_s bound; both relations are pinned by
# tests/unit/test_ik_trajectory.py
GRIP_STEP_PER_TICK = 0.010
GRIP_SEND_EVERY = 4


def park_read_reply(payload: dict, metadata: dict) -> dict:
    """Attach the completed park's TC-2 clock barrier, or refuse.

    A successful reply without a usable stamp would make the reader fall
    back to arrival order and reopen issue #153's stale-frame wall race.
    Returning an ordinary failed move keeps the service reply guarantee and
    lets the scan tour advance fail-closed."""
    park_stamp = parse_sim_stamp(metadata)
    if park_stamp is None:
        return {"ok": False, "reason": "missing_park_stamp"}
    return {**payload, "frame_after_sim_time_ns": park_stamp}


def grip_ramp_tick(current: float, target: float, tick: int) -> tuple[float, int, bool]:
    """One 100 Hz tick of the gripper ramp: returns (grip, tick, emit).
    Pure so the emitted SEQUENCE is testable: per-message step legality
    and emission cadence both regressed during review rounds 1-2."""
    if current == target:
        return current, tick, False
    step = min(GRIP_STEP_PER_TICK, abs(target - current))
    current = current + step if target > current else current - step
    tick += 1
    return current, tick, tick % GRIP_SEND_EVERY == 0 or current == target


# max per-joint jump between consecutive insertion waypoints (rad)
CONTINUITY_MAX = 1.2
# max per-joint jump for the front-mode wrist flip, held to the same
# bound as every other consecutive pair: multi-radian flips have NEVER
# executed stably (a ~2.5 rad planned flip diverged to 3.4 rad tracking
# error and wrapped the arm into a physics NaN that CRASHED the bridge —
# T09 diag runs). Until the under-board grasp strategy is resolved
# (ADR-10 section 8), an over-limit flip REFUSES the plan so the episode
# closes honestly via the verifier timeout instead of killing the sim
FLIP_MAX = CONTINUITY_MAX
# staging TCP height: above every shelf box top (max 0.44 on the
# staggered two-level shelf), reached BEFORE moving over the scene — the
# raw home->pregrasp joint sweep clipped shelf boxes (T08); kept low
# because deep rear staging poses at 0.66 failed IK (T10 probe)
STAGING_Z = 0.56
# TCP height for the lowering stage: tray base top (0.04) + tallest med
# half-extent (0.055) + finger clearance
# fallback release TCP height when grasp_pose metadata carries no per-med
# place_tcp_z (planner computes: tray top + hanging box length + drop gap
# — pressing the box down drove it THROUGH the tray slab, hovering high
# toppled it)
PLACE_TCP_Z = 0.125
TRANSFER_TCP_Z = 0.30
# flange-yaw correction realizing the planner's grip axis (issue #92).
# The Franka hand is mounted -45 degrees about the flange z (measured
# in-sim: neutral link7 yaw -45 vs hand yaw 0), so a flange-yaw target
# psi_f executes the finger-separation axis at psi_f - 45 (the
# m0-2-0d0773 seed-3 plan targeted flange yaw 90 and the executed axis
# came out 46). The planner's grip axis u(psi) = (-sin psi, cos psi)
# sits at 90 + psi; flange = psi + HAND_MOUNT_YAW makes the executed
# axis coincide with u mod pi (the -45 branch keeps IK solutions near
# the old ones). Uncompensated, every "axis-aligned" top-down grip was
# a DIAGONAL pinch: the T10 "diagonal detents" at close and the seed-3
# hand-corner topple of a taller neighbour both trace here. Front mode
# (FRONT_QUAT in grasp_topdown) composes the same offset about its own
# local z (issue #92 follow-up, closed).
HAND_MOUNT_YAW = -math.pi / 4

_RZ_PI = np.diag([-1.0, -1.0, 1.0])  # local z spin: box-symmetric grasp flip
# canonical retry seeds (CON-5: a FIXED list, tried in order): DLS from the
# home posture stalls in a local minimum for horizontal-wrist targets; a
# wrist-forward posture reaches them in <150 iterations
_CANONICAL_SEEDS = (
    np.array([0.0, 0.2, 0.0, -2.6, 0.0, 1.2, 0.785], dtype=np.float32),
    np.array([0.0, -0.4, 0.0, -2.8, 0.0, 2.4, 0.785], dtype=np.float32),
)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _so101_seeds() -> tuple[np.ndarray, ...]:
    """Fixed low-discrepancy starts over the official joint limits (CON-5)."""
    limits = load_limits("so101")
    lower = np.asarray(limits.q_min[: limits.n_arm_dof], dtype=np.float64)
    upper = np.asarray(limits.q_max[: limits.n_arm_dof], dtype=np.float64)
    primes = (2, 3, 5, 7, 11)
    return tuple(
        (lower + (upper - lower) * np.array([_van_der_corput(i, p) for p in primes])).astype(
            np.float32
        )
        for i in range(1, 65)
    )


_SO101_CANONICAL_SEEDS = _so101_seeds()


#: G1-E: the harness accepts g1_right, but this module's kinematics are
#: Franka-DH and SO-101-URDF only — G1-F owns the G1 chain. An unknown
#: embodiment MUST raise here instead of silently solving the Franka chain
#: (HAR-1; the pre-G1-E fall-through was the bug).
_SUPPORTED_EMBODIMENTS = ("franka", "mobile", "so101")


def _require_supported_embodiment(embodiment: str, fn: str) -> None:
    if embodiment not in _SUPPORTED_EMBODIMENTS:
        raise ValueError(f"{fn}: unsupported embodiment {embodiment!r}")


def _arm_bounds(embodiment: str) -> tuple[np.ndarray, np.ndarray]:
    limits = load_limits(embodiment)
    return (
        np.asarray(limits.q_min[: limits.n_arm_dof], dtype=np.float64),
        np.asarray(limits.q_max[: limits.n_arm_dof], dtype=np.float64),
    )


def fk_tcp(q_arm: np.ndarray, embodiment: str = "franka") -> np.ndarray:
    _require_supported_embodiment(embodiment, "fk_tcp")
    if embodiment == "so101":
        return fk_ee_pose(q_arm, embodiment)[0]
    pos, rotation = fk_flange(q_arm)
    return pos + rotation[:, 2] * TCP_OFFSET


def quat_to_rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Matrix -> xyzw quaternion (Shepperd, branch-stable)."""
    r = rotation
    trace = np.trace(r)
    if trace > 0:
        w = math.sqrt(1.0 + trace) / 2
        x, y, z = (
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
        )
        return np.array([x, y, z, w])
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1e-12)) * 2
    vec = [0.0, 0.0, 0.0]
    vec[i] = s / 4
    vec[j] = (r[j, i] + r[i, j]) / s
    vec[k] = (r[k, i] + r[i, k]) / s
    w = (r[k, j] - r[j, k]) / s
    return np.array([*vec, w])


def _slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    if float(np.dot(qa, qb)) < 0:
        qb = -qb
    dot = min(1.0, max(-1.0, float(np.dot(qa, qb))))
    theta = math.acos(dot)
    if theta < 1e-6:
        return qa
    return (math.sin((1 - t) * theta) * qa + math.sin(t * theta) * qb) / math.sin(theta)


def topdown_rotation(yaw: float) -> np.ndarray:
    """Rz(yaw + HAND_MOUNT_YAW) @ Rx(pi): flange z straight down, FINGER
    axis at `yaw` — the mount offset maps finger yaw to flange yaw."""
    cy, sy = math.cos(yaw + HAND_MOUNT_YAW), math.sin(yaw + HAND_MOUNT_YAW)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return rz @ rx


def so101_radial_rotation(target_pos: np.ndarray, jaw_sign: float = 1.0) -> np.ndarray:
    """Feasible front-grasp frame for SO-101's five-axis serial geometry.

    The tool points radially away from the base, the official jaw-motion axis
    is tangential (horizontal, so it can straddle a shelf box), and the
    remaining axis points up. World-yaw is coupled to target azimuth by the
    base pan joint, so IK constrains the other two orientation-error
    components and leaves that coupled yaw free.
    """
    target = np.asarray(target_pos, dtype=float)
    radial = np.array([target[0], target[1], 0.0])
    norm = float(np.linalg.norm(radial))
    if norm < 1e-9:
        radial = np.array([1.0, 0.0, 0.0])
    else:
        radial /= norm
    sign = 1.0 if jaw_sign >= 0 else -1.0
    tangential = sign * np.array([-radial[1], radial[0], 0.0])
    up = np.cross(radial, tangential)
    # Official STL/URDF geometry: the moving jaw sweeps along frame X.
    return np.column_stack((tangential, up, radial))


# -- T2 read poses (design doc §3, idea I13) ---------------------------------
# Park the WRIST CAMERA looking at the box face. The ladder is
# deterministic (CON-5): fixed (range_m, azimuth_rad, pitch_rad)
# entries, walked in order — an entry is taken iff every earlier entry
# had no IK solution or did not TRACK. All three axes are measured
# (offline streamed repro, seed 3): flat 0.13 m reads at min margin
# +0.27; far-side boxes (+y) jam the arm against the shelf at every
# FLAT entry (0.3-0.9 rad terminal error) and read only from the
# PITCHED rungs (camera above the face looking down — rectification
# absorbs the tilt; cetirizine +0.27 at pitch 0.35). Pitch is CAPPED at
# 0.35: from 0.55 up the face quad slides onto the box behind and reads
# it confidently wrong — the one failure mode T2 must never have.
# FAR-FIRST reorder (campaign t2_breakthrough agent-2 audit, ported r2):
# every audited collision was a read-park approach/advance at the 0.13 m
# rung (fingertips ~7.7 cm nominal from the face); flat entries at
# 0.20/0.24 m solve and stage for the same faces at the same rate with
# 14.7/18.7 cm TCP-to-face clearance, and rectified reads stay above the
# margin floor out to 0.24 m. Read capability is a superset — only the
# PREFERENCE moves outward; the stock close entries stay as last-resort
# rungs so the refusal-retry walk ends where the old ladder began.
READ_LADDER = (
    (0.20, 0.0, 0.0),
    (0.16, 0.0, 0.35),
    (0.20, 0.25, 0.0),
    (0.20, -0.25, 0.0),
    (0.16, 0.25, 0.35),
    (0.16, -0.25, 0.35),
    (0.24, 0.0, 0.0),
    (0.16, 0.0, 0.0),
    (0.13, 0.0, 0.0),
)
# jam chirality (measured, seed 3): flat near entries TRACK for faces on
# the -y side but JAM into the shelf for +y faces (0.3-0.9 rad terminal
# error, 4 s of pressing that KNOCKED boxes to 40-125 degree tilts and
# poisoned every later read of the episode). For +y faces the pitched
# entries are moved to the FRONT of the ladder so the jamming entries
# never execute; a deterministic function of the face (CON-5).
FAR_SIDE_Y = 0.05
_PITCHED_FIRST_LADDER = (
    (0.16, 0.0, 0.35),
    (0.16, 0.25, 0.35),
    (0.16, -0.25, 0.35),
    (0.20, 0.0, 0.35),
    (0.20, 0.0, 0.0),
    (0.20, 0.25, 0.0),
    (0.20, -0.25, 0.0),
    (0.24, 0.0, 0.0),
    (0.16, 0.0, 0.0),
    (0.13, 0.0, 0.0),
)
# near-side faces under this clearance over their board also take the
# pitched-first ladder (v7's measured jam class: flat entries press into
# the shelf — arm-L run 73d6d1: 56/115 parks bailed); the same
# deterministic face-based choice already made for far-side (+y) faces
PITCH_FIRST_CLEARANCE_M = 0.05
_BOARD_TOPS: tuple[float, ...] | None = None
# a read parks or it retries: terminal tracking error above this means
# the camera is NOT looking at the planned face. Wide on purpose: the
# rectified read projects through the ACHIEVED pose, so closeness only
# matters for keeping the face in frame — good parks measure 0.03-0.11
# rad, shelf-jam failures 0.3-0.9, and a 0.12 tol rejected a good 0.107
# park (offline streamed repro)
READ_TRACK_TOL = 0.20
# bounded ladder walk per read_move — each attempt costs a retreat+park
READ_MAX_ATTEMPTS = 6
# staged read approach: park first at the same view backed off along the
# view axis, then advance — keeps the home->read joint hop clear of the
# shelf face (upper-level transits swept boxes 12-28 cm without it).
# Deeper backoff preferred (0.15 audited fully clean on the knock seeds);
# 0.10 is the fallback where the deep far pose has no IK (the far-side
# pitched entries seed 3's tour needs), still clean bar one 8.6 cm case
# the 0.15 rung absorbs. An entry with NEITHER is dropped.
READ_STAGE_BACKOFFS_M = (0.15, 0.10)
READ_STAGE_BACKOFF_M = READ_STAGE_BACKOFFS_M[0]  # spy/test anchor
# the frozen desk wrist camera (scenes/pharmacy.py add_camera): 320x240
# at 70 degree vertical fov — the aim-prior projection must match the
# rendered geometry exactly
WRIST_FX = WRIST_FY = 120.0 / math.tan(math.radians(35.0))
WRIST_CX, WRIST_CY = 160.0, 120.0
# hand = flange @ Rz(HAND_MOUNT_YAW), zero translation — pinned against
# the Genesis link poses to 6e-7 (test_read_pose)
_FLANGE_TO_HAND = np.array(
    [
        [math.cos(HAND_MOUNT_YAW), -math.sin(HAND_MOUNT_YAW), 0.0],
        [math.sin(HAND_MOUNT_YAW), math.cos(HAND_MOUNT_YAW), 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def read_flange_targets(
    face: np.ndarray,
    range_m: float,
    azimuth_rad: float,
    mount: np.ndarray,
    pitch_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(tcp_pos, flange_rotation) parking the wrist camera `range_m`
    from `face`, looking at it from `azimuth_rad` off the -x normal and
    `pitch_rad` above the horizontal. Chain: desired camera pose (CV
    look-at converted to the mount's GL frame via GL_TO_CV — omit that
    factor and the camera stares at the robot's own hand) -> hand pose
    through the inverse mount -> flange frame -> TCP position for the
    standard IK entry point."""
    from aisle.verifier.calibration import GL_TO_CV, lookat_rotation_cv

    face = np.asarray(face, dtype=np.float64)
    direction = np.array(
        [
            -math.cos(azimuth_rad) * math.cos(pitch_rad),
            math.sin(azimuth_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
        ]
    )
    eye = face + range_m * direction
    camera = np.eye(4)
    camera[:3, :3] = lookat_rotation_cv(eye, face) @ GL_TO_CV
    camera[:3, 3] = eye
    t_hand = camera @ np.linalg.inv(mount)
    r_flange = t_hand[:3, :3] @ _FLANGE_TO_HAND.T
    tcp = t_hand[:3, 3] + r_flange[:, 2] * TCP_OFFSET
    return tcp, r_flange


def solve_read_poses(
    face: np.ndarray, mount: np.ndarray, q0: np.ndarray, embodiment: str = "franka"
) -> list[tuple[np.ndarray, float]]:
    """Every IK-solvable ladder entry as (q, range_m), ladder order.
    NO 180-degree flip retry (unlike ik_solve): spinning the flange flips
    the off-axis camera away from the face. franka-only — the SO-101
    read chain is unmeasured, so it refuses rather than guesses.

    The FULL list, not the first hit: IK feasibility is not
    trackability — the first live T2 tour found read poses the
    controller could not reach (0.45 rad terminal error, shelf
    contact), so the executor walks this list until one TRACKS."""
    if embodiment != "franka":
        return []
    face = np.asarray(face, dtype=np.float64)
    pitched_first = face[1] > FAR_SIDE_Y
    if not pitched_first:
        global _BOARD_TOPS
        if _BOARD_TOPS is None:
            from aisle.nodes.label_reader import shelf_board_tops

            _BOARD_TOPS = tuple(shelf_board_tops())
        below = [t for t in _BOARD_TOPS if t <= face[2] + 0.01]
        board = max(below) if below else min(_BOARD_TOPS)
        if face[2] - board < PITCH_FIRST_CLEARANCE_M:
            pitched_first = True
    ladder = _PITCHED_FIRST_LADDER if pitched_first else READ_LADDER
    solutions = []
    for range_m, azimuth, pitch in ladder:
        tcp, r_flange = read_flange_targets(face, range_m, azimuth, mount, pitch)
        for seed in (q0, *_CANONICAL_SEEDS):
            q = _ik_once(tcp, r_flange, seed, embodiment)
            if q is not None:
                # staged approach: the same view retracted along its axis
                # (READ_STAGE_BACKOFF_M), IK'd from q first so the branch
                # matches — the direct home->read joint hop swept boxes
                # on upper-level layouts (T2 curve run 1: collisions at
                # t=2.6-3.6 s, a target knocked clean off the shelf;
                # displacement audit: 12-28 cm knocks, every one on an
                # UNSTAGED entry). An entry with no staged approach is
                # DROPPED: an unprotected transit is how boxes get
                # knocked, and the ladder has more entries.
                q_far = None
                for backoff in READ_STAGE_BACKOFFS_M:
                    far_tcp, far_rot = read_flange_targets(
                        face, range_m + backoff, azimuth, mount, pitch
                    )
                    for far_seed in (q, *_CANONICAL_SEEDS):
                        q_far = _ik_once(far_tcp, far_rot, far_seed, embodiment)
                        if q_far is not None and wrist_hop_flips(q_far, q):
                            q_far = None  # flipped staged->read hop: next seed
                            continue
                        if q_far is not None:
                            break
                    if q_far is not None:
                        break
                if q_far is not None:
                    solutions.append((q, range_m, pitch, q_far))
                break
    return solutions


def solve_read_pose(
    face: np.ndarray, mount: np.ndarray, q0: np.ndarray, embodiment: str = "franka"
) -> tuple[np.ndarray, float, float] | None:
    """(q, range_m, pitch) for the first reachable ladder entry, else None."""
    solutions = solve_read_poses(face, mount, q0, embodiment)
    return solutions[0] if solutions else None


def wrist_camera_pose(q_arm: np.ndarray, mount: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cam_pos, cam_rot_cv) of the wrist camera at an arm pose, via the
    executor's own FK — the reader rectifies the face quad through THIS
    (the ACHIEVED pose, not the planned one: the tracked pose sags a few
    degrees, which at read range shifts the face tens of pixels; the
    first live T2 tour centred on a NEIGHBOUR box and read it
    confidently wrong)."""
    from aisle.verifier.calibration import GL_TO_CV

    pos_f, rot_f = fk_flange(np.asarray(q_arm, dtype=np.float64))
    t_hand = np.eye(4)
    t_hand[:3, :3] = rot_f @ _FLANGE_TO_HAND
    t_hand[:3, 3] = pos_f
    camera = t_hand @ mount
    return camera[:3, 3], camera[:3, :3] @ GL_TO_CV


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion-based orientation error (2*sign(w)*vec of target@current.T).

    The naive skew-symmetric rotation-vector (0.5*vee(R_err)) is ZERO at a
    180-degree error (sin(pi)=0) — the first live T08 runs 'converged' onto
    a pi-flipped wrist because of exactly that blindness. The quaternion
    vector part is sin(theta/2)*axis: non-degenerate at pi."""
    r = target @ current.T
    trace = np.trace(r)
    # Shepperd's method, branch-stable
    if trace > 0:
        w = math.sqrt(1.0 + trace) / 2
        vec = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (4 * w)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1e-12)) * 2
        vec = np.empty(3)
        vec[i] = s / 4
        vec[j] = (r[j, i] + r[i, j]) / s
        vec[k] = (r[k, i] + r[i, k]) / s
        w = (r[k, j] - r[j, k]) / s
    return 2.0 * (vec if w >= 0 else -vec)


def _pose_error(
    q: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray, embodiment: str
) -> np.ndarray:
    _require_supported_embodiment(embodiment, "_pose_error")
    if embodiment == "so101":
        tcp, rotation = fk_ee_pose(q, embodiment)
        # SO-101's base pan couples world yaw to target azimuth. Its five
        # independent task constraints are TCP position plus pitch/roll;
        # compare against the radial front frame but leave the world-z
        # rotation-error component free. This is the same underactuated
        # semantics used by Genesis, stated explicitly rather than asking a
        # five-axis chain to realize a six-axis pose.
        base = so101_radial_rotation(target_pos)
        jaw_sign = 1.0 if float(np.dot(target_rot[:, 0], base[:, 0])) >= 0 else -1.0
        desired = so101_radial_rotation(target_pos, jaw_sign)
        orientation = _rotation_error(rotation, desired)[:2]
    else:
        pos, rotation = fk_flange(q)
        tcp = pos + rotation[:, 2] * TCP_OFFSET
        orientation = _rotation_error(rotation, target_rot)
    return np.concatenate([target_pos - tcp, orientation])


def _dls(
    q: np.ndarray,
    target_pos,
    target_rot,
    rows: slice,
    iters: int,
    embodiment: str,
) -> np.ndarray:
    """Damped-least-squares descent on the selected error rows (position
    rows only for the bootstrap, every profile constraint for the full solve). Clamped
    error (CLIK-style) plus a deterministic backtracking line search keep
    the descent stable near joint limits, where the raw DLS oscillates."""
    damping = 0.05
    eps = 1e-5
    q_min, q_max = _arm_bounds(embodiment)
    err = _pose_error(q, target_pos, target_rot, embodiment)[rows]
    n = len(err)
    reg = damping**2 * np.eye(n)
    for _ in range(iters):
        clamped = err.copy()
        pos_norm = np.linalg.norm(clamped[:3])
        if pos_norm > 0.08:
            clamped[:3] *= 0.08 / pos_norm
        if n > 3:
            rot_norm = np.linalg.norm(clamped[3:])
            if rot_norm > 0.4:
                clamped[3:] *= 0.4 / rot_norm
        jac = np.empty((n, len(q)))
        for j in range(len(q)):
            dq = q.copy()
            dq[j] += eps
            jac[:, j] = (err - _pose_error(dq, target_pos, target_rot, embodiment)[rows]) / eps
        step = jac.T @ np.linalg.solve(jac @ jac.T + reg, clamped)
        base_norm = np.linalg.norm(err)
        for _halving in range(4):
            candidate = np.clip(q + step, q_min, q_max)
            candidate_err = _pose_error(candidate, target_pos, target_rot, embodiment)[rows]
            if np.linalg.norm(candidate_err) < base_norm:
                break
            step = step / 2
        q = candidate
        err = candidate_err  # carry: next iteration reuses the accepted error
    return q


def _ik_once(
    target_pos: np.ndarray, target_rot: np.ndarray, q0: np.ndarray, embodiment: str
) -> np.ndarray | None:
    _require_supported_embodiment(embodiment, "_ik_once")
    q = np.asarray(q0, dtype=np.float64).copy()
    for bootstrap in (False, True):
        if bootstrap:
            # position-only descent first: pulls the arm into the right
            # region, where the full-pose solve has a clean basin
            q = _dls(
                np.asarray(q0, dtype=np.float64).copy(),
                target_pos,
                target_rot,
                slice(0, 3),
                80 if embodiment == "so101" else 60,
                embodiment,
            )
        q = _dls(
            q,
            target_pos,
            target_rot,
            slice(0, 5 if embodiment == "so101" else 6),
            250 if embodiment == "so101" else 150,
            embodiment,
        )
        err = _pose_error(q, target_pos, target_rot, embodiment)
        pos_tol = 0.015 if embodiment == "so101" else 5e-4
        rot_tol = 0.02 if embodiment == "so101" else 1e-3
        if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
            return q.astype(np.float32)
    return None


def ik_continuation(
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    target_rot: np.ndarray,
    q_start: np.ndarray,
    step_m: float = 0.04,
    embodiment: str = "franka",
) -> list[np.ndarray] | None:
    """Solve a Cartesian straight-line move by numerical continuation and
    return EVERY substep config: the executor tracks the planned line
    through these waypoints (discarding them and joint-interpolating
    endpoint-to-endpoint lets the TCP bow off the line — PR #10 review),
    and chaining keeps each config on q_start's branch (a single far
    solve can land wrist-flipped and sweep the arm through the shelf)."""
    from_pos = np.asarray(from_pos, dtype=np.float64)
    to_pos = np.asarray(to_pos, dtype=np.float64)
    n = max(1, int(math.ceil(np.linalg.norm(to_pos - from_pos) / step_m)))
    q = q_start
    path: list[np.ndarray] = []
    for i in range(1, n + 1):
        q = _ik_once(from_pos + (to_pos - from_pos) * (i / n), target_rot, q, embodiment)
        if q is None:
            return None
        path.append(q)
    return path


def ik_solve(
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q0: np.ndarray,
    embodiment: str = "franka",
) -> np.ndarray | None:
    """DLS-IK for a TCP pose. Deterministic (CON-5): fixed seed pose, fixed
    iteration budget; a box is symmetric under a 180-degree spin about the
    approach axis, so the flipped grasp is tried in fixed order before
    reporting failure."""
    _require_supported_embodiment(embodiment, "ik_solve")
    target_pos = np.asarray(target_pos, dtype=np.float64)
    seeds = _SO101_CANONICAL_SEEDS if embodiment == "so101" else _CANONICAL_SEEDS
    for rot in (target_rot, target_rot @ _RZ_PI):
        for seed in (q0, *seeds):
            q = _ik_once(target_pos, rot, seed, embodiment)
            if q is not None:
                return q
    return None


def interpolate_step(
    current: np.ndarray, target: np.ndarray, max_vel: float, dt: float
) -> np.ndarray:
    """One velocity-bounded step of joint-space interpolation."""
    delta = np.clip(target - current, -max_vel * dt, max_vel * dt)
    return (current + delta).astype(np.float32)


@dataclass(frozen=True)
class Stage:
    name: str
    path: tuple  # (n, 7) waypoint chain; the LAST entry is the stage target
    gripper: float  # 0 open .. 1 closed
    settle_s: float  # dwell after reaching, letting physics catch up
    vel: float = 1.0  # joint-velocity scale; carry stages move gently
    # completion tolerance (rad): the 0.10 default is centimeters at the
    # TCP — release-critical stages need the arm actually AT the hover
    # pose, or the box grounds/tips (M0 runs m0-1-2f6716 + offline sweep)
    track_tol: float = TRACK_TOL

    @property
    def q(self) -> np.ndarray:
        return self.path[-1]


class StagedPlan:
    """The full pick-place waypoint sequence, solved once per grasp_pose.
    Chained seeding (each stage seeds the next solve) keeps arm
    configurations consistent across stages. The pregrasp sits
    approach_height back along the grasp's own approach axis, so a tilted
    grasp descends along its tilt."""

    def __init__(
        self,
        grasp_pose: np.ndarray,
        tray_xy: tuple[float, float],
        approach_m: float,
        q_seed: np.ndarray,
        place_z: float = PLACE_TCP_Z,
        embodiment: str = "franka",
    ) -> None:
        from aisle.scenes.pharmacy import load_physics

        grasp_pose = np.asarray(grasp_pose, dtype=np.float32).reshape(7)
        grasp_pos = grasp_pose[:3].astype(np.float64)
        grasp_rot = quat_to_rotation(grasp_pose[3:7])
        approach_axis = grasp_rot[:, 2]  # flange z: points from wrist to fingertips
        limits = load_limits(embodiment)
        home = np.asarray(q_seed, dtype=np.float32)[: limits.n_arm_dof]
        profile = load_physics()["embodiment"][embodiment]
        grasp_cmd = float(profile.get("gripper_grasp_cmd", 1.0))
        pregrasp_cmd = float(profile.get("gripper_pregrasp_cmd", 0.0))
        staging_height = float(profile.get("trajectory_staging_z", STAGING_Z))
        transfer_height = float(profile.get("trajectory_transfer_z", TRANSFER_TCP_Z))
        lift_height = float(profile.get("trajectory_lift_m", LIFT_H))
        retract_vel = float(profile.get("trajectory_retract_vel_scale", 0.5))
        transfer_vel = float(profile.get("trajectory_transfer_vel_scale", 0.35))
        pre_pos = grasp_pos - approach_axis * approach_m
        up = np.array([0.0, 0.0, lift_height])
        self.stages: list[Stage] = []
        self.error: str | None = None

        place_rot = grasp_rot if embodiment == "so101" else topdown_rotation(0.0)
        transfer_pos = np.array([tray_xy[0], tray_xy[1], transfer_height])
        lower_pos = np.array([tray_xy[0], tray_xy[1], place_z])
        # approach entirely in free space: rise vertically over the home
        # footprint, traverse at height, then descend — the raw joint-space
        # sweep from home crossed the shelf volume and clipped boxes (T08)
        home_tcp = fk_tcp(home, embodiment)
        staging_z = max(staging_height, float(pre_pos[2]))
        rise_pos = np.array([home_tcp[0], home_tcp[1], staging_z])
        staging_pos = np.array([pre_pos[0], pre_pos[1], staging_z])
        # a FRONT grasp holds its horizontal wrist only from the pregrasp
        # on; the high approach flies with the neutral top-down wrist (a
        # 0.5 m descent holding a horizontal wrist does not converge), and
        # the wrist flip happens in free air ahead of the shelf
        front_mode = abs(float(approach_axis[2])) < 0.5  # metadata carries the flag too
        approach_rot = (
            grasp_rot if embodiment == "so101" or not front_mode else topdown_rotation(0.0)
        )
        q_rise = ik_solve(rise_pos, approach_rot, home, embodiment)
        if q_rise is None:
            self.error = f"IK failed for waypoint 'rise' at {np.round(rise_pos, 3).tolist()}"
            return
        staging_path = ik_continuation(
            rise_pos, staging_pos, approach_rot, q_rise, embodiment=embodiment
        )
        if staging_path is None:
            self.error = f"IK failed for waypoint 'staging' at {np.round(staging_pos, 3).tolist()}"
            return
        q_staging = staging_path[-1]
        self.flip_pair: tuple | None = None
        if front_mode and embodiment == "so101":
            # The compact five-axis arm is designed for radial front
            # approaches. Keep the feasible horizontal pitch throughout
            # the free-space move; its base-coupled yaw follows each TCP
            # waypoint inside the SO-101 IK constraint.
            pregrasp_path = ik_continuation(
                staging_pos, pre_pos, grasp_rot, q_staging, embodiment=embodiment
            )
        elif front_mode:
            # descend most of the way neutral, then flip the wrist to
            # horizontal. Slerped orientation continuation does NOT
            # converge through the intermediate tilts here, so the flip
            # executes as ONE joint-interpolated move whose swept hand
            # path is explicitly VERIFIED to stay in the free half-space
            # ahead of the shelf (the PR #10 review measured the raw jump
            # at 2.18 rad; unverified it could sweep anywhere)
            drop_pos = np.array([pre_pos[0], pre_pos[1], min(staging_z, pre_pos[2] + 0.15)])
            drop_path = ik_continuation(
                staging_pos, drop_pos, approach_rot, q_staging, embodiment=embodiment
            )
            q_pre = (
                ik_solve(pre_pos, grasp_rot, drop_path[-1], embodiment)
                if drop_path is not None
                else None
            )
            if drop_path is not None and q_pre is not None:
                if np.abs(q_pre - drop_path[-1]).max() > FLIP_MAX:
                    self.error = "front flip jump exceeds FLIP_MAX (infeasible reorientation)"
                    return
                limit_x = float(pre_pos[0]) + 0.04  # 2 cm shy of the shelf front
                for f in np.linspace(0.0, 1.0, 21):
                    q_sweep = drop_path[-1] + f * (q_pre - drop_path[-1])
                    flange_pos, rotation = fk_ee_pose(q_sweep, embodiment)
                    tcp = fk_tcp(q_sweep, embodiment)
                    if tcp[0] > limit_x or flange_pos[0] > limit_x:
                        self.error = "front flip sweep enters the shelf half-space"
                        return
                self.flip_pair = (drop_path[-1], q_pre)
            pregrasp_path = drop_path + [q_pre] if q_pre is not None else None
        else:
            pregrasp_path = ik_continuation(
                staging_pos, pre_pos, grasp_rot, q_staging, embodiment=embodiment
            )
        if pregrasp_path is None:
            self.error = f"IK failed for waypoint 'pregrasp' at {np.round(pre_pos, 3).tolist()}"
            return
        q_pre = pregrasp_path[-1]
        # the insertion (advance/lift/retract) and placement descents are
        # continuation PATHS the executor follows waypoint by waypoint
        advance_path = ik_continuation(pre_pos, grasp_pos, grasp_rot, q_pre, embodiment=embodiment)
        if advance_path is None:
            self.error = "IK failed for stage 'advance'"
            return
        lift_path = (
            ik_continuation(
                grasp_pos,
                grasp_pos + up,
                grasp_rot,
                advance_path[-1],
                embodiment=embodiment,
            )
            if advance_path is not None
            else None
        )
        if lift_path is None:
            self.error = "IK failed for stage 'lift'"
            return
        retract_path = (
            ik_continuation(
                grasp_pos + up,
                pre_pos + up,
                grasp_rot,
                lift_path[-1],
                embodiment=embodiment,
            )
            if lift_path is not None
            else None
        )
        if retract_path is None:
            self.error = "IK failed for stage 'retract'"
            return
        # transfer as a CARTESIAN continuation, not a bare joint waypoint:
        # a joint-space swing leaves the TCP orientation unconstrained
        # mid-path — the wrist tilts, gravity torques the box about the
        # pinch line, and it creep-rotates flat before release (T10
        # telemetry, seed 3 omeprazole; slower swings made it WORSE
        # because the wrist spent longer tilted)
        if embodiment == "so101" and retract_path is not None:
            # A straight shelf->tray chord crosses a narrow singular basin
            # in the official five-axis chain.  Stay outside it with the
            # profile's measured free-space radial route: first rise at the
            # shelf-front clearance, then sweep around the base to the tray.
            route = [
                np.array([pre_pos[0], pre_pos[1], transfer_height]),
                *(
                    np.asarray(point, dtype=float)
                    for point in profile.get("trajectory_transfer_route", [])
                ),
                transfer_pos,
            ]
            transfer_path = []
            route_start = pre_pos + up
            route_q = retract_path[-1]
            for route_end in route:
                segment = ik_continuation(
                    route_start,
                    route_end,
                    place_rot,
                    route_q,
                    embodiment=embodiment,
                )
                if segment is None:
                    transfer_path = None
                    break
                transfer_path.extend(segment)
                route_start = route_end
                route_q = segment[-1]
        else:
            transfer_path = (
                ik_continuation(
                    pre_pos + up,
                    transfer_pos,
                    place_rot,
                    retract_path[-1],
                    embodiment=embodiment,
                )
                if retract_path is not None
                else None
            )
        if transfer_path is None:
            self.error = "IK failed for stage 'transfer'"
            return
        q_transfer = transfer_path[-1] if transfer_path is not None else None
        lower_path = (
            ik_continuation(
                transfer_pos,
                lower_pos,
                place_rot,
                q_transfer,
                embodiment=embodiment,
            )
            if q_transfer is not None
            else None
        )
        if lower_path is None:
            self.error = "IK failed for stage 'lower'"
            return
        # release opens the fingers STATIONARY at the hover pose: the old
        # open-while-rising release lifted the still-gripped box during
        # the ~1 s finger ramp and it slipped off raised with pendulum
        # energy — tall meds toppled on landing (offline 50-pair sweep;
        # the T08 shear concern applied to a SEATED box, and the box now
        # hovers PLACE_DROP_GAP above the tray instead)
        release_path = (lower_path[-1],) if lower_path is not None else None
        self.stages = [
            Stage("rise", (q_rise,), 0.0, 0.1),
            Stage("staging", tuple(staging_path), 0.0, 0.1),
            Stage("pregrasp", tuple(pregrasp_path), 0.0, 0.2),
            # Narrow the official 138 mm one-sided jaw in free space before
            # entering the shelf. The measured 0.60 command remains clear
            # of every half-scale carton; final contact begins near 0.75.
            Stage("preclose", (pregrasp_path[-1],), pregrasp_cmd, 0.2),
            Stage("advance", tuple(advance_path), pregrasp_cmd, 0.3),
            Stage("close", (advance_path[-1],), grasp_cmd, 0.5),
            Stage("lift", tuple(lift_path), grasp_cmd, 0.2, vel=0.5),
            Stage("retract", tuple(retract_path), grasp_cmd, 0.2, vel=retract_vel),
            # the transfer swing is where the box shifts in the grip:
            # carry it gently
            # vel 0.2: at 0.35 the long swing (far +y shelf to the tray)
            # whipped the box to ~45 degrees inside even a full-force
            # pinch, and pad friction held the tilt to release (T10
            # telemetry, seed 3 omeprazole)
            Stage("transfer", tuple(transfer_path), grasp_cmd, 0.3, vel=transfer_vel),
            # lower/release: tight tolerance + a real settle so the box
            # hovers converged and motionless before the fingers open —
            # releasing mid-motion or centimeters off grounds or tips it
            Stage("lower", tuple(lower_path), grasp_cmd, 1.0, vel=0.35, track_tol=0.03),
            # settle covers the full 1 s finger-open ramp plus box drop
            Stage("release", tuple(release_path), 0.0, 1.5, vel=0.35, track_tol=0.03),
            # rise clear of the tray walls before the home swing: the raw
            # release->home sweep dragged the fingers through the tray and
            # jammed the arm (T08 live run)
            Stage("clear", (q_transfer,), 0.0, 0.1),
            Stage("home", (home,), 0.0, 0.0),
        ]
        # continuity invariant over every consecutive waypoint pair of the
        # SHELF-PROXIMATE stages (rise..retract, incl. within-path steps
        # and the front-mode wrist flip): a branch flip there sweeps the
        # arm through the shelf. The transfer/home swings are deliberate
        # large free-space moves over open ground and are exempt.
        retract_index = next(i for i, stage in enumerate(self.stages) if stage.name == "retract")
        flat = [q for stage in self.stages[: retract_index + 1] for q in stage.path]
        for a, b in zip(flat, flat[1:], strict=False):
            if self.flip_pair is not None and a is self.flip_pair[0] and b is self.flip_pair[1]:
                continue  # the flip jump is sweep-verified above instead
            if np.abs(np.asarray(a) - np.asarray(b)).max() > CONTINUITY_MAX:
                self.error = "discontinuous waypoint chain"
                self.stages = []
                return

    @property
    def ok(self) -> bool:
        return not self.error


class StageStreamer:
    """The proven per-joint_state streaming step over a Stage list —
    waypoint marching, gravity-sag integral correction, grip ramp, settle/
    bail advancement. Extracted from main() unchanged so the S1 expert
    (T15) executes its split pick/place stage lists with the SAME battle-
    tested execution semantics. Pure (no dora): the caller sends.

    step(qpos) -> (full_joint_cmd | None, gripper_value | None, logs):
    joint_cmd is None once the list is finished; gripper_value is emitted
    only on ramp ticks; logs are stderr-worthy stage transitions."""

    def __init__(
        self,
        stages: list,
        home: np.ndarray,
        dt: float,
        max_vel: float,
        integ_cap: float = 0.15,
        embodiment: str = "franka",
        max_waypoints: int | None = None,
    ) -> None:
        # H6 F3 (ADR-h6-operation-protocol): with a cap, the executor
        # holds pose after marching that many waypoints and never
        # finishes; None (the default) is byte-identical pre-H6 behavior
        self.max_waypoints = max_waypoints
        self.marched = 0
        self.stages = list(stages)
        self.home = np.asarray(home, dtype=np.float32)
        self.limits = load_limits(embodiment)
        self.n_arm = self.limits.n_arm_dof
        self.q_min = self.limits.q_min_arr[: self.n_arm]
        self.q_max = self.limits.q_max_arr[: self.n_arm]
        self.dt = dt
        self.max_vel = max_vel
        # gravity-sag integral cap: 0.15 suits the desk poses; the store's
        # yawed long-reach advance sags ~0.27 raw on the wrist pitch (T15
        # round 12), so the S1 expert passes a higher cap
        self.integ_cap = integ_cap
        self.stage_idx = 0
        self.wp_idx = 0
        self.settle_ticks = 0
        self.at_target_ticks = 0
        self.diverge_ticks = 0
        self.current_cmd: np.ndarray | None = None
        self.integ = np.zeros(self.n_arm, dtype=np.float32)
        self.current_grip = 0.0
        self.grip_tick = 0

    @property
    def done(self) -> bool:
        return self.stage_idx >= len(self.stages)

    def step(self, qpos: np.ndarray) -> tuple[np.ndarray | None, float | None, list[str]]:
        if self.done:
            return None, None, []
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if self.current_cmd is None:
            self.current_cmd = qpos[: self.n_arm].copy()
        if self.max_waypoints is not None and self.marched >= self.max_waypoints:
            fingers = gripper_to_fingers(self.current_grip, self.limits).astype(np.float32)
            arm = np.clip(self.current_cmd + self.integ, self.q_min, self.q_max)
            return np.concatenate([arm, fingers]).astype(np.float32), None, []
        logs: list[str] = []
        stage = self.stages[self.stage_idx]
        # ramp the gripper (unit-tested via grip_ramp_tick)
        self.current_grip, self.grip_tick, emit = grip_ramp_tick(
            self.current_grip, stage.gripper, self.grip_tick
        )
        grip_out = self.current_grip if emit else None
        # march the stage's waypoint chain: track the PLANNED Cartesian
        # path, not a straight joint-space line between stage endpoints
        waypoint = stage.path[self.wp_idx]
        self.current_cmd = interpolate_step(
            self.current_cmd, waypoint, self.max_vel * stage.vel, self.dt
        )
        if self.wp_idx < len(stage.path) - 1 and np.abs(self.current_cmd - waypoint).max() < 1e-6:
            self.wp_idx += 1
            self.marched += 1
        # integral correction: the MJCF actuators sag ~0.08 rad under
        # gravity (their gains are baked into the asset) — integrate the
        # tracking error into the COMMAND so the sim settles on target
        self.integ = np.clip(
            self.integ + 0.004 * (self.current_cmd - qpos[: self.n_arm]),
            -self.integ_cap,
            self.integ_cap,
        )
        corrected = np.clip(self.current_cmd + self.integ, self.q_min, self.q_max).astype(
            np.float32
        )
        # finger targets FOLLOW the stage's gripper intent (BRG-1 last-wins)
        fingers = gripper_to_fingers(self.current_grip, self.limits).astype(np.float32)
        full_cmd = np.concatenate([corrected, fingers]).astype(np.float32)
        # stage completion: command at target AND sim tracked within
        # tolerance; a bounded at-target dwell advances anyway so a
        # contact-blocked joint cannot stall the plan forever
        if np.abs(self.current_cmd - stage.q).max() < 1e-6 and self.current_grip == stage.gripper:
            self.at_target_ticks += 1
            track_err = np.abs(qpos[: self.n_arm] - stage.q)
            tracked = track_err.max() < stage.track_tol
            if tracked:
                self.settle_ticks += 1
            if track_err.max() > DIVERGE_TOL:
                self.diverge_ticks += 1
            else:
                self.diverge_ticks = 0
            if (
                self.settle_ticks * self.dt >= stage.settle_s
                or self.at_target_ticks * self.dt >= STAGE_BAIL_S
                or self.diverge_ticks * self.dt >= DIVERGE_BAIL_S
            ):
                if not tracked:
                    kind = (
                        "diverged" if self.diverge_ticks * self.dt >= DIVERGE_BAIL_S else "bailed"
                    )
                    logs.append(
                        f"stage {stage.name} {kind} at joint {int(track_err.argmax())} "
                        f"err {float(track_err.max()):.3f}"
                    )
                    # T2 registration seeds 10/15: after a contact-blocked
                    # bail the command sits at a stage target the arm never
                    # reached, and the NEXT stage would interpolate from
                    # that phantom — dragging the pressed arm across the
                    # shelf (the measured transit-collision class). Hand
                    # the next stage the arm's ACTUAL position; tracked
                    # completions keep the command lead (gravity-sag
                    # design) untouched.
                    self.current_cmd = qpos[: self.n_arm].copy()
                    self.integ[:] = 0.0
                logs.append(f"stage done: {stage.name}")
                self.stage_idx += 1
                self.marched += 1
                self.wp_idx = 0
                self.settle_ticks = 0
                self.at_target_ticks = 0
                self.diverge_ticks = 0
        return full_cmd, grip_out, logs


def main() -> None:
    import json
    import os
    import sys

    import pyarrow as pa

    from aisle.nodes.h6_fault import armed_fault, plan_waypoint_cap
    from aisle.scenes.pharmacy import load_physics, resolve_layout, wrist_mount_transform
    from aisle.topics import env_accepts, env_pin_from_env, make_sender
    from aisle.turn_node import Node

    h6_fault = armed_fault("ik-trajectory")
    embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
    physics = load_physics()
    layout = resolve_layout(physics, embodiment)
    tray_pos = layout["tray"]["pos"]
    tray_xy = (float(tray_pos[0]), float(tray_pos[1]))
    home = np.asarray(physics["embodiment"][embodiment]["home_qpos"], dtype=np.float32)
    max_vel = float(os.environ.get("AISLE_MAX_JOINT_VEL", "1.0"))
    dt = 0.01  # joint_state contract cadence (TC-4)
    mount = wrist_mount_transform(physics["cameras"], physics["embodiment"][embodiment]).astype(
        np.float64
    )

    env_pin = env_pin_from_env(os.environ)
    node = Node()
    send = make_sender(node, env_pin)
    streamer: StageStreamer | None = None
    # while a read move runs: {request_id, face, attempts: [(q, range_m)],
    # attempt_idx} — the executor walks the ladder until one attempt TRACKS
    pending_read: dict | None = None
    parked_at_read = False  # the arm sits at a shelf-front read pose

    def start_read_attempt(pending: dict) -> None:
        """Arm the streamer for the pending read's current ladder attempt.
        EVERY read move routes via home: a direct joint-space interpolation
        between two shelf-front poses sweeps the open fingers laterally
        through neighbour boxes at ~13 cm range (first live T2 episode:
        verdict `collision`)."""
        nonlocal streamer
        q_read, range_m, pitch, q_far = pending["attempts"][pending["attempt_idx"]]
        print(
            f"read attempt {pending['attempt_idx']}: range {range_m} pitch {pitch} "
            f"face {pending['face'].tolist()}",
            file=sys.stderr,
        )
        home_arm = home[: q_read.shape[0]].astype(np.float32)
        stages = [
            Stage(name="read-retreat", path=(home_arm,), gripper=0.0, settle_s=0.1),
            # staged approach (T2 curve run 1): park backed-off first so
            # the joint hop stays clear of the shelf, then advance —
            # solve_read_poses drops any entry without a staged pose
            Stage(name="read-stage", path=(q_far.astype(np.float32),), gripper=0.0, settle_s=0.1),
            Stage(name="read", path=(q_read.astype(np.float32),), gripper=0.0, settle_s=0.4),
        ]
        streamer = StageStreamer(stages, home, dt, max_vel, embodiment=embodiment)

    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        if not env_accepts(metadata, env_pin):
            continue  # fleet mode (BRG-5): another env's stream
        if event["id"] == "reset_done":
            # episode boundary: NEVER keep executing a stale plan — in the
            # first live run a stale stream fought the post-reset guard
            # reference until the wall timeout froze everything
            streamer = None
            pending_read = None
            parked_at_read = False
        elif event["id"] == "read_move":
            if streamer is not None and not streamer.done:
                print("read_move refused: a plan is executing", file=sys.stderr)
                continue
            request = json.loads(event["value"][0].as_py())
            face = np.asarray(request["face"], dtype=np.float64).reshape(-1)[:3]
            # a refused READ retries the same candidate from the next
            # ladder entry: the state machine passes back attempt_used+1
            offset = int(request.get("attempt_offset", 0))
            attempts = solve_read_poses(face, mount, home[: len(home) - 2].astype(np.float64))
            if request.get("flat_only"):
                # pitched-refusal retry (t2-scan-tsm): walk only the flat
                # entries, whose margin floor (0.04) is the trustworthy one
                attempts = [a for a in attempts if a[2] == 0.0]
            if offset >= len(attempts):
                # ladder exhausted (or unreachable face / non-franka):
                # REPLY with failure so the tour advances instead of
                # hanging on a silent drop
                send(
                    "move_done",
                    pa.array([json.dumps({"ok": False})]),
                    {"request_id": metadata.get("request_id", "")},
                )
                print(f"read_move exhausted: face {face.tolist()}", file=sys.stderr)
                continue
            pending_read = {
                "request_id": metadata.get("request_id", ""),
                "face": face,
                "attempts": attempts[: offset + READ_MAX_ATTEMPTS],
                "attempt_idx": offset,
            }
            start_read_attempt(pending_read)
        elif event["id"] == "grasp_pose":
            if streamer is not None and not streamer.done:
                continue  # one plan at a time; re-plan only after home
            # T4 inc-2 (return_item): a plan whose metadata carries
            # place_xy delivers to THAT destination instead of the tray
            # -- the return executor sends tray-pick plans placing at a
            # shelf slot; everything else about the plan is identical
            dest_xy = tray_xy
            if metadata.get("place_xy") is not None:
                px, py = metadata["place_xy"]
                dest_xy = (float(px), float(py))
            candidate = StagedPlan(
                event["value"].to_numpy(zero_copy_only=False),
                dest_xy,
                float(metadata.get("approach_m", 0.15)),
                home,
                place_z=float(metadata.get("place_tcp_z", PLACE_TCP_Z)),
                embodiment=embodiment,
            )
            if not candidate.ok:
                print(f"grasp plan failed: {candidate.error}", file=sys.stderr)
                continue
            stages = list(candidate.stages)
            if parked_at_read:
                # the arm is parked at a shelf-front read pose; going
                # straight to pregrasp drags the fingers through the
                # shelf. T0/T1 never park (flag never set): byte-identical
                # behavior on the frozen tiers.
                n_arm = stages[0].path[0].shape[0]
                stages.insert(
                    0,
                    Stage(
                        name="grasp-retreat",
                        path=(home[:n_arm].astype(np.float32),),
                        gripper=0.0,
                        settle_s=0.1,
                    ),
                )
                parked_at_read = False
            streamer = StageStreamer(
                stages,
                home,
                dt,
                max_vel,
                embodiment=embodiment,
                max_waypoints=plan_waypoint_cap(stages) if h6_fault == "traj_short" else None,
            )
            print(f"plan ready: {len(stages)} stages", file=sys.stderr)
        elif event["id"] == "joint_state" and streamer is not None and not streamer.done:
            qpos = np.asarray(
                event["value"].to_numpy(zero_copy_only=False), dtype=np.float32
            ).reshape(-1)
            full_cmd, grip_out, logs = streamer.step(qpos)
            if grip_out is not None:
                send("gripper_cmd", pa.array(np.array([grip_out], dtype=np.float32)), metadata)
            if full_cmd is not None:
                send("joint_cmd", pa.array(full_cmd), metadata)
            for line in logs:
                print(line, file=sys.stderr)
            if streamer.done:
                streamer = None  # finished: idle until the next episode
                if pending_read is None:
                    # a GRASP plan completed (read parks reply on
                    # move_done): tell the state machine so it can arm
                    # an in-context retry if no verdict lands (HAR-3)
                    send("plan_done", pa.array([json.dumps({"ok": True})]), metadata)
                if pending_read is not None:
                    q_read, range_m, pitch, _ = pending_read["attempts"][
                        pending_read["attempt_idx"]
                    ]
                    track_err = float(np.abs(qpos[: q_read.shape[0]] - q_read).max())
                    if track_err > READ_TRACK_TOL:
                        # IK-feasible but untrackable (shelf contact, 0.45
                        # rad terminal error in the first live tour): walk
                        # the ladder instead of reading a wild view
                        pending_read["attempt_idx"] += 1
                        if pending_read["attempt_idx"] < len(pending_read["attempts"]):
                            print(
                                f"read pose untracked (err {track_err:.2f}); "
                                f"ladder attempt {pending_read['attempt_idx']}",
                                file=sys.stderr,
                            )
                            start_read_attempt(pending_read)
                            continue
                        send(
                            "move_done",
                            pa.array([json.dumps({"ok": False})]),
                            {"request_id": pending_read["request_id"]},
                        )
                        print(f"read_move untrackable (err {track_err:.2f})", file=sys.stderr)
                        pending_read = None
                        continue
                    parked_at_read = True
                    cam_pos, cam_rot_cv = wrist_camera_pose(
                        qpos[: q_read.shape[0]].astype(np.float64), mount
                    )
                    payload = {
                        "ok": True,
                        "range_m": range_m,
                        "attempt_used": pending_read["attempt_idx"],
                        # a pitched view carries a wrong-read hazard the
                        # reader must guard with a higher margin floor
                        # (measured: amoxicillin-at-pitch reads a
                        # neighbour at +0.093)
                        "pitched": pitch > 0.0,
                        "face": [float(v) for v in pending_read["face"]],
                        "cam_pos": [float(v) for v in cam_pos],
                        "cam_rot_cv": [float(v) for v in cam_rot_cv.reshape(-1)],
                    }
                    # The reader must choose a frame by SIM order, not
                    # whichever queued RGB event wins a wall-clock race with
                    # this six-hop read dialogue (CON-5/TC-2). Strictly-newer
                    # eligibility also guarantees one rendered tick after the
                    # terminal tracked pose.
                    #
                    # Defaulting a missing stamp to 0 would arm a barrier that
                    # EVERY stamped frame clears, including frames captured
                    # during the park motion. park_read_reply instead turns an
                    # unusable stamp into an explicit failed move so the tour
                    # advances without ever entering an unbarriered read.
                    payload = park_read_reply(payload, metadata)
                    if not payload["ok"]:
                        print(
                            "read park refused: no usable sim stamp for "
                            f"{pending_read['request_id']}",
                            file=sys.stderr,
                        )
                    send(
                        "move_done",
                        pa.array([json.dumps(payload)]),
                        {"request_id": pending_read["request_id"]},
                    )
                    pending_read = None


if __name__ == "__main__":
    main()
