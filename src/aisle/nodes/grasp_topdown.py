"""grasp-planner-topdown node (CAP-5): target_pose -> grasp_pose.

Two approach modes, chosen by the shelf geometry (ADR-10):
- TOP level (no board above): classic top-down — fingers descend onto the
  box's top section, yaw across the narrower horizontal axis.
- LOWER levels: a board above makes any from-above descent collide (the
  hand column crosses the board; proven in the T08 live runs), so the
  grasp is a FRONT approach: wrist horizontal, the hand slides into the
  inter-board gap from the shelf front, fingers straddling the box sides.
The approach DISTANCE rides in grasp_pose metadata (approach_m) so
ik-trajectory can start its pregrasp clear of the shelf front. Med
dimensions come from meds.toml via the target_med metadata that
oracle-pose forwards.
"""

from __future__ import annotations

import math

import numpy as np

from aisle.nodes.ik_trajectory import (
    HAND_MOUNT_YAW,
    rotation_to_quat,
    so101_radial_rotation,
)
from aisle.scenes.pharmacy import level_x_span

# how far the fingertips engage below the box TOP (top-down mode). 0.045
# assumed 5 mm palm clearance from 0.05-long fingers, but the REAL
# fingertip-to-palm distance is shorter: the palm plate pressed on the
# box top, shoving it sideways during the descent, spinning near-square
# meds into diagonal detents at close, and ratcheting a pitch tilt
# through the carry that toppled tall meds at release (T10 renders of
# the cetirizine grasp; the "shallow grips pitch" note from T08 was this
# same palm contact misattributed). 0.035 keeps a genuinely clear palm;
# the box hangs pendulum-stable from the deeper-set centroid.
GRIP_ENGAGEMENT = 0.035
# gripper geometry (finger half-open along the grip axis, and the clearance
# the open fingers need around the gripped face) is per-embodiment config
# in physics.toml (gripper_open_m / gripper_finger_clear_m), passed into
# plan_grasp from main() — never an inline planner constant (SCN-2).
# clearance between the shelf front plane and the front-mode pregrasp TCP
FRONT_CLEARANCE = 0.06
# the tray is a flat slab (no walls): release the box from a drop gap
# above it — pressing it down drives it THROUGH the slab (T08 replays).
# release hover: small enough that the drop cannot tip a tall med,
# large enough that lower-stage tracking error cannot ground the box
# while gripped (ik-trajectory lower/release track_tol 0.03 + settle
# bound the error well under 0.02). With the palm clear
# (GRIP_ENGAGEMENT) and the soft close (grip_close_for), the box stays
# axis-aligned in the grip, so the rising clear stage passes the
# still-flanking fingertips without touching it.
PLACE_DROP_GAP = 0.02
# the wrist's radius below the flange axis: inserting at box-center height
# scraped the wrist on the board's front edge (T08 live run 6), so the
# front grasp rides high enough for the wrist to clear the board top
WRIST_CLEARANCE = 0.065
# the fingers must keep at least this much box below the grasp line
MIN_FINGER_ON_BOX = 0.015


# front-approach flange orientation: the intended HAND pose is Ry(pi/2)
# (flange z horizontal +x into the shelf, finger travel horizontal), and
# the flange target composes away the local hand mount —
# Ry(pi/2) @ Rz(-HAND_MOUNT_YAW) (issue #92 follow-up: the bare Ry(pi/2)
# executed the fingers 45 degrees DIAGONAL in the y-z plane, the
# front-mode twin of the m0 seed-3 top-down bug; correction sign
# verified in Genesis — tilt -45.1 -> -0.1 deg. Physical gate:
# tests/sim/test_hand_mount.py).
_FRONT_BASE = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)  # xyzw Ry(pi/2)


def _qz(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def needs_front(box_x: float, box_z: float, shelf: dict) -> bool:
    """Safety net for out-of-band poses (ADR-12): the sampler's open bands
    guarantee sky above every SAMPLED box, so this fires only for drifted
    or nudged poses. Top-down is unsafe when the box sits below a higher
    board within that board's span PLUS the hand_clearance_m strip the
    sampler reserves — the descending hand column clips the board's front
    edge there (T10 physics replay) — with less than hand_column_m of
    vertical clearance."""
    board_half = shelf["board_thickness"] / 2
    for level, height in enumerate(shelf["level_heights"]):
        board_bottom = shelf["pos"][2] + height - board_half
        if (
            box_z < board_bottom
            and box_x >= level_x_span(shelf, level)[0] - shelf["hand_clearance_m"]
            and (board_bottom - box_z) < shelf["hand_column_m"]
        ):
            return True
    return False


def _quat_mul(a, b) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# composed after _quat_mul is defined; the local (right-hand) factor
# spins about the flange z so the hand lands on the intended pose
FRONT_QUAT = _quat_mul(_FRONT_BASE, _qz(-HAND_MOUNT_YAW))


def yaw_of(quat_xyzw) -> float:
    """Yaw (rotation about world z) of a pose quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def topdown_quat(yaw: float) -> tuple[float, float, float, float]:
    """qz(yaw + HAND_MOUNT_YAW) * qx(pi): flange z pointing straight down
    with the FINGER axis at `yaw` (TC-1 xyzw) — the flange target carries
    the hand-mount offset (issue #92, see HAND_MOUNT_YAW)."""
    flange_yaw = yaw + HAND_MOUNT_YAW
    qz = (0.0, 0.0, math.sin(flange_yaw / 2), math.cos(flange_yaw / 2))
    qx = (1.0, 0.0, 0.0, 0.0)
    return _quat_mul(qz, qx)


def finger_yaw_of(quat_xyzw) -> float:
    """The PHYSICAL finger-straddle yaw a flange quaternion realizes —
    yaw_of minus the hand-mount offset."""
    return yaw_of(quat_xyzw) - HAND_MOUNT_YAW


def _fingertip_clearance(box_xy, u, size_xyz, neighbours, finger_open) -> float:
    """Min horizontal clearance from the OPEN fingertips (box centre +/-
    finger_open along grip-axis unit vector u) to any neighbour's AABB.
    Large => the fingers sweep clear of same-level boxes."""
    if not neighbours:
        return math.inf
    tips = (
        (box_xy[0] + finger_open * u[0], box_xy[1] + finger_open * u[1]),
        (box_xy[0] - finger_open * u[0], box_xy[1] - finger_open * u[1]),
    )
    best = math.inf
    for nx, ny, nhx, nhy in neighbours:
        for tx, ty in tips:
            dx = max(abs(tx - nx) - nhx, 0.0)
            dy = max(abs(ty - ny) - nhy, 0.0)
            best = min(best, math.hypot(dx, dy))
    return best


def plan_grasp(
    target_pose: np.ndarray,
    size_xyz,
    grip: float = GRIP_ENGAGEMENT,
    front: bool = False,
    shelf_front_x: float = 0.0,
    *,
    tray_top_z: float,
    neighbours: list | None = None,
    finger_open: float | None = None,
    finger_clear: float | None = None,
    topdown_approach: float = 0.15,
    radial_front: bool = False,
    front_clearance: float = FRONT_CLEARANCE,
    front_tcp_overshoot: float = 0.0,
    front_jaw_center_offset: float = 0.0,
    front_vertical_offset: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Pure plan: (grasp_pose7, approach_m).

    Top-down: TCP at the box's top section, fingers across the narrower
    horizontal axis by default, but rotated 90 degrees when that keeps the
    open fingers clear of a same-level neighbour (the default sweep grazed
    a box ~3 cm away and the live pipeline clipped it; t10-m0-full seed 8).
    Neighbour-aware selection runs only when the caller supplies the
    gripper geometry (finger_open, finger_clear from physics.toml);
    otherwise the legacy narrow-axis grip stands. Front: TCP at the box
    center, wrist horizontal, approach from shelf_front_x minus clearance."""
    pose = np.asarray(target_pose, dtype=np.float32).reshape(7)
    if front:
        half_z = float(size_xyz[2]) / 2
        bottom, top = float(pose[2]) - half_z, float(pose[2]) + half_z
        # ride high: wrist over the board edge, fingers still on the box
        z = (
            float(pose[2]) + float(front_vertical_offset)
            if radial_front
            else min(max(float(pose[2]), bottom + WRIST_CLEARANCE), top - MIN_FINGER_ON_BOX)
        )
        if radial_front:
            base_rotation = so101_radial_rotation(pose[:3])
            jaw_sign = 1.0
            if neighbours:
                # The official jaw is one-sided: the fixed fingertip extends
                # on frame -X while the moving jaw retracts from +X. Choose
                # the 180-degree wrist-roll parity whose fixed-finger side
                # has the larger empty corridor. This leaves the radial tool
                # axis unchanged and is deterministic on ties.
                target_xy = pose[:2].astype(float)

                def fixed_side_clearance(sign: float) -> float:
                    fixed_dir = -sign * base_rotation[:2, 0]
                    clearances = []
                    for nx, ny, nhx, nhy in neighbours:
                        delta = np.array([nx, ny], dtype=float) - target_xy
                        projected = float(np.dot(delta, fixed_dir))
                        if projected <= 0:
                            continue
                        clearances.append(
                            projected
                            - abs(float(fixed_dir[0])) * float(nhx)
                            - abs(float(fixed_dir[1])) * float(nhy)
                        )
                    return min(clearances, default=math.inf)

                if fixed_side_clearance(-1.0) > fixed_side_clearance(1.0):
                    jaw_sign = -1.0
            rotation = so101_radial_rotation(pose[:3], jaw_sign)
            tcp_xy = (
                pose[:2]
                + rotation[:2, 2] * float(front_tcp_overshoot)
                + rotation[:2, 0] * float(front_jaw_center_offset)
            )
            # Re-anchor the feasible radial frame at the actual offset TCP.
            rotation = so101_radial_rotation([tcp_xy[0], tcp_xy[1], z], jaw_sign)
            quat = rotation_to_quat(rotation)
            radial_x = max(float(rotation[0, 2]), 1e-6)
            approach = (float(tcp_xy[0]) - (shelf_front_x - front_clearance)) / radial_x
        else:
            quat = FRONT_QUAT
            tcp_xy = pose[:2]
            approach = float(pose[0]) - (shelf_front_x - front_clearance)
        grasp = np.array([tcp_xy[0], tcp_xy[1], z, *quat], dtype=np.float32)
        return grasp, approach, place_tcp_z(size_xyz, top - z, tray_top_z)
    legacy_yaw = yaw_of(pose[3:7]) + (math.pi / 2 if size_xyz[0] < size_xyz[1] else 0.0)
    if neighbours and finger_open is not None and finger_clear is not None:
        base = yaw_of(pose[3:7])
        box_xy = (float(pose[0]), float(pose[1]))
        # candidate grip yaws: straddle box-y (fingers protrude along box-y)
        # or box-x. A candidate is feasible only if the open fingers clear
        # the gripped face (elongated meds can only grip their narrow axis).
        candidates = []
        for straddle_axis, yaw in ((1, base), (0, base + math.pi / 2)):
            if float(size_xyz[straddle_axis]) / 2 > finger_open - finger_clear:
                continue
            u = (-math.sin(yaw), math.cos(yaw))  # world dir the fingers protrude along
            clearance = _fingertip_clearance(box_xy, u, size_xyz, neighbours, finger_open)
            narrow = size_xyz[straddle_axis] <= size_xyz[1 - straddle_axis]
            candidates.append((clearance, narrow, yaw))
        # prefer clearance; on a tie prefer the narrower (more stable) grip
        yaw = max(candidates, key=lambda c: (round(c[0], 3), c[1]))[2] if candidates else legacy_yaw
    else:
        yaw = legacy_yaw
    z = float(pose[2]) + float(size_xyz[2]) / 2 - grip
    grasp = np.array([pose[0], pose[1], z, *topdown_quat(yaw)], dtype=np.float32)
    return grasp, float(topdown_approach), place_tcp_z(size_xyz, grip, tray_top_z)


def place_tcp_z(size_xyz, grip_from_top: float, tray_top_z: float) -> float:
    """Release TCP height: box bottom hovers PLACE_DROP_GAP above the tray
    slab when the TCP is here (the box hangs (size_z - grip_from_top)
    below the TCP). The slab height comes from the embodiment's layout
    (SCN-2: scene constants live in physics.toml, and so101's tray
    differs)."""
    return float(tray_top_z) + (float(size_xyz[2]) - float(grip_from_top)) + PLACE_DROP_GAP


def neighbour_constraints(centres: list, target_med: str, med_names: list, meds: dict) -> list:
    """(x, y, half_x, half_y) grip-axis constraints from a `neighbours`
    payload: every box except the target.

    The payload is POSITIONAL — one row per med in scene-manifest order, so
    `strict=True` pins producer drift (a short list is a producer bug, not a
    plan input). A row of None is a neighbour the L1 estimator REFUSED to
    place (mask under the occlusion floor, TC-9) and is omitted from the
    constraints — fewer constraints means a permissive plan, which is why
    the refusal count travels in the metadata — never unpacked as a centre:
    at L0 every row is a real centre and this reduces to the old parse."""
    return [
        [row[0], row[1], meds[name]["size"][0] / 2, meds[name]["size"][1] / 2]
        for name, row in zip(med_names, centres, strict=True)
        if name != target_med and row is not None
    ]


#: G1-E/F2: g1_right plans top-down on its own physics profile (gripper
#: geometry and pregrasp height from physics.toml, TCP convention per C1);
#: any other unknown embodiment is still refused by name before the
#: profile lookup (HAR-1).
def require_supported_embodiment(embodiment: str) -> None:
    if embodiment not in ("franka", "mobile", "so101", "g1_right"):
        raise ValueError(f"require_supported_embodiment: unsupported embodiment {embodiment!r}")


def main() -> None:
    import json
    import os
    import sys

    import pyarrow as pa

    from aisle.nodes.h6_fault import armed_fault, raise_grasp
    from aisle.scenes.pharmacy import MED_NAMES, load_meds, load_physics, resolve_layout
    from aisle.topics import env_accepts, env_pin_from_env, make_sender
    from aisle.turn_node import Node

    h6_fault = armed_fault("grasp-planner-topdown")
    embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
    require_supported_embodiment(embodiment)
    physics = load_physics()
    layout = resolve_layout(physics, embodiment)
    meds = load_meds()
    shelf = layout["shelf"]
    shelf_front_x = shelf["pos"][0] - shelf["level_size"][0] / 2
    tray = layout["tray"]
    tray_top_z = tray["pos"][2] + tray["size"][2] / 2
    # SCN-2: the gripper geometry (finger half-open along the grip axis and
    # the clearance the open fingers need) is physics.toml config, not a
    # planner constant
    profile = physics["embodiment"][embodiment]
    finger_open = float(profile["gripper_open_m"])
    finger_clear = float(profile["gripper_finger_clear_m"])
    topdown_approach = float(profile.get("pregrasp_height_m", physics["ik"]["pregrasp_height_m"]))
    env_pin = env_pin_from_env(os.environ)
    node = Node()
    send = make_sender(node, env_pin)
    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        if not env_accepts(metadata, env_pin):
            continue  # fleet mode (BRG-5): another env's stream
        if event["id"] == "target_pose":
            med = metadata.get("target_med")
            if med not in meds:
                print(f"target_pose without a known target_med ({med!r})", file=sys.stderr)
                continue
            pose = event["value"].to_numpy(zero_copy_only=False)
            flat = np.asarray(pose).reshape(-1)
            front = bool(profile.get("force_front_grasp", False)) or needs_front(
                float(flat[0]), float(flat[2]), shelf
            )
            # same-level neighbours (x, y, half_x, half_y) for grip-axis
            # selection — every box except the target
            neighbours = None
            if "neighbours" in metadata:
                neighbours = neighbour_constraints(
                    json.loads(metadata["neighbours"]), med, MED_NAMES, meds
                )
            grasp, approach, place_z = plan_grasp(
                pose,
                meds[med]["size"],
                front=front,
                shelf_front_x=shelf_front_x,
                tray_top_z=tray_top_z,
                neighbours=neighbours,
                finger_open=finger_open,
                finger_clear=finger_clear,
                topdown_approach=topdown_approach,
                radial_front=bool(profile.get("force_front_grasp", False)),
                front_clearance=float(profile.get("front_clearance_m", FRONT_CLEARANCE)),
                front_tcp_overshoot=float(profile.get("front_tcp_overshoot_m", 0.0)),
                front_jaw_center_offset=float(profile.get("front_jaw_center_offset_m", 0.0)),
                front_vertical_offset=float(profile.get("front_vertical_offset_m", 0.0)),
            )
            if h6_fault == "grasp_high":
                grasp = raise_grasp(grasp)
            send(
                "grasp_pose",
                pa.array(grasp),
                {**metadata, "approach_m": approach, "front": front, "place_tcp_z": place_z},
            )


if __name__ == "__main__":
    main()
