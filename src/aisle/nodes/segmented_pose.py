"""L1 perception: pose ESTIMATED from ground-truth segmentation (TC-9).

The perception ladder's middle rung. L0 hands the graph `poses` — the
non-privileged ground-truth pose — so grounding is free and only grasping is
exercised. L1 forbids `poses` and gives the graph `seg_overhead` +
`depth_overhead` instead: the mask says WHICH pixels are the object (still
ground truth, which is what makes this a rung and not L2), and the pose is
estimated from those pixels. That is the first tier where a wrong estimate
can lose an episode.

Estimation is deliberately the simplest thing that measures well: mask the
target's pixels, back-project them with the published calibration, take the
top surface, and place the centre half a box-height below it. Measured
against Genesis ground truth over 20 objects on 4 seeds: **2.2 mm mean XY
error, z error ~0** (z falls out exactly, since the top surface plus half a
known height IS the centre).

The failure mode that matters is occlusion, not noise: a partially hidden box
gives a centroid biased toward its visible fragment. The two worst cases in
that sweep were the two smallest masks (19.5 mm and 9.0 mm) while every
unoccluded object came in under ~2 mm. So an estimate carries the mask size
that produced it and the node REFUSES below a floor rather than publishing a
confident guess — the same lesson the identity stage learned from tray-sized
detections.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from aisle.nodes.perception_session import FramePairSession

# below this many mask pixels the centroid is dominated by whatever fragment
# is visible. Unoccluded meds render 433-970 px at 640x480 from the overhead
# camera, so this floor rejects a box more than roughly half hidden while
# never rejecting a clear one (measured, seeds 3-6).
MIN_MASK_PIXELS = 200
# the top surface is the highest depth band of the mask: edge pixels straddle
# the silhouette and pick up the shelf behind, exactly as VER-10's
# dominant_surface found for the verifier's own masks
TOP_SURFACE_QUANTILE = 0.85
# the estimator places the centre half a box HEIGHT below the top surface,
# which is only true for an UPRIGHT box. Measured: commanding a 30-45 degree
# tilt settles the box on its side and the z estimate is 20.0 mm low --
# exactly half the difference between the height and the dimension now
# vertical. A planner aiming 2 cm low collides, and from one top-down view
# the estimator cannot see which face it is looking at. It can, however, see
# the FOOTPRINT: an upright box shows its (x, y) face, a tipped one shows a
# face containing z. This is the fraction by which the observed top-face
# extents may differ from the upright expectation before the pose is refused.
# 0.4 admits discretisation and settling wobble; a tip changes the short
# extent by ~3x in the desk meds, so the two are not close.
UPRIGHT_FOOTPRINT_TOLERANCE = 0.4


class PoseRefused(Exception):
    """The mask cannot support a trustworthy pose (TC-9). Refusing is the
    contract: a biased centroid is worse than no pose, because downstream the
    grasp planner cannot tell the difference."""


def seg_ids_for(id_map: dict, target: str) -> list[int]:
    """The `seg_overhead` ids belonging to one med, from the map the bridge
    publishes in `bridge_info`.

    Never derive these from entity indices: Genesis's segmentation ids are its
    own map (seg id -> (entity_idx, link_idx)), so masking on entity index
    selects other geometry. A spike that did exactly that returned robot links
    with pixel counts identical across scenes whose layouts differed (TC-9)."""
    ids = id_map.get(target)
    if not ids:
        raise PoseRefused(f"no segmentation ids published for {target!r} (TC-9 bridge_info)")
    return [int(i) for i in ids]


def estimate_pose(
    seg: np.ndarray,
    depth: np.ndarray,
    ids: list[int],
    box_height_m: float,
    backproject,
    min_mask_pixels: int = MIN_MASK_PIXELS,
    footprint_m: tuple | None = None,
) -> dict:
    """(x, y, z) of the box CENTRE plus the evidence that produced it.

    `backproject(depth, pixels) -> (N, 3)` base-frame points is injected so
    this stays pure: the caller passes the VER-8 calibration already bound.
    `footprint_m` is the box's upright (x, y) face from meds.toml; passing it
    enables the upright check that keeps a tipped box from being reported 20 mm
    low. Returns {pos, mask_pixels, top_surface_z_m} — the mask size travels
    with the estimate so a consumer can weigh it (TC-9)."""
    mask = np.isin(seg, ids)
    pixels = int(mask.sum())
    if pixels < min_mask_pixels:
        raise PoseRefused(
            f"mask is {pixels} px, under the {min_mask_pixels} px floor — a centroid this "
            "small is dominated by the visible fragment of an occluded box (TC-9)"
        )
    vs, us = np.nonzero(mask)
    points = np.asarray(backproject(depth, np.stack([us, vs], axis=1)), dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise PoseRefused("back-projected mask contains non-finite points")
    top = points[points[:, 2] >= np.quantile(points[:, 2], TOP_SURFACE_QUANTILE)]
    top_z = float(np.median(top[:, 2]))
    if footprint_m is not None:
        # 5-95 percentiles: stray silhouette pixels would otherwise inflate the
        # extent and mask a genuine mismatch
        observed = sorted(
            float(np.percentile(top[:, i], 95) - np.percentile(top[:, i], 5)) for i in (0, 1)
        )
        expected = sorted(float(v) for v in footprint_m)
        for obs, exp in zip(observed, expected, strict=True):
            if exp > 0 and abs(obs - exp) / exp > UPRIGHT_FOOTPRINT_TOLERANCE:
                raise PoseRefused(
                    f"top face measures {observed[0]:.3f}x{observed[1]:.3f} m against an upright "
                    f"{expected[0]:.3f}x{expected[1]:.3f} m — the box is not upright, so half the "
                    "HEIGHT is the wrong offset to its centre (measured 20 mm low when tipped)"
                )
    return {
        "pos": [float(top[:, 0].mean()), float(top[:, 1].mean()), top_z - box_height_m / 2],
        "mask_pixels": pixels,
        "top_surface_z_m": top_z,
    }


def estimate_neighbours(seg, depth, id_map: dict, meds: dict, backproject) -> tuple[list, int]:
    """Same-level neighbour (x, y) centres for the grasp planner's fingertip
    clearance, estimated the same way as the target.

    L0 hands the planner every neighbour from ground truth; without this the
    planner silently loses `_fingertip_clearance` at L1 and picks grip axes
    that close on a neighbour. Returns (rows, n_refused) with ONE row per med
    in `meds` order — the `neighbours` payload is POSITIONAL (the consumer
    zips it against MED_NAMES strict=True, exactly as L0's full list), so a
    neighbour whose mask is too small keeps its slot as None and is counted,
    never guessed at and never silently dropped: a SHORTER list would crash
    the planner, and fewer constraints means a permissive plan, not a safe
    one, so the count travels in the metadata."""
    rows, refused = [], 0
    for name, spec in meds.items():
        try:
            est = estimate_pose(
                seg, depth, seg_ids_for(id_map, name), float(spec["size"][2]), backproject
            )
        except PoseRefused:
            refused += 1
            rows.append(None)
            continue
        rows.append([est["pos"][0], est["pos"][1]])
    return rows, refused


@dataclass(kw_only=True)
class L1Session(FramePairSession):
    """The L1 rung's estimate over the shared frame-pairing lifecycle
    (perception_session.FramePairSession carries the review-hardened rules:
    once-per-request, reset clears target AND buffers, symmetric same-stamp
    pairing, unknown-med refusal).

    `backprojector` maps the published calibration to a
    `backproject(depth, pixels) -> (N, 3)` callable, bound per bridge_info so
    a republish can never be shadowed by an earlier closure (B023)."""

    backprojector: Callable[[dict], Callable] | None = None
    id_map: dict = field(default_factory=dict)
    # H6 F1 (ADR-h6-operation-protocol): "pose_bias" shifts the TARGET
    # estimate only; None (the default) is byte-identical pre-H6 behavior
    fault: str | None = None

    def on_bridge_info(self, info: dict) -> None:
        super().on_bridge_info(info)
        self.id_map = info.get("segmentation_ids") or {}

    # the L1 observation frame is the segmentation mask
    on_seg = FramePairSession.on_obs

    @property
    def latest_seg(self):
        return self.latest_obs

    def _estimate(self, seg: np.ndarray, depth: np.ndarray) -> dict:
        project = self.backprojector(self.calibration)
        estimate = estimate_pose(
            seg,
            depth,
            seg_ids_for(self.id_map, self.target),
            float(self.meds[self.target]["size"][2]),
            project,
            footprint_m=tuple(self.meds[self.target]["size"][:2]),
        )
        if self.fault == "pose_bias":
            from aisle.nodes.h6_fault import bias_pose

            estimate = bias_pose(estimate)
        neighbours, refused = estimate_neighbours(seg, depth, self.id_map, self.meds, project)
        return {
            **estimate,
            "target_med": self.target,
            "neighbours": neighbours,
            "neighbours_refused": refused,
        }


def main() -> None:  # pragma: no cover — dora runtime
    import json
    import sys

    import pyarrow as pa

    from aisle.scenes.pharmacy import load_meds
    from aisle.topics import make_sender
    from aisle.turn_node import Node
    from aisle.verifier.stages import backproject_overhead

    node = Node()
    # TC-2's per-topic monotonic seq lives in one place for every AISLE node;
    # hand-building metadata here dropped `seq` entirely
    send = make_sender(node)
    from aisle.nodes.h6_fault import armed_fault

    session = L1Session(
        meds=load_meds(),
        fault=armed_fault("segmented-pose"),
        # bound per bridge_info by the session (B023): a closure over a loop
        # variable here would follow a later republish
        backprojector=lambda calibration: (
            lambda depth, pixels: backproject_overhead(depth, calibration, pixels)
        ),
    )

    request_identity = {}
    for event in node:
        if event["type"] != "INPUT":
            continue
        topic, metadata = event["id"], (event.get("metadata") or {})
        if topic == "bridge_info":
            session.on_bridge_info(json.loads(event["value"][0].as_py()))
        elif topic == "target_request":
            request = json.loads(event["value"][0].as_py())
            if not session.on_target_request(request):
                print(
                    f"target_request refused: unknown med {request.get('target_med')!r}",
                    file=sys.stderr,
                )
            else:
                request_identity = {
                    k: metadata[k] for k in ("goal_id", "request_id") if k in metadata
                }
        elif topic == "reset_done":
            request_identity = {}
            session.on_reset_done()
        elif topic in ("depth_overhead", "seg_overhead"):
            h, w = int(metadata.get("h", 0)), int(metadata.get("w", 0))
            if h <= 0 or w <= 0:
                # a frame without its stamped dimensions violates the bridge
                # contract (BRG-2 metadata); skip it LOUDLY rather than die on
                # reshape — a dead pose source times out every later episode
                # in the least legible way (round-2 review)
                print(f"{topic} frame skipped: h={h} w={w}", file=sys.stderr)
                continue
            stamp = int(metadata.get("sim_time_ns", -1))
            frame = np.asarray(event["value"].to_numpy(zero_copy_only=False))
            try:
                if topic == "depth_overhead":
                    out = session.on_depth(stamp, frame.astype(np.float32).reshape(h, w))
                else:
                    out = session.on_seg(stamp, frame.reshape(h, w))
            except PoseRefused as exc:
                # TC-9: refuse rather than publish a biased pose. The episode
                # then closes honestly on the verifier's timeout instead of the
                # arm diving at a phantom.
                print(f"L1 pose refused for {session.target}: {exc}", file=sys.stderr)
                continue
            if out is None:
                continue
            # `target_pose` is the topic the graph's grasp planner consumes
            # (oracle_pose publishes the same name at L0) -- an L1 node must be
            # a drop-in replacement, not a new edge name. The frame's image
            # keys (enc/h/w) stay behind: a 7-float pose that claims to be a
            # 640x480 seg_i32 frame misleads any consumer that keys on them.
            send(
                "target_pose",
                pa.array(np.asarray(out["pos"] + [0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
                {
                    **{
                        k: v
                        for k, v in metadata.items()
                        if k not in ("enc", "h", "w", "goal_id", "request_id")
                    },
                    **request_identity,
                    "target_med": out["target_med"],
                    "neighbours": json.dumps(out["neighbours"]),
                    "neighbours_refused": out["neighbours_refused"],
                    "mask_pixels": out["mask_pixels"],
                    "perception": "L1",
                },
            )


if __name__ == "__main__":  # pragma: no cover
    main()
