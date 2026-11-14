"""return-planner node (T4 increment two, ADR-32 §3; VER-3 amendment):
turns the state machine's `return_request` into a standard grasp plan
that picks the named med FROM THE TRAY and places it at a free shelf
slot — the `return_item` behavior the spec names, riding the existing
guard-gated grasp/IK stack unchanged (the plan's `place_xy` metadata
redirects the place destination; nothing else differs from a pick).

Not the reset service, deliberately: this is an in-episode manipulation
with the episode's own perception constraints. Pose source: the tray
geometry itself — the box to return IS the tray's occupant (goal-start
tray set, VER-1 amendment), so its grasp pose is the tray centre at the
med's known half-height (SCN-2 public geometry), no privileged topic.
"""

from __future__ import annotations

from aisle.nodes.grasp_topdown import topdown_quat


def _return_orientation(embodiment: str) -> tuple:
    if embodiment == "franka":
        return topdown_quat(0.0)
    if embodiment == "so101":
        return (0.0, 0.0, 0.0, 1.0)  # separate embodiment; retain its existing path
    raise ValueError(f"unsupported return-planner embodiment: {embodiment}")


def return_slot(med: str, meds: dict, layout: dict) -> list | None:
    """The return slot: the shelf's front-left corner column at level 0 —
    empty by construction in T4 scenes (SCN-2 samples interior slots)."""
    spec = meds.get(med)
    if spec is None:
        return None
    shelf = layout["shelf"]
    sx = float(shelf["pos"][0]) - float(shelf["level_size"][0]) / 2.0 + float(spec["size"][0])
    sy = float(shelf["pos"][1]) - float(shelf["level_size"][1]) / 2.0 + float(spec["size"][1])
    return [sx, sy]


def return_grasp_from_estimate(
    estimate: dict, med: str, meds: dict, layout: dict, *, embodiment: str = "franka"
) -> tuple[list, list] | None:
    """(grasp_pose7, place_xy) grasping the MEASURED box pose (L1
    estimate) instead of the assumed tray centre — t4-inc2-recovery-r4
    seeds 4/8: the delivered box lies wherever release dropped it, and a
    centre-assuming grasp strikes the edge or closes on air."""
    slot = return_slot(med, meds, layout)
    if slot is None:
        return None
    x, y, z = (float(v) for v in estimate["pos"])
    return [x, y, z, *_return_orientation(embodiment)], slot


def carry_request_meta(request_meta: dict, place_xy: list) -> dict:
    """The outgoing plan's metadata: the request's CORRELATION keys plus
    the place redirect — never its turn stamp. The estimate arrives frames
    after the request, and re-sending the request's full stamp is an
    ADR-30 'unrelated turn stamp' ProtocolError (run 20260826-173702);
    the turn_node wrapper stamps outputs with the active turn itself."""
    carried = {
        k: v
        for k, v in request_meta.items()
        if k not in ("turn_epoch", "turn_id", "sim_time_ns", "seq")
    }
    carried["place_xy"] = place_xy
    return carried


def return_grasp_and_slot(
    med: str, meds: dict, layout: dict, *, embodiment: str = "franka"
) -> tuple[list, list] | None:
    """(grasp_pose7, place_xy) for returning `med` from the tray to the
    front-most free shelf slot column on level 0. Pure (CON-12)."""
    spec = meds.get(med)
    if spec is None:
        return None
    tray = layout["tray"]
    tx, ty = float(tray["pos"][0]), float(tray["pos"][1])
    tray_top = float(tray["pos"][2]) + float(tray["size"][2]) / 2.0
    grasp_z = tray_top + float(spec["size"][2]) / 2.0
    grasp = [tx, ty, grasp_z, *_return_orientation(embodiment)]
    return grasp, return_slot(med, meds, layout)


def main() -> None:  # pragma: no cover — graph-tested
    import json
    import os
    import sys

    import numpy as np
    import pyarrow as pa

    from aisle.nodes.segmented_pose import L1Session, PoseRefused
    from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout
    from aisle.topics import env_accepts, env_pin_from_env, make_sender
    from aisle.turn_node import Node
    from aisle.verifier.stages import backproject_overhead

    meds = load_meds()
    embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
    layout = resolve_layout(load_physics(), embodiment)
    env_pin = env_pin_from_env(os.environ)
    node = Node()
    send = make_sender(node, env_pin)
    # the return med's pose is MEASURED from the same L1 frames the
    # delivery leg uses (rung-honest): the delivered box lies wherever
    # release dropped it, and a centre-assuming grasp strikes the edge
    # or closes on air (t4-inc2-recovery-r4 seeds 8/4)
    session = L1Session(
        meds=meds,
        backprojector=lambda calibration: (
            lambda depth, pixels: backproject_overhead(depth, calibration, pixels)
        ),
    )
    pending: dict | None = None  # {med, meta, frames}
    FALLBACK_FRAMES = 75  # ~5 s at 15 Hz: the centre grasp is the floor, not the plan

    def plan_and_send(grasp: list, place_xy: list, meta: dict, med: str, how: str) -> None:
        out_meta = carry_request_meta(meta, place_xy)
        send("grasp_pose", pa.array(np.asarray(grasp, dtype=np.float32)), out_meta)
        print(f"return plan ({how}): {med} tray->slot {place_xy}", file=sys.stderr)

    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        if not env_accepts(metadata, env_pin):
            continue
        topic = event["id"]
        if topic == "bridge_info":
            session.on_bridge_info(json.loads(event["value"][0].as_py()))
        elif topic == "reset_done":
            session.on_reset_done()
            pending = None
        elif topic == "return_request":
            payload = json.loads(event["value"][0].as_py())
            med = payload.get("return_med", "")
            if return_slot(med, meds, layout) is None:
                print(f"return refused: unknown med {med!r}", file=sys.stderr)
                continue
            session.on_target_request({"target_med": med})
            pending = {"med": med, "meta": metadata, "frames": 0}
        elif topic in ("seg_overhead", "depth_overhead") and pending is not None:
            h, w = int(metadata.get("h", 0)), int(metadata.get("w", 0))
            if h <= 0 or w <= 0:
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
                print(f"return estimate refused for {pending['med']}: {exc}", file=sys.stderr)
                out = None
            if out is not None:
                pair = return_grasp_from_estimate(
                    out, pending["med"], meds, layout, embodiment=embodiment
                )
                plan_and_send(pair[0], pair[1], pending["meta"], pending["med"], "estimate")
                pending = None
                continue
            pending["frames"] += 1
            if pending["frames"] >= FALLBACK_FRAMES:
                # the pre-estimate behavior as the floor: grasp the tray
                # centre rather than never returning at all — LOUD, so a
                # fallback-heavy run is visible in the record
                pair = return_grasp_and_slot(pending["med"], meds, layout, embodiment=embodiment)
                print(
                    f"return estimate unavailable after {pending['frames']} frames — "
                    "falling back to tray centre",
                    file=sys.stderr,
                )
                plan_and_send(pair[0], pair[1], pending["meta"], pending["med"], "centre-fallback")
                pending = None


if __name__ == "__main__":
    main()
