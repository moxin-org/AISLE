"""Reset service node (SPEC 040 RST-1/RST-2).

Teleport requests (mode 0) pass through to the bridge, which owns state
injection (BRG-4); replies flow back with metadata intact so the <2 s
RST-1 budget is auditable end-to-end. Behavioral mode (RST-2) runs the
attempt loop from reset/behavioral.py — retry <=3, then TELEPORT
fallback with `fallback: true` in the reply metadata; the loop never
hangs on a reset. Invalid requests are refused per-request — the service
never forwards anything the bridge would reject, and never dies (TC-6).

`reset_done` carries exactly one meaning: the episode boundary passed and
the sim was touched. Refusals answer on `reset_refused`, whose audience is
the one requester correlating on `request_id` (ADR-34, issue #195) — so a
boundary consumer needs no discriminator, and cannot forget one.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from aisle.topics import parse_sim_stamp, stamp

TELEPORT, BEHAVIORAL = 0, 1

#: RST-2 motion chains verified for behavioral reset (G1-C2): reset/motion.py
#: pins Franka limits and a 7-dof qpos layout. so101/g1_right would be driven
#: by Franka limits and a wrong-length home — refuse explicitly instead.
BEHAVIORAL_RESET_EMBODIMENTS = ("franka",)


def behavioral_reset_supported(embodiment: str) -> bool:
    """Pure gate (unit-tested): True iff the RST-2 attempt loop has a
    verified motion chain for this embodiment."""
    return embodiment in BEHAVIORAL_RESET_EMBODIMENTS


def route_reset(mode: int) -> str:
    """Pure dispatch (unit-tested): teleport -> bridge directly;
    behavioral -> the RST-2 attempt loop (whose exhaustion also ends at
    the bridge, with fallback metadata)."""
    if mode == TELEPORT:
        return "bridge"
    if mode == BEHAVIORAL:
        return "behavioral"
    raise ValueError(f"reset mode must be 0 or 1, got {mode}")


def refusal_reply_metadata(request_meta: dict, payload: np.ndarray, error: str) -> dict:
    """TC-6 reply keys for a refused request: echo request_id, seed/mode
    when the payload was well-formed enough to carry them, t_reset_ms=0
    (the sim was never touched), and the error (ADR-8).

    `env_id` rides through (issue #192) for the same reason the behavioral
    reply carries it: every episode-state consumer now takes the boundary
    from this output, and the guard slices per env (BRG-5).

    Deliberately NO `sim_time_ns` — a refused reset never touched the sim,
    so there is none to report and inventing one would be a lie the
    verifier's episode baseline depends on.

    This rides `reset_refused`, never `reset_done` (ADR-34, issue #195).
    While it shared the boundary topic, the missing stamp was a hazard every
    consumer had to defend against: it drove `label_reader.on_reset_done`
    down its UNFENCED branch and would have had the guard re-reference hold
    state to a home the robot is not at. Both nodes filtered on `error`;
    both filters are now DELETED, because a refusal no longer reaches
    them. The `error` key is still here — it tells the requester what went
    wrong — but nothing routes on it any more."""
    meta = {
        "request_id": request_meta.get("request_id", ""),
        "t_reset_ms": 0,
        "error": error,
    }
    if "env_id" in request_meta:
        meta["env_id"] = request_meta["env_id"]
    if payload.shape[0] == 2:
        meta["seed"], meta["mode"] = int(payload[0]), int(payload[1])
    return meta


def main(clock=None) -> None:
    """The clock is injected (CON-5): `t_reset_ms` is a WALL measurement and
    must be reproducible in tests without sleeping."""
    import json
    import time
    import traceback

    import pyarrow as pa

    from aisle.turn_node import Node

    clock = clock or time.monotonic
    node = Node()
    seq_reply = 0
    seq_forward = 0
    seq_cmd = 0
    # its OWN sequence, because TC-2 defines `seq` as PER-TOPIC monotonic
    # and these are now two topics. Consumers read `reset_done`'s seq as the
    # episode epoch (nav_action, s1_expert), so keeping the counters joined
    # would leave gaps in the boundary sequence that ADR-7 §13 says exist to
    # distinguish drops from normal flow — false gaps, not lost messages.
    # (An earlier revision of this comment claimed the shared counter caused
    # epoch DRIFT between consumers. It did not: the refusal was delivered on
    # `reset_done`, so every consumer read the same seq off the same message.
    # The reason is TC-2 conformance, not drift — round-2 review.)
    seq_refused = 0
    runtime = None  # BehavioralRuntime, built lazily on first behavioral request
    latest_sim_ns = 0
    # when the CURRENT behavioral request started, for its TC-6 t_reset_ms.
    # RST-2 is the route where this matters most: it can take three real
    # motion attempts, and it is the only route whose reply carried no
    # timing at all (issue #194).
    behavioral_started = None
    # the fallback whose bridge reply has not come back yet (RST-2, #205):
    # {request_id, attempts, started}. One slot, because TC-6 resets never
    # overlap; correlated on request_id so a stray reply cannot pick it up.
    pending_fallback = None

    def get_runtime():
        nonlocal runtime
        if runtime is None:
            from aisle.reset.runtime import BehavioralRuntime
            from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout
            from aisle.verifier.models import load_pinned

            embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
            physics = load_physics()
            runtime = BehavioralRuntime(
                layout=resolve_layout(physics, embodiment),
                meds=load_meds(),
                home_q=np.asarray(physics["embodiment"][embodiment]["home_qpos"], dtype=np.float32),
                model_pair=load_pinned("identity"),
            )
        return runtime

    def fallback_teleport(seed: int, request_meta: dict, attempts: int) -> None:
        nonlocal seq_forward, pending_fallback
        from aisle.reset.behavioral import BehavioralOutcome, behavioral_reply_metadata

        # RST-2/#205: the bridge answers this teleport, and it builds its
        # reply metadata FRESH — echoing only request_id. So the forward's
        # audit keys never come back on their own; remember them here and
        # merge them onto the relay, which this node already owns (#192).
        pending_fallback = {
            "request_id": request_meta.get("request_id", ""),
            "attempts": attempts,
            "started": behavioral_started,
        }
        forward_meta = dict(request_meta)
        forward_meta.update(
            behavioral_reply_metadata(
                request_meta, BehavioralOutcome(fallback=True, attempts=attempts)
            )
        )
        seq_forward += 1
        node.send_output(
            "bridge_reset",
            pa.array(np.array([seed, TELEPORT], dtype=np.uint32)),
            stamp(forward_meta, seq_forward),
        )

    for event in node:
        if event["type"] != "INPUT":
            continue
        metadata = event.get("metadata") or {}
        event_id = event["id"]
        is_reset_request = event_id == "reset" or (
            event_id.startswith("reset_")
            and event_id != "reset_done"
            and event_id.rsplit("_", 1)[1].isdigit()
        )
        if is_reset_request:
            if not metadata.get("request_id"):
                # TC-6 correlates request/reply via request_id: with none
                # there is nothing to reply TO — drop loudly; forwarding
                # would trip the bridge's own TC-6 validation
                print("reset refused: missing request_id metadata (TC-6)", file=sys.stderr)
                continue
            payload = np.zeros(0, dtype=np.uint32)
            try:
                # inside the try (issue #192 review): a non-numeric payload
                # raised straight out of the loop and killed the node, which
                # since #192 strands every consumer in the graph rather than
                # just the requester. The refusal path below already exists
                # for exactly this class of input.
                payload = np.asarray(
                    event["value"].to_numpy(zero_copy_only=False), dtype=np.uint32
                ).reshape(-1)
                if payload.shape[0] != 2:
                    raise ValueError(f"reset payload must be UInt32[2], got {payload.shape}")
                route = route_reset(int(payload[1]))
            except (ValueError, TypeError, OverflowError) as refusal:
                # refuse THIS request loudly without killing the service: the
                # requester gets a correlated error reply, later teleports
                # still work (TC-6; ADR-8). The type list mirrors
                # topics.parse_sim_stamp's. Measured: the payloads reachable
                # over the wire (strings, nested lists) all raise ValueError,
                # so TypeError/OverflowError are unexercised belt — kept
                # because this node is the single boundary authority since
                # issue #192 and dora does not restart nodes, but NOT claimed
                # as tested.
                print(f"reset refused: {refusal}", file=sys.stderr)
                seq_refused += 1
                node.send_output(
                    "reset_refused",
                    pa.array(np.array([0], dtype=np.uint32)),
                    stamp(refusal_reply_metadata(metadata, payload, str(refusal)), seq_refused),
                )
                continue
            if route == "behavioral":
                embodiment = os.environ.get("AISLE_EMBODIMENT", "franka")
                if not behavioral_reset_supported(embodiment):
                    # G1-C2: RST-2's motion chain is Franka-verified (limits,
                    # 7-dof home). Driving another embodiment with it would
                    # command wrong limits and a wrong-length home — refuse
                    # this request loudly; teleport (mode 0) is unaffected.
                    print(
                        f"reset refused: behavioral reset not implemented for "
                        f"{embodiment!r} (RST-2 chain is franka-only, G1-C2)",
                        file=sys.stderr,
                    )
                    seq_refused += 1
                    node.send_output(
                        "reset_refused",
                        pa.array(np.array([0], dtype=np.uint32)),
                        stamp(
                            refusal_reply_metadata(
                                metadata,
                                payload,
                                f"behavioral reset not implemented for {embodiment!r}; "
                                f"use teleport (mode 0)",
                            ),
                            seq_refused,
                        ),
                    )
                    continue
                # RST-2: start the attempt runtime; commands stream on
                # joint_state events through the guard; the outcome
                # settles asynchronously (success replies directly,
                # exhaustion falls back to teleport — never hangs:
                # every stage has a bounded bail and every attempt a
                # bounded budget)
                # the clock starts ABOVE get_runtime(): when `bridge_info`
                # has not landed first, the FIRST behavioral request pays
                # the ~2 s model load inside itself, and RST-1's budget is
                # <2 s — a t_reset_ms that excludes the load cannot see the
                # breach it exists to report (round-2 review)
                behavioral_started = clock()
                try:
                    rt = get_runtime()
                    rt.start(int(payload[0]), metadata)
                except Exception as broken:
                    # issue #206: building the runtime does file I/O and an
                    # Owlv2 from_pretrained, and starting it plans. Any of
                    # that can fail on a first run with a cold or corrupt
                    # model cache — and this node's death has NO recovery:
                    # since #192 every consumer in every graph takes its
                    # episode boundary from here, and dora does not restart
                    # nodes. Broad on purpose: the alternative to catching an
                    # unlisted exception is stranding the run.
                    #
                    # Refused, not fallen back: RST-2's teleport fallback
                    # reports `fallback: true, behavioral_attempts: N`, and
                    # dressing an infrastructure failure as a behavioral
                    # outcome would put a lie in the A6 measurement. Teleport
                    # needs no runtime, so the service keeps serving.
                    traceback.print_exc()
                    print(f"behavioral reset unavailable: {broken!r}", file=sys.stderr)
                    runtime = None  # rebuild next time rather than reuse a half-started one
                    behavioral_started = None
                    seq_refused += 1
                    node.send_output(
                        "reset_refused",
                        pa.array(np.array([0], dtype=np.uint32)),
                        stamp(
                            refusal_reply_metadata(
                                metadata, payload, f"behavioral runtime unavailable: {broken}"
                            ),
                            seq_refused,
                        ),
                    )
                    continue
                if rt.outcome == "exhausted":  # unplannable from the start
                    print("behavioral reset: unplannable, fallback", file=sys.stderr)
                    fallback_teleport(int(payload[0]), metadata, rt.attempts)
                continue
            seq_forward += 1
            node.send_output("bridge_reset", pa.array(payload), stamp(dict(metadata), seq_forward))
        elif event["id"] == "reset_done":
            reply_meta = dict(metadata)
            if pending_fallback and reply_meta.get("request_id") == pending_fallback["request_id"]:
                # RST-2 (#205): restore the audit trail the bridge could not
                # carry. RST-1 (#194): t_reset_ms spans the WHOLE reset —
                # the behavioral attempts plus this teleport — because the
                # bridge only ever timed its own hop, on the route where the
                # reset is most expensive. `mode` is the mode REQUESTED, on
                # every route (ADR-35); `fallback` is what says a teleport
                # actually ran.
                reply_meta["fallback"] = True
                reply_meta["behavioral_attempts"] = pending_fallback["attempts"]
                reply_meta["mode"] = BEHAVIORAL
                started = pending_fallback["started"]
                if started is not None:
                    reply_meta["t_reset_ms"] = int((clock() - started) * 1000)
                pending_fallback = None
                behavioral_started = None
            seq_reply += 1
            node.send_output("reset_done", event["value"], stamp(reply_meta, seq_reply))
        elif event["id"] == "bridge_info":
            # builds the runtime early so the model load (~2 s) never
            # lands inside a reset request — which makes THIS the first
            # place a cold or corrupt model cache surfaces (issue #206).
            # Nothing is owed a reply on a broadcast input; the requirement
            # is that the boundary authority survives to serve teleports,
            # and that the next behavioral request retries the build rather
            # than inheriting a half-constructed runtime.
            try:
                get_runtime().on_bridge_info(json.loads(event["value"][0].as_py()))
            except Exception:
                # Deliberately does NOT discard the runtime (cross-review of
                # #223). `bridge_info` is emitted ONCE, before the bridge's
                # event loop, and `on_bridge_info` is the only writer of
                # `calibration` — so dropping a runtime that BUILT would
                # throw calibration away permanently, and every later
                # behavioral reset would burn all three attempts on
                # `calibration is None` and report `fallback: true,
                # behavioral_attempts: 3`. That is precisely the silent A6
                # void `validate.py`'s REFUSAL of unwired graphs exists to
                # prevent, and precisely the lie the behavioral branch above
                # refuses to write. A failed build leaves `runtime` unset
                # anyway, so the next request retries it.
                traceback.print_exc()
                print("behavioral pre-warm failed; teleport is unaffected", file=sys.stderr)
        elif event["id"] == "rgb_overhead":
            h, w = int(metadata.get("h", 0)), int(metadata.get("w", 0))
            if runtime is not None and h > 0 and w > 0:
                frame = np.asarray(event["value"].to_numpy(zero_copy_only=False))
                runtime.on_rgb(frame.astype(np.uint8).reshape(h, w, 3))
        elif event["id"] == "depth_overhead":
            h, w = int(metadata.get("h", 0)), int(metadata.get("w", 0))
            if runtime is not None and h > 0 and w > 0:
                frame = np.asarray(event["value"].to_numpy(zero_copy_only=False))
                runtime.on_depth(frame.astype(np.float32).reshape(h, w))
        elif event["id"] == "joint_state":
            # TOTAL read (BG-3, issue #160): a malformed stamp from any
            # upstream node must degrade this decision, never raise out of
            # the event loop. That rule binds hardest HERE: since issue
            # #192 every consumer in every graph takes its episode
            # boundary from this node, and dora does not restart nodes, so
            # one bad stamp would strand the whole run.
            stamped = parse_sim_stamp(metadata)
            latest_sim_ns = stamped if stamped is not None else latest_sim_ns
            if runtime is None or not runtime.active:
                continue
            qpos = np.asarray(
                event["value"].to_numpy(zero_copy_only=False), dtype=np.float32
            ).reshape(-1)
            cmd, grip = runtime.on_joint_state(qpos)
            if grip is not None:
                seq_cmd += 1
                node.send_output(
                    "reset_gripper_cmd",
                    pa.array(np.array([grip], dtype=np.float32)),
                    stamp(dict(metadata), seq_cmd),
                )
            if cmd is not None:
                seq_cmd += 1
                node.send_output("reset_joint_cmd", pa.array(cmd), stamp(dict(metadata), seq_cmd))
            if runtime.outcome == "success":
                # the box is verified back on the shelf: reply directly,
                # no state injection (BRG-4 untouched). The reply's
                # sim_time_ns is post-motion so the verifier's episode
                # baseline can never predate the reset (BRG-4 parity)
                from aisle.reset.behavioral import BehavioralOutcome, behavioral_reply_metadata

                reply = behavioral_reply_metadata(
                    runtime.request_meta,
                    BehavioralOutcome(fallback=False, attempts=runtime.attempts),
                )
                reply["sim_time_ns"] = latest_sim_ns
                # TC-6: every reply carries seed/mode/t_reset_ms. This route
                # carried none of the three (issue #194) -- and it is the one
                # that can take three real motion attempts, so it is exactly
                # where RST-1's <2 s budget most needs to be auditable.
                reply["seed"] = int(runtime.seed)
                reply["mode"] = BEHAVIORAL
                reply["t_reset_ms"] = (
                    0 if behavioral_started is None else int((clock() - behavioral_started) * 1000)
                )
                print(
                    f"behavioral reset: success in {runtime.attempts} attempt(s)",
                    file=sys.stderr,
                )
                seq_reply += 1
                node.send_output(
                    "reset_done", pa.array(np.array([1], dtype=np.uint32)), stamp(reply, seq_reply)
                )
                runtime.outcome = None
                # cleared WITH the outcome: `0` already means "the sim was
                # never touched" on the refusal path, so a stale anchor
                # would report the PREVIOUS request's start as this one's
                # duration rather than reading as missing. Unreachable
                # today (success requires a start), but that is a property
                # of runtime.py, not of this loop (round-2 review)
                behavioral_started = None
            elif runtime.outcome == "exhausted":
                print(
                    f"behavioral reset: exhausted after {runtime.attempts}, fallback",
                    file=sys.stderr,
                )
                fallback_teleport(runtime.seed, runtime.request_meta, runtime.attempts)
                runtime.outcome = None


if __name__ == "__main__":
    main()
