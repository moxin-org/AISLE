"""SO-101 SPEC-VENDOR-NODE-V1 vendor node as an ADR-30 lockstep participant.

Runs the shipping ``so_arm101_node`` controller (dorarobotics/so-arm101-dora-node)
with its ``aisle`` backend, so AISLE evaluates the same code that drives the real
arm. Only the transport changes: ``aisle.turn_node.Node`` replaces the dora Node,
and the controller's clock is the turn's simulation time, so trajectories, settle
timeouts and stale-state checks run in sim time whatever the real-time factor.

    in   cmd_request  json_utf8   SPEC cmd_request envelopes (from a planner)
    in   joint_state  jointvec_f32 radians, TC-5 order
    in   reset_done   reset_done_u32  cancels a move in flight
    in   turn         clock
    out  cmd_response, capabilities, state, safety_event  json_utf8 envelopes
    out  joint_cmd    jointvec_f32 radians (route through budget-guard)
    out  gripper_cmd  scalar_f32 in [0, 1], 0 open
    out  turn_done

The package must be importable: install it, or point ``AISLE_SO101_NODE_SRC`` at a
checkout. ``ROBOT_ID``, ``TCP_Z_FLOOR_M`` and ``MAX_SPEED_DEG_S`` configure it as on
the robot. The heartbeat watchdog is off: no network bridge sits in between.
"""

from __future__ import annotations

import os
import sys


def _import_vendor():
    src = os.environ.get("AISLE_SO101_NODE_SRC")
    if src:
        sys.path.insert(0, os.path.abspath(src))
    from so_arm101_node import aisle, arm, config, motion, node, runtime

    return aisle, arm, config, motion, node, runtime


class SimClock:
    """Seconds of simulation time, from the latest turn stamp."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def observe(self, metadata: dict | None) -> None:
        ns = (metadata or {}).get("sim_time_ns")
        if ns is not None:
            self.t = max(self.t, int(ns) / 1e9)


def build(send, environ=None):
    """Controller + pump for one SO-101 arm; ``send`` is the node's send_output."""
    env = os.environ if environ is None else environ
    aisle, arm, config, motion, node, runtime = _import_vendor()
    clock = SimClock()
    cfg = config.from_env(
        {**env, "SO101_BACKEND": "aisle", "SO101_ARMS": "arm:1", "SO101_BASE": "none"}
    )
    link = aisle.AisleLink(stale_s=cfg.aisle_stale_s, clock=clock)
    controller = node.So101Controller(
        {"arm": aisle.AisleArm(link, arm.ArmConfig.from_first_id("arm", 1))},
        robot_id=cfg.robot_id,
        limits=motion.Limits(max_speed_deg_s=cfg.max_speed_deg_s),
        rest=cfg.rest_pose or aisle.AISLE_HOME,
        heartbeat_timeout_ms=0,
        tcp_floor_m=cfg.tcp_floor_m,
        clock=clock,
    )
    pump = runtime.Pump(controller, send, state_hz=cfg.state_hz, clock=clock, aisle=link)
    return clock, pump, runtime.decode


class _RawEventLog:
    """``AISLE_SPEC_DEBUG=1``: print every raw dora event before the turn wrapper buffers
    it (input, turn stamp, seq, and this node's ready-plan entry), to see what a stuck
    turn is waiting for."""

    def __init__(self, raw, node_id: str) -> None:
        self.raw = raw
        self.node_id = node_id

    def send_output(self, *args, **kwargs):
        return self.raw.send_output(*args, **kwargs)

    def __iter__(self):
        import json as _json

        for event in self.raw:
            meta = event.get("metadata") or {}
            fields = {
                k: meta[k] for k in ("turn_epoch", "turn_id", "seq", "sim_time_ns") if k in meta
            }
            if event.get("id") == "turn" and isinstance(meta.get("ready_plan"), str):
                own = _json.loads(meta["ready_plan"]).get(self.node_id)
                fields["ready"] = own and {
                    k: own.get(k)
                    for k in ("expected_inputs", "expected_counts", "expected_turn_ids")
                }
            print(
                f"spec-so101 raw: {event.get('type')} {event.get('id')} {fields}",
                file=sys.stderr,
                flush=True,
            )
            yield event


def main() -> None:
    from aisle.topics import make_sender
    from aisle.turn_node import Node

    if os.environ.get("AISLE_SPEC_DEBUG") == "1":
        from dora import Node as DoraNode

        dora_node = Node(
            raw_node=_RawEventLog(DoraNode(), os.environ.get("AISLE_TURN_NODE", "spec-so101"))
        )
    else:
        dora_node = Node()
    sender = make_sender(dora_node)  # TC-2: per-topic seq; the turn wrapper adds the turn stamp
    clock, pump, decode = build(lambda topic, value: sender(topic, value, {}))
    for event in dora_node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        clock.observe(event.get("metadata"))
        input_id = event["id"]
        if input_id == "joint_state":
            pump.on_joint_state(event["value"])
        elif input_id == "cmd_request":
            pump.on_request(decode(event["value"]))
        elif input_id == "reset_done":
            pump.on_reset_done()
        elif input_id == "turn":
            pump.on_tick()


if __name__ == "__main__":
    main()
