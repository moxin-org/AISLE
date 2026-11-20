"""Trusted Arrow adapter for P3's public task boundary.

This codec does not attest graph routing, confinement or physical completion.
The caller owns an outer process deadline: a Node iterator can block. Authored
code must receive TaskSession RPC, never this object or its Node.
"""

from __future__ import annotations

import json
from collections import deque

import pyarrow as pa

from aisle.harness.p3_task_boundary import _identifier, _require
from aisle.topics import make_sender
from aisle.turn_node import Node

OUTPUTS = frozenset({"robot_msg", "target_request", "return_request", "turn_done"})
INPUTS = frozenset({"reset_done", "human_msg", "plan_done", "violation"})
STAMP_KEYS = ("env_id", "seq", "sim_time_ns", "turn_epoch", "turn_id")


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate task JSON field")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite task JSON constant")


def _payload(topic, value):
    _require(isinstance(value, pa.Array) and len(value) == 1, "invalid task Arrow payload")
    if topic == "reset_done":
        _require(value.type == pa.uint32() and value.to_pylist() == [1], "invalid reset payload")
        return [1]
    _require(value.type == pa.string(), "task JSON requires Arrow Utf8[1]")
    text = value[0].as_py()
    _require(isinstance(text, str) and len(text.encode()) <= 8192, "invalid task JSON size")
    decoded = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    _require(isinstance(decoded, dict), "task JSON must be an object")
    return decoded


class DoraTaskTransport:
    """Only fixed request outputs and explicit public inputs on one lockstep incarnation.

    Public ticks derive from the simulation turn at 1 Hz. Unstamped wall ticks
    are ignored; no authored action is serviced from a wall handler. Private
    metadata is discarded before TaskSession's input log and public projector.
    Only closure of the coordinator's turn input is a normal end; bare EOF,
    STOP, ERROR, malformed traffic and incarnation drift close authority.
    """

    def __init__(self, node, *, turn_epoch):
        _require(
            isinstance(node, Node)
            and node.lockstep
            and set(node.outputs) == OUTPUTS
            and len(node.outputs) == len(OUTPUTS)
            and not node.wall_outputs,
            "P3 transport requires lockstep and exact output grants",
        )
        _require(type(turn_epoch) is int and turn_epoch >= 0, "invalid bound turn epoch")
        self.node, self.turn_epoch = node, turn_epoch
        self.events, self.sender = iter(node), make_sender(node, env_pin=0)
        self.closed, self.ready = False, False
        self.sim_time_ns, self.last_tick_ns, self.tick_seq = 0, None, 0
        self.pending = deque()
        self.last_turn_id = -1

    def _metadata(self, topic, metadata):
        _require(isinstance(metadata, dict), "invalid task transport metadata")
        keys = ("sim_time_ns", "turn_epoch", "turn_id") if topic == "turn" else STAMP_KEYS
        _require(
            all(type(metadata.get(k)) is int and metadata[k] >= 0 for k in keys),
            "invalid task transport stamp",
        )
        _require(metadata["turn_epoch"] == self.turn_epoch, "task producer epoch changed")
        _require(topic == "turn" or metadata["env_id"] == 0, "task environment differs")
        clean = {k: metadata[k] for k in keys}
        identities = {"human_msg": ("goal_id",), "plan_done": ("goal_id", "request_id")}
        for key in identities.get(topic, ()):
            value = metadata.get(key)
            _require(
                isinstance(value, str) and 0 < len(value) <= 256,
                "invalid task correlation identity",
            )
            clean[key] = value
        return clean

    def next_event(self):
        _require(not self.closed, "task transport is closed")
        try:
            return self._next_event()
        except StopIteration:
            raise
        except Exception:
            self.closed = True
            raise

    def _next_event(self):
        inputs = []
        while True:
            if self.pending:
                self.ready = True
                return self.pending.popleft()
            event = next(self.events, None)
            _require(event is not None, "task transport ended without coordinator closure")
            kind, topic = event.get("type"), event.get("id")
            if kind == "INPUT_CLOSED":
                _require(topic in INPUTS | {"turn", "tick"}, "undeclared task input closure")
                if topic == "turn":
                    _require(not inputs, "coordinator closed during a task turn")
                    self.closed = True
                    raise StopIteration
                continue
            _require(kind == "INPUT", "task transport stopped or failed")
            _require(topic in INPUTS | {"turn", "tick"}, "undeclared task input")
            if topic == "tick":
                _require(
                    not any(k in (event.get("metadata") or {}) for k in ("turn_epoch", "turn_id")),
                    "stamped wall tick refused",
                )
                continue
            metadata = self._metadata(topic, event.get("metadata"))
            if topic == "turn":
                _require(
                    metadata["turn_id"] > self.last_turn_id
                    and metadata["sim_time_ns"] >= self.sim_time_ns,
                    "task turns must be monotonic",
                )
                self.last_turn_id = metadata["turn_id"]
                self.sim_time_ns = metadata["sim_time_ns"]
                # Node yields inputs in lexical topic order. Reset must precede
                # same-turn dialogue even though 'human_msg' sorts before it.
                self.pending.extend(sorted(inputs, key=lambda row: row[0] != "reset_done"))
                if any(row[0] == "reset_done" for row in inputs):
                    self.last_tick_ns = self.sim_time_ns
                inputs = []
                if (
                    self.last_tick_ns is not None
                    and self.sim_time_ns - self.last_tick_ns >= 1_000_000_000
                ):
                    self.last_tick_ns = self.sim_time_ns
                    self.tick_seq += 1
                    self.pending.append(
                        ("tick", None, {**metadata, "env_id": 0, "seq": self.tick_seq})
                    )
            else:
                _require(len(inputs) < 1024, "task turn input budget exhausted")
                inputs.append((topic, _payload(topic, event["value"]), metadata))

    def send(self, topic, payload, metadata):
        _require(not self.closed and self.ready, "task transport is closed or has no input")
        try:
            _require(topic in OUTPUTS - {"turn_done"}, "undeclared task output")
            fields = {
                "target_request": {"target_med"},
                "return_request": {"return_med"},
                "robot_msg": {"kind", "med", "text"},
            }[topic]
            identities = {"goal_id"} if topic == "robot_msg" else {"goal_id", "request_id"}
            _require(
                isinstance(payload, dict) and set(payload) == fields, "invalid task output payload"
            )
            _require(
                isinstance(metadata, dict)
                and set(metadata) == identities
                and all(isinstance(v, str) and 0 < len(v) <= 256 for v in metadata.values()),
                "invalid task output identities",
            )
            if topic == "robot_msg":
                _require(
                    payload["kind"] == "confirm"
                    and _identifier(payload["med"])
                    and isinstance(payload["text"], str)
                    and len(payload["text"]) <= 2048,
                    "invalid task dialogue output",
                )
            else:
                _require(_identifier(next(iter(payload.values()))), "invalid task medicine")
            self.sender(
                topic,
                pa.array([json.dumps(payload, allow_nan=False)], type=pa.string()),
                {**metadata, "sim_time_ns": self.sim_time_ns},
            )
        except Exception:
            self.closed = True
            raise
