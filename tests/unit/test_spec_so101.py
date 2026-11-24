"""spec-so101 adapter: the shipping SO-101 vendor node driven by turn sim time.

Skipped unless the vendor package is importable (install it or set
AISLE_SO101_NODE_SRC to a so-arm101-dora-node checkout).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pyarrow as pa
import pytest

from aisle.nodes import spec_so101

try:
    spec_so101._import_vendor()
except ImportError:
    pytest.skip("so_arm101_node not importable", allow_module_level=True)

NS = "vendor.lerobot.so101"
HOME_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 1.74533]


class Sent:
    def __init__(self) -> None:
        self.items: list[tuple[str, object]] = []

    def __call__(self, output, value, metadata=None) -> None:
        self.items.append((output, value))

    def responses(self) -> list[dict]:
        return [json.loads(v.to_pylist()[0]) for o, v in self.items if o == "cmd_response"]

    def commands(self) -> list[np.ndarray]:
        return [
            np.asarray(v.to_numpy(zero_copy_only=False)) for o, v in self.items if o == "joint_cmd"
        ]


def _turn(clock, pump, t: float, q) -> None:
    clock.observe({"sim_time_ns": int(t * 1e9)})
    pump.on_joint_state(pa.array(q, type=pa.float32()))
    pump.on_tick()


def test_sim_clock_only_moves_forward():
    clock = spec_so101.SimClock()
    clock.observe({"sim_time_ns": 2_000_000_000})
    clock.observe({"sim_time_ns": 1_000_000_000})
    clock.observe({})
    assert clock() == 2.0


def test_verbs_become_tc5_commands_in_sim_time():
    sent = Sent()
    clock, pump, _ = spec_so101.build(sent, environ={"ROBOT_ID": "so101-aisle"})
    pump.on_request({"request_id": "early", "verb": f"{NS}.enable_torque"})
    assert sent.responses()[-1]["code"] == "NOT_READY"

    q = np.asarray(HOME_RAD, np.float32)
    t = 0.0
    _turn(clock, pump, t, q)
    pump.on_request({"request_id": "on", "verb": f"{NS}.enable_torque"})
    pump.on_request(
        {
            "request_id": "mv",
            "verb": f"{NS}.move_to_joint_state",
            "params": {"joints": {"shoulder_pan": 20.0}},
        }
    )
    for _ in range(400):  # 4 s of sim time at 10 ms turns
        t += 0.01
        cmds = sent.commands()
        if cmds:
            q = q + 0.5 * (cmds[-1].astype(np.float32) - q)
        _turn(clock, pump, t, q)
        if any(r["request_id"] == "mv" for r in sent.responses()):
            break
    done = next(r for r in sent.responses() if r["request_id"] == "mv")
    assert done["ok"], done
    cmds = sent.commands()
    assert all(c.dtype == np.float32 and c.shape == (6,) for c in cmds)
    assert (
        float(np.max(np.abs(np.diff(np.stack(cmds), axis=0)))) < 0.1
    )  # below the guard's per-step limit
    assert np.degrees(q[0]) == pytest.approx(20.0, abs=2.0)


def test_into_the_table_is_refused_by_the_node():
    sent = Sent()
    clock, pump, _ = spec_so101.build(sent, environ={})
    _turn(clock, pump, 0.0, np.asarray(HOME_RAD, np.float32))
    pump.on_request({"request_id": "on", "verb": f"{NS}.enable_torque"})
    pump.on_request(
        {
            "request_id": "down",
            "verb": f"{NS}.move_to_joint_state",
            "params": {"joints": {"shoulder_lift": 60.0, "elbow_flex": 30.0}},
        }
    )
    down = next(r for r in sent.responses() if r["request_id"] == "down")
    assert down["code"] == "LIMIT_VIOLATION"


def test_vendor_source_env_is_honoured(monkeypatch):
    src = os.environ.get("AISLE_SO101_NODE_SRC")
    if not src:
        pytest.skip("vendor package installed, not loaded from a checkout")
    monkeypatch.setenv("AISLE_SO101_NODE_SRC", src)
    assert spec_so101._import_vendor()[0].__name__ == "so_arm101_node.aisle"


def test_outputs_carry_the_seq_a_lockstep_consumer_requires():
    """Every output goes through aisle.topics.make_sender, as in main(): a downstream
    ParticipantTurn must accept each message (it refused seq=None in a live run)."""
    from aisle.topics import make_sender
    from aisle.turns import ParticipantTurn

    captured: list[tuple[str, object, dict]] = []

    class RawNode:
        def send_output(self, topic, value, metadata=None):
            captured.append((topic, value, metadata or {}))

    sender = make_sender(RawNode())
    clock, pump, _ = spec_so101.build(lambda t, v: sender(t, v, {}), environ={})
    _turn(clock, pump, 0.0, np.asarray(HOME_RAD, np.float32))
    pump.on_request({"request_id": "on", "verb": f"{NS}.enable_torque"})
    pump.on_request(
        {
            "request_id": "mv",
            "verb": f"{NS}.move_to_joint_state",
            "params": {"joints": {"shoulder_pan": 5.0}},
        }
    )
    _turn(clock, pump, 0.01, np.asarray(HOME_RAD, np.float32))
    topics = {t for t, _, _ in captured}
    assert {"cmd_response", "capabilities", "state", "joint_cmd", "gripper_cmd"} <= topics

    consumer = ParticipantTurn("consumer", ["turn_done"])
    turn = {"turn_epoch": 1, "turn_id": 1, "sim_time_ns": 10_000_000}
    for topic, value, metadata in captured:
        consumer.buffer(topic, {"id": topic, "value": value}, {**metadata, **turn})
    seqs = [m["seq"] for t, _, m in captured if t == "joint_cmd"]
    assert seqs == list(range(1, len(seqs) + 1))
