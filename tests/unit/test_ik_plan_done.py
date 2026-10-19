"""ik-trajectory sends plan_done only when the graph declares it (G1-F4;
HAR-3 retry arming vs ADR-30 lockstep output accounting).

The bug (lane1 smoke 20261019-224058-539bc6): a grasp plan that runs to
completion WITHOUT a verdict makes the node emit `plan_done` — but
expert_t0's ik-trajectory declares only joint_cmd,gripper_cmd,turn_done,
so under lockstep aisle.turns raises ProtocolError "unknown or reserved
output ik-trajectory/plan_done" and the node dies into wall clamp.
franka/so101 verdicts normally land first, which is why main never saw
it; the G1 failure set exposed a latent main defect.

The fix follows the dora-genesis bridge precedent (ADR-30): the node
treats AISLE_TURN_OUTPUTS as the DECLARED output set. Under lockstep the
host always exports it (typed_node_host pins it from the graph config);
free-run reads the same variable when present. plan_done is sent only
when declared; graphs that wire it (15 eval/shield/expert graphs) keep
the HAR-3 retry arming unchanged.

The drive harness is an IDEAL ACTUATOR: joint_state echoes the last
joint_cmd one tick later (perfect tracking, no gravity sag), generated
on demand so the streamer always has ticks until the plan completes —
pre-scripted finite event lists starved mid-plan (the bail at advance
and the truncated stage list) before this.

Why a unit test suffices instead of a tests/sim or tests/graph smoke:
the crash mechanism lives in the SEND DECISION, not in transport or
physics — aisle.turns raises ProtocolError the moment node.send_output
fires for an undeclared topic, so the observable is fully determined at
this module's call site under the exact env the host exports
(AISLE_TURN_OUTPUTS, which typed_node_host pins from the graph config in
both modes). These tests drive the REAL main() event loop through a
completed plan under the expert_t0 declaration and a declaring
declaration, pinning both sides of the guard; a graph-level smoke would
add the whole runtime (graph deploy, genesis, wall clock) to observe the
same branch."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aisle.nodes.ik_trajectory import plan_done_declared

pytestmark = pytest.mark.unit

_MAX_TICKS = 12000  # full 13-stage plans converge well inside 3k ticks


class IdealActuatorNode:
    """A dora Node over which ik-trajectory.main() runs a whole plan:
    yields the grasp_pose event, then joint_state events that echo the
    last commanded pose (perfect actuator), generated lazily until the
    plan finishes or the tick budget runs out."""

    def __init__(self, grasp_event: dict, home_full: np.ndarray):
        self._grasp_event = grasp_event
        self._home = np.asarray(home_full, dtype=np.float32)
        self.last_cmd: np.ndarray | None = None
        self.sent: list[tuple[str, dict]] = []

    def __iter__(self):
        import pyarrow as pa

        yield self._grasp_event
        qpos = self._home
        for _ in range(_MAX_TICKS):
            yield {
                "type": "INPUT",
                "id": "joint_state",
                "value": pa.array(qpos.astype(np.float32)),
                "metadata": {},
            }
            qpos = self.last_cmd if self.last_cmd is not None else self._home

    def send_output(self, topic, value, metadata=None):
        self.sent.append((topic, value, dict(metadata or {})))
        if topic == "joint_cmd":
            arr = value.to_numpy(zero_copy_only=False).reshape(-1).astype(np.float32)
            self.last_cmd = arr


def _g1_grasp_event() -> tuple[dict, np.ndarray]:
    """A plannable g1_right grasp_pose event (slot 0, amoxicillin) plus the
    profile home — the same planner entry the expert graph feeds."""
    from aisle.nodes.grasp_topdown import GRIP_ENGAGEMENT, plan_grasp
    from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout

    physics = load_physics()
    profile = physics["embodiment"]["g1_right"]
    layout = resolve_layout(physics, "g1_right")
    meds = load_meds()
    med = "amoxicillin"
    slot = (float(profile["placement_slots_xy"][0][0]), float(profile["placement_slots_xy"][0][1]))
    size = meds[med]["size"]
    shelf = layout["shelf"]
    z_lo = shelf["pos"][2] + shelf["level_heights"][0] + shelf["board_thickness"] / 2
    pose = np.array([slot[0], slot[1], z_lo + size[2] / 2, 0, 0, 0, 1], dtype=np.float32)
    grasp, approach, place_z = plan_grasp(
        pose,
        size,
        grip=GRIP_ENGAGEMENT,
        shelf_front_x=shelf["pos"][0] - shelf["level_size"][0] / 2,
        tray_top_z=profile["tray_pos"][2] + profile["tray_size"][2] / 2,
        finger_open=float(profile["gripper_open_m"]),
        finger_clear=float(profile["gripper_finger_clear_m"]),
        topdown_approach=float(profile["pregrasp_height_m"]),
    )
    import pyarrow as pa

    event = {
        "type": "INPUT",
        "id": "grasp_pose",
        "value": pa.array(np.asarray(grasp, dtype=np.float32)),  # main() calls .to_numpy
        "metadata": {"target_med": med, "approach_m": approach, "place_tcp_z": place_z},
    }
    return event, np.asarray(profile["home_qpos"], dtype=np.float32)


def _so101_grasp_event() -> tuple[dict, np.ndarray]:
    """The RECORDED seed-0 radial-front grasp from test_so101_motion — the
    official five-axis chain cannot plan top-down at this shelf, so the
    fixture uses the same proven radial pose the so101 stack tests."""
    from aisle.nodes.grasp_topdown import place_tcp_z
    from aisle.scenes.pharmacy import load_meds, load_physics

    physics = load_physics()
    profile = physics["embodiment"]["so101"]
    size = load_meds()["amoxicillin"]["size"]
    tray_top = profile["tray_pos"][2] + profile["tray_size"][2] / 2
    grasp = np.array(
        [
            0.39851335,
            0.00547770,
            0.095,
            0.49655211,
            0.50342429,
            0.50342429,
            0.49655211,
        ],
        dtype=np.float32,
    )
    import pyarrow as pa

    event = {
        "type": "INPUT",
        "id": "grasp_pose",
        "value": pa.array(grasp),  # main() calls .to_numpy
        "metadata": {
            "target_med": "amoxicillin",
            "approach_m": 0.13852643,
            "place_tcp_z": place_tcp_z(size, 0.0, tray_top),
        },
    }
    return event, np.asarray(profile["home_qpos"], dtype=np.float32)


def _run_node(monkeypatch, env: dict, grasp_event: dict, home_full: np.ndarray):
    import sys

    import aisle.turn_node

    class _Silence:
        def write(self, _):
            return None

        def flush(self):
            return None

    monkeypatch.setattr(sys, "stderr", _Silence())
    fake = IdealActuatorNode(grasp_event, home_full)
    monkeypatch.setattr(aisle.turn_node, "Node", lambda *a, **k: fake)
    import os

    import aisle.nodes.ik_trajectory as ikt

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        ikt.main()  # one drive; the iterator ends at StopIteration
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return fake


def test_plan_done_declared_parses_the_env_list():
    """HAR-3/ADR-30: the declared set comes from AISLE_TURN_OUTPUTS —
    the same variable the typed host pins from the graph config and the
    dora-genesis bridge already guards against."""
    assert plan_done_declared("joint_cmd,gripper_cmd,plan_done,turn_done") is True
    assert plan_done_declared("joint_cmd,gripper_cmd,turn_done") is False
    assert plan_done_declared("") is False


def test_grasp_completion_does_not_send_undeclared_plan_done(monkeypatch):
    """G1-F4 (lane1 smoke): with the expert_t0 declaration — no plan_done
    in AISLE_TURN_OUTPUTS — a grasp plan running to completion without a
    verdict sends joint_cmd/gripper_cmd only: no ProtocolError path, no
    plan_done emission. The node idles instead of dying."""
    event, home = _g1_grasp_event()
    fake = _run_node(
        monkeypatch,
        {
            "AISLE_EMBODIMENT": "g1_right",
            "AISLE_LOCKSTEP": "1",
            "AISLE_TURN_NODE": "ik-trajectory",
            "AISLE_TURN_OUTPUTS": "joint_cmd,gripper_cmd,turn_done",
        },
        event,
        home,
    )
    topics = [t for t, _, _ in fake.sent]
    assert "joint_cmd" in topics and "gripper_cmd" in topics, topics[:8]
    assert "plan_done" not in topics


def test_grasp_completion_sends_plan_done_when_declared(monkeypatch):
    """HAR-3 unchanged: with plan_done declared (the 15 eval/shield/expert
    graphs that wire it), the completed grasp plan still emits it — the
    retry-arming signal those graphs consume."""
    event, home = _g1_grasp_event()
    fake = _run_node(
        monkeypatch,
        {
            "AISLE_EMBODIMENT": "g1_right",
            "AISLE_LOCKSTEP": "1",
            "AISLE_TURN_NODE": "ik-trajectory",
            "AISLE_TURN_OUTPUTS": "joint_cmd,gripper_cmd,plan_done,turn_done",
        },
        event,
        home,
    )
    dones = [v for t, v, _ in fake.sent if t == "plan_done"]
    assert dones, [t for t, _, _ in fake.sent][:8]
    assert json.loads(dones[0].to_pylist()[0]) == {"ok": True}


def test_so101_grasp_completion_respects_the_declaration(monkeypatch):
    """Regression guard: the same declaration logic governs so101 —
    verdict-first successes were never affected, but the failure path
    (plan runs out, no verdict) must not crash it either. Drives the
    recorded radial-front grasp to completion."""
    event, home = _so101_grasp_event()
    fake = _run_node(
        monkeypatch,
        {
            "AISLE_EMBODIMENT": "so101",
            "AISLE_LOCKSTEP": "1",
            "AISLE_TURN_NODE": "ik-trajectory",
            "AISLE_TURN_OUTPUTS": "joint_cmd,gripper_cmd,turn_done",
        },
        event,
        home,
    )
    topics = [t for t, _, _ in fake.sent]
    assert "joint_cmd" in topics, topics[:8]
    assert "plan_done" not in topics
