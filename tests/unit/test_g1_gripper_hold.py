"""The gripper holds through transfer: no triangle-wave open (G1-F6;
HAR-3 retry arming vs ADR-18 latch semantics).

The lane1 smoke 20261020-011256-c50a80 saw ik-trajectory emit a
TRIANGLE gripper_cmd (0.00 -> 0.90 over 23 ticks, straight back down 23
ticks) while the box was lifted to retract height and never carried
across — the latch released mid-air when the ramp fell below its close
floor and the box fell back onto its own slot.

Node-level reproduction with the F4 IdealActuatorNode driving the REAL
main() loop proves the single-plan stage table is CORRECT: the ramp
0.04 -> 0.90 (close) holds flat through lift/retract/transfer (1014
ticks) and only ramps down at the tray during release. The triangle's
down-edge is a RESTART: a plan that finishes via stage bails (tracking
failure in sim) leaves the fingers closed; the NEXT grasp_pose is then
accepted, and its free-space stages (rise/staging/pregrasp/preclose)
command gripper 0.0 — on g1 that is the profile's own
gripper_pregrasp_cmd = 0.0 — opening the still-latched box in mid-air.

Pinned here (SCN-4, BG-3, TC-5):
- the hold window: from close through the tick before release, the
  emitted gripper_cmd is monotone non-decreasing and never dips below
  carry_latch_close once it has reached it (g1_right, so101, franka);
- a duplicate grasp_pose arriving mid-execution does NOT restart the
  plan (one up-ramp, one down-ramp — no second triangle);
- a replan that starts while the gripper is closed keeps grasp_cmd on
  every stage before close (the fix), so the box stays latched through
  the re-approach."""

from __future__ import annotations

import numpy as np
import pytest
from test_ik_plan_done import IdealActuatorNode  # tests/unit is on sys.path (cli_helpers pattern)

pytestmark = pytest.mark.unit


class _Sink:
    def write(self, _):
        return None

    def flush(self):
        return None


def _drive(monkeypatch, env: dict, grasp_event: dict, home_full: np.ndarray, extra_events=None):
    """Run ik-trajectory.main() over one grasp plan (plus optional extra
    events injected AFTER the plan completes); returns the node carrying
    every sent (topic, value, metadata)."""
    import sys

    import aisle.turn_node

    monkeypatch.setattr(sys, "stderr", _Sink())
    node = _SequenceNode(grasp_event, home_full, extra_events or [])
    monkeypatch.setattr(aisle.turn_node, "Node", lambda *a, **k: node)
    import os

    import aisle.nodes.ik_trajectory as ikt

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        ikt.main()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return node


class _SequenceNode(IdealActuatorNode):
    """IdealActuatorNode that also records gripper_cmd values and counts
    plan starts (via the node's own stderr channel, silenced)."""

    def __init__(self, grasp_event, home_full, extra_events):
        super().__init__(grasp_event, home_full)
        self.grip_values: list[float] = []
        self.joint_cmds = 0
        self._extra = extra_events

    def __iter__(self):
        yield self._grasp_event
        import pyarrow as pa

        qpos = self._home
        for _ in range(24000):  # enough ticks for two full plans
            yield {
                "type": "INPUT",
                "id": "joint_state",
                "value": pa.array(qpos.astype(np.float32)),
                "metadata": {},
            }
            qpos = self.last_cmd if self.last_cmd is not None else self._home
            # inject extras after the first plan's window
            if self._extra and self.joint_cmds > 600:
                while self._extra:
                    yield self._extra.pop(0)

    def send_output(self, topic, value, metadata=None):
        super().send_output(topic, value, metadata)
        if topic == "gripper_cmd":
            self.grip_values.append(float(value.to_pylist()[0]))
        if topic == "joint_cmd":
            self.joint_cmds += 1


def _assert_hold_window(grip_values: list[float], latch_close: float) -> None:
    """The close->pre-release window: monotone rise, then NEVER a dip
    below latch_close once reached — the only legal descent starts at the
    release ramp (detected as the final sustained fall to 0)."""
    peak_i = int(np.argmax(grip_values))
    rise = grip_values[: peak_i + 1]
    assert rise == sorted(rise), f"gripper ramp not monotone: {rise[:8]}..."
    assert max(rise) >= latch_close, f"never latched: max {max(rise)} < {latch_close}"
    # every value after the peak stays >= latch_close until the release
    # descent (the final run toward 0); the release ramp itself is the
    # sustained fall, allowed to cross — but ONLY once, monotonically
    below = [i for i, v in enumerate(grip_values[peak_i:], peak_i) if v < latch_close]
    if below:
        first_below = below[0]
        tail = grip_values[first_below:]
        assert tail == sorted(tail, reverse=True) or tail == sorted(tail), (
            "gripper dipped below the latch floor and came back — a reopen "
            f"while holding: {grip_values[first_below - 3 : first_below + 8]}"
        )
        assert 0.0 in tail or tail[-1] < latch_close, tail[-4:]


@pytest.mark.parametrize(
    ("embodiment", "event_builder"),
    [
        ("g1_right", "g1"),
        ("so101", "so101"),
        ("franka", "franka"),
    ],
)
def test_gripper_holds_through_transfer_until_release(monkeypatch, embodiment, event_builder):
    """SCN-4, BG-3, TC-5: from the close ramp to the tick before release,
    the emitted gripper_cmd is monotone non-decreasing and never dips
    below carry_latch_close once reached — the anti-triangle invariant on
    all three desk embodiments (the lane1 smoke saw exactly this dip)."""
    from aisle.scenes.pharmacy import load_physics

    profile = load_physics()["embodiment"][embodiment]
    event, home = _grasp_event_for(embodiment, event_builder)
    node = _drive(
        monkeypatch,
        {
            "AISLE_EMBODIMENT": embodiment,
            "AISLE_LOCKSTEP": "1",
            "AISLE_TURN_NODE": "ik-trajectory",
            "AISLE_TURN_OUTPUTS": "joint_cmd,gripper_cmd,turn_done",
        },
        event,
        home,
    )
    assert node.grip_values, "no gripper_cmd emitted"
    latch_close = float(profile.get("carry_latch_close", profile.get("gripper_grasp_cmd", 0.55)))
    _assert_hold_window(node.grip_values, latch_close)


def test_duplicate_grasp_pose_mid_execution_does_not_restart(monkeypatch):
    """(a)-trigger pin: a grasp_pose arriving while the plan executes is
    IGNORED (the one-plan-at-a-time guard) — the grip sequence shows a
    single up-ramp and a single release descent, no second triangle."""
    from test_ik_plan_done import _g1_grasp_event

    event, home = _g1_grasp_event()
    dup = dict(event)
    node = _drive(
        monkeypatch,
        {
            "AISLE_EMBODIMENT": "g1_right",
            "AISLE_LOCKSTEP": "1",
            "AISLE_TURN_NODE": "ik-trajectory",
            "AISLE_TURN_OUTPUTS": "joint_cmd,gripper_cmd,turn_done",
        },
        event,
        home,
        extra_events=[dup, dup],
    )
    peaks = [
        i
        for i in range(1, len(node.grip_values))
        if node.grip_values[i] >= 0.85 and node.grip_values[i - 1] < 0.85
    ]
    assert len(peaks) == 1, f"duplicate grasp_pose restarted the plan: {node.grip_values}"


def test_replan_while_held_keeps_the_grip_closed():
    """The fix, unit-pinned: `_held_replan_stages` rewrites every stage
    BEFORE close (and close itself) to command grasp_cmd when the replan
    starts with the gripper closed — the box stays latched through the
    re-approach instead of opening in mid-air (G1's profile
    gripper_pregrasp_cmd is 0.0, which is what made the restart open)."""
    from aisle.nodes.ik_trajectory import Stage, _held_replan_stages

    stages = [
        Stage("rise", (np.zeros(7, dtype=np.float32),), 0.0, 0.1),
        Stage("pregrasp", (np.zeros(7, dtype=np.float32),), 0.0, 0.2),
        Stage("preclose", (np.zeros(7, dtype=np.float32),), 0.0, 0.2),
        Stage("close", (np.zeros(7, dtype=np.float32),), 0.55, 0.5),
        Stage("lift", (np.zeros(7, dtype=np.float32),), 0.55, 0.2),
        Stage("release", (np.zeros(7, dtype=np.float32),), 0.0, 1.5),
    ]
    held = _held_replan_stages(stages, grasp_cmd=0.55)
    before_close = [s for s in held if s.name in ("rise", "pregrasp", "preclose")]
    assert all(s.gripper == 0.55 for s in before_close), [s.gripper for s in held]
    # from close on: unchanged semantics (close holds, release still opens)
    assert [s.gripper for s in held if s.name in ("close", "lift", "release")] == [
        0.55,
        0.55,
        0.0,
    ]
    # untouched when not held: identity
    same = _held_replan_stages(stages, grasp_cmd=0.55, held=False)
    assert [s.gripper for s in same] == [s.gripper for s in stages]


def _grasp_event_for(embodiment: str, kind: str):
    if kind == "g1":
        from test_ik_plan_done import _g1_grasp_event

        return _g1_grasp_event()
    if kind == "so101":
        from test_ik_plan_done import _so101_grasp_event

        return _so101_grasp_event()
    # franka: the recorded top-down probe pose from test_so101-style units,
    # planned through the same planner entry the expert graph feeds
    import pyarrow as pa

    from aisle.nodes.grasp_topdown import GRIP_ENGAGEMENT, plan_grasp
    from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout

    physics = load_physics()
    profile = physics["embodiment"]["franka"]
    layout = resolve_layout(physics, "franka")
    meds = load_meds()
    size = meds["amoxicillin"]["size"]
    shelf = layout["shelf"]
    z_lo = shelf["pos"][2] + shelf["level_heights"][0] + shelf["board_thickness"] / 2
    pose = np.array([0.42, 0.0, z_lo + size[2] / 2, 0, 0, 0, 1], dtype=np.float32)
    grasp, approach, place_z = plan_grasp(
        pose,
        size,
        grip=GRIP_ENGAGEMENT,
        shelf_front_x=shelf["pos"][0] - shelf["level_size"][0] / 2,
        tray_top_z=profile["tray_pos"][2] + profile["tray_size"][2] / 2,
        finger_open=float(profile["gripper_open_m"]),
        finger_clear=float(profile["gripper_finger_clear_m"]),
        topdown_approach=float(
            profile.get("pregrasp_height_m", physics["ik"]["pregrasp_height_m"])
        ),
    )
    event = {
        "type": "INPUT",
        "id": "grasp_pose",
        "value": pa.array(np.asarray(grasp, dtype=np.float32)),
        "metadata": {
            "target_med": "amoxicillin",
            "approach_m": approach,
            "place_tcp_z": place_z,
        },
    }
    return event, np.asarray(profile["home_qpos"], dtype=np.float32)
