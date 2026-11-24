"""spec-so101-planner: StagedPlan waypoints -> SPEC move requests."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from aisle.nodes.spec_so101_planner import NS, Planner, gripper_percent, plan_steps


def stage(name, path_deg, gripper=0.0, vel=1.0):
    return SimpleNamespace(
        name=name,
        path=tuple(np.radians(np.asarray(p, dtype=np.float64)) for p in path_deg),
        gripper=gripper,
        vel=vel,
    )


def test_gripper_mapping_inverts_aisle_closure():
    assert gripper_percent(0.0) == 100.0
    assert gripper_percent(1.0) == 0.0
    assert gripper_percent(0.6) == pytest.approx(40.0)
    assert gripper_percent(1.5) == 0.0


def test_thinning_keeps_stage_ends_and_bounds_each_hop():
    dense = [[i, 0, 0, 0, 0] for i in range(0, 31)]  # 1 deg steps to 30 deg
    steps = plan_steps([stage("advance", dense), stage("close", [[30, 0, 0, 0, 0]], 1.0)], 8.0, 60)
    pans = [s.params["joints"]["shoulder_pan"] for s in steps]
    assert pans[0] == pytest.approx(0.0) and pans[-2] == pytest.approx(30.0)
    assert all(b - a <= 8.0 + 1e-6 for a, b in zip(pans, pans[1:], strict=False))
    assert steps[-1].stage == "close" and steps[-1].params["joints"]["gripper"] == 0.0
    assert all(s.verb == f"{NS}.move_to_joint_state" for s in steps)


def test_speed_follows_stage_velocity_scale_and_cap():
    steps = plan_steps([stage("transfer", [[10, 0, 0, 0, 0]], vel=0.35)], 8.0, 60)
    assert steps[0].params["speed_deg_s"] == pytest.approx(math.degrees(0.35), abs=0.01)
    fast = plan_steps([stage("rise", [[10, 0, 0, 0, 0]], vel=2.0)], 8.0, 60)
    assert fast[0].params["speed_deg_s"] == 60


def test_planner_sends_one_request_at_a_time_and_aborts_on_tracking_error():
    sent = []
    planner = Planner(
        lambda topic, value: sent.append(json.loads(value.to_pylist()[0])),
        stride_deg=8.0,
        max_speed_deg_s=60,
    )
    planner.queue = plan_steps(
        [stage("a", [[0, 0, 0, 0, 0]]), stage("b", [[20, 0, 0, 0, 0]])], 8.0, 60
    )
    planner.on_turn()
    planner.on_turn()
    assert len(sent) == 1
    line = planner.on_response({"request_id": sent[0]["request_id"], "ok": True, "code": "0"}, 1.0)
    assert line.startswith("stage done: a")
    planner.on_turn()
    assert len(sent) == 2
    line = planner.on_response(
        {"request_id": sent[1]["request_id"], "ok": False, "code": "TRACKING_ERROR", "msg": "x"},
        2.0,
    )
    assert line.startswith("plan aborted at b") and not planner.busy
