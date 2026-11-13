"""Unit tests for the return planner's pure geometry (T4 inc-2,
VER-3 amendment `return_item`; CON-12 — no sim, no dora)."""

import pytest

from aisle.nodes.return_planner import return_grasp_and_slot
from aisle.scenes.pharmacy import load_meds, load_physics, resolve_layout

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def geometry():
    return load_meds(), resolve_layout(load_physics(), "franka")


def test_grasp_is_at_the_tray_at_the_meds_half_height(geometry):
    meds, layout = geometry
    grasp, _ = return_grasp_and_slot("amoxicillin", meds, layout)
    tray = layout["tray"]
    assert grasp[0] == pytest.approx(tray["pos"][0])
    assert grasp[1] == pytest.approx(tray["pos"][1])
    tray_top = tray["pos"][2] + tray["size"][2] / 2.0
    assert grasp[2] == pytest.approx(tray_top + meds["amoxicillin"]["size"][2] / 2.0)
    assert grasp[3:] == [0.0, 0.0, 0.0, 1.0]


def test_return_slot_is_inside_the_shelf_front_corner(geometry):
    meds, layout = geometry
    _, (sx, sy) = return_grasp_and_slot("metformin", meds, layout)
    shelf = layout["shelf"]
    assert abs(sx - shelf["pos"][0]) <= shelf["level_size"][0] / 2.0
    assert abs(sy - shelf["pos"][1]) <= shelf["level_size"][1] / 2.0


def test_unknown_med_refuses(geometry):
    meds, layout = geometry
    assert return_grasp_and_slot("aspirin", meds, layout) is None


def test_estimate_grasp_targets_the_measured_pose_not_the_centre(geometry):
    """t4-inc2-recovery-r4 seeds 4/8: the delivered box lies wherever
    release dropped it — a centre-assuming grasp strikes the edge
    (collision) or closes on air (not_returned). With an estimate the
    grasp targets the MEASURED pose; the slot is unchanged."""
    from aisle.nodes.return_planner import return_grasp_from_estimate

    meds, layout = geometry
    est = {"pos": [0.62, -0.13, 0.093]}
    grasp, place_xy = return_grasp_from_estimate(est, "amoxicillin", meds, layout)
    assert grasp[:3] == pytest.approx([0.62, -0.13, 0.093])
    assert grasp[3:] == [0.0, 0.0, 0.0, 1.0]
    _, centre_slot = return_grasp_and_slot("amoxicillin", meds, layout)
    assert place_xy == pytest.approx(centre_slot)


def test_estimate_grasp_refuses_unknown_med(geometry):
    from aisle.nodes.return_planner import return_grasp_from_estimate

    meds, layout = geometry
    assert return_grasp_from_estimate({"pos": [0.6, 0.0, 0.09]}, "aspirin", meds, layout) is None


def test_carried_request_metadata_drops_the_stale_turn_stamp():
    """Run 20260826-173702: the estimate arrives frames after the
    return_request, and re-sending the request's FULL turn stamp is an
    ADR-30 ProtocolError ('unrelated turn stamp') that kills the node —
    the wrapper stamps outputs with the ACTIVE turn itself, so the
    carried metadata must keep only correlation keys."""
    from aisle.nodes.return_planner import carry_request_meta

    request_meta = {
        "goal_id": "ep-0001r",
        "request_id": "ep-0001r/return_request/1",
        "env_id": 0,
        "seq": 7,
        "turn_epoch": 1,
        "turn_id": 2093,
        "sim_time_ns": 20920000000,
    }
    carried = carry_request_meta(request_meta, [0.31, -0.18])
    assert carried["goal_id"] == "ep-0001r"
    assert carried["request_id"] == "ep-0001r/return_request/1"
    assert carried["env_id"] == 0
    assert carried["place_xy"] == [0.31, -0.18]
    for stale in ("turn_epoch", "turn_id", "sim_time_ns", "seq"):
        assert stale not in carried
    assert request_meta["turn_id"] == 2093  # input not mutated
