"""G1-F2: ik-trajectory and grasp-planner-topdown on g1_right (SCN-4,
BG-3, TC-5; ADR-g1-embodiment scope A).

The 7-DOF right arm solves full-pose DLS-IK on the G1-F1 world-frame
chain, plans the top-down staged pick/place the C1 scene assumes (rpy
[0, pi/2, 0] palm-down TCP), and every planned waypoint must satisfy the
G1-F1 guard: inside the workspace AABB, zero violations along the path."""

from __future__ import annotations

import numpy as np
import pytest
from cli_helpers import REPO_ROOT, run_json

from aisle.nodes.budget_guard import clamp_joint_cmd, fk_ee_pose, load_limits
from aisle.nodes.grasp_topdown import GRIP_ENGAGEMENT, plan_grasp
from aisle.nodes.ik_trajectory import StagedPlan, ik_solve, interpolate_step, topdown_rotation
from aisle.scenes.pharmacy import (
    MED_NAMES,
    load_meds,
    load_physics,
    resolve_layout,
    sample_placements,
)

pytestmark = pytest.mark.unit


def run_validate(graph, *args: str) -> tuple[int, dict]:
    """The same CLI wrapper test_validator.py uses."""
    return run_json("aisle.harness.cli", "validate", str(graph), *args)


PHYSICS = load_physics()
PROFILE = PHYSICS["embodiment"]["g1_right"]
LAYOUT = resolve_layout(PHYSICS, "g1_right")
MEDS = load_meds()
LIMITS = load_limits("g1_right")
HOME = np.asarray(LIMITS.fallback_qpos, dtype=np.float32)
TRAY_TOP = PROFILE["tray_pos"][2] + PROFILE["tray_size"][2] / 2


def _grasp_pose(placements, med_name: str, grip: float | None = None):
    target = next(p for p in placements if p.name == med_name)
    pose = np.array([target.x, target.y, target.z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    neighbours = [
        [p.x, p.y, MEDS[p.name]["size"][0] / 2, MEDS[p.name]["size"][1] / 2]
        for p in placements
        if p.name != target.name
    ]
    return plan_grasp(
        pose,
        MEDS[med_name]["size"],
        grip=float(PROFILE.get("grip_engagement_m", GRIP_ENGAGEMENT)) if grip is None else grip,
        shelf_front_x=LAYOUT["shelf"]["pos"][0] - LAYOUT["shelf"]["level_size"][0] / 2,
        tray_top_z=TRAY_TOP,
        neighbours=neighbours,
        finger_open=float(PROFILE["gripper_open_m"]),
        finger_clear=float(PROFILE["gripper_finger_clear_m"]),
        topdown_approach=float(PROFILE["pregrasp_height_m"]),
    )


@pytest.mark.parametrize(
    ("seed", "med"),
    [(0, "cetirizine"), (2, "amoxicillin"), (7, "metformin")],
)
def test_g1_ik_reaches_scene_placements_topdown(seed, med):
    """SCN-4, TC-5, CON-5: deterministic 7-DOF full-pose IK reaches the C1
    scene's top-down grasp poses — position residual inside the profile's
    ik_pos_tol_m, tool z pointing down (palm-down TCP), same input =>
    same output."""
    placements = sample_placements(seed, MED_NAMES, LAYOUT)
    grasp, approach, _ = _grasp_pose(placements, med)
    target = grasp[:3].astype(np.float64)
    desired = topdown_rotation(0.0)
    q = ik_solve(target, desired, HOME[:7], embodiment="g1_right")
    assert q is not None, (seed, med)
    pos, rotation = fk_ee_pose(q, embodiment="g1_right")
    assert float(np.linalg.norm(pos - target)) < float(PROFILE["ik_pos_tol_m"]), (
        seed,
        med,
        float(np.linalg.norm(pos - target)),
    )
    assert float(rotation[2, 2]) == pytest.approx(-1.0, abs=0.05), "tool z must point down"
    again = ik_solve(target, desired, HOME[:7], embodiment="g1_right")
    assert np.array_equal(q, again)
    assert q.shape == (7,)


def test_g1_ik_reaches_tray_transfer_envelope():
    """SCN-4, BG-3: the tray hover band (transfer height over the delivery
    tray) is reachable with the palm-down place orientation."""
    target = np.array(
        [PROFILE["tray_pos"][0], PROFILE["tray_pos"][1], float(PROFILE["trajectory_transfer_z"])],
        dtype=np.float64,
    )
    q = ik_solve(target, topdown_rotation(0.0), HOME[:7], embodiment="g1_right")
    assert q is not None
    pos, _ = fk_ee_pose(q, embodiment="g1_right")
    assert float(np.linalg.norm(pos - target)) < float(PROFILE["ik_pos_tol_m"])


def test_g1_staged_plan_topdown_uses_seven_joints_and_profile_commands():
    """SCN-4, TC-5, BG-3: the unchanged staged pick/place sequence solves
    for the G1 chain — every waypoint is 7 joints, gripper stages carry
    the profile's commands, the lift keeps height, and the plan returns
    to the wire home. Slot: seed-7 amoxicillin at the near column
    [0.28, 0.08] (the so101 mirror tests ONE pose too). KNOWN LIMITATION
    (G1-F2, measured): the cross-body far column [0.24, +0.16] — right
    arm reaching across the torso — stalls the transfer-stage DLS in an
    elbow-family switch; every Cartesian route and the joint-space sweep
    fail a safety gate (clearance or palm flip), so the plan fails LOUD
    there (plan.error set, nothing executes) pending a routed profile;
    see ADR-g1-embodiment."""
    placements = sample_placements(7, MED_NAMES, LAYOUT)
    grasp, approach, place_z = _grasp_pose(placements, "amoxicillin")
    plan = StagedPlan(
        grasp,
        tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
        approach_m=approach,
        q_seed=HOME,
        place_z=place_z,
        embodiment="g1_right",
    )
    assert plan.ok, plan.error
    assert all(np.asarray(q).shape == (7,) for stage in plan.stages for q in stage.path), (
        "the G1 wire arm is seven joints"
    )
    stages = {stage.name: stage for stage in plan.stages}
    assert stages["close"].gripper == pytest.approx(float(PROFILE["gripper_grasp_cmd"]))
    assert stages["preclose"].gripper == pytest.approx(float(PROFILE["gripper_pregrasp_cmd"]))
    assert stages["advance"].gripper == pytest.approx(float(PROFILE["gripper_pregrasp_cmd"]))
    advance_q = stages["advance"].q
    lift_q = stages["lift"].q
    lift_gain = float(
        np.linalg.norm(fk_ee_pose(lift_q, embodiment="g1_right")[0])
        - np.linalg.norm(fk_ee_pose(advance_q, embodiment="g1_right")[0])
    )
    assert lift_gain > 0.0, "the lift stage must rise"
    assert (
        fk_ee_pose(lift_q, embodiment="g1_right")[0][2]
        > fk_ee_pose(advance_q, embodiment="g1_right")[0][2]
    )
    assert plan.stages[-1].q == pytest.approx(HOME[:7])
    assert stages["transfer"].vel == pytest.approx(float(PROFILE["trajectory_transfer_vel_scale"]))


def test_g1_plan_waypoints_stay_in_workspace_and_guard_clean():
    """BG-3 (G1-F1 chained in): every planned waypoint's TCP lies inside
    the workspace AABB, and the STREAM the executor actually publishes —
    per-tick velocity-bounded interpolation between waypoints (StageStreamer
    semantics: max_vel * dt steps, dt = TC-4 cadence) — produces ZERO guard
    violations. Feeding waypoint jumps straight to the guard would test the
    executor's interpolator away; this tests the guard against the real
    command sequence."""
    placements = sample_placements(2, MED_NAMES, LAYOUT)
    grasp, approach, place_z = _grasp_pose(placements, "cetirizine")
    plan = StagedPlan(
        grasp,
        tuple(np.asarray(PROFILE["tray_pos"][:2], dtype=float)),
        approach_m=approach,
        q_seed=HOME,
        place_z=place_z,
        embodiment="g1_right",
    )
    assert plan.ok, plan.error
    lo = np.asarray(LIMITS.workspace_min)
    hi = np.asarray(LIMITS.workspace_max)
    fingers_open = float(PROFILE["gripper_open_qpos"])
    prev = np.concatenate([HOME[:7], [fingers_open, fingers_open]]).astype(np.float32)
    current = HOME[:7].copy()
    max_vel, dt = 1.0, 0.01  # the manifest's max_joint_vel_rad_s at TC-4 cadence
    for stage in plan.stages:
        for q in stage.path:
            target_q = np.asarray(q, dtype=np.float32)
            tcp = fk_ee_pose(target_q, embodiment="g1_right")[0]
            assert np.all(lo <= tcp) and np.all(tcp <= hi), (stage.name, tcp)
            while not np.allclose(current, target_q, atol=1e-6):
                current = interpolate_step(current, target_q, max_vel, dt)
                full = np.concatenate([current, [fingers_open, fingers_open]]).astype(np.float32)
                safe, violations = clamp_joint_cmd(full, prev, LIMITS, timed_out=False)
                assert violations == [], (stage.name, violations)
                assert np.all(safe >= np.asarray(LIMITS.q_min, dtype=np.float32) - 1e-6)
                prev = safe


def test_g1_manifests_open_and_only_bridge_manifests_still_mismatch():
    """VAL-2: with ik-trajectory and grasp-planner-topdown open to
    g1_right, the T0 graph under --embodiment g1_right reports
    EMBODIMENT_MISMATCH for at most the seven bridge manifests lane1's
    G1-C2 owns (dora-genesis, arm-driver-sim, gripper-driver-sim,
    camera-source, reset, verifier-oracle, task-state-machine) and
    nothing else."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t0.yaml", "--embodiment", "g1_right")
    mismatched = {e["node"] for e in report["errors"] if e["code"] == "EMBODIMENT_MISMATCH"}
    bridge = {
        "dora-genesis",
        "arm-driver-sim",
        "gripper-driver-sim",
        "camera-source",
        "reset",
        "verifier-oracle",
        "task-state-machine",
    }
    assert mismatched <= bridge, sorted(mismatched - bridge)
    assert "ik-trajectory" not in mismatched and "grasp-planner-topdown" not in mismatched


def test_executor_speed_reads_the_g1_profile_unless_env_overrides():
    """G1-F11 (per-episode wall budget): the executor's joint-velocity cap
    comes from AISLE_MAX_JOINT_VEL when set, else the embodiment profile's
    executor_max_joint_vel_rad_s, else the historical 1.0 — so franka and
    so101 (no key) keep 1.0, and g1_right stays under its 2.0 rad/s guard
    envelope (qdot_max)."""
    from aisle.nodes.budget_guard import load_limits
    from aisle.nodes.ik_trajectory import executor_max_vel

    g1 = PHYSICS["embodiment"]["g1_right"]
    assert executor_max_vel({}, g1) == pytest.approx(float(g1["executor_max_joint_vel_rad_s"]))
    assert executor_max_vel({}, g1) < max(load_limits("g1_right").qdot_max[:7])
    assert executor_max_vel({"AISLE_MAX_JOINT_VEL": "0.7"}, g1) == pytest.approx(0.7)
    for other in ("franka", "so101"):
        assert executor_max_vel({}, PHYSICS["embodiment"][other]) == pytest.approx(1.0)
