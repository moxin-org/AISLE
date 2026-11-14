"""SPEC 210 MOB-4 acceptance (named by the spec) + the MOB-3 motion-sink
gate for base_cmd. No dora, no sim (CON-12): validator logic only."""

import pytest
import yaml
from cli_helpers import run_json

pytestmark = pytest.mark.unit


def _validate(graph, embodiment="mobile"):
    return run_json("aisle.harness.cli", "validate", str(graph), "--embodiment", embodiment)


def _codes(report: dict) -> set[str]:
    return {e["code"] for e in report.get("errors", [])}


def _write(tmp, nodes) -> str:
    path = tmp / "g.yaml"
    path.write_text(yaml.safe_dump({"nodes": nodes}))
    return path


def test_franka_arm_validates_under_mobile():
    """MOB-4: mobile resolves to the franka arm, so a franka-arm node
    validates unchanged under the mobile profile."""
    from aisle.harness.validate import validate_nodes

    manifests = {"ik-trajectory": {"embodiment": {"arm": ["franka"], "gripper": "parallel"}}}
    errors, _ = validate_nodes(
        [{"id": "ik-trajectory"}], manifests, set(), "mobile", allow_unproven=True
    )
    assert not [e for e in errors if e["code"] == "EMBODIMENT_MISMATCH"]


def test_base_requiring_node_mismatches_on_fixed_base():
    """MOB-4: a base-requiring node validates under mobile but is an
    EMBODIMENT_MISMATCH on a fixed-base (franka) graph."""
    from aisle.harness.validate import validate_nodes

    manifests = {"nav-action": {"embodiment": {"arm": ["franka"], "base": ["mobile"]}}}
    ok, _ = validate_nodes([{"id": "nav-action"}], manifests, set(), "mobile", allow_unproven=True)
    assert not [e for e in ok if e["code"] == "EMBODIMENT_MISMATCH"]
    bad, _ = validate_nodes([{"id": "nav-action"}], manifests, set(), "franka", allow_unproven=True)
    assert [e for e in bad if e["code"] == "EMBODIMENT_MISMATCH"]


@pytest.mark.parametrize("port", ["base_cmd", "base_cmd_0"])
def test_unguarded_base_cmd_is_rejected(tmp_path, port):
    """MOB-3 (PR #14 review): base_cmd is a motion sink — a base command
    reaching the bridge without traversing the budget guard is MOTION_UNGATED."""
    graph = _write(
        tmp_path,
        [
            {"id": "nav-action", "outputs": ["base_cmd"]},
            {
                "id": "dora-genesis",
                "inputs": {port: "nav-action/base_cmd"},
                "outputs": ["base_pose"],
            },
        ],
    )
    rc, report = _validate(graph)
    assert "MOTION_UNGATED" in _codes(report)
    assert rc != 0


@pytest.mark.parametrize("port", ["base_cmd", "base_cmd_0"])
def test_guarded_base_cmd_passes_the_motion_gate(tmp_path, port):
    """MOB-3: routed through the budget guard, base_cmd is NOT MOTION_UNGATED."""
    graph = _write(
        tmp_path,
        [
            {"id": "nav-action", "outputs": ["base_cmd"]},
            {
                "id": "budget-guard",
                "inputs": {"base_cmd": "nav-action/base_cmd"},
                "outputs": ["base_cmd_safe"],
            },
            {
                "id": "dora-genesis",
                "inputs": {port: "budget-guard/base_cmd_safe"},
                "outputs": ["base_pose"],
            },
        ],
    )
    _, report = _validate(graph)
    assert "MOTION_UNGATED" not in _codes(report)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"base_pose": "dora-genesis/base_pose"},
        {"tick": "dora/timer/millis/5000"},
    ],
)
def test_mobile_guard_must_wire_pose_and_tick(tmp_path, extra):
    """MOB-3 (PR #14 re-review, retimed by ADR-29): on a mobile graph the
    guard (it outputs base_cmd_safe) MUST wire base_pose (keep-out + the
    watchdog's sim clock) AND tick (BG-5 stats + the fail-closed wall-net
    sweep) — missing either (the realistic partial regression is a stale
    pre-ADR-29 graph shape) silently disables a safety mechanism."""
    graph = _write(
        tmp_path,
        [
            {"id": "nav-action", "outputs": ["base_cmd"]},
            {
                "id": "budget-guard",
                "inputs": {"base_cmd": "nav-action/base_cmd", **extra},
                "outputs": ["base_cmd_safe"],
            },
            {
                "id": "dora-genesis",
                "inputs": {"base_cmd": "budget-guard/base_cmd_safe"},
                "outputs": ["base_pose"],
            },
        ],
    )
    rc, report = _validate(graph)
    assert "MOBILE_GUARD_INCOMPLETE" in _codes(report)
    assert rc != 0


@pytest.mark.parametrize(
    "tick_source",
    [
        "dora/timer/millis/60000",  # slower than the wall-net latency story
        "some-node/heartbeat",  # not a timer at all: never guaranteed to fire
        # issue #160 item 3: the old ad-hoc regex ACCEPTED millis/0 (0 <= max),
        # disagreeing with _parse_timer_hz, which rejects a zero period as
        # malformed everywhere else — the shared parser now governs here too
        "dora/timer/millis/0",
        # PR #177 review: the `dora` prefix is part of the contract. The
        # dedup briefly dropped it, so a node-sourced pseudo-timer parsed
        # as a real 100 Hz timer and passed this safety check — the ADR-29
        # wall-net sweep wired to something that may never fire.
        "some-node/timer/millis/10",
        "evil/timer/millis/1000",
    ],
)
def test_mobile_guard_tick_must_be_a_bounded_timer(tmp_path, tick_source):
    """PR #156 review: the guard's tick is the wall-net sweep's clock — a
    name-only check would let a graph wire it from a never-firing node or a
    10-minute timer and pass validation with the fail-closed net disabled."""
    graph = _write(
        tmp_path,
        [
            {"id": "nav-action", "outputs": ["base_cmd"]},
            {
                "id": "budget-guard",
                "inputs": {
                    "base_cmd": "nav-action/base_cmd",
                    "base_pose": "dora-genesis/base_pose",
                    "tick": tick_source,
                },
                "outputs": ["base_cmd_safe"],
            },
            {
                "id": "dora-genesis",
                "inputs": {"base_cmd": "budget-guard/base_cmd_safe"},
                "outputs": ["base_pose"],
            },
        ],
    )
    rc, report = _validate(graph)
    assert "MOBILE_GUARD_INCOMPLETE" in _codes(report)
    assert rc != 0


def test_mobile_guard_pose_must_come_from_the_bridge(tmp_path):
    """PR #156 review: base_pose is the watchdog's staleness clock and the
    keep-out's ground truth — wired from an arbitrary node, forged or
    absent stamps defeat both. Only a sim_bridge provider qualifies."""
    graph = _write(
        tmp_path,
        [
            {"id": "nav-action", "outputs": ["base_cmd", "base_pose"]},
            {
                "id": "budget-guard",
                "inputs": {
                    "base_cmd": "nav-action/base_cmd",
                    # nav-action does not provide sim_bridge
                    "base_pose": "nav-action/base_pose",
                    "tick": "dora/timer/millis/5000",
                },
                "outputs": ["base_cmd_safe"],
            },
            {
                "id": "dora-genesis",
                "inputs": {"base_cmd": "budget-guard/base_cmd_safe"},
                "outputs": ["base_pose"],
            },
        ],
    )
    rc, report = _validate(graph)
    assert "MOBILE_GUARD_INCOMPLETE" in _codes(report)
    assert rc != 0


def test_complete_mobile_guard_passes_the_wiring_rule(tmp_path):
    """MOB-3: a guard wiring base_pose + tick is complete — the pose
    stream carries keep-out feedback AND the watchdog clock, the stats
    tick carries the wall-net sweep (ADR-29); no dedicated watchdog
    wall-timer input exists anymore (CON-5)."""
    graph = _write(
        tmp_path,
        [
            {"id": "nav-action", "outputs": ["base_cmd"]},
            {
                "id": "budget-guard",
                "inputs": {
                    "base_cmd": "nav-action/base_cmd",
                    "base_pose": "dora-genesis/base_pose",
                    "tick": "dora/timer/millis/5000",
                },
                "outputs": ["base_cmd_safe"],
            },
            {
                "id": "dora-genesis",
                "inputs": {"base_cmd": "budget-guard/base_cmd_safe"},
                "outputs": ["base_pose"],
            },
        ],
    )
    _, report = _validate(graph)
    assert "MOBILE_GUARD_INCOMPLETE" not in _codes(report)
