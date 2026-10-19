"""g1-driver contract behavior over loopback buses (G1-D; HWP-5/6/7/8,
TC-6; ADR-g1-driver).

CON-12: no hardware, no unitree_sdk2py import. The bus is injected
(`make_buses()`), so every lease/weight/mode/reset behavior below is
testable without a robot, and the loopback doubles are seeded and
clock-injected (CON-5)."""

import sys

import numpy as np
import pytest

from aisle.embodiment import G1_RIGHT_ARM_JOINTS, G1_RIGHT_JOINTS
from aisle.hardware.adapters import SCENARIOS
from aisle.hardware.adapters import run_scenario as run_hw_scenario
from aisle.nodes.g1_driver import (
    DT_NS,
    JOINT_NAMES,
    MODE_MACHINE_REQUIRED,
    G1Driver,
    LoopbackArmBus,
    LoopbackGripperBus,
    clamp_step,
    fingers_from_grip,
    grip_from_fingers,
    make_buses,
)

pytestmark = pytest.mark.unit


def test_clamp_step_bounds_every_joint_per_tick():
    """HWP-5: the driver-level per-tick delta clamp bounds each of the 7
    right-arm joints whatever arrives on the wire."""
    current = np.zeros(7, dtype=np.float32)
    wild = np.array([2.0, -2.0, 0.01, 0.0, 3.0, -0.02, 1.0], dtype=np.float32)
    stepped = clamp_step(current, wild)
    assert float(np.abs(stepped).max()) <= 0.05 + 1e-7
    assert stepped[2] == pytest.approx(0.01)


def test_loopback_arm_tracks_commands_with_first_order_lag():
    """HWP-8: the loopback double is a first-order lag over the frozen
    g1_right wire order — one tick, one bounded step."""
    bus = LoopbackArmBus()
    assert bus.read_state()["q"].shape == (7,)
    bus.write_arm(np.zeros(7, dtype=np.float32), 1.0)
    bus.write_arm(np.full(7, 0.3, dtype=np.float32), 1.0)
    first = bus.read_state()["q"]
    assert float(np.abs(first).max()) <= 0.05 + 1e-7
    for _ in range(40):
        state = bus.read_state()
    assert np.allclose(state["q"], 0.3, atol=1e-6)
    assert state["mode_machine"] == MODE_MACHINE_REQUIRED


def test_make_buses_loopback_never_imports_the_sdk():
    """HWP-8/CON-12: `loopback` buses are pure doubles — CI has no
    unitree_sdk2py, so importing it there must be impossible in principle
    (same guard as test_so101_driver.py:39)."""
    with pytest.raises(ValueError, match="AISLE_G1_NETWORK_INTERFACE"):
        make_buses("", "")
    before = set(sys.modules)
    arm, gripper = make_buses("loopback", "loopback")
    assert isinstance(arm, LoopbackArmBus) and isinstance(gripper, LoopbackGripperBus)
    added = set(sys.modules) - before
    assert not any(m.startswith("unitree_sdk2py") or m.startswith("cyclonedds") for m in added), (
        "loopback must never import unitree_sdk2py/cyclonedds (CI has no G1 hardware extra)"
    )


def test_weight_ramps_gradually_on_takeover_and_release():
    """G1-D: arm_sdk takeover weight ramps 0→1 over the configurable
    takeover window (never a step), and release ramps back down — no
    torque step in either direction."""
    clock = [0]
    arm, grip = LoopbackArmBus(), LoopbackGripperBus()
    driver = G1Driver(arm, grip, lease_s=10.0, takeover_s=0.5, now_ns=lambda: clock[0])
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    weights = []
    for _ in range(60):
        clock[0] += DT_NS
        out = driver.tick()
        weights.append(out["weight"])
    assert weights[0] < 0.05  # first tick is a fraction of the window
    assert weights[-1] == pytest.approx(1.0)
    assert all(
        w2 - w1 <= DT_NS / 0.5e9 + 1e-9 for w1, w2 in zip(weights, weights[1:], strict=False)
    )
    clock[0] += int(10e9)  # lease expires
    driver.tick()
    released = []
    for _ in range(60):
        clock[0] += DT_NS
        released.append(driver.tick()["weight"])
    assert released[-1] == 0.0


def test_lease_expiry_stops_tracking_and_never_auto_rearms():
    """HWP-6/G1-D: past AISLE_G1_LEASE_S with no new command the driver
    stops following and ramps the weight to 0 with the reason recorded;
    later commands do NOT retake the arm — only an explicit re-arm does."""
    clock = [0]
    arm, grip = LoopbackArmBus(), LoopbackGripperBus()
    driver = G1Driver(arm, grip, lease_s=0.2, takeover_s=0.1, now_ns=lambda: clock[0])
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    for _ in range(20):
        clock[0] += DT_NS
        driver.tick()
    assert driver.weight > 0.9
    clock[0] += int(0.5e9)  # past the 0.2 s lease
    out = driver.tick()
    assert "lease_expired" in out["reason"]
    for _ in range(30):
        clock[0] += DT_NS
        out = driver.tick()
    assert out["weight"] == 0.0 and driver.armed is False
    driver.on_joint_cmd(np.full(7, 0.4, dtype=np.float32))  # new traffic arrives…
    for _ in range(30):
        clock[0] += DT_NS
        out = driver.tick()
    assert out["weight"] == 0.0, "a fresh command must not silently re-take the arm"
    assert not np.allclose(arm.read_state()["q"], 0.4)
    rows = [r for r in driver.telemetry if r["reason"]]
    assert rows and all("lease_expired" in r["reason"] for r in rows)


def test_reset_refuses_but_is_the_explicit_rearm_path():
    """TC-6: hardware has no teleport, so reset always REFUSES; the refusal
    is also the only explicit re-arm after a lease expiry (ADR-g1-driver)."""
    clock = [0]
    arm, grip = LoopbackArmBus(), LoopbackGripperBus()
    driver = G1Driver(arm, grip, lease_s=0.2, takeover_s=0.1, now_ns=lambda: clock[0])
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    clock[0] += int(0.5e9)
    driver.tick()
    assert driver.armed is False
    payload, metadata = driver.on_reset(request_id="r7")
    assert payload["refused"] is True and "no teleport" in payload["reason"]
    assert metadata["request_id"] == "r7"
    assert driver.armed is True and driver.weight == 0.0
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    for _ in range(12):  # stay inside the 0.2 s lease while the weight ramps
        clock[0] += DT_NS
        driver.tick()
    assert driver.weight > 0.9


def test_mode_machine_mismatch_refuses_torque():
    """G1-D: mode_machine must be 15 (g1_29dof_mode_15_with_dex1_1);
    anything else refuses torque and names the reason in bridge_info."""
    clock = [0]
    arm = LoopbackArmBus(mode_machine=7)
    grip = LoopbackGripperBus()
    driver = G1Driver(arm, grip, lease_s=10.0, takeover_s=0.1, now_ns=lambda: clock[0])
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    for _ in range(40):
        clock[0] += DT_NS
        out = driver.tick()
    assert out["weight"] == 0.0
    assert "mode_mismatch" in out["reason"] and "15" in out["reason"]
    assert arm.last_weight == 0.0, "no nonzero arm_sdk weight may reach the bus"


def test_joint_state_is_nine_and_names_match_the_wire():
    """TC-5/HWP-7: joint_state carries wire 7+2 in the frozen g1_right
    order — 7 right-arm radians then the two Dex1-1 finger meters."""
    clock = [0]
    arm, grip = LoopbackArmBus(), LoopbackGripperBus()
    driver = G1Driver(arm, grip, lease_s=10.0, now_ns=lambda: clock[0])
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    driver.on_gripper_cmd(1.0)
    for _ in range(80):
        clock[0] += DT_NS
        out = driver.tick()
    q = out["joint_state"]
    assert q.shape == (9,)
    assert JOINT_NAMES == G1_RIGHT_JOINTS  # the wire order the node speaks
    assert G1_RIGHT_ARM_JOINTS == tuple(JOINT_NAMES[:7])
    assert np.allclose(q[:7], 0.0, atol=1e-6)  # the arm tracks its commanded zeros
    assert out["gripper_state"] == pytest.approx(1.0, abs=1e-3)
    assert q[7] == pytest.approx(q[8], abs=1e-6)  # symmetric fingers
    assert q[7] > 0.0  # open end of the URDF travel


def test_gripper_normalization_round_trips():
    """G1-D: gripper_cmd/gripper_state are NORMALIZED 0..1 (closed→open);
    the finger positions derive from the URDF travel and invert exactly."""
    for norm in (0.0, 0.25, 0.5, 0.9, 1.0):
        fingers = fingers_from_grip(norm)
        assert fingers.shape == (2,)
        assert grip_from_fingers(fingers) == pytest.approx(norm, abs=1e-6)
    assert fingers_from_grip(0.0)[0] == pytest.approx(-0.02)  # URDF lower limit
    assert fingers_from_grip(1.0)[0] == pytest.approx(0.0245)  # URDF upper limit
    assert 0.0 <= grip_from_fingers(np.array([0.5, 0.5])) <= 1.0


def test_loopback_gripper_lags_and_is_seeded_deterministic():
    """HWP-8/CON-5: the gripper double is a seeded first-order lag — same
    seed, same current/temperature telemetry."""
    grip = LoopbackGripperBus(seed=3)
    grip.write(1.0)
    assert abs(grip.read() - 0.0) < 0.15  # not instantaneous
    for _ in range(50):
        value = grip.read()
    assert value == pytest.approx(1.0, abs=1e-3)
    a = LoopbackGripperBus(seed=9)
    b = LoopbackGripperBus(seed=9)
    a.write(0.7), b.write(0.7)
    assert [a.read_state() for _ in range(5)] == [b.read_state() for _ in range(5)]


def test_bridge_info_clock_is_hardware_monotonic():
    """HWP-7: the driver's stamps are the hardware monotonic clock,
    honestly labeled — never described as simulator time."""
    clock = [123_000]
    arm, grip = LoopbackArmBus(), LoopbackGripperBus()
    driver = G1Driver(arm, grip, lease_s=10.0, now_ns=lambda: clock[0])
    driver.on_joint_cmd(np.zeros(7, dtype=np.float32))
    out = driver.tick()
    assert out["bridge_info"]["clock"] == "hardware_monotonic"
    assert out["sim_time_ns"] >= 123_000


def test_hardware_scenarios_run_at_g1_dof_and_mode_mismatch_covers_g1():
    """HWP-8/G1-D: all 15 scenarios plus the G1-only mode_mismatch run at
    dof=9 with byte-reproducible telemetry (same seed ⇒ same result)."""
    names = list(SCENARIOS) + ["mode_mismatch"]
    assert len(names) == 16
    for name in names:
        first = run_hw_scenario(name, seed=5, dof=9)
        again = run_hw_scenario(name, seed=5, dof=9)
        assert first == again, name
        expected_rows = 6 if name == "evidence_sink_failure" else 12  # sink caps at 6 by design
        assert len(first["rows"]) == expected_rows, name
        assert len(first["rows"][0]["requested_positions"]) == 9
    mismatch = run_hw_scenario("mode_mismatch", seed=5, dof=9)
    assert mismatch["observations"]["holds"] == 12 and mismatch["observations"]["halted"]
    assert all(r["device_error"] == "mode_mismatch" for r in mismatch["rows"])
    with pytest.raises(ValueError, match="unknown scenario"):
        run_hw_scenario("nonsense", dof=9)
    legacy = run_hw_scenario("command_receipt_correlation", seed=0)
    assert len(legacy["rows"][0]["requested_positions"]) == 6  # default dof unchanged
