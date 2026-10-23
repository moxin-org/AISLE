"""Hardware-independent adapters and deterministic doubles (HWP-8; SPEC
520, issue #356).

Injectable bus, camera, clock, estop, and telemetry interfaces with
loopback and replay doubles so the driver contract, telemetry schema, and
drill procedures can be exercised before equipment exists. Every double is
seeded and clock-injected (CON-5); nothing here is physical evidence and
every row it produces carries `evidence_kind: loopback`.

Fixture scenarios cover command/receipt correlation, actuator lag,
saturation, stale and missing state, disconnect/reconnect, dropped frames,
clock skew and reset, calibration mismatch, overcurrent and
overtemperature, lease expiry, estop, and evidence-sink failure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

EVIDENCE_KIND = "loopback"
SO101_DOF = 6
# G1-D: the g1_right wire is 9 DOF (7 right-arm joints + 2 Dex1-1 fingers);
# run_scenario's `dof` parameter selects it, the SO-101 default never moves.
G1_DOF = 9
MAX_STEP_RAD = 0.05
SCENARIOS = (
    "command_receipt_correlation",
    "actuator_lag",
    "saturation",
    "stale_state",
    "missing_state",
    "disconnect_reconnect",
    "dropped_frames",
    "clock_skew",
    "clock_reset",
    "calibration_mismatch",
    "overcurrent",
    "overtemperature",
    "lease_expiry",
    "estop",
    "evidence_sink_failure",
)
# G1-only scenario (G1-D): mode_machine != 15 refuses torque. Kept out of
# SCENARIOS so the SO-101 contract surface (test_hardware_phase pins
# set(out["scenarios"]) == set(SCENARIOS)) is untouched.
G1_SCENARIOS = ("mode_mismatch",)


class Bus(Protocol):
    def write_positions(self, q_cmd: np.ndarray) -> str | None: ...
    def read_state(self) -> dict: ...
    def connected(self) -> bool: ...


class Camera(Protocol):
    def frame(self) -> dict | None: ...


class Clock(Protocol):
    def device_ns(self) -> int | None: ...
    def host_ns(self) -> int: ...


class Estop(Protocol):
    def engaged(self) -> bool: ...


class TelemetrySink(Protocol):
    def write(self, row: dict) -> bool: ...


@dataclass
class DoubleClock:
    """Injected clock with optional skew and a scripted reset."""

    host: int = 0
    device_offset_ns: int = 0
    step_ns: int = 10_000_000
    reset_at_tick: int | None = None
    tick: int = 0
    device_available: bool = True

    def advance(self) -> None:
        self.tick += 1
        self.host += self.step_ns
        if self.reset_at_tick is not None and self.tick == self.reset_at_tick:
            self.device_offset_ns = -self.host  # device clock restarts at zero

    def device_ns(self) -> int | None:
        return self.host + self.device_offset_ns if self.device_available else None

    def host_ns(self) -> int:
        return self.host


@dataclass
class LoopbackBusDouble:
    """First-order lag toward the last command with scripted faults."""

    dof: int = SO101_DOF
    home: np.ndarray | None = None
    lag_steps: int = 1
    saturate_at: float | None = None
    disconnect_ticks: tuple[int, int] | None = None
    stale_after: int | None = None
    current_a: float = 0.4
    temperature_c: float = 35.0
    tick: int = 0
    q: np.ndarray = field(init=False)
    target: np.ndarray = field(init=False)
    _receipts: int = 0

    def __post_init__(self) -> None:
        base = (
            np.zeros(self.dof, dtype=np.float32)
            if self.home is None
            else np.asarray(self.home, dtype=np.float32)
        )
        self.q = base.copy()
        self.target = base.copy()

    def connected(self) -> bool:
        if self.disconnect_ticks is None:
            return True
        start, end = self.disconnect_ticks
        return not (start <= self.tick < end)

    def write_positions(self, q_cmd: np.ndarray) -> str | None:
        self.tick += 1
        if not self.connected():
            return None
        cmd = np.asarray(q_cmd, dtype=np.float32)
        if self.saturate_at is not None:
            cmd = np.clip(cmd, -self.saturate_at, self.saturate_at)
        self.target = cmd
        self._receipts += 1
        return f"rcpt-{self._receipts}"

    def read_state(self) -> dict:
        if self.stale_after is not None and self.tick > self.stale_after:
            return {"joint_state": None, "encoder_state": None, "stale": True}
        step = MAX_STEP_RAD / max(self.lag_steps, 1)
        delta = np.clip(self.target - self.q, -step, step)
        self.q = (self.q + delta).astype(np.float32)
        return {
            "joint_state": self.q.copy().tolist(),
            "encoder_state": [int(v * 4096) for v in self.q],
            "current_a": self.current_a,
            "temperature_c": self.temperature_c,
            "stale": False,
        }


@dataclass
class ReplayCamera:
    frames: list[dict]
    drop_every: int | None = None
    index: int = 0

    def frame(self) -> dict | None:
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        if self.drop_every and self.index % self.drop_every == 0:
            return None
        return frame


@dataclass
class ScriptedEstop:
    engage_at_tick: int | None = None
    tick: int = 0

    def advance(self) -> None:
        self.tick += 1

    def engaged(self) -> bool:
        return self.engage_at_tick is not None and self.tick >= self.engage_at_tick


@dataclass
class ListSink:
    rows: list[dict] = field(default_factory=list)
    fail_after: int | None = None

    def write(self, row: dict) -> bool:
        if self.fail_after is not None and len(self.rows) >= self.fail_after:
            return False
        self.rows.append(row)
        return True


def run_scenario(name: str, *, seed: int = 0, ticks: int = 12, dof: int = SO101_DOF) -> dict:
    """Drive the doubles through one scenario and return the telemetry rows
    plus the observation the drill or fixture asserts on. `dof` sizes the
    command vectors (SO101_DOF default; G1 runs pass 9) — the scenario
    semantics and the 12-row pin never change with it."""
    if name not in SCENARIOS and name not in G1_SCENARIOS:
        raise ValueError(f"unknown scenario: {name!r}")
    rng = random.Random(seed)
    clock = DoubleClock(
        device_offset_ns=5_000_000 if name == "clock_skew" else 0,
        reset_at_tick=6 if name == "clock_reset" else None,
    )
    bus = LoopbackBusDouble(
        dof=dof,
        lag_steps=4 if name == "actuator_lag" else 1,
        saturate_at=0.2 if name == "saturation" else None,
        disconnect_ticks=(4, 7) if name == "disconnect_reconnect" else None,
        stale_after=5 if name == "stale_state" else None,
        current_a=3.5 if name == "overcurrent" else 0.4,
        temperature_c=80.0 if name == "overtemperature" else 35.0,
    )
    camera = ReplayCamera(
        [{"id": i} for i in range(ticks)], drop_every=3 if name == "dropped_frames" else None
    )
    estop = ScriptedEstop(engage_at_tick=5 if name == "estop" else None)
    sink = ListSink(fail_after=6 if name == "evidence_sink_failure" else None)
    lease_ttl = 3 if name == "lease_expiry" else 10_000
    silence = range(4, 9) if name == "lease_expiry" else range(0)  # producer goes quiet
    calibration_ok = name != "calibration_mismatch"
    mode_ok = name != "mode_mismatch"  # G1-only: mode_machine must be 15 (G1-D)
    current_limit_a, temperature_limit_c = 3.0, 70.0
    last_command_tick, sink_failures, frames_dropped, halted = 0, 0, 0, False
    for tick in range(1, ticks + 1):
        clock.advance()
        estop.advance()
        target = np.array([0.3 * rng.uniform(0.5, 1.0)] * dof, dtype=np.float32)
        lease_expired = tick - last_command_tick > lease_ttl
        over_limit = bus.current_a > current_limit_a or bus.temperature_c > temperature_limit_c
        if tick in silence:
            decision, receipt = "hold", None  # held command: nothing transmitted
        elif estop.engaged() or lease_expired or not calibration_ok or not mode_ok or over_limit:
            decision, receipt = "hold", None
            halted = True
        else:
            receipt = bus.write_positions(target)
            decision = "pass" if receipt else "refuse"
            if receipt:
                last_command_tick = tick
        state = (
            bus.read_state()
            if name != "missing_state"
            else {"joint_state": None, "encoder_state": None, "stale": True}
        )
        if camera.frame() is None:
            frames_dropped += 1
        row = {
            "proposal_id": f"p-{tick}",
            "gateway_decision": decision,
            "receipt_id": receipt,
            "requested_positions": target.tolist(),
            "transmitted_positions": bus.target.tolist() if receipt else None,
            "gripper_command": 0.0,
            "joint_state": state["joint_state"],
            "encoder_state": state["encoder_state"],
            "current_a": state.get("current_a"),
            "temperature_c": state.get("temperature_c"),
            "connection_state": "connected" if bus.connected() else "disconnected",
            "torque_state": "off" if halted else "on",
            "lease_state": "expired" if lease_expired else "held",
            "over_limit": over_limit,
            "device_error": (
                "mode_mismatch"
                if not mode_ok
                else "calibration_mismatch"
                if not calibration_ok
                else None
            ),
            "source_sequence": tick,
            "device_timestamp": clock.device_ns(),
            "host_monotonic_ns": clock.host_ns(),
            "clock_domain": "double",
            "alignment_uncertainty_ns": 1_000_000,
            "evidence_kind": EVIDENCE_KIND,
        }
        if not sink.write(row):
            sink_failures += 1
    return {
        "scenario": name,
        "evidence_kind": EVIDENCE_KIND,
        "rows": sink.rows,
        "observations": {
            "receipts": sum(1 for r in sink.rows if r["receipt_id"]),
            "holds": sum(1 for r in sink.rows if r["gateway_decision"] == "hold"),
            "refusals": sum(1 for r in sink.rows if r["gateway_decision"] == "refuse"),
            "frames_dropped": frames_dropped,
            "sink_failures": sink_failures,
            "halted": halted,
            "lease_expired_ticks": sum(1 for r in sink.rows if r["lease_state"] == "expired"),
            "max_current_a": max((r["current_a"] or 0.0) for r in sink.rows) if sink.rows else None,
            "max_temperature_c": max((r["temperature_c"] or 0.0) for r in sink.rows)
            if sink.rows
            else None,
            "device_clock_regressed": any(
                (a["device_timestamp"] or 0) > (b["device_timestamp"] or 0)
                for a, b in zip(sink.rows, sink.rows[1:], strict=False)
                if a["device_timestamp"] is not None and b["device_timestamp"] is not None
            ),
        },
    }
