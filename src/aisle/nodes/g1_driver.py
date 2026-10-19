"""g1-driver node (G1-D, ADR-g1-driver).

The Unitree G1 hardware bridge behind the SAME topic surface as
so101-driver (SPEC 010 sim->real node swap): `joint_state`/`gripper_state`
out at the TC-4 cadence, `joint_cmd`/`gripper_cmd`/`reset` in, TC-2 stamps
on a hardware monotonic clock (HWP-7). The wire order is the frozen
`g1_right` profile (TC-5): 7 right-arm joints + 2 Dex1-1 finger positions.

Three SO-101 gaps are closed here by design (outer-loop entry #2):
- a lease watchdog: no command within AISLE_G1_LEASE_S stops following and
  ramps the arm_sdk weight to 0, and a LATER command never silently
  re-takes the arm — only the reset reply path re-arms explicitly;
- `gripper_cmd` is normalized 0..1 (closed→open), never radians in a joint
  slot, and `gripper_state` reports the SAME normalized scalar back;
- `mode_machine` must be 15 (asset `g1_29dof_mode_15_with_dex1_1`); any
  other value refuses torque and says why in `bridge_info`/telemetry.

Upstream facts cited as file:line@commit:
- unitre_sdk2_python @ 814556d15970dd2ecf1c9984e845ca02ab07e206
  (`unitre_sdk2py`, install_requires `cyclonedds==0.10.2`, setup.py): the
  arm-SDK surface is `rt/arm_sdk` (`LowCmd_`) / `rt/lowstate` (`LowState_`)
  — example/g1/high_level/g1_arm7_sdk_dds_example.py:107,111; right-arm
  motor indices 22-28 and weight slot 29 — G1JointIndex :55-64; gradual
  weight takeover/release — :135,:168; kp 60 / kd 1.5, control_dt 0.02 —
  :74-75,:69; CRC on every LowCmd — :173.
- dex1_1_service @ ced3216290bb22dae356213993786beeaa7143ab: the Dex1-1
  gripper is an RS485 device behind a C++ serial<->DDS bridge (main.cpp
  MotorUnit) speaking `rt/dex1/<side>/{cmd,state}` with
  `unitree_go::msg::dds_::MotorCmds_/MotorStates_` (test/test_gripper.cpp:37,
  main.cpp:125); finger travel is prismatic -0.02..0.0245 m
  (urdf/dex1_1.urdf:54,:139). No official Python service interface is
  confirmed, so the real GripperBus raises HardwarePending instead of
  guessing a protocol.
"""

from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

from aisle.embodiment import G1_RIGHT_ARM_JOINTS, G1_RIGHT_JOINTS
from aisle.nodes.so101_driver import clamp_step

DT_NS = 10_000_000  # node tick cadence (100 Hz, manifest parity with so101)
MAX_STEP_RAD = 0.05  # per-tick delta clamp, the driver-level belt under the guard
N_ARM = len(G1_RIGHT_ARM_JOINTS)
JOINT_NAMES = G1_RIGHT_JOINTS  # re-export, never redefined (outer-loop addendum)
MODE_MACHINE_REQUIRED = 15  # g1_29dof_mode_15_with_dex1_1
#: Dex1-1 prismatic finger travel, urdf/dex1_1.urdf:54@ced3216 (-0.02..0.0245 m)
FINGER_LOWER_M = -0.02
FINGER_UPPER_M = 0.0245
#: normalized 0..1 (closed -> open) mapped linearly onto the finger travel
FINGER_TRAVEL_M = FINGER_UPPER_M - FINGER_LOWER_M
SDK2_PIN = "814556d15970dd2ecf1c9984e845ca02ab07e206"  # unitree_sdk2_python


class HardwarePending(RuntimeError):
    """The real device path exists upstream but is not implementable yet."""


def fingers_from_grip(grip: float) -> np.ndarray:
    """Both Dex1-1 finger positions (meters) for a normalized 0..1 command
    (closed -> open); symmetric by construction — one DOF, two fingers."""
    finger = FINGER_LOWER_M + float(np.clip(grip, 0.0, 1.0)) * FINGER_TRAVEL_M
    return np.array([finger, finger], dtype=np.float32)


def grip_from_fingers(fingers: np.ndarray) -> float:
    """The normalized 0..1 opening implied by finger positions (meters);
    the inverse of fingers_from_grip, clamped to the physical travel."""
    fingers = np.asarray(fingers, dtype=np.float32).reshape(-1)
    mean = float(np.clip(fingers, FINGER_LOWER_M, FINGER_UPPER_M).mean())
    return (mean - FINGER_LOWER_M) / FINGER_TRAVEL_M


class LoopbackArmBus:
    """Pure-python G1 arm double: first-order lag over the 7 right-arm
    joints, seeded telemetry, injectable mode_machine (CON-5). A write
    with weight 0 never moves the target — weight 0 means hands off."""

    def __init__(
        self,
        mode_machine: int = MODE_MACHINE_REQUIRED,
        seed: int = 0,
        now_ns=time.monotonic_ns,
    ) -> None:
        self._rng = random.Random(seed)
        self.q = np.zeros(N_ARM, dtype=np.float32)
        self.target = self.q.copy()
        self.mode_machine = mode_machine
        self.last_weight = 0.0
        self.current_a = 0.3
        self.temperature_c = 32.0
        self.connected = True
        self._now_ns = now_ns

    def write_arm(self, q_cmd: np.ndarray, weight: float) -> None:
        self.last_weight = float(weight)
        if weight > 0.0:
            self.target = np.asarray(q_cmd, dtype=np.float32)[:N_ARM].copy()

    def read_state(self) -> dict:
        self.q = clamp_step(self.q, self.target, MAX_STEP_RAD)
        return {
            "q": self.q.copy(),
            "mode_machine": self.mode_machine,
            "current_a": self.current_a + self._rng.uniform(-0.02, 0.02),
            "temperature_c": self.temperature_c + self._rng.uniform(-0.5, 0.5),
        }

    def disconnect(self) -> None:
        self.connected = False


class LoopbackGripperBus:
    """Pure-python Dex1-1 double: normalized 0..1 in and out with a
    first-order lag; seeded telemetry rows are reproducible (CON-5)."""

    STEP = 0.05

    def __init__(self, seed: int = 0, now_ns=time.monotonic_ns) -> None:
        self._rng = random.Random(seed)
        self.q = 0.0
        self.target = 0.0
        self.connected = True
        self._now_ns = now_ns

    def write(self, grip: float) -> None:
        self.target = float(np.clip(grip, 0.0, 1.0))

    def read(self) -> float:
        self.q += float(np.clip(self.target - self.q, -self.STEP, self.STEP))
        return self.q

    def read_state(self) -> dict:
        q = self.read()
        return {
            "q": q,
            "current_a": 0.1 + self._rng.uniform(-0.01, 0.01),
            "temperature_c": 30.0 + self._rng.uniform(-0.3, 0.3),
        }

    def disconnect(self) -> None:
        self.connected = False


class Sdk2ArmBus:  # pragma: no cover — hardware-only by construction
    """The real arm bus: unitree_sdk2py behind write_arm/read_state.

    Message construction follows example/g1/high_level/
    g1_arm7_sdk_dds_example.py@814556d1 line for line: publisher
    `rt/arm_sdk` with LowCmd_ (:107), right-arm motor_cmd slots 22-28 with
    tau=0, kp=60, kd=1.5 (:74-75, indices :55-64), the 0..1 takeover
    weight in motor_cmd[29].q (:135,:168), and CRC over the whole LowCmd
    before every write (:173). State comes from the `rt/lowstate`
    LowState_ subscription (:111-112), including mode_machine."""

    def __init__(self, network_interface: str, now_ns=time.monotonic_ns) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, network_interface)
        self._LowCmd = LowCmd_
        self._crc = CRC()
        self._kp, self._kd = 60.0, 1.5  # g1_arm7_sdk_dds_example.py:74-75
        self.low_state = None
        self.last_weight = 0.0
        self.connected = False
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)  # :107
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)  # :111
        self._sub.Init(self._on_low_state, 10)
        self.connected = True

    def _on_low_state(self, msg) -> None:
        self.low_state = msg

    def write_arm(self, q_cmd: np.ndarray, weight: float) -> None:
        cmd = self._LowCmd()
        for i in range(N_ARM):  # right-arm slots 22-28 (:55-64)
            slot = cmd.motor_cmd[G1_MOTOR_OFFSET + i]
            slot.tau = 0.0
            slot.q = float(q_cmd[i])
            slot.dq = 0.0
            slot.kp = self._kp
            slot.kd = self._kd
        cmd.motor_cmd[G1_WEIGHT_SLOT].q = float(np.clip(weight, 0.0, 1.0))  # :135
        cmd.crc = self._crc.Crc(cmd)  # :173
        self._pub.Write(cmd)
        self.last_weight = float(weight)

    def read_state(self) -> dict:
        if self.low_state is None:
            return {"q": np.zeros(N_ARM, dtype=np.float32), "mode_machine": None}
        q = np.array(
            [float(self.low_state.motor_state[G1_MOTOR_OFFSET + i].q) for i in range(N_ARM)],
            dtype=np.float32,
        )
        state = {"q": q, "mode_machine": int(getattr(self.low_state, "mode_machine", 0) or 0)}
        motor = self.low_state.motor_state[G1_MOTOR_OFFSET]
        for key, attr in (("current_a", "tau_est"), ("temperature_c", "temperature")):
            value = getattr(motor, attr, None)
            if value is not None:
                state[key] = float(value)
        return state

    def disconnect(self) -> None:
        self.connected = False


#: right-arm motor indices inside LowCmd_.motor_cmd (22-28) and the
#: 0..1 arm_sdk weight slot (29) — G1JointIndex@814556d1 :55-64
G1_MOTOR_OFFSET = 22
G1_WEIGHT_SLOT = 29


class Dex1GripperBus:  # pragma: no cover — hardware pending, never guessed
    """The real Dex1-1 path is NOT implementable yet (HardwarePending).

    unitreerobotics/dex1_1_service@ced3216 is a C++ serial<->DDS bridge:
    RS485 motor over USB (main.cpp MotorUnit), topics
    `rt/dex1/right/cmd` / `rt/dex1/right/state` carrying
    unitree_go MotorCmds_/MotorStates_ (test_gripper.cpp:37, main.cpp:125),
    finger travel -0.02..0.0245 m (urdf/dex1_1.urdf:54). No official
    Python service/topic interface is confirmed, so constructing this bus
    raises instead of inventing a protocol. Missing for implementation:
    either (a) unitree_sdk2py MotorCmds_ bindings + the dex1_1_service
    daemon running, pinned to a commit, or (b) an official Python gripper
    service — whichever lands first."""

    def __init__(self, port: str) -> None:
        raise HardwarePending(
            "Dex1GripperBus: no confirmed Python interface for the Dex1-1 gripper "
            "(unitreerobotics/dex1_1_service@ced3216 is a C++ serial<->DDS bridge; "
            "topics rt/dex1/right/{cmd,state}; finger travel -0.02..0.0245 m). "
            f"Serial port {port!r} recorded; refusing to guess the protocol."
        )


def make_buses(
    network_interface: str, gripper_port: str, *, now_ns=time.monotonic_ns
) -> tuple[LoopbackArmBus | Sdk2ArmBus, LoopbackGripperBus | Dex1GripperBus]:
    """The injection point (HWP-8): `loopback` selects the pure doubles —
    importing unitree_sdk2py ONLY on the real path, so CI never needs it."""
    if not network_interface:
        raise ValueError("AISLE_G1_NETWORK_INTERFACE must be set (interface name or 'loopback')")
    if network_interface == "loopback":
        arm: LoopbackArmBus | Sdk2ArmBus = LoopbackArmBus(now_ns=now_ns)
    else:
        arm = Sdk2ArmBus(network_interface, now_ns=now_ns)
    if gripper_port == "loopback":
        gripper: LoopbackGripperBus | Dex1GripperBus = LoopbackGripperBus(now_ns=now_ns)
    else:
        gripper = Dex1GripperBus(gripper_port)  # raises HardwarePending (documented)
    return arm, gripper


class G1Driver:
    """The contract brain, bus-agnostic and clock-injected (HWP-8/CON-5):
    lease watchdog, gradual arm_sdk weight takeover/release, mode_machine
    gate, normalized gripper round-trip, and the reset refusal that is
    also the explicit re-arm path."""

    def __init__(
        self,
        arm: LoopbackArmBus | Sdk2ArmBus,
        gripper: LoopbackGripperBus | Dex1GripperBus,
        *,
        lease_s: float = 0.2,
        takeover_s: float = 0.5,
        now_ns=time.monotonic_ns,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.lease_ns = int(float(lease_s) * 1e9)
        self.takeover_ns = int(float(takeover_s) * 1e9)
        self._now_ns = now_ns
        self.armed = True
        self.weight = 0.0
        self.target: np.ndarray | None = None
        self.grip_target = 0.0
        self.last_cmd_ns: int | None = None
        self.last_tick_ns: int | None = None
        self.reason: str | None = None
        self.telemetry: list[dict] = []
        self._tick_count = 0
        self._sent_info = False

    # -- inputs -----------------------------------------------------------
    def on_joint_cmd(self, wire: np.ndarray) -> None:
        """joint_cmd takes the FIRST 7 wire slots (arm only; fingers ride
        gripper_cmd)."""
        self.target = np.asarray(wire, dtype=np.float32).reshape(-1)[:N_ARM].copy()
        self.last_cmd_ns = self._now_ns()

    def on_gripper_cmd(self, grip: float) -> None:
        self.grip_target = float(np.clip(grip, 0.0, 1.0))  # normalized, never radians
        self.last_cmd_ns = self._now_ns()

    def on_reset(self, request_id: str = "") -> tuple[dict, dict]:
        """TC-6: hardware has no teleport — always refuse; the refusal is
        the explicit re-arm after a lease expiry (ADR-g1-driver)."""
        self.armed = True
        self.weight = 0.0
        self.reason = None
        payload = {
            "refused": True,
            "reason": "hardware has no teleport (G1): behavioral reset only; lease re-armed",
        }
        return payload, {"request_id": request_id, **payload}

    # -- one control tick --------------------------------------------------
    def tick(self) -> dict:
        now = self._now_ns()
        dt = DT_NS if self.last_tick_ns is None else max(now - self.last_tick_ns, 0)
        self.last_tick_ns = now
        self._tick_count += 1
        state = self.arm.read_state()
        mode_ok = state["mode_machine"] == MODE_MACHINE_REQUIRED
        lease_ok = self.last_cmd_ns is None or (now - self.last_cmd_ns) <= self.lease_ns
        if not mode_ok:
            self.reason = (
                f"mode_mismatch: mode_machine={state['mode_machine']} "
                f"but {MODE_MACHINE_REQUIRED} (g1_29dof_mode_15_with_dex1_1) is required; "
                "refusing torque"
            )
            self.armed = False
        elif self.armed and not lease_ok:
            self.reason = (
                f"lease_expired: no command within {self.lease_ns / 1e9:.3f}s "
                "(AISLE_G1_LEASE_S); weight ramped to 0, explicit reset required to re-arm"
            )
            self.armed = False
        following = self.armed and mode_ok and self.target is not None
        step = dt / self.takeover_ns if self.takeover_ns > 0 else 1.0
        self.weight = float(np.clip(self.weight + (step if following else -step), 0.0, 1.0))
        q7 = state["q"]
        if following:
            q_cmd = clamp_step(q7, self.target, MAX_STEP_RAD)
            self.arm.write_arm(q_cmd, self.weight)
            self.gripper.write(self.grip_target)
        else:
            self.arm.write_arm(q7, 0.0)  # hold where we are, hands off
        grip_q = self.gripper.read()
        joint_state = np.concatenate([q7, fingers_from_grip(grip_q)])
        self.telemetry.append(
            {"tick": self._tick_count, "weight": self.weight, "reason": self.reason}
        )
        out = {
            "joint_state": joint_state,
            "gripper_state": grip_q,
            "weight": self.weight,
            "reason": self.reason,
            "sim_time_ns": int(now),  # hardware monotonic, HWP-7 wording in bridge_info
        }
        if not self._sent_info:
            out["bridge_info"] = {
                "backend": "g1-hardware",
                "clock": "hardware_monotonic",  # HWP-7: never described as sim time
                "perception": "hardware",
                "n_envs": 1,
                "mode_machine_required": MODE_MACHINE_REQUIRED,
                "sdk2_pin": SDK2_PIN,
            }
            self._sent_info = True
        return out


def main() -> None:  # pragma: no cover — dora runtime, exercised on hardware
    import json

    import pyarrow as pa
    from dora import Node

    from aisle.topics import stamp

    network = os.environ.get("AISLE_G1_NETWORK_INTERFACE", "")
    gripper_port = os.environ.get("AISLE_G1_GRIPPER_PORT", "")
    lease_s = float(os.environ.get("AISLE_G1_LEASE_S", "0.2"))
    takeover_s = float(os.environ.get("AISLE_G1_TAKEOVER_S", "0.5"))
    arm, gripper = make_buses(network, gripper_port)
    print(
        f"g1-driver: arm={'loopback' if network == 'loopback' else network} "
        f"gripper={'loopback' if gripper_port == 'loopback' else gripper_port}",
        file=sys.stderr,
    )
    driver = G1Driver(arm, gripper, lease_s=lease_s, takeover_s=takeover_s)

    node = Node()
    seq: dict[str, int] = {}

    def send(topic: str, value, metadata: dict | None = None) -> None:
        seq[topic] = seq.get(topic, 0) + 1
        node.send_output(topic, value, stamp({**(metadata or {})}, seq[topic]))

    try:
        for event in node:
            if event["type"] != "INPUT":
                continue
            eid = event["id"]
            metadata = event.get("metadata") or {}
            if eid == "joint_cmd":
                cmd = np.asarray(event["value"].to_numpy(zero_copy_only=False), dtype=np.float32)
                driver.on_joint_cmd(cmd)
            elif eid == "gripper_cmd":
                raw = np.asarray(event["value"].to_numpy(zero_copy_only=False)).reshape(-1)[0]
                driver.on_gripper_cmd(float(raw))  # normalized 0..1
            elif eid == "reset":
                payload, meta = driver.on_reset(metadata.get("request_id", ""))
                send(
                    "reset_refused",
                    pa.array(np.array([0], dtype=np.uint32)),
                    {"request_id": meta["request_id"], "reason": payload["reason"]},
                )
            elif eid == "tick":
                out = driver.tick()
                meta = {"sim_time_ns": out["sim_time_ns"], "env_id": 0}
                send("joint_state", pa.array(out["joint_state"]), meta)
                send(
                    "gripper_state",
                    pa.array(np.array([out["gripper_state"]], dtype=np.float32)),
                    meta,
                )
                if "bridge_info" in out:
                    send("bridge_info", pa.array([json.dumps(out["bridge_info"])]), meta)
    finally:
        arm.disconnect()
        gripper.disconnect()


if __name__ == "__main__":
    main()
