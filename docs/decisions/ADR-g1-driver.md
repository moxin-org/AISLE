# ADR-g1-driver — G1 hardware driver (`g1-driver`)

Status: PROPOSED (G1-D/G1-D-r1, outer-loop entries #2/#3; draft amendment lane, not for main)

- Date: 2026-10-19
- Deciders: inner loop (octos lane2); outer loop (claude) reviews
- Requirement IDs: TC-2, TC-5, TC-6, HWP-5, HWP-6, HWP-7, HWP-8 (SPEC 010, SPEC 520)

## Context

AISLE's sim->real node-swap claim (SPEC 010) has one hardware bridge so far:
`so101-driver` (ADR-phase6-prep). The Unitree G1 campaign needs the same topic
surface on the `g1_right` embodiment: 7 right-arm joints + 2 Dex1-1 fingers
(`g1_29dof_mode_15_with_dex1_1`). The SO-101 driver carries three known gaps
the outer loop explicitly refused to inherit: no lease watchdog (the last
target keeps acting forever), `gripper_cmd` treated as radians, and
`gripper_state` published un-normalized.

Upstream facts (cited as `file:line@commit` in the code):

- `unitree_sdk2py` @ `814556d1` (`unitreerobotics/unitree_sdk2_python`), which
  pins `cyclonedds==0.10.2` (setup.py). Arm SDK = publisher `rt/arm_sdk`
  (`LowCmd_`), state = subscriber `rt/lowstate` (`LowState_`); right-arm
  motor slots 22–28, 0..1 takeover weight in `motor_cmd[29].q`; kp 60/kd 1.5,
  control_dt 0.02; CRC over every LowCmd
  (example/g1/high_level/g1_arm7_sdk_dds_example.py:69,74-75,107,111,135,168,173).
- Dex1-1 gripper: `unitreerobotics/dex1_1_service` @ `ced3216` is a C++
  serial↔DDS bridge over RS485↔USB speaking `rt/dex1/<side>/{cmd,state}`
  (`unitree_go::MotorCmds_/MotorStates_`, test_gripper.cpp:37, main.cpp:125);
  finger travel prismatic −0.02..0.0245 m (urdf/dex1_1.urdf:54). **No official
  Python interface is confirmed.**

## Decision

1. **Interface selection — injected protocol buses, not robot objects.**
   `g1_driver.py` defines `ArmBus`-shaped `write_arm(q7, weight)` /
   `read_state() -> {q, mode_machine, current_a?, temperature_c?}` and
   `GripperBus`-shaped `write(0..1)` / `read() -> 0..1`. `make_buses()` reads
   `AISLE_G1_NETWORK_INTERFACE` (interface name or `loopback`) and
   `AISLE_G1_GRIPPER_PORT` (serial or `loopback`). `loopback` never imports
   `unitree_sdk2py`/`cyclonedds` (asserted via `sys.modules` diff, the
   test_so101_driver.py:39 pattern).
2. **Weight ramp + lease semantics.** Takeover weight ramps 0→1 over
   `AISLE_G1_TAKEOVER_S` (default 0.5 s), release ramps 1→0; no torque step in
   either direction (mirrors the example's stage-4 release). The lease
   watchdog: no new command within `AISLE_G1_LEASE_S` (default 0.2 s) → stop
   following, ramp weight to 0, record `lease_expired` in telemetry/
   `bridge_info`. A later command never silently re-takes the arm; the ONLY
   re-arm path is the reset service reply (which always refuses the teleport,
   TC-6/ADR-34, and re-arms the lease).
3. **mode_machine gate.** `mode_machine != 15` → zero weight out, torque
   refused, `mode_mismatch` reason in `bridge_info` and telemetry.
4. **Finger mapping (CONTRACT semantics, G1-D-r1).**
   `gripper_cmd`/`gripper_state` are normalized with the AISLE CONTRACT
   semantics (env/CONTRACT.md:19,41; env/limits.toml:24-26):
   **0.0 = open … 1.0 = closed**. Which finger position is "open" is a
   geometry fact: unitree_ros@5994d4f
   `robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf:1113-1119` gives
   `right_dex1_finger_joint_1` axis `(0,-1,0)` and `:1142-1147` joint_2 axis
   `(0,1,0)` over prismatic travel −0.02..0.0245 m — +q moves finger1 −y and
   finger2 +y, so the fingers DIVERGE: the UPPER limit +0.0245 m is OPEN,
   the lower −0.02 m is CLOSED (same travel in dex1_1_service@ced3216
   urdf/dex1_1.urdf:54,139; the unitree_ros URDF is the geometry authority
   because it carries the axes). The two Dex1-1 finger positions in
   `joint_state[7:9]` are meters, symmetric:
   `finger = 0.0245 − grip × 0.0445`; `grip = (0.0245 − mean(fingers)) / 0.0445`.
   Exact inverse by construction; the same rule both directions.
5. **Dex1-1 real bus = HardwarePending.** `Dex1GripperBus.__init__` raises
   `HardwarePending` naming what is missing (either unitree_sdk2py
   MotorCmds_ bindings + the dex1_1_service daemon pinned to a commit, or an
   official Python service). We refuse to guess a protocol the vendor has not
   confirmed in Python.
6. **SDK2 install = out-of-lock (ADR-67/ADR-68 pattern).** `unitree_sdk2py` +
   `cyclonedds==0.10.2` are hardware-only; they never touch `pyproject.toml`/
   `uv.lock`. `tools/g1_runtime.py` (`install`/`verify`/`receipt`) installs
   via `uv pip` pinned to the SDK2 commit and writes a gitignored
   `.g1-runtime-receipt.json`; CON-8 (JSON stdout, logs stderr, exit 0 iff ok).
7. **Registry.** `g1-driver.yaml` manifest mirrors so101's topics
   (`arm: [g1_right]`, `safety_class: motion`); `g1_right` added to the
   capability schema arm enum (real + 7 byte-identical fixture copies),
   `curated_core.toml`, `EXPECTED_IDS`, the reserved_dists mirror, and the
   confinement `DENIED_IMPORTS` / fault-injector `FROZEN_ASSETS` tuples
   (symmetric with so101). `run_scenario(dof=...)` defaults to `SO101_DOF`
   (rows == 12 pin unchanged) and gains the G1-only `mode_mismatch` scenario
   outside the SO-101 `SCENARIOS` tuple.

## Consequences

- The G1 driver closes all three inherited SO-101 gaps; the lease/re-arm
  semantics are testable without hardware over loopback doubles (seeded,
  clock-injected, CON-5).
- `hardware_monotonic` (HWP-7 wording) labels every stamp; `joint_state` is
  wire 7+2 in the frozen `G1_RIGHT_JOINTS` order (TC-5).
- The real gripper path stays explicitly blocked rather than speculative;
  unblocking needs one upstream fact, not rework here.
- Freeze registry tests are expected red until G-I rolls the successor
  registration (outer-loop addendum): do not self-register.
- Drivers remain free-run; the budget guard stays the frozen authority and
  `clamp_step` (0.05 rad/tick) the belt under it.
