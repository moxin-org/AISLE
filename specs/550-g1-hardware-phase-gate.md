# SPEC 550 — G1 hardware phase evidence gate (draft)

Status: PROPOSED, HUMAN-REVIEW GATED. This contract supplements
ADR-g1-driver and adapts SPEC 520 to the Unitree G1 right-arm embodiment
(`g1_right`, asset `g1_29dof_mode_15_with_dex1_1`). It inherits every SPEC
520 requirement not restated here (HWP-1..HWP-20 remain binding for the
so101 lane and, unless a G1H clause overrides, for the G1 lane). Drafted
under the ADR-66 freeze-period discipline: nothing here is accepted, and no
realized-hardware observation exists; every empirical field stays
`hardware_pending` until measured on a pinned G1 station.

Dependencies unchanged from SPEC 520: issues #344 shared control surface,
#345 statistics, #346 non-oracle tasks, #350 actuation boundary, #351
exposure ledger, #352 authorization disposition, #354 instrument audit.
Passing a unit or loopback gate (including the g1 dry-run) satisfies none
of them.

## Station identity

- G1H-1: The G1 station manifest MUST pin the robot serial, the
  `mode_machine=15` station profile (`g1_29dof_mode_15_with_dex1_1`), the
  Dex1-1 gripper firmware version and RS485-to-USB serial, the PC2/Jetson
  compute unit, the wired `192.168.123.x` control network interface, the
  remote-controller channel with its physical e-stop path, and the
  `unitree_sdk2py` install commit recorded in the runtime receipt, in
  addition to every HWP-1 field. A station whose `mode_machine` is not 15
  MUST NOT receive torque.
- G1H-2: The Dex1-1 gripper interface is `hardware_pending`: no official
  Python service interface is confirmed (unitreerobotics/dex1_1_service is
  a C++ serial-to-DDS bridge). Physical gripper trials MUST NOT begin
  before a pinned, reviewed interface lands; the driver's
  `HardwarePending` refusal MUST stay in place until then.

## Arm hand-over, balance, and safety authority

- G1H-3: The driver MUST take over `rt/arm_sdk` with a gradual weight ramp
  (0 to 1 over the configured window) and release it the same way while
  holding the current pose; a lease expiry MUST ramp the weight to zero,
  freeze the gripper at its current position, and require an explicit
  re-arm. A one-beat weight drop from 1 to 0 is forbidden. These drills
  MUST be exercised and retained before any load-bearing trial.
- G1H-4: Balance and fall safety authority stays with Unitree's motion
  controller; no AISLE node MUST command legs, torso, or balance, and the
  safety case MUST name the motion controller (not any AISLE guard) as the
  balance/fall authority, per the HWP-12 non-oracle boundary wording.
- G1H-5: Joint zero calibration MUST read `rt/lowstate` radians (never raw
  encoder counts) for the right-arm motor slots, and the Dex1-1 travel
  calibration MUST pin the CONTRACT gripper mapping endpoints (0.0 = open
  at the upper finger limit +0.0245 m, 1.0 = closed at the lower -0.02 m)
  against physical reference blocks.
- G1H-6: The PC2/Jetson host clock and the `rt/lowstate` timestamp source
  are two device clock domains; the workspace calibration MUST record a
  time mapping between them, and telemetry MUST keep both stamps with an
  explicit reconciliation uncertainty, never silently adopting one domain.

## Trials, reset, and execution ladder

- G1H-7: Reset on the G1 is behavioral only (TC-6/ADR-34): the driver
  refuses teleports, the refusal is the explicit lease re-arm, and the
  reset/intervention protocol MUST keep the SPEC 520 scoring consequences
  verbatim.
- G1H-8: The execution ladder MUST run torque-off telemetry drills first,
  then hoist no-load, then representative load; weight-ramp and
  lease-expiry drills MUST precede any load-bearing trial, and the
  fourteen-step SPEC 520 checklist order MUST hold unchanged for G1.
