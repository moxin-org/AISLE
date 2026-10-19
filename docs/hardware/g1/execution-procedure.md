# G1 hardware phase: exact execution procedure (G1H-8, HWP-20 analog)

Blocker: no G1 station is available. Every step below is `pending` until
measured on the pinned station (mode_machine=15, g1_29dof_mode_15_with_dex1_1);
`harness hardware dry-run --embodiment g1_right` prints the checklist with
live statuses and refuses to mark a later step passed while an earlier one
is not.

| step | what | passes when | artifact |
|---|---|---|---|
| 1 | acquire and inventory the G1 station | `station-manifest.json` validates with no `hardware_pending` field (G1H-1) | `docs/hardware/g1/station-manifest.template.json` filled |
| 2 | rig the hoist/protective frame; anchor the workspace; verify the remote-controller stop and physical estop | anchoring and estop rows signed in the safety case (G1H-4) | `safety-case.template.json` filled and signed |
| 3 | pin firmware, unitree_sdk2py commit, cyclonedds, power, USB/network topology | hashes present in the station manifest | station manifest |
| 4 | measure joint-zero (rt/lowstate radians), Dex1, workspace, camera, and time calibration; verify independently | calibration artifacts validate; raw records regenerate them (G1H-5) | `calibration-procedure.md` outputs |
| 5 | sign the safety case and preflight checklist | every preflight item `passed`; three roles named | safety case |
| 6 | torque-off telemetry drill, then hoist no-load, then representative load | drill records retained with traces and final state (G1H-8) | drill records |
| 7 | weight-ramp takeover/release drill, lease-expiry drill, held-command, estop, power-loss, evidence-sink drills | each drill retained with operator action and stop latency (G1H-3) | drill records |
| 8 | freeze the physical scorer and measure its fidelity against an independent audit | fidelity estimate retained (HWP-13 analog) | `scorer-contract.json` |
| 9 | freeze reset and human-intervention protocols | protocols hashed (G1H-7) | `reset-intervention-protocol.json` |
| 10 | freeze the physical protocol | filled; registered with `harness freeze` (HWP-14 analog) | registration |
| 11 | run the paired physical task instances in randomized blocks | analyzer report over retained trials | report |
| 12 | blinded live-fault cell if the claim is kept | sealing followed | sealed record |
| 13 | regenerate the analyzer report and sim-to-hardware deltas | `harness hardware report --embodiment g1_right` from raw records | report |
| 14 | update the claim matrix and public status | rows move off `hardware_pending` only with retained evidence | `docs/claim-evidence.yaml` |

Arm hand-over rules baked into steps 6-7: the driver takes over rt/arm_sdk
with a gradual weight ramp (never a step), releases the same way while
holding the current pose, and a lease expiry freezes the gripper and ramps
the weight to zero without silent re-take. Balance and fall protection stay
with the motion controller in damping mode; the AISLE controller never
commands legs or torso.

Roles on equipment day: physical safety operator (estop authority),
trial conductor (start/stop authority), software controller (torque
enable and evidence sink). One person may not hold all three.
