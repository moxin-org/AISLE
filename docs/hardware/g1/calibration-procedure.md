# G1 calibration procedure (G1H-5, frozen before station results)

Fixtures: the pharmacy tray fixture id `tray-fixture-v1`, a printed
checkerboard (9x6, 20 mm squares) for intrinsics, a 4-marker AprilTag board
on the tray for extrinsics, a machinist's rule (0.5 mm) and a dial indicator
for repeatability, and the Unitree remote controller for damping-mode moves.

Environmental conditions recorded per session: ambient temperature,
lighting (lux at the tray), and time since power-on (at least 15 minutes).

Joint zero calibration (G1H-5, HWP-2 analog): joint zeros are read from
`rt/lowstate` radians (motor_state[i].q for the right-arm slots 22-28 of the
pinned unitree_sdk2py example), NEVER from raw encoder counts — the SDK
already reports SI units. Procedure: in damping mode, move each right-arm
joint to its marked mechanical zero, record 20 consecutive lowstate samples,
and store the median as zero with the observed range as repeatability. The
Dex1-1 travel calibration pins the two prismatic fingers against the URDF
travel (-0.02..0.0245 m) and records the CONTRACT mapping endpoints
(0.0 = open at the upper limit, 1.0 = closed at the lower limit) against a
40 mm reference block. Expiry: 30 days or any mechanical work. Retry rule:
a repeat outside 2 percent of range invalidates the joint and restarts its
sequence.

Workspace calibration (HWP-3 analog): 20 checkerboard poses per camera for
intrinsics (reprojection error under 0.5 px, else redo); 12 AprilTag board
poses for extrinsics with residuals retained — one extrinsic set for the
fixed overhead camera over the tray, one for the G1 head camera — and a
time mapping from 100 synchronized frame and joint-state pairs, recording
the PC2/Jetson clock and the lowstate timestamp source so the two device
clock domains stay reconcilable (G1H-6). Split: poses 1 to 12 calibrate,
13 to 20 evaluate; the evaluation residual is the reported uncertainty.

Raw record schema: `raw/<session>/{images/, joint_samples.jsonl,
measurements.jsonl, residuals.json}`; `tools/hw_calibration.py --build`
regenerates the signed artifacts from them and `--check` runs the VER-8
stage-0 refusal before any robot motion.
