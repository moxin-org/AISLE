# ADR-73: Experimental Adora Nano, native Rust execution

Status: proposed; separate from the frozen pharmacy benchmarks.

The requested Nano integration uses an AISLE-owned Rust policy, kinematics,
RGB/depth perception, guard and independent task judge. The existing DoRobot
Rust Nexus simulator supplies physics, three cameras and recording through Dora.
The two projects exchange typed Arrow topics, not a Python API dependency.

Requirements: CON-1/3/4/5/8/9/15, TC-1/2/3/8/9, MOB-2/3/4/5, BG-1/2.
Acceptance starts in `native/nano/tests/contracts.rs`.
These are design targets for this experimental surface, not a declaration of
full normative conformance. Rust unit/CLI tests run in `ci/nano-rust`; the legacy
Python HAR-9 trace scanner does not enumerate Rust tests. Actual GPU acceptance
is opt-in through `tests/evidence.rs` after a retained native rollout.

The Nano profile is experimental, not a CAP-7 registered manipulation skill.
It does not alter the curated registry, frozen scenes, verifier, reset, or existing
Franka/SO-101 contracts. A future promotion requires measured acceptance and
human review. It does not claim the standard TC-4 sensor-rate acceptance: the
initial capture/control cadence is 5 Hz with a 100 Hz physics integration clock.

Navigation uses a supplied static furniture map and simulated base localization;
this is not SLAM. RGB color components identify colored test cubes; synchronized
depth and camera extrinsics supply metric geometry. Policy receives no object
poses or segmentation. The separate judge receives physics object poses and
contacts, and requires contact-supported lift and released placement. No object
attachment, root teleport, or kinematic cube updates are allowed during rollout.
Topic-level oracle separation is audited, but processes share a local filesystem;
this profile does not claim adversarial policy sandboxing. Contact-supported lift
is historical evidence; arrival and robot-contact-free placement must remain
valid at termination. The supervisor also requires normal Dora process exit.

The local symmetric Nano URDF is a simulation model, not measured hardware
calibration. Rust FK/IK uses that URDF including both j5 joints. Asset hashes,
seed, executable hashes, scene and graph are retained for each run. A new process
is the experimental reset boundary; no in-episode TC-6 reset service is claimed.
