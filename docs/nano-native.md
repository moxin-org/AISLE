# Adora Nano: native Rust indoor experiments

This experimental profile connects AISLE's Rust policy, RGB/depth perception,
URDF FK/IK, motion guard and independent judge to DoRobot's Rust Nexus simulator
through Dora 1.0.1. No Python process participates in a rollout. The existing
pharmacy benchmark, registered skills and hardware drivers are unchanged.

Use a macOS Apple Silicon machine, the DoRobot Studio `feat/nano-autonomy`
checkout, and AISLE `feat/nano-rust-indoor`. Build the native Studio runtime
using its release instructions (FFmpeg 8 and cargo-gpu **0.10.0**, with its pinned
shader toolchain). Supply the full `data/adora_nano/adora_nano.urdf` path, including
meshes; the AISLE copy is sufficient for kinematic tests only.

```sh
cargo test --locked --manifest-path native/nano/Cargo.toml
cargo build --release --locked --manifest-path native/nano/Cargo.toml
native/nano/target/release/aisle-nano rollout \
  --model "$DOROBOT_STUDIO/data/adora_nano/adora_nano.urdf" \
  --simulator "$NANO_RUNTIME" --dora "$DORA" \
  --output runs/nano-right-0 --task pickup --arm right --seed 0
```

To reproduce the validated navigation-and-pickup task across the room, use a
fresh output directory and the farther work-surface goal:

```sh
native/nano/target/release/aisle-nano rollout \
  --model "$DOROBOT_STUDIO/data/adora_nano/adora_nano.urdf" \
  --simulator "$NANO_RUNTIME" --dora "$DORA" \
  --output runs/nano-complex-right-0 --task pickup --arm right --seed 0 \
  --goal-x 2.3 --goal-y 3.9 --frames 1000 --wall-seconds 900
```

The retained test travelled 5.93 m around the furniture, lifted the cube 9.37 cm,
and released it onto the table without furniture contact. A separate local-origin
pickup failed, so this result does not establish reliable grasping at arbitrary
base poses. See the validation report for all retained attempts.

`--task inspect` captures a short three-camera scene inspection. `--task navigate`
tests navigation only. `--task pickup` navigates to the work surface, finds a red
cube from RGB, estimates its position with matching depth, solves the selected
arm's IK, attempts grasp/lift/place/release, then settles. `--arm left` mirrors the
work surface. `--goal-x` / `--goal-y`, `--frames` and `--wall-seconds` bound runs.
Always use a new output directory; unsuccessful runs are retained.

The furnished 12 × 10 m room contains walls, sofa, coffee table and legs, chairs,
kitchen island, bookshelf, divider, a low pickup table and colored distractors.
Navigation uses a known static map and simulator localization, not SLAM. The
solid-wheel simulation drives forward and turns; it does not model omni-wheel
rollers. Navigation-only runs keep the pickup station at its fixed room location;
pickup runs approach the station along its docking axis. The
red-cube detector is color-specific, not an open-vocabulary model. Its depth
input supplies geometry only. The policy never subscribes to segmentation or
object ground-truth poses.

The generated `dataflow.yaml` contains four native processes:

```text
Nexus simulator ── RGB + depth + camera pose + robot state ──▶ policy
       ▲                                                       │
       └──────────────── guard ◀──── joint velocities ─────────┘
       └── object poses + physical contacts ──▶ independent judge
```

Hot topics use Arrow arrays: `state` Float32[24] is world XYZ + quaternion XYZW +
17 URDF-ordered joints; RGB is UInt8[640×480×3], depth Float32[640×480], camera
pose Float64[16] row-major OpenGL world-from-camera, commands Float32[17] joint
velocities. Every sample carries `seq`, `observation_id`, `sim_time_ns`, `env_id`. All three cameras
are published and recorded; the initial policy uses the head camera. A single
observation/command turn is in flight. The simulator waits for both a guarded
command and the judge's receipt before advancing. Missing messages trigger an
immutable observation retry after one wall second, with a new delivery `seq`
and the same `observation_id` and simulation timestamp. Cached commands and
oracle samples make retries idempotent; altered commands under the same ID are
rejected. A 30-second delivery timeout fails the episode. `--drop-depth-once 5`
exercises this path by omitting one head-depth delivery, without dropping the
recorded frame. Physics runs at 100 Hz; capture and control
are 5 Hz. This is not the standard TC-4 rate acceptance or ADR-30 turn protocol.

The independent judge requires stationary arrival, physical contact with **both fingers**,
at least 7 cm lift held for 0.4 s, release with no robot link touching the cube,
and 1 s of stable placement. Arrival and placement must still hold at termination.
The cube must enter the placement zone while supported by both fingers; a dropped
cube that happens to settle in the zone cannot pass.
Controller completion cannot set success. Furniture contact fails the
overall task. Contact auditing reads only the active GPU contact range, requires contact separation ≤1.5 mm, and checks both fingers in the same physics step. Missing judge observations fail evaluation. Robot self-collision remains disabled in this inherited simulation model. There is no weld, object attachment or root teleport during rollout.
The guard enforces velocity/position limits, coarse reach/floor sanity bounds and
base/arm mutual exclusion. These bounds do not prevent arm–furniture collisions.
IK plans reject branch jumps over 1.5 rad between waypoints. Loaded arm motion is
limited to 0.20 rad/s; this remains a waypoint controller, not a collision planner.

The experimental bridge applies URDF effort limits (the underlying URDF importer
does not apply them), including the declared 1 Nm gripper limit, and bounds position-target
windup at 0.10 rad. These are explicit simulation settings, not measured hardware
control parameters. Wheel targets restart at measured angles each observation,
so zero commands discard stalled wheel-target backlog. Cube pickup uses the
declared 4 cm target size.

Each run keeps `provenance.json`, scene, graph, policy/IK plan, guard and judge
traces, three latest camera images, `result.json` and `rollout.json`. `ok` reports
whether the evaluation completed; `success` reports the physical task result.
`rollout.json` is authoritative: it combines physical judgment with capture and
normal Dora exit; run-root `result.json` contains the physics judge alone.
Raw captures and all three videos are saved in `dataset/`, using
Studio's replay format. The preview socket is `<run>/preview.sock`.

After a rollout, rerun its evidence acceptance entirely in Rust:

```sh
AISLE_NANO_EVIDENCE=runs/nano-right-0 cargo test --release --locked \
  --manifest-path native/nano/Cargo.toml \
  --test evidence -- --ignored
```

This recomputes the independent verdict and checks observation continuity,
guard coverage, motor envelopes, graph isolation and retained executable hashes.
It deliberately fails unsuccessful tasks. Dataset metadata keeps capture
completion separate from `aisle-result.json`, the supervisor-checked task evaluation.

Codex can execute this same CLI, inspect camera images and structured results,
and repeat seeds or arms. Preserve its JSON transcript alongside the run evidence;
never replace a failed result with a controller's assertion of success.

The local symmetric URDF and nominal camera calibration remain simulation
assumptions. Passing these tests does not certify a real Nano. See
[the decision record](decisions/ADR-73.md) for the acceptance boundary.
Measured attempts and independent audits are retained in the
[validation report](validation/nano-native-2026-11-17/README.md).
