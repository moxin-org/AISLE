# ADR-67 — A second physics engine (Nexus) behind the scene contract

Status: PROPOSED — owner review required under CON-14; the spec-change PR it
owes (see below) must land before it can be accepted.
Amended by: ADR-70 (the wheel comes from PyPI inside the lock; the build
receipt, `engine-runtime.json` and `tools/nexus_runtime.py` are retired).

## Scope: development-only engines

Nexus, and rapier after it (ADR-68), are development engines. Their wheels
are built from dimforge branches pinned by commit in `engine-runtime.json`,
outside `pyproject.toml` and `uv.lock`, so no run on them can pass
`--env-baseline origin/main`: every such run is `--env-baseline local`,
recorded as unattested, with the engine build receipt as its only
provenance. No result from either engine enters the measured record until
all three of these hold: the engines are installed from released versions
inside the lock, a `spec-change` PR generalizes the Genesis wording of
SPEC 020 and SPEC 030 (CON-14), and the frozen baseline is re-established
under human review (CON-7). Whether that path is worth taking is a
maintainer decision this ADR does not make.

## Context

SPEC 020 names Genesis World as the simulator and SPEC 030 calls the bridge
`dora-genesis`, but the contracts themselves are engine neutral at the object
level: the bridge talks to `robot`, entity, link and camera objects through a
small duck-typed surface (`get_qpos`, `control_dofs_position`, `set_pos`,
`render(rgb, depth, segmentation)`, ...), and everything that decides a
scene (layout, seeded placements, DR draws, label textures, camera mounts)
is a pure function in the frozen `aisle.scenes` modules.

We want to run the same experiments on Nexus (`dimforge/nexus`, a GPU
rigid-body engine with Python bindings) without disturbing the measured
Genesis record.

## Decision

1. **The engine is an explicit, attested choice.** `AISLE_SIM_ENGINE`
   (`genesis` | `nexus`, default `genesis`) is declared on the bridge node
   like the perception rung, injected by `harness rollout --sim-engine`,
   and attested in `bridge_info` (`sim_engine`, with `genesis_version`
   keeping its BRG-6 name and carrying whichever engine's version ran) and
   in the run manifest (`sim_engine`). Unknown names are refused, never
   defaulted (the TC-9 rule).
2. **The frozen set is untouched (CON-7).** `src/aisle/scenes/*` keeps the
   Genesis builders byte-for-byte. The Nexus realization lives in
   `src/aisle/sim/nexus_backend.py`, outside the fence, and REUSES the
   frozen pure functions (`resolve_layout`, `sample_placements`,
   `apply_occlusion`, `label_texture_image`, `wrist_mount_transform`,
   `_assert_reachable`, the `SceneHandle`/`StoreHandle` dataclasses). Only
   the object construction is re-implemented. A sim test pins the Nexus
   placements to the frozen sampler's output.
3. **Same object surface, same conventions.** Nexus wrappers return numpy
   arrays (Genesis returned tensors; `to_numpy` passes both), quaternions in
   Genesis's (w, x, y, z) order, the same single-env/batched shape rules,
   `segmentation_idx_dict` in the `link`-level shape TC-9 documents, and a
   camera `transform` in the OpenGL look-at convention VER-8 pins, so the
   verifier's calibration module needs no change.
4. **Rendering lives in the Nexus viewer, not the physics engine.** Sensor
   cameras (offscreen surfaces with RGB, metric depth and per-body
   segmentation passes, optionally attached to a link) were added to
   `nexus_viewer3d`; the physics core only gained body/joint state read and
   write access.
5. **Nexus-only constants** (Genesis-equivalent default PD gains for URDF
   joints, the ground slab replacing the infinite plane, camera clip planes)
   live in `src/aisle/sim/nexus_physics.toml`, outside the frozen set.
6. **The engine realization is attested by its own digest, not by widening
   the fence.** `env_hash.sim_engine_hash(root, engine, build=None)` is one
   sha256 over the engine name, `src/aisle/sim/**` (so nexus_physics.toml's
   `substeps`, `internal_pgs_iterations`, `contact_natural_frequency` and
   `friction_combine_rule` are inside it) and the engine build provenance.
   It returns `{"engine", "sim_engine_hash", "n_files", "build"}` for the
   run manifest, and `tools/env_hash.py --sim-engine <engine>` emits the
   same block as `report["sim"]`. Recorded, never a gate: the CON-7 verdict
   stays the frozen set's.
   Adding `src/aisle/sim` to `FROZEN_DIRS` was rejected. It moves the
   Genesis `env_hash` from `1d83efde` (87 files) to `4cf1d9f9` (90),
   i.e. it spends the attestation discontinuity that issue #283 reserves
   for the composite `env_hash` with its bridging measurement (re-run M0
   and one tier curve at the new hash, show identity). Until then every
   recorded Genesis manifest, `tools/env_hash.json` and every
   `--env-baseline origin/main` run would have to be reissued for a change
   that buys nothing on the Genesis side.
7. **The engine build is provenance, not a version string.**
   `nexus_runtime.read_receipt(root=None)` is the public reader for
   `.nexus-runtime-receipt.json`, returning
   `{"installed", "path", "receipt", "problem"}` (`receipt` = wheel,
   version, cargo feature, platform, and the nexus/rapier/kiss3d commits
   with their dirty flags), never raising; `tools/nexus_runtime.py receipt`
   is its CLI (exit 0 iff installed). The receipt is gitignored and local
   to the machine that built the wheel, so the manifest's copy is the only
   durable trace, and it is what `sim_engine_hash`'s `build` argument
   takes. ADR-24's attested set is engine-aware to match: `ENGINE_DISTS`
   adds the running engine's own distribution (`dimforge-nexus3d` for
   Nexus) to the engine-neutral core, which previously named `genesis-world`
   whatever engine ran.
8. **The realization cannot be hot-swapped.** `src/aisle/sim` joins
   `harness/swap.py`'s `FROZEN_ROOTS` beside the bridge: unfenced code that
   decides the physics would otherwise be the one way to change what a
   running attested dataflow measures (HAR-10).

## Consequences

- Results across engines are NOT comparable: contact models, solver and
  renderer differ. A Nexus run is a different environment; the frozen-set
  hash does not encode the engine, so the manifest's `sim_engine` field and
  decision 6's `sim_engine_hash` are the discriminators. The frozen set does
  not encode the solver settings either, which is why the engine digest
  exists. Before any Nexus result enters the measured record,
  the frozen baseline must be re-established under human review (CON-7) and
  SPEC 020 / SPEC 030 wording generalized by a `spec-change` PR (CON-14).
- The Genesis-fit constants in `physics.toml` (gripper gains, the SO-101
  kinematic carry latch, convex decomposition threshold) were not re-tuned;
  the Nexus path inherits them.
- CON-5 on Nexus: the GPU broad phase and constraint coloring use atomics,
  so bitwise run-to-run reproducibility is not established. `build_scene`
  determinism (same seed, same initial state) holds; stepping determinism
  must be measured before Nexus evidence is trusted.
- Fixed upstream while doing this, both consumed by Nexus through
  `[patch.crates-io]` path overrides until releases carry them:
  rapier3d-urdf composed URDF `rpy` as intrinsic XYZ Euler angles instead of
  fixed-axis roll-pitch-yaw (`Rz * Ry * Rx`), misplacing every SO-101 link
  below the shoulder (rapier branch `fix-urdf-rpy`, released in rapier 0.36.0,
  which Nexus now takes from crates.io); and kiss3d replaced its
  global mesh/texture/material managers whenever a second window or
  offscreen surface was created, orphaning the material of objects built
  before the sensor cameras existed, whose per-object uniform buffer then
  grew until wgpu rejected the offsets (kiss3d branch
  `fix-shared-window-managers`).
- Solver settings are the engine's, not the scene's
  (`src/aisle/sim/nexus_physics.toml [sim]`). The first Nexus runs lost the
  Franka pinch on lift, showed boxes creeping on their boards, and needed 16
  substeps with eight PGS iterations to come close. All of it traced to the
  Nexus solver, fixed on the `aisle-backend` branch: contacts between a
  multibody link and a rigid body were solved twice, once by the multibody
  solver and once by the rigid-body solver against a zero-inverse-mass copy of
  the link that never moved, so the fingers' friction was cancelled by a ghost
  of themselves; the rigid-body sweeps ran one PGS iteration per substep
  against eight for the multibody (now interleaved); contact warmstarting
  matched every point of a small manifold to the first old point within 10 cm,
  so a fingertip pad's four points all inherited one impulse and the solver
  had to redistribute them every step (now nearest-point matching); friction
  rows were only solved in the once-per-substep stabilization sweep (now
  optionally in every biased iteration, `friction_in_bias_pass`); and Nexus
  averaged the friction of a touching pair where Genesis takes the maximum
  (`friction_combine_rule`). Replaying the recorded T0 commands
  (`tools/nexus_grasp_replay.py`) the box now follows the hand with a tilt
  under 3.2 degrees at 8, 4 or 2 substeps, and at Genesis's single substep
  with 30 Hz contacts; resting boxes drift 0.001 mm in 3 s. AISLE runs 2
  substeps with 240 Hz contacts, 0.5 mm allowance and eight iterations, about
  3 ms of GPU per 10 ms tick on an M-series laptop.
- A Nexus run cannot pass `--env-baseline origin/main`. The wheel is
  installed out of lock (`uv pip install` of a locally built archive), so
  ADR-24's `uv sync --locked --check` fails and the trusted gate refuses
  with DIST_DRIFT; with the attested set now engine-aware, the wheel's own
  PEP 610 record (`archive_info` with no hash, a `file://` path) adds
  `dimforge-nexus3d: archive install without a hash`. Nexus runs are
  therefore `--env-baseline local`, recorded honestly as unattested, and
  the receipt digest is the substitute provenance. Putting Nexus inside the
  lock is the precondition for any attested Nexus run.
- Editing `tools/env_hash.py` at all makes the trusted mode refuse on a
  branch until the change merges ("the gate tooling itself is not
  trusted"), Genesis runs included. The engine attestation above therefore
  lands as a normal env-change PR, not mid-campaign.
- `sim_engine_hash` covers the whole realization package, so a Nexus-only
  edit also moves the Genesis digest. Over-inclusive by design: it may
  report a change that did not affect the run, and can never report two
  different physics as the same environment.
- The rollout manifest records `sim_engine_hash` together with the engine
  build receipt. Because that receipt is the only source provenance for an
  out-of-lock wheel, the gate refuses Nexus or rapier when a required solver
  or renderer receipt is absent, malformed, or identifies dirty sources.
- Nexus reads generalized robot velocities back from its DoF state, and both
  alternative backends re-base a fixed-root robot through the body buffer for
  the SPEC 210 mobile embodiment. The published rapier 0.36.0 loader carries
  the URDF fixed-axis `rpy` correction described above.

## Alternatives rejected

- Refactoring `pharmacy.py`/`store.py` into an engine-neutral builder: the
  clean design, but it edits the frozen set and re-freezes the baseline for
  a change with no measured benefit yet.
- Selecting the engine from ambient environment only: violates the
  graph-attests-everything rule that the rung and the backend already
  follow.
