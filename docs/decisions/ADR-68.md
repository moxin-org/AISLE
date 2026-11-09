# ADR-68 — A CPU engine: rapier physics behind the Nexus renderer

Status: PROPOSED — owner review required under CON-14, with ADR-67.
Amended by: ADR-70 (`rapier3d` comes from PyPI inside the lock;
`tools/rapier_runtime.py` and the source pins are retired).
Trigger: ADR-67 left stepping determinism unestablished on the only
alternative engine.
Scope: a development-only engine, under the conditions ADR-67's scope
section sets for Nexus (pinned out-of-lock wheel, `--env-baseline local`
runs only, no measured-record results).

## Context

ADR-67 put a second physics engine behind the scene contract: Nexus, a GPU
rigid-body solver selected with `AISLE_SIM_ENGINE=nexus`. Genesis remains the
default and the only engine behind the measured record.

Two engines gave AISLE a cross-check on physics but not on reproducibility.
Genesis reduces on Metal or CUDA, and Nexus colours constraints with GPU
atomics, so neither can promise a bitwise-identical step, which is what CON-5
asks for. Nexus is rapier on the GPU: scenes are authored as rapier worlds and
baked into GPU buffers. That makes rapier on the CPU the natural third engine,
with the same contact model, the same MJCF and URDF loaders and the same joint
semantics, executing single-threaded where bitwise reproducibility is
achievable and where the physics tick is far cheaper.

## Decision

1. **A third engine name, attested like the others.** `AISLE_SIM_ENGINE=rapier`
   joins `genesis` and `nexus` in `aisle.sim.ENGINES`; it is declared on the
   bridge node, asserted by `harness rollout --sim-engine`, and recorded in the
   run manifest with its own `sim_engine_hash` and build receipt.
2. **rapier owns the physics, Nexus owns the pixels.** The backend steps
   `rapier3d.PhysicsWorld` on the CPU and writes the resulting poses into a
   Nexus scene that is built but never stepped, then renders through the same
   sensor cameras. Measured before the backend was written: writing a body pose
   and a robot configuration with no solver step moves the RGB, the metric
   depth and the segmentation exactly as a stepped scene does.

   The alternatives were rejected. rapier's own Panda3D testbed is windowed
   only, with no offscreen buffer, no depth, no segmentation and no camera
   intrinsics. A new offscreen renderer would mean calibrating a third sensor
   model that VER-8 would then have to re-derive.

   The consequence is explicit: the rapier engine needs both wheels, and
   `engine_available("rapier")` checks `rapier3d` and `nexus3d`. In exchange, a
   rapier run and a Nexus run share the identical renderer, shadows,
   antialiasing, segmentation ids and camera model, so a comparison between
   them isolates the solver as the only variable.
3. **The realization lives outside the frozen set**, next to the Nexus one, and
   rebuilds the same scenes from the frozen pure functions (layout, placements,
   occlusion, label textures, wrist mount, reachability). Nothing under CON-7
   moves. `src/aisle/sim/**` is covered by the ADR-67 engine digest, so the
   solver settings in `rapier_physics.toml` are attested even though the frozen
   set does not contain them.
4. **Missing rapier Python bindings are added upstream**, on a branch in the
   rapier checkout, never worked around downstream. The engine needed multibody
   state writing (`apply_displacements`, `generalized_position`, the link motor
   setters, armature), the MJCF name tables, an IK degree-of-freedom filter and
   a counters toggle.

## Consequences

- Genesis stays the only engine behind the measured record. Three engines now
  produce evidence that is not comparable across engines.
- `rapier_physics.toml` holds the engine-realization constants. rapier has no
  counterpart to the Nexus scheduling knobs (`friction_in_bias_pass`,
  `implicit_coriolis`, `substep_refresh`), and its substep analogue is the
  solver iteration count. The camera and light blocks are copied from
  `nexus_physics.toml` verbatim, because the renderer is the same one.
- Two out-of-lock wheels now have to be installed by hand, and any `uv sync`
  removes both.
- The engine sources are pinned, not tracked. `engine-runtime.json` names the
  repository, branch and full commit of the two repositories AISLE builds
  wheels from, and `tools/engine_sources.py` fetches them by exact commit, the
  same discipline `dora-runtime.json` applies to the Dora CLI. The crates the
  engine merely links against are pinned once, in nexus's own Cargo manifest:
  rapier by its published release, kiss3d by git revision rather than branch
  until its fix is released: pinning them here as well would be two
  sources of truth for one dependency, and a branch would make the build
  irreproducible. The cost is that a new engine commit has to be pushed and
  the pin bumped before a pinned install can use it; an unpushed pin fails
  closed rather than building something else. `tools/rapier_runtime.py verify` reports the renderer's receipt
  alongside the solver's so a half-installed environment is visible.
- A multibody is built with a 6-DoF free root that rapier collapses only during
  the first `step()`, so the backend takes one warm-up step before reading the
  degree-of-freedom layout. `Multibody::update_root_type` is `pub(crate)`
  upstream; making it callable would remove the warm-up step.
