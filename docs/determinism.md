# Determinism notes (SCN-7, CON-5)

Contract: same (seed, cfg, platform) ⇒ bitwise-identical initial
`oracle_state`. Verified by `tests/sim/test_scene.py::test_build_determinism`
on macOS arm64 (Metal backend, float32), and on the optional Nexus engine
(ADR-67) by `tests/sim/test_nexus_scene.py::test_build_determinism`.

Backend selection is explicit and recorded. `uv sync --extra sim` plus
`harness rollout --sim-extra sim` uses Metal on Darwin and CPU elsewhere,
even when a CUDA device is visible. `uv sync --extra cuda` plus
`harness rollout --sim-extra cuda` uses CUDA on Linux and fails closed when
the device is unavailable; it never silently retries on CPU. The selected
extra, engine, resolved backend, and device are persisted in `manifest.json`
(`sim_extra`, `sim_engine`, `sim_backend`, `sim_device`) beside the
environment fingerprint. `sim_engine` is the discriminator between the two
engines' evidence: the frozen-set hash does not encode the engine.
On Nexus the same two extras resolve through that engine's own table
(`aisle.sim.select_nexus_backend`): `sim` is Metal on Darwin and WebGPU
elsewhere, `cuda` is Linux-only and fails closed the same way.
A pre-attestation development build executed on
an NVIDIA GeForce RTX 5090 with driver 580.126.09 and PyTorch 2.13.0+cu130;
that historical run is not evidence for the corrected attested path. A fresh
hardware run is required before making a performance or reproducibility claim.

Known platform caveats — recorded here rather than hidden (SCN-7):

- Initial oracle_state is placement-derived (pure Python RNG → float32), so
  it is expected to be bitwise-identical across backends. POST-STEP state is
  not yet covered by any contract: Metal vs CUDA vs CPU floating-point
  reduction order may diverge once physics steps run (relevant from T05
  onward; measure before promising cross-platform reproducibility).
- genesis is initialized once per process (backend fixed at first
  build_scene call); mixing backends in one process is unsupported. Nexus
  does the same in `_ensure_nexus` (`src/aisle/sim/nexus_backend.py`): one
  engine per process, and a later build asking for another backend is
  refused. That engine also holds one live renderable scene: a newer
  `build_scene` supersedes the previous scene's render nodes and frees its
  cameras, so the older handle keeps its physics readbacks and camera poses
  but its cameras raise on render.
- Nexus steps in its deterministic mode (`deterministic = true` in
  `src/aisle/sim/nexus_physics.toml`), which sorts contacts, constraint colors
  and solver order canonically so the GPU atomics no longer decide the result.
  `tests/sim/test_nexus_scene.py::test_stepping_is_reproducible` drops a box
  pile under a moving arm twice and requires bitwise identical joint
  coordinates and `oracle_state`. The guarantee is same machine, same wheel
  and same backend only; Genesis remains the only engine behind the measured
  record.
- The rapier engine (ADR-68) is the one with stepping determinism, and it is
  measured: `tests/sim/test_rapier_scene.py::test_bitwise_step_determinism`
  drops a box into contact in two identically seeded worlds, steps both 200
  times and requires bitwise identical `oracle_state` and joint coordinates.
  It steps single-threaded on the CPU by configuration
  (`num_threads = 1` in `src/aisle/sim/rapier_physics.toml`), which is what
  removes the only ordering variation it has. Cross-platform bit equality is
  a separate claim and is not measured; it would need a `rapier3d` wheel
  built with rapier's `determinism` cargo feature, which the published one is
  not.
- CUDA startup errors propagate; AISLE never silently retries initialization
  on CPU. Metal-vs-CUDA post-step divergence has not yet been quantified.
