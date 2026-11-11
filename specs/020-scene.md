# SPEC 020 — Pharmacy scene

Status: DRAFT until M0. Frozen set after M0 (CON-7). Module: `src/aisle/scenes/pharmacy.py`.

Engines (ADR-67, ADR-68, ADR-70, ADR-71): the scene contract below is engine
neutral. `src/aisle/scenes/pharmacy.py` is the Genesis World realization and
stays in the frozen set. The `nexus` and `rapier` realizations live in
`src/aisle/sim/`, outside the frozen set, and reuse the frozen pure functions
that decide layout, placements, domain-randomization draws and camera mounts,
so a seed means the same initial scene on every engine. Every run names its
engine (`sim_engine`, BRG-6). Results from different engines are different
environments and are not pooled or compared as one measurement. Which engine
is the default is an ADR-level decision (ADR-71), not part of this contract.

- SCN-1: `build_scene(seed:int, embodiment:str="franka", n_envs:int=1, headless:bool=True, cfg:SceneCfg|None=None) -> SceneHandle` MUST be a pure function of its arguments (CON-5). `SceneHandle` exposes: `scene`, `robot`, `boxes: dict[str, Entity]` (insertion order = oracle_state order, TC table), `tray`, `cams: dict[str, Camera]`.
- SCN-2: Five medicine boxes (names fixed: amoxicillin, ibuprofen, cetirizine, omeprazole, metformin); sizes/colors from `scenes/meds.toml`, physics params from `scenes/physics.toml`. NO physics constants inline in code.
- SCN-3: Box placement randomized per seed on a staggered **two-level** shelf (M0 env-change, owner-approved; see ADR-12 — the original three-level stack physically blocked the top-down grasp on lower levels, making M0-1 unreachable); rejection-sample any initial interpenetration; placements MUST be reachable (IK solution exists) — assert at build time.
- SCN-4: Embodiments: `franka` (the Genesis-bundled MJCF, loaded by every engine) and `so101` (URDF under `assets/so101/`); both MUST place the tray and shelf inside the arm's workspace (asserted).
- SCN-5: Cameras: `overhead` 640x480 fov 55 fixed; `wrist` 320x240 fov 70 attached to EE link. Rendering MUST use the engine's rasterizer path by default (Metal-safe, CON-1).
- SCN-6: Domain-randomization toggles (lighting, textures, friction jitter, camera jitter) behind `SceneCfg`, all default OFF; each toggle independently seedable.
- SCN-7: Determinism: same (seed,cfg,platform,engine,backend) ⇒ bitwise-identical initial `oracle_state`; document any Metal-vs-CUDA or cross-engine divergence in `docs/determinism.md` rather than hiding it.

Acceptance: `tests/sim/test_scene.py::test_build_determinism` (SCN-1,7), `::test_reachability` (SCN-3,4), `::test_no_interpenetration` (SCN-3), `tests/unit/test_scene_cfg.py` (SCN-2,6 — config parsing only, no sim).
