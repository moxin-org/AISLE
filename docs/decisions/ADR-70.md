# ADR-70 — Engine wheels from PyPI, inside the lock

Status: PROPOSED — owner review required under CON-14, with ADR-67 and ADR-68.
Amends: ADR-67 (scope, attestation) and ADR-68 (installation, pinning).
Trigger: `dimforge-nexus3d` 0.2.1 and `rapier3d` 0.36.1 were published to PyPI
with every binding AISLE uses.

## Context

ADR-67 and ADR-68 built the Nexus and rapier wheels from source:
`engine-runtime.json` pinned a commit of each repository,
`tools/nexus_runtime.py` and `tools/rapier_runtime.py` built and installed the
wheels outside `uv.lock`, and a gitignored build receipt was the only provenance
a run could record. The trusted checker refused a Nexus or rapier run whose
receipt was missing or named a dirty checkout, and any `uv sync` removed the
wheels. ADR-67 named "the engines are installed from released versions inside
the lock" as the first precondition for engine results entering the measured
record.

Both engines now have releases on PyPI, with wheels for macOS arm64, Linux
x86_64 and aarch64, and Windows x64 (rapier3d also covers musl). Every Nexus
wheel has the WebGPU and CPU backends; the macOS wheel also has native Metal.
`NexusViewer.with_backend(name)` selects one by name and raises `ValueError`
for a backend the wheel lacks (`nexus3d.available_backends()` lists them). No
published wheel has CUDA.

## Decision

1. **The wheels are locked dependencies of the `sim` extra.**
   `dimforge-nexus3d==0.2.1` and `rapier3d==0.36.1` join the `sim` extra in
   `pyproject.toml`, under a marker limited to the platforms Nexus ships wheels
   for (rapier renders through Nexus, so it follows the same set).
   `uv sync --extra sim` installs both; the CUDA extra does not, because the
   published Nexus wheel has no CUDA feature.
2. **The source-build machinery is retired.** `engine-runtime.json`,
   `tools/engine_sources.py`, `tools/nexus_runtime.py`, `tools/rapier_runtime.py`
   and the receipts they wrote are removed. An unreleased engine change is tried
   by installing a locally built wheel over the locked one; such a run is
   recorded as unattested and refused by trusted runs, like any other
   out-of-lock dist.
3. **Provenance comes from the lock.** `tools/env_hash.py --sim-engine` keeps
   the engine digest (`sim_engine_hash` over the engine name and
   `src/aisle/sim/**`) but no longer folds in or requires a build receipt. The
   engine's wheel is identified the way Genesis is: by `uv.lock`, the
   `uv sync --locked --check` selection check, and the PEP 610 record of every
   dist in the engine-aware attested set. The manifest keeps its
   `sim_engine_build` key, now `{"engine", "sim_engine_hash", "n_files"}`.
4. **Backends are selected by name.** `select_nexus_backend("sim", ...)` keeps
   resolving native Metal on macOS and WebGPU elsewhere, and the Nexus backend
   passes the name to `NexusViewer.with_backend`, so a backend the installed
   wheel lacks (`cuda` on any published wheel) fails at the first scene build
   with the wheel's own message instead of a `hasattr` probe.

## Consequences

- A Nexus or rapier run can now pass the dist attestation, so the first of
  ADR-67's three preconditions holds. The other two (the `spec-change` PR
  generalizing SPEC 020/030, and a re-established frozen baseline) still stand,
  so neither engine enters the measured record yet.
- `uv sync --extra sim` no longer removes the engines, and no install step
  remains beyond the sync.
- WebGPU validates buffer usages where native Metal does not. Nexus 0.2.0
  read multibody joint velocities from a buffer without `COPY_SRC`, which
  panicked on WebGPU; 0.2.1 fixes it, which is why the pin starts there.
- Determinism (`deterministic = true`) is per backend: two runs on one
  backend replay bit for bit, but a WebGPU run is not expected to match a
  native Metal run of the same seed.
- On a platform outside the marker the `sim` extra installs without the
  engines, and `--sim-engine nexus` or `rapier` refuses at the `sim_engine`
  gate. The CUDA extra does not include the engines either.
- Editing `tools/env_hash.py` makes trusted runs refuse on this branch until it
  merges, so this lands as an env-change PR, not mid-campaign (ADR-67).
