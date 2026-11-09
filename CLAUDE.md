# CLAUDE.md — AISLE (development phase)

You are implementing AISLE via spec-driven development. This file governs
DEVELOPMENT agents. (The RESEARCH-agent contract that runs inside experiments
is a separate, later artifact: harness/CLAUDE.research.md — do not conflate.)

## Read first, in order
1. specs/000-constitution.md  — invariants; RFC 2119; quality gates
2. The spec you were asked to implement (specs/NNN-*.md)
3. TASKS.md — where your task sits in the dependency order
4. docs/Project_AISLE_Experiment_Design.md — background WHY (skim §2, §3, §8)

## The loop (spec-driven)
For every task: (1) restate the requirement IDs you will satisfy;
(2) WRITE THE ACCEPTANCE/UNIT TESTS FIRST, each citing its IDs in the docstring;
(3) implement until tests pass; (4) run gates (below); (5) PR with IDs listed.
If a spec is ambiguous: pick an interpretation, record docs/decisions/ADR-<n>.md,
proceed (CON-15). If spec and test conflict: STOP, open `spec-conflict` issue (CON-13).

## Environment
macOS arm64, Python via uv only. Setup: `uv sync` (dev) / `uv sync --extra sim`
(anything that runs the sim — plain sync REMOVES the sim extras). dora uses
the Python API pinned in pyproject. The CLI may use the matching 1.0.1 release
(`uv tool install dora-rs-cli==1.0.1`) or the explicit source pin in
`dora-runtime.json`, installed/verified by `tools/dora_runtime.py`. See
`docs/benchmark/v1/dora-runtime.md`; candidate source pins do not authorize
benchmark acceptance. Run things:
`uv run pytest -m unit`, `uv run harness validate graphs/expert_t0.yaml`,
`dora run graphs/expert_t0.yaml --uv`. Never install with bare pip/conda.
CUDA-only deps are forbidden in default extras (CON-1).
Optional engines (ADR-67, ADR-68): `harness rollout --sim-engine nexus` (or
`rapier`) runs the same scene on Nexus (or rapier); both wheels come from PyPI
through the `sim` extra (ADR-70, docs/getting-started.md §3b). Genesis stays
the default and the only engine behind the measured record.

## Quality gates before EVERY commit (inherited from dora-rs/dora practice)
Class B/C (nodes, harness, frozen set, contracts): /review on the diff →
/simplify → `uv run ruff format --check .` → `uv run ruff check .` →
`uv run pytest -m unit` → (+ `-m "sim or graph"` if you touched sim/graph
code). Class A docs-only (docs/, ADRs, README, generated): format → lint →
unit only. Class A tests/ or tools/: /review → format → lint → unit.
Conventional commits. One spec concern per PR. (CON-9 as amended by ADR-65.)

## Hard rules
- Never edit specs/ without a `spec-change` PR (CON-14).
- Never edit the frozen set after M0 without human review (CON-7).
- Every CLI: JSON to stdout, logs to stderr, exit 0 iff ok (CON-8).
- Every MUST you implement needs a test citing its ID; tools/trace_check.py
  will fail CI otherwise (HAR-9).
- Determinism is a feature: inject RNG/time; same seed ⇒ same result (CON-5).

## dora specifics you will need
- Dataflow YAML nodes with inputs/outputs; timers `dora/timer/millis/<N>`.
- Python node API: `from dora import Node`; events loop; pyarrow payloads.
- Service pattern: request/reply via `request_id` metadata (used by reset, TC-6).
- Action pattern: goal/feedback/result via `goal_id` metadata (episodes, TC-7).
- Reference the dora repo's CLAUDE.md and docs/patterns.md when unsure; prefer
  reading dora source over guessing API signatures.
