# Getting started

Primary platform: macOS arm64 (M-series). Linux supports CPU simulation with
the `sim` extra and GPU simulation on supported NVIDIA hardware with the
explicit `cuda` extra. The signed-off M0 experiments ran on an M3 MacBook; the Linux CUDA
measurements described in `docs/demo.md` are development evidence, not a
cross-backend reproducibility claim.
Install Git and rustup as well as Python >= 3.11, managed exclusively through
[uv](https://docs.astral.sh/uv/) — never bare pip/conda (CON-2).

## 1. Install

```bash
git clone https://github.com/moxin-org/aisle && cd aisle
uv sync --extra sim --locked
```

The `sim` extra pulls Genesis, torch, and the Dora Python API.
Things to know before anything else:

- **Plain `uv sync` REMOVES the sim extras.** If a sim test suddenly
  can't import `genesis` or `dora`, this is why. Re-run with
  `--extra sim`.
- **On an NVIDIA host, use `--extra cuda` instead.** `sim` resolves the
  CPU torch on Linux (CON-1 keeps CUDA wheels out of the default set), so
  Genesis runs on CPU even with a GPU present. `uv sync --extra cuda`
  installs the same stack with the CUDA torch; the two extras are
  mutually exclusive. Pass `--sim-extra cuda` to `harness rollout`; the
  request fails closed if CUDA is unavailable and is recorded in the run
  manifest. The default `--sim-extra sim` never auto-upgrades to CUDA.
- **The Python API remains pinned to 1.0.1; the CLI uses the declared source
  revision.** The release CLI 1.0.1 loses required events under timer pressure
  (#516). Install the corrected source pin rather than relying on version
  output, which is also 1.0.1 for the corrected binary.

Genesis is the default simulation engine. Nexus and rapier come with the same
locked `sim` extra (on macOS arm64, Linux x86_64 and aarch64, and Windows
x64); see the
[simulation backend guide](simulation-backends.md) for a comparison and the
selection path.

On Ubuntu 24.04, install the CPU quickstart's rendering prerequisites. Genesis
constructs an offscreen renderer even when physics runs on CPU:

```bash
sudo apt-get update
sudo apt-get install --yes --no-install-recommends libegl1 libgl1 libgl1-mesa-dri
```

Build and verify the declared runtime in a fresh directory outside the checkout:

```bash
rustup toolchain install 1.97.1
export AISLE_DORA_PREFIX="$PWD/../aisle-dora-runtime"
uv run --extra sim --locked python tools/dora_runtime.py install --prefix "$AISLE_DORA_PREFIX"
uv run --extra sim --locked python tools/dora_runtime.py verify --prefix "$AISLE_DORA_PREFIX"
```

Choose another new prefix if that directory exists. On an NVIDIA setup, use
`--extra cuda` in these uv commands to preserve the selected torch build.
The installer checks the source commit, Cargo.lock, compiler and paired API,
and retains the executable hash in a receipt. See
[the runtime guide](benchmark/v1/dora-runtime.md). A changed pin requires a new
installation; do not edit an old receipt to make it match.

For the supported CPU/Metal quickstart, while the checkout has no existing runs:

```bash
uv run --extra sim --locked python tools/quickstart.py --runtime-prefix "$AISLE_DORA_PREFIX"
```

This executes graph validation, a public task, bundle validation and reporting.
The quickstart uses the `sim` extra; CUDA experiments use the manual rollout
path below. The Linux clone and archive paths passed with verified source
receipts in [run 34082153283](https://github.com/moxin-org/aisle/actions/runs/34082153283).
That evidence does not establish independent reproduction or release readiness.

For manual harness and graph commands, make the verified binary available to
child processes in the current shell:

```bash
export PATH="$AISLE_DORA_PREFIX/bin:$PATH"
```

Stop any older daemon/coordinator you started before switching runtime builds.

## 2. Verify the install

```bash
uv run --extra sim --locked pytest -m unit    # no simulator, several minutes
uv run --extra sim --locked pytest -m "sim or graph"   # brings up Genesis; several minutes
```

Every harness CLI prints JSON to stdout, logs to stderr, and exits 0
iff ok (CON-8) — pipe anything to `jq`.

```bash
uv run --extra sim --locked harness validate graphs/expert_t0.yaml
```

## 3. Run the expert graph

The hand-written T0 baseline (pick a known box into the tray) is the
repo's integration test and your first end-to-end run:

```bash
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local
```

- On NVIDIA hosts, replace `--extra sim` with `--extra cuda` in the
  verification and manual harness commands, and add `--sim-extra cuda` to
  the rollout command.
- `--no-idea-gate` and `--env-baseline local` are the human/dev
  overrides (both recorded in the run manifest). Research agents run
  without them: rollouts then require an open idea-tree entry (HAR-8)
  and a trusted frozen-set baseline (ADR-21).
- While the run waits on dora, a progress line goes to stderr every 15 s:
  the phase (scene build before the first trace byte, then the running
  episode against its wall budget), the last verdict, and the run deadline.
  `AISLE_ROLLOUT_PROGRESS_S` changes the cadence; `0` silences it. Stdout
  stays the single JSON report (CON-8). Once physics runs, the line also
  carries the recent performance: wall time of the physics step call and of
  the whole bridge tick (the step plus the state reads that block on the
  GPU, which is the honest cost when the step call is an asynchronous
  submit), the engine's GPU time per step when it reports one (Nexus does,
  Genesis does not), render time per frame, and the real-time factor.
- The bridge writes those numbers per 100-step window to
  `runs/<run-id>/sim_timing.jsonl`, and the manifest's `sim_timing` block
  aggregates them over the run, so two engines can be compared on the same
  graph and seeds.
- The per-episode wall clamp is the tier's budget (150 s for T0/T1) plus a
  scene-build grace on the first episode of a launch. The grace is
  engine-derived: 420 s for Genesis, which compiles kernels (9m30s for T0),
  and 60 s for Nexus and rapier, which build a scene in seconds, so a wedged
  Nexus episode clamps about a minute past its tier budget.
  `--per-episode-wall-s N` overrides the tier budget.
- `AISLE_DEBUG_VIEW=side` (either engine) adds an operator camera after the
  build, looking at the shelf front from the tray side, and
  writes `runs/<run-id>/debug_view.mp4` at 10 fps, useful for seeing a grasp
  slip in profile. `AISLE_DEBUG_VIEW=px,py,pz;lx,ly,lz` sets an explicit eye
  and look-at in the base frame. The recorded traces and `overhead.mp4` are
  unchanged: this camera is not a topic.
- Results land in `runs/<run-id>/`: per-episode results JSON, Arrow
  traces, and videos. `runs/` is gitignored; every run is reproducible
  from (graph hash, env hash, seed list) (CON-5).

You can also launch a graph directly, without the rollout wrapper:

```bash
dora run graphs/expert_t0.yaml --uv
```

Sim runs want the machine to themselves — close other GPU/CPU-heavy
work, and see `docs/troubleshooting.md` if runs behave strangely
(leaked simulator processes from a previous killed run are the most
common cause).

## 3b. Optional: run the scene on the Nexus engine (ADR-67)

Genesis is the default and the only engine behind the measured record. The
graphs can also run on [Nexus](https://github.com/dimforge/nexus) (GPU
rigid bodies, native Metal on macOS and WebGPU elsewhere) for development:
the bridge picks the engine
from `AISLE_SIM_ENGINE`, which `harness rollout --sim-engine nexus` injects
into the bridge node and records in the manifest. Results are not
comparable across engines.

How far the Nexus path is actually exercised, as of today:

- The **pharmacy desk** scene with the franka and so101 embodiments, the
  **retail store** scene and the **mobile** embodiment are all covered by
  `tests/sim/test_nexus_scene.py` (placements, IK and home pose, the
  overhead/wrist passes, the realized wrist calibration, stepping, teleport
  reset, re-basing, batched builds).
- The **L0 and L1** rungs run end to end: single-seed expert rollouts of
  T0, T1 and T4 succeed, and `test_l1_estimate_matches_nexus_ground_truth`
  pins the estimator against ground truth inside 1 mm.
- The **L2** rung does not work on Nexus. The open-vocabulary detector
  refuses every frame (`identity margin -0.016 under the 0.01 floor`), so
  expert_t1_l2 and expert_t2 fail with `never_grasped`. The cause is render
  fidelity rather than physics: Nexus box pixels come back at 0.47
  saturation against a declared 0.73 albedo, and the frame is flatter than
  Genesis's.
- Nexus steps in its **deterministic mode**: the same seed and commands
  replay bit for bit on one machine, wheel and backend. See
  [determinism](determinism.md) for what is and is not covered.

Nexus is part of the lock (ADR-70): `uv sync --extra sim` installs the
published `dimforge-nexus3d` wheel (WebGPU everywhere, plus native Metal
on macOS). Then:

```bash
uv run --extra sim --locked pytest -m sim tests/sim/test_nexus_scene.py
```

```bash
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local --sim-engine nexus
```

`AISLE_SIM_BACKEND` accepts `metal`, `webgpu`, `cuda` or `cpu` for Nexus, as
far as the installed wheel supports them (`nexus3d.available_backends()`):
the macOS wheel has `metal`, `webgpu` and `cpu`, the others `webgpu` and
`cpu`; none has `cuda`. The
[simulation backend guide](simulation-backends.md) covers development builds
of unreleased Nexus changes.

The solver settings Nexus needs for the pick-and-place (substeps, contact
stiffness, PGS iterations) live in `src/aisle/sim/nexus_physics.toml`;
`tools/nexus_grasp_replay.py --run runs/<id>` replays a recorded run's joint
and gripper commands into a fresh Nexus scene under overrides, which is how
those values were chosen and how a grasp regression is reproduced offline.

## 3c. Optional: run the scene on the rapier CPU engine (ADR-68)

The third engine steps the same scenes with
[rapier](https://github.com/dimforge/rapier) on the CPU and renders them
through the Nexus viewer, so a rapier run and a Nexus run differ only in the
solver. Select it with `harness rollout --sim-engine rapier`.

It needs both wheels, the Nexus one above for the renderer and `rapier3d`
for the solver; the `sim` extra installs both. Its engine constants live in
`src/aisle/sim/rapier_physics.toml`, alongside the Nexus ones.

```bash
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local --sim-engine rapier
```

## 4. Where to go next

- `docs/physical-ai-primer.md` — the concepts behind the project
  (Physical AI, VLM/VLA/world models/WAMs, sim-to-real, agentic
  auto-research) mapped to where each lives in this repo — start here
  if you are new to the field itself.
- `docs/architecture.md` — what the nodes, graphs, and harness are and
  how they fit together.
- `docs/development-workflow.md` — the spec-driven loop, quality gates,
  and PR conventions (read before your first change).
- `docs/experiments.md` — the hypotheses, what has been measured, and
  where findings live.
- `CLAUDE.md` — the development-agent contract; short, and humans are
  held to it too.
- `docs/Project_AISLE_Experiment_Design.md` — the full design doc
  (the WHY behind everything above).
