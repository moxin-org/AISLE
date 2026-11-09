# Troubleshooting

The known failure modes, most common first. Every one of these has cost
this project real debugging time.

## Sim imports suddenly fail (`genesis`/`dora` not found)

You ran plain `uv sync`, which REMOVES the sim extras. Re-run:

```bash
uv sync --extra sim
```

Anything that touches the simulator needs the extra; plain sync is only
for pure-unit work.

The Nexus and Rapier wheels (`dimforge-nexus3d`, `rapier3d`) are part of the
same extra, so plain sync removes them too and `--sim-engine nexus` then
refuses at the `sim_engine` gate with `simulation engine 'nexus' is not
installed in this environment`. The same sync with `--extra sim` restores
them. They are published for macOS arm64, Linux x86_64 and aarch64, and
Windows x64 only; on other platforms the extra skips them. The complete install and selection
matrix is in the [simulation backend guide](simulation-backends.md).

## Leaked simulator processes (the first thing to check)

A timeout-killed or crashed `dora run` leaves orphaned node processes
behind — historically at high CPU, silently corrupting every
measurement that follows (slow episodes, flaky timing tests, thermal
throttling). Upstream issue: dora-rs/dora#2856.

Before debugging ANY perf/timing weirdness:

```bash
uptime                          # load average sane for an idle box?
ps aux | grep -E "dora|genesis|nexus|rapier" | grep -v grep
```

Kill leftovers by their run working directory rather than pattern-
matching all of python (other work may be running):

```bash
# inspect first, then kill the pids whose cwd is the stale run
lsof -a -d cwd -c python | grep runs/
```

The campaign runners sweep their own worktrees between sessions; manual
`dora run` invocations are on you.

## Dora version, receipt or runtime mismatch

The Python API is pinned to 1.0.1, but the CLI uses the corrected source commit
in `dora-runtime.json`. Both the affected release and corrected source build
report CLI version 1.0.1, so `dora --version` alone cannot distinguish them.
Check the actual receipt against the current pin:

```bash
uv run --extra sim --locked python tools/dora_runtime.py verify --prefix "$AISLE_DORA_PREFIX"
```

`AISLE_DORA_PREFIX` is the installation directory chosen in
[getting started](getting-started.md). If the receipt, binary or source pin
has changed, install into a new prefix using those instructions. Do not
reinstall the affected release CLI as a remedy for lockstep stalls, and do not
rewrite a receipt. A failed installation keeps partial output for diagnosis;
choose a fresh prefix for the retry.

For the public quickstart, pass `--runtime-prefix "$AISLE_DORA_PREFIX"`.
For manual commands, `which dora` should resolve inside that prefix after
`export PATH="$AISLE_DORA_PREFIX/bin:$PATH"`. Older Cargo, conda or Homebrew
installations may otherwise shadow the verified executable. Use `--extra cuda`
in verification commands if that is your selected environment.

Stop and restart any older daemon/coordinator you started before switching
runtime builds. Historical pre-1.0 recordings and stores may need migration;
consult the [upstream 1.0 release notes](https://github.com/dora-rs/dora/blob/v1.0.1/Changelog.md#v100-2026-09-02)
before reusing them.

## Rollout refuses to start

The refusal JSON says why; the common ones:

- **Frozen-set drift** — your working tree differs from the trusted
  env baseline. If you intentionally changed frozen code, that is an
  `env-change` PR (CON-7), not an override. For local dev on top of a
  known-good tree, `--env-baseline local` (recorded in the manifest).
- **No open idea** (HAR-8) — research-agent branches must
  `harness report log` an idea before rolling out. Humans:
  `--no-idea-gate` (recorded).
- **Validation errors** — fix the graph; the error's `hint` field
  usually names the registry node or adapter you need.
  `INSTALL_MISSING` means the manifest's package is not in the frozen
  environment — pick an installed alternative (see `analysis/h1/` for
  why this matters).

## Episodes hang or time out

- S-tier (retail) episodes are long-horizon; give rollouts a real
  `--timeout-s` and expect minutes per episode, not seconds.
- A graph bug can leave an episode with no termination condition —
  the episode then runs until the outer timeout. If a rollout stalls,
  check whether traces are still being written (`ls -lt
  runs/<id>/traces/`) before assuming the runner is dead.
- One machine, one sim run. Parallel sim runs (or a parallel `uv sync`
  / cargo build during a run) contend for the GPU/CPU and corrupt
  timing.

## Nexus engine refusals (ADR-67, optional engine)

- **`this nexus scene was superseded by a newer build_scene in this
  process`**: the Nexus engine holds one live renderable scene per
  process. A newer build takes the viewer's render nodes, and the older
  handle's cameras raise from then on (its physics readbacks still work,
  and the viewer frees those cameras, so a process can build any number of
  scenes).
  Build one scene per process, or read from the newest handle. The sim
  tests build a fresh scene per test for exactly this reason.
- **`nexus already initialized with backend 'X'; build_scene requires
  'Y'`**: like `gs.init`, the backend is fixed at the first build and
  cannot be mixed within a process. Set `AISLE_SIM_BACKEND` once
  (`metal`, `webgpu`, `cuda` or `cpu`) and restart.
- **`this nexus3d build has no cuda backend`** (or `metal`): `AISLE_SIM_BACKEND`
  asked for a GPU API the installed wheel was not built with. The macOS wheel
  supports `metal` (its default), `webgpu` and `cpu`; the Linux and Windows
  wheels `webgpu` and `cpu`. `cuda` needs a development build of Nexus
  ([simulation backend guide](simulation-backends.md)).
- Nexus steps in its deterministic mode, so two runs of the same seed and
  commands on one machine, wheel and backend should match bit for bit; one
  that does not is a bug report. See [determinism](determinism.md).

## Nondeterminism (same seed, different result)

Determinism is a spec requirement (CON-5) with a dedicated writeup:
`determinism.md`. Short version: RNG and time are injected, the scene
is rebuilt from (seed, embodiment, tier), and known nondeterminism
sources (thread pools, unordered dict iteration, wall-clock in result
paths) are bugs — file them as such. The M0 gate includes a
same-seeds replicate check.

## CI red but local green

- Did you run the full gate chain with `&&`? A `;`-chained run can
  scroll a red step past you.
- `tools/trace_check.py` fails when an implemented MUST id has no
  citing test — grep the spec id, add the test docstring citation.
- CI has no simulator; `-m unit` must not import sim deps at
  collection time.
