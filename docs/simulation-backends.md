# Simulation backends

AISLE can realize the same scene and bridge contract with three physics
engines. Select one with `harness rollout --sim-engine`; the harness injects
the choice into the bridge and records it in the run manifest.

| Engine | Physics | Renderer | Installation | Status |
|---|---|---|---|---|
| `genesis` | Genesis | Genesis | Included in the `sim` or `cuda` extra | Default; only engine behind the measured record |
| `nexus` | Nexus GPU | Nexus | `dimforge-nexus3d`, included in the `sim` extra | Development only |
| `rapier` | Rapier CPU | Nexus | `rapier3d` and `dimforge-nexus3d`, included in the `sim` extra | Development only |

Results from different engines are not directly comparable. Each run records
the engine, resolved backend and device, and the engine realization digest.
The engine wheels come from `uv.lock` like Genesis, so the environment
attestation covers them too ([ADR-70](decisions/ADR-70.md)).

The Nexus and Rapier wheels are published for macOS arm64, Linux x86_64 and
aarch64, and Windows x64; the `sim` extra skips them on other platforms.

## Genesis

Genesis is installed with the normal simulation environment:

```bash
uv sync --extra sim --locked
```

It is the default when `--sim-engine` is omitted. An explicit development
rollout is:

```bash
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local \
    --sim-engine genesis
```

On NVIDIA Linux, use `uv sync --extra cuda --locked`, use `--extra cuda` on
the `uv run`, and add `--sim-extra cuda`. The CUDA request fails closed if no
CUDA device is available.

## Nexus

Nexus is installed by the normal simulation environment:

```bash
uv sync --extra sim --locked
```

Run a graph with Nexus:

```bash
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local \
    --sim-engine nexus
```

The harness resolves native Metal on macOS and WebGPU elsewhere (wgpu then
picks the platform's GPU API). The macOS wheel supports `metal`, `webgpu` and
`cpu`; the Linux and Windows wheels `webgpu` and `cpu`
(`nexus3d.available_backends()` lists them). `AISLE_SIM_BACKEND` overrides the
choice; a backend the wheel lacks, such as `cuda`, fails at the first scene
build. Determinism holds per backend: a Metal run and a WebGPU run of the same
seed are not expected to match bit for bit.

## Rapier

Rapier steps physics on the CPU and uses Nexus for rendering. Both wheels come
with the `sim` extra:

```bash
uv sync --extra sim --locked
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local \
    --sim-engine rapier
```

## Engine development builds

To try an unreleased Nexus or Rapier change, build its wheel with maturin from
the checkout and install it over the locked one:

```bash
uv pip install --reinstall-package dimforge-nexus3d ../nexus/target/wheels/dimforge_nexus3d-*.whl
```

Use `uv run --no-sync` afterwards, since a sync puts the locked wheel back.
Such a wheel is not from the lock, so the run records it as unattested
(`env_attested: false`) and trusted runs refuse it.

## Selection rules

- `--sim-engine {genesis,nexus,rapier}` is available on `harness rollout`,
  `harness fleet`, `harness monolith run`, `harness fault calibrate`, and
  `harness skill register`.
- A graph can declare `AISLE_SIM_ENGINE` in its simulation bridge node. A
  conflicting CLI selection is refused instead of silently overriding the
  graph.
- For a direct `dora run`, set `AISLE_SIM_ENGINE` in the bridge node's `env`
  mapping. Prefer the harness for recorded work because it validates, injects,
  hashes, and records the engine choice.

More detail:

- [Getting started](getting-started.md), including platform prerequisites
- [Harness CLI guide](harness-guide.md)
- [Troubleshooting](troubleshooting.md)
- [ADR-67: Nexus engine](decisions/ADR-67.md)
- [ADR-68: Rapier engine](decisions/ADR-68.md)
- [ADR-70: Engine wheels from PyPI](decisions/ADR-70.md)
