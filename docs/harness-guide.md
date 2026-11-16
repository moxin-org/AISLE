# Harness CLI guide

Every subcommand prints JSON to stdout, logs to stderr, exits 0 iff ok
(CON-8) — `| jq .` is your friend. Spec: `specs/070-harness-clis.md`.
Humans and agents use the same tools; if a command is annoying for you,
file that as a bug.

## validate — the compile loop

```bash
uv run harness validate graphs/expert_t0.yaml
uv run harness validate mygraph.yaml --embodiment mobile
```

Loads the graph plus all registry manifests and checks edge by edge:
unique ids, every input has a producer, Arrow schema and rate
compatibility, oracle isolation (VAL-6), motion gating through
`budget-guard`, embodiment consistency, evalcards for motion nodes.
Errors carry a `code` and a `hint`:

```json
{"ok": false, "errors": [{"code": "SCHEMA_MISMATCH",
  "edge": "perception/object_pose -> grasp_planner/object_pose",
  "produced": "pose6d_f32", "expected": "pose7d_f32",
  "hint": "insert an adapter node or use pose-convert from the registry"}]}
```

`--allow-unproven` downgrades the missing-evalcard refusal to a warning
— dev-only; the research harness never sets it.

## rollout — seeded episodes through a graph

```bash
uv run harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 8 --seeds 0..7 --no-idea-gate --env-baseline local
```

Order of operations: frozen-set hash check (ADR-21 baseline) →
validate → launch per episode batch → record → write
`runs/<run-id>/` (results JSON, Arrow traces, videos, manifest).
Reproducible from (graph hash, env hash, seed list); `--run-id` pins
the id when you need a stable path.

Flags you will actually use: `--tier` (T0..T4, S1..S3), `--embodiment
{franka,so101,mobile}`, `--seeds a..b|comma list`, `--reset
{teleport,behavioral}`, `--timeout-s`. The two human-override flags —
`--no-idea-gate` (skip the open-idea requirement, HAR-8) and
`--env-baseline local` (trust the local frozen set) — are recorded in
the run manifest; research agents run without them.

`--sim-extra {sim,cuda}` picks the attested dependency/backend selection.
`--sim-engine {genesis,nexus,rapier}` picks the physics engine realizing
the scene (ADR-67, ADR-68); it defaults to `nexus` (ADR-71) and is recorded in the
manifest as `sim_engine`, beside a digest of the engine realization and the
wheel's build receipt. `nexus` (GPU) and `rapier` (CPU, rendering through
the Nexus viewer) refuse before launch when their wheels are not installed
(see the [simulation backend guide](simulation-backends.md)), and results are not
comparable across engines. A graph whose bridge node declares
`AISLE_SIM_ENGINE` owns the choice: passing a different `--sim-engine` is
refused rather than silently overridden. The same flag is on
`harness monolith run`, `harness fault calibrate`, `harness skill register`
and `harness fleet`.

```bash
uv run --no-sync harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local --sim-engine nexus
```

The scene-build wall grace added to the first episode of each launch is
engine-derived: 420 s for Genesis, which compiles kernels, and 60 s for the
other two, which build a scene in seconds.

## traces — query recorded evidence

```bash
uv run harness traces query --run <run-id> --topic budget-guard/joint_cmd_safe \
    --episode 3 --summarize
uv run harness traces query --run <run-id> --topic dora-genesis/joint_state \
    --format npz --out /tmp/js.npz
```

Arrow slices by run/topic/episode/time-window (`--t0/--t1`, sim ns).
`--summarize` returns compact stats instead of raw samples — reach for
it first; raw topic dumps are large and mostly noise.

## report — the idea tree

```bash
uv run harness report log --idea "descend at 0.4x speed near shelf" \
    --parent I3 --expect "+10pp pass@1"
uv run harness report close --id I7 --observed "+12pp" --verdict up
```

Appends to the per-branch idea-tree JSONL. For research agents this is
not optional bookkeeping: `rollout` refuses to run on a branch with no
open idea (HAR-8), which is what makes campaign logs legible as
pre-registered hypotheses.

## skill — register a reusable skill

```bash
uv run harness skill register skills/my-skill/
```

Validates the skill's manifest, runs its bundled eval suite, writes the
evalcard, and installs it into the registry. A skill = node or subgraph
code + `capability.yaml` + eval config; until an evalcard exists a
motion-class skill cannot be used in a validated graph.

## swap / probe — live-dataflow ops (HAR-10..12)

```bash
uv run harness swap --graph graphs/agent.yaml --dataflow <df-id> \
    --replace grasp_planner --with skills/grasp_v2/node.yaml
uv run harness probe --dataflow <df-id> --topic dora-genesis/joint_state --for 30
```

`swap` validates the FULL post-swap graph before any runtime mutation,
then does remove→add (restoring the original node if the add fails) and
writes the post-swap doc back to the graph file. Refused outright:
swapping the budget guard or any frozen-set node (trust anchors), and
any replacement that fails validation. `probe` attaches a temporary
read-only trace recorder to a topic and detaches after the window;
`oracle_state` is refused (VAL-6 has no probe exemption).

Every swap/probe ATTEMPT — success, failure, or refusal — appends an
event to `runs/swaps/<branch>.jsonl` (HAR-12). Those events are the raw
material for the H4 iteration-latency comparison.
