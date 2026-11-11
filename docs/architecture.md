# Architecture tour

A newcomer-altitude walk through the system. Depth on any topic: the
design doc (`Project_AISLE_Experiment_Design.md`, §2–§5) and the spec
that owns it (`specs/`).

## External benchmark trust boundary

<!-- claim:architecture/four-zone-boundary -->
The **coding-agent session** is the benchmark experimental unit. AISLE names
four trust zones so “the agent cannot” is never used where the repository only
says “the protocol forbids”:

1. **Mutable participant** — the coding agent, its workspace, declared robot
   graph, participant nodes, and allowed configuration.
2. **Frozen evaluator** — tracked task, scorer, admissibility, and analysis
   artifacts whose hashes are checked at session start and audit time.
3. **Trusted actuation** — the scoped validator and command guard boundary that
   gates declared graph paths and clamps commands; it is not yet a claim about
   arbitrary process, socket, or driver side channels.
4. **Hidden controller** — task selection, randomization, seeds, sealed faults,
   and treatment assignment kept outside the participant-visible workspace.

Sealed task/fault contents and held-out assignments are intended to be
**inaccessible** to the participant. Direct actuation outside declared
participant interfaces is **forbidden** by protocol but is not yet proven
inaccessible. The broader bypass boundary and attack classes are deferred to
issue **#350**; until that threat model is ratified and evaluated, AISLE claims
declared-topology gating only.

## The stack in one picture

```
BRAIN      coding agent (Claude Code / Codex) under a contract
             composes graphs, authors nodes, runs rollouts, logs ideas
ARTIFACTS  git: dataflow YAML, skills/, manifests, evalcards
EVIDENCE   runs/: Arrow traces, videos, results, idea tree
EXECUTION  dora-rs runtime — typed topics, dynamic node add/remove
WORLD      Genesis physics scene (pharmacy desk; retail store)
             optionally Nexus on the GPU (ADR-67) or rapier on the CPU
             (ADR-68): same scenes, same bridge node
```

The agent never touches the simulator directly. It edits **artifacts**
(a typed dataflow YAML plus node code), asks the harness to run them,
and reads **evidence** back. Everything in between is enforced
machinery.

## The dataflow layer (dora-rs)

A run is a dora dataflow: nodes connected by typed Arrow topics,
declared in YAML (`graphs/`). The topic contract (SPEC 010,
`src/aisle/topics.py`) fixes schemas, rates, units (radians, base
frame) for every topic class. Two hand-written expert graphs serve as
baselines and integration tests:

- `graphs/expert_t0.yaml` — fixed-pose desk pick (tier T0).
- `graphs/expert_s1.yaml` — mobile retail product-picking (S1).

Every graph passes through the **validator** (SPEC 060) before it runs:
edge-by-edge schema/rate checks, unique ids, every input has a
producer, embodiment consistency — plus the safety-structural checks
below. Errors are machine-actionable JSON with hints; this is the
agent's compile loop.

### Simulated-time lockstep (ADR-30)

Measured graphs do not run on wall time. A `turn-barrier` node opens each
simulated turn, and every node with a forward-edge path to a bridge command
or reset input participates: it declares `AISLE_LOCKSTEP` / `AISLE_TURN_NODE`
/ `AISLE_TURN_OUTPUTS`, and closes turn *k* only after receiving every count
declared by its forward upstreams for *k* and by episodic producers for
*k-1*. Cycles are legal exactly through inputs declared `turn_edge: episodic`
— the request/reply and goal/result loops that TC-6 and TC-7 need.

The barrier loads a **committed turn plan** (`graphs/turn_plans/<stem>.json`)
at runtime, which makes the plan executable scheduler topology rather than
documentation. Two consequences follow, and both are enforced: the plans are
frozen alongside the graphs they compile from, and `harness validate` refuses
a graph whose plan disagrees with it (`TURN_PLAN_STALE`) — a stale plan does
not fail at the gate, it kills the barrier on the first watermark and every
episode wall-clamps. Agents regenerate with
`harness validate <graph> --write-turn-plan`.

The validator's `CLOCK_*` family enforces all of this structurally:
participation completeness, queue policy on watermark edges, forward-DAG
acyclicity, and exactly one terminal commit back to the bridge.

### Fleet mode

Two distinct things share the name, and they are not interchangeable:

- **`src/aisle/harness/fleet.py`** fans one graph out across N environments
  (`fleet_graph` rewrites node ids per agent) — the batched-envs "virtual
  robot fleet" of design-doc §5.
- **`tools/a5_fleet.py`** runs N *independent research-agent sessions*
  concurrently, each with its own simulator on a shared host. This is what
  A5 measured, and the ADR records the deviation from §8.4.3: lanes share the
  host rather than one batched bridge, and peer cross-pollination is deferred.

A5's measured behaviour: throughput saturates near four lanes on a 16-core
host, per-agent latency degrades gracefully, and **holdout quality is
contention-invariant** — contention costs latency, not correctness.

## The frozen set (the no-cheating rule)

The frozen set (CON-7) is hash-manifested by `tools/env_hash.py`, with
changes only via human-merged `env-change` PRs. The rollout runner refuses
to start if hashes drift from the trusted baseline. Agents can read this
code; they cannot change what judges them.

A per-directory map of which frozen paths live where is **generated** at
[`docs/generated/project-inventory.md`](generated/project-inventory.md#repository-layout)
— derived from the constants below, so it cannot disagree with what is
actually attested. (The design doc's §8.0 tree predates all of this and is
marked superseded there.)

**Read `FROZEN_DIRS` / `FROZEN_FILES` / `FROZEN_GLOBS` in
`tools/env_hash.py`, not a list in prose** — a second copy of the fence goes
stale, which is exactly how `src/aisle/mobility` stayed outside it (issue
#189, ADR-33). As orientation only: the scenes, verifier and reset packages,
the mobility verdicts, the budget guard, the topic/turn/kinematics modules,
the SO-101 assets, the budget ledger, and — as the set has widened — the
expert *and* evaluation graphs together with their committed ADR-30 turn
plans.

That widening is worth understanding, because each step followed the same
argument: **the unit of the fence is what a result depends on, not the
directory it lives in.**

- **ADR-33** pulled in the mobility verdicts after PR #177 changed nav's
  stall and timeout budgets without moving a hash — two runs with different
  failure conditions attested as the same environment.
- **#197** pulled in `graphs/turn_plans/expert_*.json`: the turn barrier
  loads the committed plan at runtime, so an unfrozen plan lets the
  scheduler topology of a measured run change silently.
- **ADR-36** pulled in `graphs/eval_*.yaml` and their plans. Those graphs are
  the exam an agent-authored skill sits to enter the registry
  (`harness skill register` scores through them), and they were editable by
  the candidate. A gate the candidate can edit is not a gate.

`graphs/agent_campaign.yaml` deliberately stays **out**: it is the research
agent's own deliverable, which the campaign instructs it to keep current, so
freezing it would put CON-7 in direct conflict with the experiment.

Safety is structural, not behavioral (H5):

- **Oracle isolation** — the bridge publishes a privileged
  `oracle_state` topic (all object poses); the validator refuses any
  graph routing it to anything but the verifier (VAL-6).
- **Motion gating** — every `joint_cmd`/`gripper_cmd` path must pass
  through the `budget-guard` node (SPEC 080), which clamps joint/vel/
  workspace limits and enforces episode timeouts. Only
  `safety_class: motion` nodes may command the arm, and motion nodes
  without an evalcard are refused.
- **Trust anchors** — the budget guard, frozen-set nodes, and `env/`
  code can never be hot-swapped on a live dataflow (`harness swap`
  refuses, HAR-10).

## The world (Genesis, optionally Nexus)

`src/aisle/scenes/` builds scenes from (seed, embodiment, tier) — a
function, not a script, so reset/batching/randomization are all
reconstructible. The **pharmacy desk** (SPEC 020): shelf, five medicine
boxes, tray, distractors, fixed-base arm. The **retail store**
(SPEC 200): aisles, planogram-governed shelf slots, mobile base +
arm — scenarios S1 (order picking), S2 (restocking), S3 (misplaced-item
return), all verified by diffing live state against
`planogram.toml`.

One bridge node (`dora-genesis`, SPEC 030) owns the simulator and
speaks Arrow on the contract topics. Tiers (T0–T4 desk, S1–S3 retail)
select task difficulty; the perception ladder (L0 oracle poses → L1
ground-truth segmentation → L2 full pixels) selects which pose source a
graph may use.

The scene contract is engine neutral: the bridge only ever touches
`robot`, entity, link and camera objects through the small duck-typed
surface Genesis exposes, so another engine can realize the same scenes
behind it. **Genesis** is the only engine behind the measured record.
**Nexus** (ADR-67, `src/aisle/sim/nexus_backend.py`), the default since
ADR-71, realizes them on the GPU, and **rapier** (ADR-68,
`src/aisle/sim/rapier_backend.py`) on the CPU, rendering through the Nexus
viewer so the two share one camera model. `aisle.sim` dispatches
`build_scene` / `build_store` on `AISLE_SIM_ENGINE`, which
`harness rollout --sim-engine` injects into the bridge node and records in
the run manifest. The bridge node itself is shared, unchanged. Results are
not comparable across engines, and Nexus stepping determinism is not
established ([determinism](determinism.md)).

## Verification and reset (frozen)

The **oracle verifier** (SPEC 040) judges episodes from `oracle_state`:
success plus a failure taxonomy (`wrong_object`, `dropped`, `timeout`,
`never_grasped`, retail adds `misplaced`, `misaligned`, `overhang`).
The asymmetric goal makes `wrong_object` the metric that must stay at
zero. A **realistic verifier** (camera-based, portable to hardware) is
specified (VER-5) with its design brief at
`decisions/ADR-realistic-verifier.md`; fidelity against the oracle is a
first-class result. The realistic verifier is VER-5; the fidelity job
that measures agreement plus false-success/false-fail rates against
the oracle is VER-6. Reset is a service node: `teleport` (state reset)
now, `behavioral` (robot re-shelves the box) later.

## The capability registry

`registry/manifests/*.yaml` — one typed manifest per node: provides/
requires, input/output schemas and rates, params with ranges,
embodiment support, `safety_class`, eval status. The manifests cover
the desk stack, the retail additions (base driver, waypoint nav,
patrol, order reader, stock/misplacement detectors, placement
controller, task planner), the perception rungs, and system nodes —
the exact, current catalog is generated into
[`generated/project-inventory.md`](generated/project-inventory.md)
rather than counted by hand here. The registry is what
makes graph composition an LLM-reliable operation — and its deliberate
gaps are experiment probes (uninstalled hub packages were H1's dominant
failure; see `analysis/h1/`).

Skills are the registry's growth path: an agent-authored node or
subgraph + manifest + evalcard, registered via `harness skill register`
(runs the skill's eval suite first). The design intent is for subgraph
skills to nest as single named nodes with trace attribution preserved;
today registered skills are used as single nodes — subgraph NESTING in
the validator/graphs is deliberately deferred (ADR-22 item 6).

## The harness

`uv run harness {validate,rollout,traces,report,skill,swap,probe}` —
every command JSON-to-stdout (CON-8). `rollout` is the workhorse: it
verifies env hashes, validates the graph, launches per-episode, records
traces, and writes a reproducible run manifest (CON-5: same seed, same
result). `swap`/`probe` mutate or inspect a LIVE dataflow (HAR-10..12)
— the H4 iteration-latency mechanism. Full CLI walkthrough:
`harness-guide.md`.

Research agents run under a separate contract
(`harness/CLAUDE.research.md`) with an idea gate: no rollout without an
open idea-tree entry stating hypothesis and expected effect. That log
is what makes a campaign auditable afterward.
