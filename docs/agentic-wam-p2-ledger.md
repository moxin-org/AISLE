# P2 offline execution ledger

P2 implements the execution-history portion of the
[adoption plan](agentic-wam-adoption-plan.md) on the merged
[P1 dataset](agentic-wam-p1-dataset.md). It is a Class A offline tool under
CON-5/8/12 and HAR-4/6, with VAL-6's oracle-isolation boundary preserved by
the projection. No runtime topics, controller, scorer, specification or
frozen file changes. This is an engineering replay instrument, not evidence
that memory improves task performance.

## Run and reconstruct

```bash
uv run --no-sync python tools/execution_ledger.py \
  --dataset /tmp/aisle-p1-dataset --out /tmp/aisle-ledger.json
uv run --no-sync python tools/execution_ledger.py \
  --fixture tests/fixtures/execution_ledger/success_recovery.json
uv run --no-sync python tools/execution_ledger.py \
  --fixture tests/fixtures/execution_ledger/failure.json
```

The CLI emits one JSON object and exits zero exactly when `ok` is true.
Without `--out`, stdout includes the report. With `--out`, it names and hashes
a new report file. Existing files and output paths inside a source dataset
are refused. No model, simulator, provider session or GPU is started.

There are two intentionally distinct inputs:

- `--dataset` verifies the P1 bundle through `demo_dataset.read_dataset`,
  then reads its selected `reset_done` and `target_request` streams. Requests
  are read directly from source, including corrections with no subsequent
  command. A malformed request invalidates the previous instruction; there
  is no privileged `episode_goal` fallback.
- `--fixture` accepts version 1 `aisle_execution_events` with an explicit
  `provenance: synthetic` label and an ordered `records` array. These normalized
  records test semantics for a possible future capture adapter. They cannot
  be relabeled as retained skill execution through this input mode.

Reports identify the source manifest or canonical fixture digest. Every
transition references its source row and hash. Fixture hashes use canonical
JSON (sorted keys, compact separators, finite numbers); dataset references
also preserve Arrow file, row, sequence, simulation timestamp and environment.
The committed fixtures and [replay checks](../analysis/aw0_p2/replay_checks.json)
let a reviewer reconstruct the resulting histories. Hashes identify content;
they do not attest that an event occurred on a robot.

## What each state establishes

An entry binds invocation ID, environment, episode, goal revision, skill
name/version, parameters, completion condition, proposal/start/update times
and evidence references. Missing skill identity remains `unconfirmed`.
An observation is not used to invent an invocation or start time.

| State | Required evidence | Meaning |
| --- | --- | --- |
| `proposed` | Explicit proposal with skill name/version | Intent only |
| `started` | Explicit matching start after proposal | Execution reported starting; physical progress unknown |
| `unconfirmed` | Return without confirmation, missing identity, or invalidated unstarted intent | Evidence is insufficient for physical completion |
| `interrupted` | Matching interruption after start, or reset/goal invalidation of started work | Execution was reported interrupted or its intent was invalidated; no delivery claim |
| `failed` | Matching execution failure after start | Execution reported failure, independently of the scorer |
| `completed` | Matching post-start joint observation satisfying the declared predicate | The named local joint configuration was observed within tolerance |

The only supported completion predicate in v1 is `joint_position`, with
ordered joint names, target radians and a positive tolerance at most 0.1 rad.
Observations must name the same joints in the same order, use radians in the
joint frame, and have compatible finite vectors. A missing condition stays
unknown. These are synthetic declared conditions, not calibrated tolerances
for a real skill or a new physical verifier.

**Every entry keeps `delivery_completion: unknown`, including `completed`
entries.** Reaching a joint configuration does not establish object identity,
grasp, release, delivery, or that the skill caused the observed configuration.
An executor return alone produces `unconfirmed`. Interrupted and failed
invocations cannot be promoted by late sensor evidence. Recovery requires a
new invocation ID and fresh start/observation evidence.

## Replay and bounds

Synthetic records replay in delivery order. Each environment starts with
episode zero's reset at simulation time zero. Resets advance the episode
ordinal by one, archive the old history and clear active entries and goals.
Reset/goal invalidation is logical; this offline tool does not send a stop
command or prove that hardware stopped.

Goal revisions cannot be reused within an episode. A new revision marks old
intent obsolete and interrupts started work. Completed local observations
remain historical facts but are no longer current intent. Old-episode,
old-goal, wrong-version and noncausal events produce explicit rejections.
A newer matching-invocation observation makes older arrivals stale even
when it fails the completion predicate. Replaying the same event ID and
content changes neither state nor evidence; the report counts duplicates.
Conflicting content under the same ID is refused.

Distinct events at the same environment/timestamp are refused: the available
evidence supplies no cross-producer ordering. P1 source rows replay in
simulation-time order after this ambiguity check; their transport arrival
order is unknown. Synthetic delivery-order tests validate delayed-event
semantics, not historical transport behavior.

Defaults are 10,000 records per replay and 128 invocations per episode.
`--max-events` accepts 1–100,000; `--max-invocations` accepts 1–10,000.
Overflow refuses the report instead of silently evicting identities and
allowing a delayed event to revive forgotten work. Record payloads are capped
at 64 KiB, fixture input at 64 MiB, and joint vectors at 256 elements. The
source P1 bundle is still validated in full; these are ledger-state bounds,
not a streaming-memory guarantee for the existing dataset reader.

The `active` projection contains current per-environment state. `histories`
contains reset-closed episodes; `timeline` records applied/rejected evidence.
These are analyst-readable offline reports. Passing the whole P1 source
archive to a policy would disclose privileged data; the projection does not
turn that archive into a policy-safe filesystem or implement live access
control. No scorer results, oracle poses, episode goals, or outcome-derived
skill labels enter the projected state. An outcome-swap test checks this
independence explicitly.

## Evidence and limits

The retained `runs/t1-genesis-l2-spot` P1 bundle yields 24 events: 12 resets
and 12 policy-visible requests over 12 recorded episodes. Eleven histories
close at a following reset; the last remains open, consistent with P1's
missing-final-reset finding. **Identified invocations: zero.** It proves
source-linked request/reset reconstruction only. Successful and unsuccessful
source outcomes are retained by P1 but cannot supply missing skill spans.

The synthetic success fixture records partial execution, interruption, a
rejected late observation, then a separately identified recovery with a
return and confirming joint observation. Final states are `interrupted` and
`completed` (local configuration only). The failure fixture records an
execution failure, a rejected late observation, a changed goal and a proposal
with missing skill identity. Final states are `failed` and `unconfirmed`.
Neither fixture is a measured skill success or a scored robot delivery.

Acceptance tests were written before implementation. They cover deterministic
replay and evidence reconstruction, duplicates/conflicts, reset and environment
isolation, goal changes and delayed replies, partial execution and recovery,
missing identity/conditions, wrong skill versions, incompatible observations,
privileged-field refusal, source-hash checks, bounds and JSON CLI behavior.

Validation completed on 2026-11-12:

- Manual diff review covered state transitions, source attribution, temporal
  ambiguity, bounded identities and the scorer/projection boundary.
- All 87 focused P0/P1/P2 tests passed, including 32 new P2 cases.
- Repository-wide Ruff formatting and lint, traceability, documentation
  inventory and claim-evidence checks passed.
- `UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked pytest -m unit`:
  **4,665 passed, 419 deselected** in 1,636.58 seconds (exit 0).
- Repeated synthetic and retained replay produced identical reports; fixture
  references, report hashes and implementation hashes match the evidence.

The full unit gate used the documented lean environment outside the sandbox
because existing process-cleanup tests need `/bin/ps`. No test was disabled
or weakened.

P2's offline semantics can support P3 planning. A useful P3 pilot still needs
real versioned invocation spans, a defined observation-based completion
contract for the chosen skill, a task with usable success variation, and the
separate campaign entry/budget. Live capture, live ledger integration and
non-oracle delivery confirmation remain separately reviewed work. The owner
has authorized this offline P2 work; it does not change ADR-71's engine and
hardware order or authorize a provider-funded campaign.
