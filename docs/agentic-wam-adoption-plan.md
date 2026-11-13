# Agentic WAM adoption plan for AISLE

Status: proposed work breakdown, 2026-11-11. Repository baseline: `ed70b59`.

P0 completed on 2026-11-11: see the [data audit](agentic-wam-p0-audit.md).
It verifies the shared pipeline using retained stock-controller recordings
and documents the missing skill-specific recordings. P1's offline contract
repair merged in [PR #616](https://github.com/moxin-org/AISLE/pull/616): see the
[dataset report](agentic-wam-p1-dataset.md).
P2's offline execution ledger is implemented and validated for review: see
the [ledger report](agentic-wam-p2-ledger.md).
Runtime capture/skill-span instrumentation remains a separate follow-up;
P3–P6 remain proposed.

Start with execution-aligned skill evidence and explicit memory. Test their
value before building a controller selector, and require a competent learned
branch before evaluating hybrid control. Treat full AW-0 integration as a
separate, budgeted GPU experiment.

The first milestone is a trustworthy dataset and a reproducible replay report
for one existing skill. It does not require AW-0 weights or model training.

## Scope and project order

This plan adds no normative requirements and changes no spec, frozen path,
benchmark registration, or execution priority. Implementation follows the
tests-first contract in [CLAUDE.md](../CLAUDE.md), with existing requirement
IDs cited where applicable; proposed checks below are not new spec IDs.

[ADR-71](decisions/ADR-71.md) amends [ADR-66](decisions/ADR-66.md): engine
qualification precedes the owner-started #347 pilot, and the hardware priority
is G1 EDU U2 23 DOF, G1 29 DOF, then SO-101. Genesis remains the qualified
default until the Nexus gates and default switch complete. Consult the
[current status](../README.md#status) before starting a measurement campaign.
Do not pool results across engine baselines. The spec moratorium is lifted
only for ADR-71's named work, not for this proposal.

The audit below can prepare work without spending provider or GPU accounts.
Implementation and new campaigns must be scheduled against the existing
work order and ADR-66's measurement cadence. New contract work remains a
separate spec-change concern when eligible. This plan does not authorize a
paid campaign, change the owner-started pilot rule, or assume hardware access.

## Deliverables and dependencies

Effort estimates are planning ranges for one implementer with review support.
They exclude review queues, simulation runtime, provider sessions, and GPU
training. Re-estimate after the data audit.

| Item | Deliverable | Depends on | Estimated engineering effort |
|---|---|---|---|
| P0 | Data and contract audit with one replay fixture | Existing repository and retained runs | 1–2 days |
| P1 | Complete trajectory export and skill attribution | P0; eligible implementation slot | 3–5 days |
| P2 | Execution ledger with replay validation | P1 | 2–4 days |
| P3 | Small matched memory/reuse pilot and report | P2; campaign entry and budget | 2–3 days setup and analysis |
| P4 | Controller selector in shadow mode | P2; useful candidate policies | 3–5 days |
| P5 | Guarded hybrid experiment | P4; learned-branch competence; campaign entry | 3–5 days integration and analysis |
| P6 | Remote AW-0 feasibility report | P1; GPU allocation and task adaptation scope | 2–4 days for smoke test only |

P0 → P1 → P2 → P3 is the first work package. P4/P5 are conditional follow-ups.
P6 may inform later policy selection but is not a dependency of the first package.

## P0 Audit one existing skill and its recordings

Use `ik-transfer-v2` as the first attribution target because its transfer
stages and collision motivation are explicit. Use a retained successful run
and a retained failure from a compatible environment if both are available.
Inventory available evidence before collecting anything new.

Read the entire data path together:

- [Trace recorder](../src/aisle/harness/trace_recorder.py).
- [Demo export](../tools/demo_export.py).
- [Training loader](../tools/finetune_smolvla.py).
- [Skill manifest](../skills/ik-transfer-v2/skill.yaml) and execution code.
- [Skill registration](../src/aisle/harness/skill.py).

The current exporter writes indices and metadata; the training loader reads
pixels and commands from the original run. Audit that combined path rather
than assuming training data is absent or that the export is self-contained.

Check camera-event to video-frame correspondence, recording cadence, episode
boundaries, state/action alignment, guard outputs, goal revisions, missing
frames, and whether stage events actually identify the code that executed.
An event index is not sufficient proof of a decoded video-frame match.

Deliver an inventory of usable fields, missing fields, artifact hashes and
provenance limitations, plus a minimal fixture with manually checked alignment.
Distinguish historical unattested data usable for engineering from admissible
evaluation data. Never infer unavailable stage identity from outcome alone.

**Exit:** one trace can be reconstructed causally, or the report identifies
the exact recording changes needed. Missing data leads to a bounded capture
task, not fabricated labels or a model-training launch.

## P1 Export trajectories with skill attribution

Extend the existing exporter and its consumers compatibly. Avoid a second
competing dataset pipeline. Export a versioned dataset plus a manifest of
source runs, graph/skill/weight hashes, environment and engine identity,
embodiment, perception rung, split membership, and transformation settings.

Each record should bind episode and goal revision, observation capture times,
joint state, proposed command when available, guard-forwarded command, active
skill/version/parameters, execution interval, and observed outcome evidence.
Forwarded commands are control targets, not proof that the robot achieved the
commanded physical state. Keep measured state separately.

Use exact execution labels where available. Otherwise mark inferred labels
and their evidence explicitly. Preserve unsuccessful episodes for reliability
and recovery analysis; maintain an explicit successful-demonstration view for
imitation learning. Missing provenance must remain visible in both views.

Split by episode and task family before deriving windows. Never split nearby
frames from one episode across training and evaluation. Any skill discovery
uses development data only, then freezes its library before evaluation.

**Acceptance checks, written before implementation:**

- No future observation is paired with an earlier action, and no pairing
  crosses an episode boundary.
- Dropped frames, duplicate timestamps, missing streams and partial files
  produce explicit refusals or documented exclusions.
- Sampled decoded frames agree with source capture identity and timestamps.
- Guard-forwarded targets and proposed targets cannot be silently exchanged.
- Identical input bytes and settings yield identical record identities and
  content hashes, excluding explicitly declared incidental metadata.
- Success and failure views retain their source lineage and split assignment.
- Re-reading the export reconstructs the selected records without hidden
  dependencies on an unlisted source file.

Relevant existing constraints: CON-4/5/8, HAR-4, and the current topic contract.
If new runtime topics are required, separate their contract change from the
offline exporter. Do not assume approval of an offline format approves new
hot-topic schemas.

**Exit:** a small dataset passes round-trip and provenance checks and a report
shows usable coverage. The first version need not support every skill.

## P2 Represent execution history explicitly

Implement a bounded ledger of proposed, started, interrupted, failed,
completed, and unconfirmed skill invocations. Bind entries to episode, goal
revision, skill version, timestamps, and evidence references. Define what
sensor or execution evidence each completion state actually establishes.

Start offline from P1 traces. Only add live integration after replay semantics
are clear. A generated plan or a function returning without exception must
not by itself establish physical completion. The final scorer remains
independent; do not leak privileged verifier information into a non-oracle
policy through the ledger.

**Acceptance:** replay gives deterministic state; duplicate events are handled
idempotently; reset clears episode history; goal changes invalidate obsolete
intent; interrupted motion cannot become a completed delivery; delayed events
cannot revive an old goal. Retain an explicit unknown state when evidence is
insufficient. Test partial execution followed by recovery.

**Exit:** the success and failure fixtures produce evidence-linked histories
that a reviewer can reconstruct. No learned memory model is needed.

## P3 Measure whether structured history helps

Select one reachable, nontrivial composition or recovery task using development
data. Do not assume T2/T3 are appropriate merely because they are difficult.
If no candidate has usable success variation, report that instrument failure
before increasing sample size.

Compare two coding-agent arms with the same model, prompt, installed skills,
raw trace access, budgets, task distribution, and held-out acceptance rule.
One receives the structured history interface; the other receives the existing
trace interface. This estimates the value of evidence organization and access,
not of extra privileged information. Library contents stay fixed in both arms.
Test library accumulation separately if this first contrast is useful.

Proposed pilot size: six fresh sessions per arm, interleaved in matched task
blocks. This is an exploratory cost/variance estimate, not a powered claim.
Freeze the task, seeds, stop rule, exclusions, budgets and selection rule
before collection; keep hidden evaluation feedback out of agent sessions.

Report session success, tokens, wall time, rollout count, repeated failed
attempts, verified skill reuse, and physical failure outcomes. Count the
session as the independent unit; episode and command counts are nested data.
Keep pilot seeds and results out of any later confirmatory estimate.

**Decision:** continue if evidence use is verifiable and the pilot shows a
plausible cost or quality benefit worth a larger test. If success is saturated,
absent in both arms, or provenance is incomplete, repair the task/instrument
instead of claiming a treatment effect. Report null and negative outcomes.

## P4 Select controllers in shadow mode

Build a selector that proposes either a registered skill or a learned action
chunk, initially without actuation authority. Inputs include applicability,
observation age, current stage, attempts, remaining budget and measured
reliability. Use a transparent fixed rule first; a learned selector requires
separate calibration data.

Reuse [ADR-38](decisions/ADR-38.md) and the
[existing action queue](../src/aisle/nodes/vla_smolvla.py). Begin with allowlisted,
typed skill invocations over existing nodes; arbitrary generated Python and
nested subgraph execution are out of scope for the first version.

**Acceptance:** at most one controller owns a motion boundary; selected
commands traverse the guard; reset and goal revision invalidate pending work;
old asynchronous replies are rejected; timeout has a defined hold/stop
behavior. After partial skill execution, reacquire observations before using
learned fallback. Log every selection and rejection reason.

Shadow traces establish interface correctness and coverage. They cannot prove
the counterfactual controller would have succeeded.

## P5 Evaluate hybrid execution under matched conditions

Enter only after both branches show useful held-out competence on the selected
task. A learned policy that fails every trial is not a meaningful fallback.
Freeze the skill library, learned checkpoint, input access and selector rules.

Compare skill-only, learned-only, a simple fixed fallback, and the proposed
selector. Hold training data, engine, scorer, seeds, observation access and
guard limits fixed. Report clean and shifted conditions separately. Distinguish
lockstep competence evaluation from live wall-clock latency evaluation.

Measure task success, collision and wrong-object outcomes, stale refusals,
guard interventions, controller ownership, inference latency and total cost.
Calibrate selection rules on development episodes; use independent episodes
and appropriate training/session replication for the claimed inference.

**Decision:** retain hybrid execution only if its measured cost/performance
trade-off improves on the best relevant baseline. A safety regression blocks
promotion pending diagnosis; zero observed incidents is not a prevention proof.
The budget guard has no semantic identity guarantee.

## P6 Test AW-0 remotely before adapting it

Pin upstream source, weights, configuration and component licenses. First
reproduce a small native RoboTwin smoke test on an allocated GPU host. Measure
peak memory, startup time, inference latency and failure behavior before
estimating a training budget. Do not install CUDA dependencies into Aisle's
default environment.

Then scope one Aisle embodiment, synchronized camera contract, joint order,
units/frames, action normalization, chunk horizon and skill API. Use a
contract-tested remote client in shadow mode before granting motion authority.
A native RoboTwin success does not demonstrate transfer to Aisle or G1.

The 23-DOF G1 wire/grasp mapping remains dependent on the mode and end-effector
information identified in ADR-71. An available simulation embodiment can be a
separate engineering probe without claiming G1 transfer or changing hardware
priority. AW-0 fine-tuning receives a new estimate after these checks.

## Review and first work assignment

The implementation agent owns P0–P2 code, tests and evidence reports. Code
review follows CON-16 and the applicable risk class; findings on another
agent's PR go into PR comments. The owner starts provider-funded campaigns,
allocates GPU budget, and handles required spec/frozen-set approvals.

Run the repository's applicable gates before each commit: docs-only changes
need format, lint and unit; tools/tests retain diff review; nodes/harness need
the full Class B gates and affected acceptance tests. Sim/graph changes add
the corresponding suites. Split PRs by concern, not by arbitrary file count.

First assignment: **complete P0 and produce the one-skill data audit with a
manually checked success/failure replay fixture.** Use retained data first.
The audit should finish with the smallest P1 patch scope, the exact tests to
write, and any required new capture. Avoid building a generic framework before
that evidence exists.

## Research references

- [AW-0 paper](https://arxiv.org/abs/2610.30136): joint code/video/action
  generation and hybrid execution. Appendix J.2 identifies the different
  training-data pipelines in the default/hybrid comparison; Aisle's P5 keeps
  training fixed to isolate execution policy.
- [Annotation guide](https://github.com/ZhuoyangLiu2005/agentic-wam/blob/main/docs/CODE_ANNOTATION.md):
  distinguishes executed simulation labels from inferred video annotations
  and highlights frame-order alignment requirements.
- [RoboTwin guide](https://github.com/ZhuoyangLiu2005/agentic-wam/blob/main/docs/ROBOTWIN.md)
  and [deployment guide](https://github.com/ZhuoyangLiu2005/agentic-wam/blob/main/docs/REALWORLD.md):
  upstream reproduction and remote-server starting points for P6.
- [Aisle research program](research-program.md): the existing staged model
  integration agenda that this plan makes actionable.
