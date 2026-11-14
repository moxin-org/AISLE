# ADR-p3-chain-completion — Require the recovery result before completion

Status: implementation interpretation of HAR-1/2/4, TC-7 and ADR-32;
no spec, frozen scene, graph or scoring-rule change.

The rollout runner could count an initial delivery row as a completed physical
episode, enter its 90-second quiet-time grace, and terminate a recovery that
was still running. Its final row-count check could then return `ok: true`
without the recovery verdict. Completion must follow the expected goal chain.

For a T4 increment-two rollout, completion now requires the initial goal and,
on corrected seeds, the recovery goal for each physical episode. Each record
must match the expected episode, seed, goal identity and recovery marker;
duplicates, unexpected identities and synthetic scores cannot establish
completion. The selected protocol comes from the captured graph's trusted
rollout-client source and environment, with ambient increment-two settings
scrubbed. Missing results remain missing in the returned report and manifest.
Legacy per-goal metrics remain descriptive; they are not a P3 chain or session
success estimator.

A partial chain cannot enter the post-completion quiet-time grace. Existing
trace-liveness, per-episode and overall wall limits still bound execution.
If an increment-two chain hits its wall limit, the run stops with incomplete
evidence and its clamped seed; it neither appends a duplicate initial verdict
nor fabricates a recovery verdict or relaunches later seeds. This is distinct
from the existing ordinary-episode clamp behavior, which is retained. Typed
host postflight still has to finish within the run deadline.

This repair does not qualify a task or complete P3. Physical qualification
requires a separately declared source pin and bounded protocol.

The generated treatment table refreshes its rollout-source hash and derived
identity. Pending CSE v33 and pilot v19 bind that table, the updated rollout
source and predecessor lineage; study rules and every pending gate remain
unchanged. Historical registrations are preserved.
