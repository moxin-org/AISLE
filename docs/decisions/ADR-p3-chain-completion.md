# ADR-p3-chain-completion — Require the recovery result before completion

Status: implementation interpretation of HAR-1/2/4, TC-7 and ADR-32;
no spec, frozen scene, graph or scoring-rule change.

The separately frozen development calibration at `389d83c` stopped after
two complete failed chains and one incomplete chain. Nine slots remained
unstarted. Seed 208's initial delivery succeeded, but its recovery result
was absent: the rollout harness stopped after 90 wall seconds without another
result row, while the recovery was still running. Its legacy row count then
reported `ok: true`. The P3 assessor rejected the incomplete scorer evidence
and stopped the batch. The retained [report](../../analysis/aw0_p3/development_calibration_389d83c.json)
preserves all attempted and unstarted slots. That batch is closed and is not
resumed or replaced by this implementation.

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

The two complete development chains also exposed infeasible return pregrasp
plans. This completion repair does not fix those plans, qualify T4, authorize
another calibration or paid session, or complete P3. Any later collection needs
its own declared source pin and bounded protocol.

The generated treatment table refreshes its rollout-source hash and derived
identity. Pending CSE v33 and pilot v19 bind that table, the updated rollout
source and predecessor lineage; study rules and every pending gate remain
unchanged. Historical registrations are preserved.
