# P3 readiness: collection is not ready

Follow-up implementation: [goal-chain capture repair](agentic-wam-p3-capture.md)
retains goal metadata and represents recovery verdicts within physical reset
windows. This repairs one evidence path; the other collection gates below
remain outstanding. P3 completion is the active objective, not readiness alone.

This is the readiness milestone for [P3](agentic-wam-adoption-plan.md), after
P2 merged in [PR #617](https://github.com/moxin-org/AISLE/pull/617). It is not
a completed pilot, registration, or memory-benefit result. **Zero P3 sessions
were assigned, launched or scored.** No paid account, simulator or GPU was
used in this audit. The owner requested the next goal; that authorizes this
readiness work without changing the separate campaign entry and budget.

The [readiness evidence](../analysis/aw0_p3/readiness.json) records source
hashes, the local search and current GitHub status. The
[pilot design draft](../analysis/aw0_p3/pilot_design_draft.json) is planning
data, not input to an existing campaign launcher. Its unresolved fields are
deliberately null; it is not a frozen protocol.

## What the evidence supports

The search repeated P0's `audit_skill_data.inventory` over local `runs/` and
`analysis/`, including ignored recordings: 19 and 34 episode files respectively.
Associated manifest/authored-graph/graph files contain no references to
`ik-transfer-v2`, `return-planner`, `t2-scan-pose`, `t2-scan-tsm` or
`expert_t4_inc2`. This bounded search does not establish that remote copies
do not exist. A textual graph hit would still not prove execution.

The retained T1/L2 Genesis run has 9 successes and 3 failures in 12 episodes.
Its [P2 replay](../analysis/aw0_p2/retained_replay.json) identifies zero skill
invocations. Its failures include policy perception/refusal rather than
demonstrated partial execution and recovery. That variation cannot qualify
the proposed memory treatment. The corresponding Nexus development records
are separate engine evidence: T1/L1 is 46/50 and T1/L2 is 37/50. Neither
establishes qualified skill-history instrumentation. Do not pool them.

| Candidate | Existing evidence | P3 disposition |
| --- | --- | --- |
| Ordinary T1 with `ik-transfer-v2` | A3's historical arms both scored 1.0; P0 could not recover the skill traces | Capture diagnostic only; not a selected recovery task |
| T4 increment-two return and redelivery | Historical complete r4 chain: 1/3 recoveries; other outcomes `not_returned` and `collision` | First candidate for repair/calibration, not admitted |
| T2 short composition with fixed scan skills | Later matched library comparison: both arms 0.25; earlier campaigns were floored or inadmissible | Second candidate, not admitted; library contrast is not a history-interface contrast |

The historical sources are [A3](../analysis/a3/a3_findings.md),
[T4 recovery](../analysis/t4_inc2/findings.md), and the
[later T2 comparison](../analysis/h3/t2_differential/findings.md). They describe
different pins, agents, tasks and budgets. The T4 result is only three chains;
the T2 comparison has one session per arm. They justify a shortlist, not an
estimate of current success probability or a powered sample size. No task is
selected until development calibration passes the criteria below.

## Concrete blockers

1. **No captured invocation history.** P2's real-data adapter projects only
   resets and `target_request`. Its invocation/recovery examples are synthetic.
   P1 has no versioned invocation IDs, lifecycle spans or recorded bindings
   from a skill to commands and observations. An episode's final verdict or a
   graph containing a skill is not a replacement.
   There is also a concrete adapter mismatch: `rollout_client.py` writes the
   initial and recovery verdict under the same episode ID, without resetting.
   P1's one-outcome-per-reset format refuses those records with `duplicate
   episode identity` (reproduced with a synthetic schema probe). Extend the
   offline representation to retain per-goal verdicts within a physical
   episode before using T4. Do not drop the first verdict, renumber outcomes
   as resets, or weaken P1's duplicate/reset checks to admit the chain.
2. **The candidate policy boundary is unsuitable.** In
   `graphs/expert_t4_inc2.yaml`, `segmented-pose` and `return-planner` consume
   simulator segmentation, and the task state machine consumes
   `verifier-oracle/episode_result`. The candidate cannot become a non-oracle
   P3 instrument by hiding just the ledger's outcome field. Recovery timing
   driven by a hidden verdict can itself disclose privileged information.
3. **Completion and task variation are unqualified.** P2 confirms only a
   declared local joint configuration in synthetic evidence, never delivery.
   Real return/redelivery needs policy-visible execution and sensor evidence;
   the held-out scorer remains separate. Existing aggregate rates do not
   qualify current partial-execution/recovery behavior.
4. **Existing launchers implement different treatments.**
   `tools/h3_campaign.py` varies library persistence. `tools/matched_campaign.py`
   and `matched_session.py` bind `typed`/`monolithic`; `pilot_assignments.py`
   and `pilot_records.py` use the same study-specific mapping. Relabeling
   these arms would change more than history access. Reuse the lower-level
   process, token accounting and sealed-block randomization components only
   through a separately reviewed P3 adapter with parity tests.
5. **Engine/program and funded-run entry remain unresolved.** GitHub confirms
   #609 and #610 merged; they are not still approval blockers. But the default
   switch [#613](https://github.com/moxin-org/AISLE/pull/613) is open/draft and
   [#614](https://github.com/moxin-org/AISLE/issues/614) is open. The
   [program update](https://github.com/moxin-org/AISLE/issues/604#issuecomment-6803885527)
   records the switch on hold. `src/aisle/sim/__init__.py` still defaults to
   Genesis. A funded P3 campaign slot, model/provider, exact instrument pin,
   seed commitments and approved cost cap have not been supplied or frozen.

This report does not reopen the held engine work or demand a Nexus change
for the offline audit. Before collection, resolve a specific slot under
ADR-71 and pin one qualified engine; do not silently substitute a development
Nexus result for the historical Genesis baseline.

## Smallest repair and calibration work

Do these as bounded follow-ups, not another generic campaign framework:

1. Recover complete historical `review-reeval-ik-transfer-v2` and
   `t4-inc2-recovery-r4` run directories from their original host if available.
   Retain manifests, source revision, graphs, Arrow streams, raw captures,
   logs and outcomes. Hash and audit them. Recovery of old files may improve
   diagnostics but cannot create absent invocation spans.
2. Add versioned lifecycle capture at the selected skill executor: proposal,
   actual start, interruption/failure and return, with environment/episode,
   policy goal revision, invocation ID, skill code/version/parameters and
   simulation timestamp. Preserve actual source command/observation identities
   and distinguish proposed from guard-forwarded targets. Retain explicit
   unknowns where the executor cannot observe a fact. The existing
   `StageStreamer`/routed plan is the first inspection target, not evidence that
   a generic `plan_done` signal already establishes delivery. Test duplicate,
   delayed, reset and goal-change cases before adding a real P2 adapter.
3. Establish policy-visible recovery evidence and isolate the final scorer.
   Inspect the T4 request/return chain and realistic perception path; no
   oracle segmentation, final goal or verdict may enter either arm's history
   or raw interface. Both arms get the same allowed source events. Capture
   identity is needed for any frame used to establish a physical condition.
   A failed or interrupted motion cannot be reported as completed delivery.
   New runtime topics or frozen graph/verifier changes require separately
   scoped contract/environment review under CON-14/7. None is changed here.
4. Run bounded development calibration only after instrumentation, isolation
   and an engine/work slot are reviewed. Proposed order: T4 recovery first,
   then T2 composition if T4 fails the task criterion. Freeze each candidate's
   graph and fixed library before calibration; no within-batch tuning.
   Proposed development-only seed sets: 200, 204, …, 244 for T4, 42–53 for T2,
   at most 12 complete chains each (24 total, up to 48 verdict records).
   The T4 set is restricted to the existing `is_corrected(seed)` predicate
   (`seed % 4 == 0`); ordinary T4 seeds do not exercise this recovery branch.
   These public seed values are never pilot or confirmatory holdouts.
5. Admit the first candidate with at least two successes and two failures in
   its 12 development chains, complete source-linked lifecycle evidence for
   every attempted invocation, and a reviewer-reconstructable partial attempt
   followed by recovery. Reject perception-only refusal variation as evidence
   of recovery. Missing capture/isolation or invalid scoring stops calibration
   for repair; it is not permission to switch tasks and hide the failure.
   If neither candidate qualifies, report no eligible task and stop. Do not
   extend the sample or manufacture a failure to create variation.

The calibration criterion is a proposed engineering screen, not a statistical
claim or a new spec requirement. It must be reviewed/frozen before capture.
The historical n=3 T4 series does not pass it. Instrument repair comes first;
running the unchanged P0 capture command alone will not make P3 ready.

## Matched pilot draft

The estimand is the effect of organizing and exposing the same execution
evidence as structured history. It is not library accumulation, extra sensor
information, or different controller competence.

Use six matched temporal blocks, each containing one fresh session per arm:
`history` and `trace`. Use the existing `create_sealed_plan` primitive with
these explicit arm names and a controller-private seed. Balance within each
block; reveal each assignment only at launch. Do not run the study-specific
`pilot_assignments.py` CLI with renamed treatment labels.

Hold model/version/settings, common task prompt, installed library hashes,
allowed action/edit surface, raw event access, task distribution, engine,
budgets and acceptance rule identical. Both arms receive the same neutral
description of available evidence. The history arm alone gets the derived
ledger interface; the trace arm uses the existing raw trace interface. Keep
the ledger implementation and prior reports out of the trace participant's
filesystem. It may independently organize its raw data; record that as
behavior rather than giving it an artificial information disadvantage.
Freeze the shared library and prohibit cross-session additions or peer memory.

Each block shares a precommitted development scenario distribution and hidden
acceptance suite across its two fresh sessions. Use disjoint seed roles:
calibration, pilot development, pilot acceptance, later confirmatory study.
For T4, every role samples the same corrected-seed subset that actually
triggers recovery, with disjoint values across roles and the same within-block
values in both arms.
The controller checks disjointness against the historical seed registry as
well as these draft development seeds. Commit hashes before assignment;
never publish hidden seed values or feed acceptance results back to agents.
No assignments or private seeds were generated in this readiness audit.

Proposed resource ceiling, awaiting a named provider/model and approval:

| Resource | Per session | Twelve-session maximum |
| --- | ---: | ---: |
| Provider-reported tokens | 400,000 | 4,800,000 |
| Wall time, including acceptance | 2.5 hours | 30 hours |
| Development verdict records, including recovery verdicts | 40 | 480 |
| Held-out chains, at most two records each | 8 | 96 chains / 192 records |

With calibration, the envelope is at most 720 verdict records, at most
600 reset episodes (a conservative bound), and
32 host-hours (two additional calibration hours); the selected task's chain
length must fit that envelope before launch. Token accounting includes the
provider's reported input/output/cache categories under one frozen meter in
both arms; ledger/tool overhead is not silently excluded. These proposed
token and reset-episode totals fit below the existing 5M-token / 2,000-episode
ceilings, but are not permission to spend them. Retain the existing
100-rollout-hour cap separately: host wall time is not interchangeable with
summed rollout time. No budget file is edited.
Dollar cost remains unknown until model, rate card and token mix are pinned.

Use the same independent hidden scorer for both arms. Draft session success:
at least 6/8 accepted held-out chains and zero unintended wrong-object events;
the staged initial delivery in the existing recovery task is classified by
the task's frozen rule, not counted as an accidental wrong-object event.
This threshold is a proposal, not a change to any verifier or registration.
Report the full chain outcomes alongside the binary session result. Passing
the chain-level calibration screen does not establish session-level variation
at 6/8: check task/budget/acceptance feasibility on development evidence before
freezing the pilot. If the acceptance rule is likely to floor both arms,
redesign and review it before collection; do not change it after pilot scores.

Stop after six sessions per arm; no outcome-dependent extension. Also stop
on wrong-object incidents outside the scripted task, guard bypass, provenance
or isolation failure, or a hard resource cap. Retain every assigned session,
including launch failures and budget stops; never replace a bad result with
a fresh session. Report unstarted/incomplete/unscorable cases explicitly with
their original reason. Do not turn missing scores or telemetry into zeros.
Budget-stopped sessions retain elapsed cost and acceptance results if safely
obtainable within the cap; otherwise mark them unscorable. Any replacement
study is a new protocol, not an invisible retry.

Report by arm and matched block: assigned/started/completed/scored sessions,
session success, provider tokens, wall time, rollout and episode counts,
repeated failed attempts, evidence-verified skill reuse, physical failure
classes and safety incidents. A repeated failed attempt is the same skill
version/parameter hash under the same goal revision after a recorded failure,
without new relevant observation; freeze that event-based derivation before
collection. Verified reuse requires a versioned invocation of the fixed
library, not an import or source-code mention. Episodes and commands are nested
observations; they do not increase the independent session count.

At n=6 per arm, report distributions and descriptive paired differences,
including null or adverse results. Do not claim a powered treatment effect,
use episode counts as independent trials, or reuse the pilot as confirmatory
data. Proceed to a larger design only if instrumentation worked and the cost
or quality difference is worth investigating; otherwise repair or stop.

## Validation and disposition

This change is documentation and generated audit evidence only. Existing
requirements affected: CON-5/8/9/11/12, HAR-4/6 and the VAL-6 information
boundary; no new normative requirements or runtime behavior are implemented.

Validation completed for this readiness change:

- Local inventory/source hashes and all report artifact hashes checked.
- The synthetic chained-verdict probe reproduces P1's explicit refusal;
  it does not represent a robot trial.
- Proposed token, verdict-record, reset-episode and wall envelopes checked;
  the T4 development seeds satisfy the existing correction predicate.
- Repository-wide Ruff formatting/lint, documentation inventory and
  claim-evidence checks passed.
- `UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked pytest -m unit`:
  **4,665 passed, 419 deselected** in 1,728.76 seconds (exit 0). The documented
  lean environment ran outside the sandbox because existing process-cleanup
  tests require `/bin/ps`; no test was skipped or weakened.

Readiness outcome: **no-go for collection; repair/capture first**. The P3
readiness goal can close with this report and draft. The measured P3 pilot
remains unrun, with its scientific question unanswered.
