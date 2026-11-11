# Agentic WAM P0 data audit

Audit date: 2026-11-11. Source baseline: `ed70b5927455e127bdc5e58a2a5abc231df60e93`.
Scope: P0 of the [adoption plan](agentic-wam-adoption-plan.md).

The current export/loader path is not ready to supply general, skill-aligned
training data. A retained stock-controller success/failure pair demonstrates
an episode-boundary image mismatch and lost instruction identity. Synthetic
fixtures additionally reproduce command-index loss and a future-state fallback.
No local retained execution of `ik-transfer-v2` was found. Its source can be
audited, but its physical execution cannot be replayed from the available
skill-specific evidence.

The smallest P1 should repair causal joins, command identity, instruction
recovery and embodiment validation in the existing offline pipeline. Runtime
skill spans and a capture manifest are separate follow-ups where recordings
cannot supply the missing evidence.

## Evidence and reproducibility

[audit_skill_data.py](../tools/audit_skill_data.py) reads retained inputs and
invokes the actual exporter and loader using temporary metadata files. It
does not launch simulation, train a model, call a provider, or alter a run.
Its `ok: true` means the audit completed, not that a dataset is certified.

Reproduce the retained-run audit from the repository root:

```bash
UV_CACHE_DIR=/private/tmp/aisle-plan-uv-cache uv run --no-sync python tools/audit_skill_data.py \
  --run runs/t1-genesis-l2-spot --episodes 0,1 \
  --search-root runs --search-root analysis \
  --search-root /Users/yonghe/src/aisle \
  > /private/tmp/aisle-p0-reproduced.json
diff -u analysis/aw0_p0/retained_run_audit.json /private/tmp/aisle-p0-reproduced.json
```

The [audit artifact](../analysis/aw0_p0/retained_run_audit.json)
contains input hashes, current source hashes, inventory results, decoded video
counts, sampled decoded pixel hashes, episode boundaries and loader output.
Inventory changes, source edits or a different video decoder can change a
subsequent report; inspect these differences rather than treating them as a
physics reproducibility result. The large recordings remain under ignored
`runs/` and are required to regenerate the real-data portion on another host.
The report alone does not replace those inputs.

Run the portable synthetic checks separately:

```bash
UV_CACHE_DIR=/private/tmp/aisle-plan-uv-cache uv run --no-sync pytest -q tests/unit/test_skill_data_audit.py
```

These fixtures write small Arrow streams with explicitly synthetic success and
failure labels. They prove software behavior, not physical task outcomes.
The characterization tests deliberately document current defects; P1 should
replace those expectations with the corrected behavior and retain the inputs
as regression fixtures. No existing spec acceptance test was weakened.

## Retained recording search

The audit searched all `episodes.jsonl` descendants under local `runs/` and
`analysis/`, including ignored runs. It inspected their manifest and captured
graphs for the skill name. It found 19 run episode files and 34 analysis
episode files, with zero skill references in those associated files. A
filename search for `transfer`, `b5176c`, `a3` and archives found the historical
A3 summaries but no corresponding raw skill run in these roots.

The historical root `/Users/yonghe/src/aisle` does not exist here. In particular,
the recorded path `runs/a3/worktree_F/runs/review-reeval-ik-transfer-v2` beneath
that root cannot be opened. This is a bounded local search, not a claim that no
copy exists on a collaborator's machine or in remote storage.

[Review notes](../analysis/reports/agent_pr_review_notes.md) retain the 2026-08-16
8/8 skill re-evaluation account, and
[judge_corpus.jsonl](../analysis/ver-vlm/judge_corpus.jsonl) retains its episode
references. Neither provides the camera/command stream required for this audit.
The skill's motivating collision, `20260814-235921-b5176c`, was a stock-controller
failure before the routed skill existed; it would not establish a failed
execution of `ik-transfer-v2` even if recovered.

## Real success and failure pair

The substitute run is `runs/t1-genesis-l2-spot`, T1/L2, Franka, Genesis/Metal.
It exercises the shared recorder/exporter/loader, not the routed skill. Its
manifest names source `baef84c2f919bab32934ef5d4e61426f80a5b423`; current audit
source identity is recorded separately. The captured authored graph matches
the manifest's graph hash. The manifest reports `env_attested: true`, but also
`env_baseline: local` and `post_run_audit: null`; this is development evidence,
not a trusted post-run-attested benchmark result.

| Quantity | Episode 0 | Episode 1 |
|---|---:|---:|
| Retained verifier outcome | success | fail, never_grasped |
| Seed | 0 | 1 |
| Reset-defined window, seconds | [0, 20.91) | [20.91, 80.96) |
| Guard-forwarded joint commands | 2,084 | 1 |
| Commands without an earlier same-episode frame | 0 | 1 |
| Commands without an earlier gripper command | 411 | 1 |
| Samples loaded at stride 1 | 1,673 | excluded by success-only export |

The outcome labels are retained verifier results, not independently judged
again. The failed episode's single command is exactly at its reset boundary;
without a command-to-goal identifier it cannot be confidently attributed to a
new skill invocation rather than residual/reset-boundary activity.

Manual timestamp checks, also encoded in the audit output:

- Episode 0 first command: 70,000,000 ns; frame index 1 has the same timestamp.
  State row 6 is available; no earlier gripper-command row exists.
- Episode 0 last command: 20,900,000,000 ns; frame index 626 has that timestamp.
- Episode 1 boundary command: 20,910,000,000 ns; export selects frame 626 from
  20,900,000,000 ns, which predates that episode's reset by 10 ms.

The complete MP4 decodes to 10,205 frames, matching 10,205 overhead camera
rows, with no duplicate camera timestamps. Frames 1 and 626 were decoded and
hashed. There are no raw capture NPZ files in this run. These checks support
the ordinal mapping assumption for this recording, but cannot prove original
pixel identity or causal event order within a shared simulation timestamp.

## Findings along the execution to training path

| Area | Evidence and implication |
|---|---|
| Video alignment | The recorder appends every successfully decoded, same-size overhead frame, while writing a metadata row even when decoding fails or size differs. It writes the MP4 at 10 fps; that is playback timing, not a 10 Hz sampling filter. Export assumes every camera row maps one-to-one to an MP4 index. Counts agree in this run, but malformed/changed-size frames can break the assumption. |
| Episode joins | `episode_window` uses reset-service boundaries and half-open time slicing. Export queries all-run frames, allowing a previous-episode image to satisfy the latest-at-or-before join. The real failed episode demonstrates this. |
| Command row identity | Export removes commands preceding the first frame but stores no original command-row index. Loader indexes unfiltered commands by the compacted export index. The synthetic fixture retains t=30 but loads the command from t=10. This defect was not triggered in the real success episode. |
| State causality | Loader uses state row 0 when no preceding state exists. The synthetic fixture pairs a t=10 action with a t=25 state. It also substitutes zeros for a null selected state; neither is evidence of the observed state. |
| Gripper and sample coverage | Loader skips commands without a preceding gripper command. The real success loses 411 of 2,084 samples at stride 1. Holding an assumed initial gripper value would need explicit evidence; neither silence nor zeros establish it. |
| Action representation | Export selects `joint_cmd_safe`, which is a guard-forwarded target, not achieved physical motion. The audited Franka commands and states are 9-wide. The SO-101-oriented loader takes five joint targets plus gripper and six state entries without checking embodiment. Thus the loader probe is diagnostic, not a valid Franka training dataset. |
| Instruction identity | Episode records lack `target_med`. Loader emits “pick the requested box…” while the actual `target_request` contains `amoxicillin`. Join the instruction to the episode/goal instead of relying on a summary field. Goal revisions require their own boundary. |
| Temporal training target | The trainer repeats one action across the model chunk horizon; this is a single-target training recipe, not a recorded future action sequence. It uses overhead imagery and zero-filled camera2/camera3, and expects a separate `stats.json` that export does not create. |
| Failure retention | Export's CLI skips failed episodes. They remain queryable in the raw run, but are absent from the training export. Recovery/reliability analysis needs a separately labeled all-outcomes view. |
| Partial traces | Trace loading keeps complete Arrow batches if a tail is truncated. The audit uses that reader and does not certify stream completeness. A dataset validator needs explicit truncation and stream-coverage reporting. |
| Metadata and identity | Trace rows retain simulation time, environment id, sequence, numeric payload and text. Arbitrary metadata such as request/goal identifiers, source version, units and camera shape is not generally persisted in columns. Some JSON payloads contain identifiers, but that does not supply them for every command. |

Primary source files are hashed in the audit artifact:
[recorder](../src/aisle/harness/trace_recorder.py),
[trace queries](../src/aisle/harness/traces.py),
[exporter](../tools/demo_export.py), and
[training loader](../tools/finetune_smolvla.py).

## Skill attribution and trust

[`ik-transfer-v2`](../skills/ik-transfer-v2/ik_transfer_v2.py) subclasses the
existing staged planner and replaces the transfer/clear stages for eligible
Franka top-down grasps. Other embodiments, front grasps, invalid plans, and
failed routed IK can retain the stock path. Presence of the skill node in a
graph therefore does not prove the routed path ran.

The reused executor emits joint/gripper commands and prints stage transitions
to stderr. `plan_done` is an overall completion signal when declared, with
`{"ok": true}` rather than a skill version, invocation identity or per-stage
physical completion proof. Read moves have their own response path. The
recorder does not turn stderr lines into simulation-stamped skill spans.
Logs can help diagnosis but cannot retrospectively establish exact aligned
stage labels. A routed-path log should also be distinguished from stock
fallback and from a successfully completed physical transfer.

[Skill registration](../src/aisle/harness/skill.py) evaluates the staged
candidate and enforces a registry floor. Its evalcard is population-level
evidence, not proof about an individual trace's executed code. P1 needs the
captured graph, skill implementation hash, embodiment and selected execution
branch bound to each data source. Missing fields must remain unknown.

## Smallest P1 scope and acceptance tests

First PR: repair the offline exporter/loader contract with an additive format
version. Keep the scope to causal, explicitly identified records; reject data
that cannot meet the contract. Existing consumers should reject unsupported
versions rather than silently applying old index semantics.

1. Preserve producer plus original command row/sequence and timestamp. Join
   frame, state and gripper within the same episode/environment using explicit
   missing-data results. Test the synthetic dropped-command and future-state
   examples, reset-boundary reuse, missing streams, and duplicate/nonmonotonic
   timestamps. Never repair causality with fabricated zeros.
2. Recover instruction from the goal/request evidence and bind goal revision.
   Test the real `amoxicillin` request and a synthetic mid-episode correction.
   Refuse ambiguity rather than insert a generic instruction.
3. Declare embodiment, joint names/order, units, frame and action semantics.
   Test that 9-wide Franka data is refused by the SO-101-specific trainer.
   Keep proposed targets, forwarded targets and measured state distinct.
4. Export all outcomes with an explicit success-only training view. Test that
   a failure remains available for analysis without entering imitation labels
   automatically. Store source hashes and a split assignment before windowing.
5. Validate pixel mapping and completeness. Where existing recordings lack a
   capture map, report ordinal mapping as unverified. Test failed decode,
   resolution change, truncated Arrow/video, missing final reset and duplicate
   source timestamps. Do not certify a file from matching counts alone.

The next, separately reviewed capture change should record an explicit
camera-row-to-frame map (or lossless frame identity) and skill invocation/stage
events with simulation timestamps, version, parameters, branch and outcome
evidence. New runtime topics require the appropriate contract review; P0
does not introduce them. Keep the scorer independent and privileged labels
out of deployable policy inputs. True future-action windows and synchronized
multi-camera training are later extensions, not implicit properties of P1.

## Exact recapture procedure if the skill recordings cannot be recovered

Prefer recovery from the original campaign host first. Request the complete
`review-reeval-ik-transfer-v2` directory, including manifest, authored and
instrumented graphs, Arrow streams, overhead video, raw captures if present,
logs, episode results and the source revision. Recompute hashes and audit it
before assigning any admissibility status. Recovery of the original baseline
collision should be labeled stock-controller evidence.

If recovery fails, capture a bounded development dataset on the simulation
machine in an eligible work slot under ADR-71. Use the currently qualified
engine explicitly; the commands below pin Genesis to match this audit's
scope. A future Nexus run is a separate dataset with its own environment
qualification, never a replacement for the historical Genesis evidence.

From a clean, pinned checkout with the locked sim environment and trusted
baseline available:

```bash
uv sync --locked --extra sim
uv run harness validate graphs/eval_ik_transfer_v2.yaml
uv run harness report log --idea "P0 bounded ik-transfer-v2 data capture" --expect "usable skill execution records"
AISLE_FRAME_CAPTURE_PERIOD_S=0.1 uv run harness rollout \
  --graph graphs/eval_ik_transfer_v2.yaml --tier T1 --perception L1 \
  --embodiment franka --episodes 8 --seeds 30..37 --reset teleport \
  --verifier oracle --sim-engine genesis --run-id aw0-p0-ik-transfer-v2-30-37
```

Do not bypass failed environment/attestation gates or change frozen geometry
to obtain a desired result. Record hashes of source and graph before and after
capture; retain full run contents, including logs and failed episodes, outside
ephemeral worktrees. The existing raw-capture option is sparse and requires
matching overhead RGB/depth; it does not supply the proposed per-command
capture map or stage spans.

Select the first recorded success and first recorded failure by episode order.
If all eight pass, permit one predeclared extension, seeds 0..15, with a new
run id `aw0-p0-ik-transfer-v2-0-15` and otherwise identical arguments. Stop
after those 24 episodes regardless of outcomes. If no failure occurs, report
that absence; a synthetic failure is not a substitute. If no success occurs,
report that instead of tuning the skill during this data audit.

Audit each selected episode with `tools/audit_skill_data.py --run <directory>
--episodes <indices>`. Keep development-only labels and distinguish old
recordings, new captures, and synthetic fixtures. Exact skill-stage alignment
still requires the capture change above; running the existing graph alone
does not cure that evidence gap. These commands are instructions for future
capture and were not executed in P0.

## Validation and completion boundary

P0 supplies a source audit, a bounded recording search, a real shared-pipeline
success/failure check, portable synthetic reproductions, artifact hashes,
the skill-specific evidence gap, and the P1 acceptance/capture procedures.
It does not certify AW-0 training readiness, replay the absent skill runs, or
measure policy improvement. The new tool/tests are Class A; runtime, specs,
frozen paths and unrelated worktree files are unchanged.

Validation completed on 2026-11-11:

- Diff review covered input immutability, diagnostic versus certification
  semantics, source attribution, and the synthetic characterization tests.
- `ruff format --check .` and `ruff check .`: passed.
- Focused P0 tests: 8 passed. Traceability check: passed, no uncovered IDs.
- `UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked pytest -m unit`:
  **4,588 passed, 419 deselected**, 2,404.37 seconds, exit 0.
- Audit rerun: byte-for-byte identical JSON. All referenced local documents,
  current source hashes and retained input hashes verified.

The full unit gate ran outside the Codex sandbox because existing process
cleanup tests require `/bin/ps`. An initial sandboxed run was interrupted after
five process-inspection failures (767 tests passed); the isolated cancellation
suite then passed 6/6 outside the sandbox. A second run in the simulation
environment was interrupted after 772 passes because of expensive runtime-tree
hashing. The final complete run used the repository's documented
[lean unit environment](troubleshooting.md#local-unit-tests-fail-with-infrastructure_exclusion-or-expired-deadlines),
preserving the simulation environment. No test was disabled or weakened.
