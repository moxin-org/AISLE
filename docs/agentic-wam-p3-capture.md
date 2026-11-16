# P3 capture repair: goal chains

This is the first implementation repair toward P3 completion. **The measured
P3 pilot is still unrun.** It addresses the recovery-record blocker in the
[readiness report](agentic-wam-p3-readiness.md), not the remaining skill
instrumentation, policy isolation, calibration, or matched-session gates.

The existing T4 increment-two client emits an initial delivery goal and a
return/redelivery goal within one physical reset episode (ADR-32). Each has
its own verdict. The recovery goal's legacy `reset_sim_ns` is the previous
verdict's simulation timestamp; it does not establish another physical reset.
P1 intentionally rejects this as a duplicate episode identity. That rejection
and P1's existing format remain unchanged.

## Retained identities and measurement companion

The recorder now writes `traces/identities.jsonl` for low-rate reset, goal,
result, request and plan-completion topics. Each versioned record retains the
observed `goal_id` and `request_id` metadata, the producer endpoint, its original
Arrow row ordinal, and a SHA-256 hash of that row's five stored fields. Missing
IDs remain missing. The Arrow schema is unchanged, so existing trace readers
continue to read the same streams. Identity rows flush immediately; Arrow
closure and row-hash checks are still required when exporting a closed bundle.
This is measurement capture, with no new runtime topic or control edge.

`tools/goal_chains.py` creates an `aisle_goal_chains` version-1 companion bundle.
It snapshots the source manifest, episode records, identity sidecar and selected
reset/goal/verdict streams, then groups distinct goals inside actual reset
windows. Both source verdicts remain intact. Each goal retains its original
legacy goal-start field separately from `physical_reset_ns`, source references,
the recorded verdict, and the episode record. A final open reset window is
represented explicitly, including an empty one.

The exporter refuses missing or mismatched goal identities, duplicate goal
outcomes, source-row hash mismatches, a goal outside its physical episode,
conflicting seeds, recovery without a preceding verdict, and recovery records
without their explicit marker. An initial goal must cite the physical reset;
a recovery must cite the preceding verdict and declare `expects_return`.
Source verdicts and episode JSONL records must agree. A goal without an outcome
remains unscored with `outcome: null`; an observed verdict without an episode
record remains explicitly incomplete. Neither is converted to a failure score.

Native realistic-verifier verdicts omit `seed`; the captured goal and rollout
record still have to bind the same seed. The missing verdict field is preserved
as missing. Select the verdict producer that drove `episodes.jsonl`. When a
realistic loop also retains an independent oracle scorer, that scorer remains
in the full raw run; this companion does not reconcile the two or substitute
the realistic verdict for ground truth.

The reader verifies all archived bytes and reconstructs the chains from the
source records. Rehashing a changed derived result does not make it valid.
Hashes establish internal consistency, not independent capture attestation.
Old runs without captured goal identities cannot be upgraded by guessing IDs
from row order or goal-name suffixes; they need recapture for this format.

**The entire companion is measurement-only and privileged.** It contains
scorer goals and verdicts and must remain outside both P3 participant views.
`policy_visible: false` labels this requirement; it is not an access-control
mechanism. The eventual campaign launcher must enforce the separation. This
companion establishes neither skill execution nor controller success, and is
not a new training dataset.

```sh
UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked python tools/goal_chains.py \
  --run runs/RECAPTURED_RUN --out /tmp/p3-goal-chains
```

Use `--endpoints mapping.json` when several producers publish a required topic;
its entries map topic names to `producer__topic.arrow` basenames. The exporter
does not silently choose between oracle and realistic verdict streams. Its
output directory must be new and outside the source run. The CLI emits one JSON
result and exits zero iff `ok` is true. A valid export may contain incomplete
goals; inspect coverage and per-goal completeness before admitting any trial.

## Validation scope

Unit tests exercise the recorder's actual event loop using injected events and
portable Arrow files, including complete and incomplete chains, corrupted
identity evidence, source/derived disagreement, malformed verdicts, and unchanged
P1 refusal. The dora acceptance test runs the actual recorder over local dora
transport with a synthetic event driver. It verifies that both goal IDs and
verdicts survive transport and export. It runs no physics, model or robot, and
is not calibration evidence.

Requirements affected: CON-5/8, HAR-4/6, TC-6/7/8; ADR-32 supplies the existing
two-goal recovery semantics. No spec, frozen path, scoring rule or campaign
assignment changes. The remaining P3 work is still real versioned invocation
capture, policy-visible recovery and scorer isolation, development calibration,
the matched history/trace adapter, and the funded measured pilot.
