# P3 isolated task-worker interface

`src/aisle/harness/p3_task_boundary.py` gives a generated worker public T4
dialogue and fixed-library request capabilities. `tools/p3_task_mock.py`
exercises that interface in fresh, verified macOS-confined processes for both
`history` and `trace`. It starts no simulator or provider and reports zero
measured sessions. The qualified L1 task and [private draft acceptance
arithmetic](agentic-wam-p3-acceptance-feasibility.md) remain separate evidence.

## Task authority

The controller binds the worker source hash, a fresh worker-incarnation ID,
the fixed medicine catalog, motion-request and call limits, evidence bundle and
arm. Authored code gets framed JSON calls, never controller Python objects,
repository access, scorer data, direct actuator outputs or reset authority.

| Operation | Arguments | Effect |
| --- | --- | --- |
| `next` | `{}` | Consume the next projected public event; `event: null` means input ended |
| `act` | `revision`, `kind`, `med` | Request `confirm`, `deliver` or `return` under the current revision |
| Existing evidence calls | Unchanged | Raw evidence in both arms; derived `history` only in the history arm |

`confirm` emits a `robot_msg`. After a matching public confirmation,
`deliver` emits `target_request` and `return` emits `return_request`.
The controller supplies goal/request metadata; the worker cannot choose
transport endpoints or override these identities. A medicine choice can still
be wrong: the boundary does not replace it with the scorer's answer.

Only one fixed-library motion can be outstanding. A recovery request may
arrive while the previous motion is finishing; confirmation of the new task
does not release the old motion. Only its matching `plan_done` with the fixed
executor's `{"ok": true}` payload does. The public event is `motion_returned`
with `physical_completion: "unknown"`. Duplicate, mismatched, pre-reset and
obsolete traffic cannot release newer motion. Reset does not replenish the
motion-request allowance. A transport or evidence failure closes the task service
and retains its reason.

The admitted input vocabulary is `reset_done`, `human_msg`, `plan_done`,
`violation` and `tick`. Public fields are projected explicitly. Oracle state,
private episode goals/results and client episode metadata are refused. The
transport must belong to one admitted producer incarnation; restart/relaunch
requires a new controller binding and source audit.

## Run the local mock

Use a locked unit environment, leaving the simulation environment intact:

```sh
UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked python tools/p3_task_mock.py \
  --bundle /path/to/qualified/policy-evidence \
  --out /private/tmp/aisle-p3-task-worker-check
```

The output path must be new and separate from the evidence tree. Each arm's
participant reads the same raw evidence, probes its history grant and writes
a distinct worker program. The controller captures those bytes after the
participant is reaped and runs them in a fresh workspace. The generated
worker confirms an initial request, delivers, receives a recovery request
before the first motion finishes, waits for the matching completion, returns
and redelivers. Stale and duplicate completions are included in the synthetic
transport. An attempted direct actuator request is refused. Public task
responses and raw evidence responses must match across arms; only the history
grant differs. Each process has a 120-second supervision limit; the task
service admits at most 64 calls and 16 fixed-library motion requests in this mock.

The controller retains source/runtime/capability receipts, exact RPC frames,
stderr, task inputs, proposals, dispatches and responses. Failed and unstarted
slots remain in the report. A child exiting successfully before consuming the
task input does not pass the boundary check. These are four unscored processes,
not four experimental sessions or evidence of successful robot manipulation.

## Remaining live integration

The transport in this executable check is synthetic. Its report therefore has
`guard_traversal_verified: false` and `physical_score: null`. The production
adapter still needs reviewed Dora graph routing through the fixed skills and
guard, outer deadlines for host iteration, live execution capture, candidate
identity across the eight private acceptance chains, and session resource and
provenance admission. The worker process timeout alone cannot interrupt a
blocking trusted host callback.

The [design decision](decisions/ADR-p3-public-task-worker.md) records the
public-lifecycle interpretation. Tests exercise CON-5/8, TC-6/7/8, VAL-2/6,
HAR-4/6 and the relevant TRT-5/6/7/9/10 boundaries. This change neither freezes
the measured action surface nor authorizes another physical batch, model call,
registration or treatment-effect claim.
