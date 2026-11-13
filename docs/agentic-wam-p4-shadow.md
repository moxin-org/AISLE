# P4 offline shadow-selector prototype

Status: preparatory P4 implementation, 2026-11-13, based on merged P3 readiness
commit `845e31f`. **Full P4 is incomplete.** The owner requested the next goal;
this deliverable prepares the interface while the candidate-policy prerequisite
remains unmet. P3's measured memory pilot remains unrun.

[`tools/shadow_selector.py`](../tools/shadow_selector.py) replays explicitly
synthetic decisions without loading a model, starting nodes, publishing motion,
or reading held-out verdicts. It is a Class A tool, not a runtime controller.
It changes no specs, frozen paths, graph wiring, benchmark entry, or engine
priority. Existing constraints: CON-5/8, TC-1/6/7/8 and
[ADR-38](decisions/ADR-38.md). BG-1 remains a runtime integration gate.

## Candidate evidence and entry decision

The registered `ik-transfer-v2` node has a typed `grasp_pose` input and direct
motion outputs; its [manifest](../registry/manifests/ik-transfer-v2.yaml) has
`eval: null`. Earlier successful composed graphs are not an invocation-level
reliability estimate. P2 found no versioned skill invocation spans in the
retained P1 recording; see the [P2 report](agentic-wam-p2-ledger.md).

SmolVLA's registered zero-shot evaluation is 0/4. The later fine-tuned adapter
also scored **0/8 in lockstep**, with 3 never-grasped, 1 dropped and 4 collision
outcomes: [M1 findings](../analysis/m1/m1_zeroshot_findings.md). These were
unattested development measurements, not a current qualified benchmark.
Lockstep removed inference staleness as the explanation for that adapter's
failure; it did not establish live performance. Neither this prototype nor the
M1 record supplies a useful learned fallback for the intended P4 task.

The prototype therefore permits only synthetic reliability inputs. It does not
promote either adapter, launch collection, or infer alternate-controller success.
The `unqualified` fixture deliberately uses unknown skill reliability and a
zero-of-eight learned count; its decision is to hold. This is a synthetic check
of the refusal rule, not a reconstructed M1 trial or a real policy evaluation.

## Bounded input and fixed rule

The version-1 JSON input contains `format: aisle_shadow_events`,
`provenance: synthetic`, a candidate list, and arrival-ordered `records`.
Only environment 0 is supported, with at most 10,000 records and a 4 MiB CLI
input limit. Each event has a unique ID and a strictly increasing simulation
arrival timestamp. Exact duplicate delivery is idempotent; conflicting IDs
and ambiguous timestamps invalidate the input. Asynchronous replies have new
arrival times and explicitly echo their original request, episode, goal, and
adapter identity. Observation capture time is separately supplied as `obs_ns`.

Two adapters are allowlisted: `ik-transfer-v2` and `vla-policy-smolvla`.
Their `version` hashes bind the registered manifest and direct adapter source.
These are **interface pins**, not transitive implementation, model-weight,
training-data or environment attestations. Changing either pinned file makes
old fixtures fail validation until intentionally refreshed.

Skill parameters are exactly `grasp_pose`: seven finite numbers representing
base-frame meters and an XYZW unit quaternion. Learned request parameters are
empty; the synthetic reply contains 1–128 actions, each with seven finite arm
joint targets in radians and one gripper fraction in [0, 1]. This intermediate
Franka-shaped representation is not a TC-5 Arrow message or a SO-101 adapter.
There is no generated Python, dynamic import, path supplied by the fixture,
nested subgraph execution, or arbitrary skill invocation.

The rule first requires a current, fresh observation. It then filters every
candidate by declared applicability, stage, remaining attempts, remaining
budget, typed parameters, and reliability. A candidate's entire timeout must
fit the reported remaining budget. Unknown reliability is ineligible; a known
synthetic success fraction must be at least 1/2. Eligible skills have priority
over learned chunks; IDs provide a stable tie break. The fraction is an
**uncalibrated fixture rule**, not a recommended deployment threshold or a
confidence bound. Every evaluated candidate records its rejection or priority
reason. Global holds also have explicit reasons.

Attempts are consumed at selection, including cancelled or timed-out requests,
and reset only with a new goal or episode. Remaining budget, stage and
applicability are caller-supplied context, not quantities this tool measures.
There is no cost allocator, inference scheduler, or automatic retry loop.

## Ownership, cancellation and observations

Selection reserves a single request. Another selection holds while that request
is pending. A `boundary` proposes one skill invocation, or pops one learned
action from the existing ADR-38 `ChunkQueue`. Active skills retain the reservation
until a matching return, interruption, timeout, reset or goal change. A learned
request releases it after its last action. Explicit interruption flushes the old
chunk before a replacement selection; an old reply cannot refill the new queue.

The `owner` field is the reservation **after** each event. The final action may
therefore have an owner of null; its proposal still names the request that
owned that boundary. All proposals say `actuation_authority: false`.
`required_motion_route: budget_guard_inputs` describes a future routing
requirement, not an observed route. Skill proposals would enter the skill's
typed input; the skill's resulting commands would need the guard. The report
always says `guard_traversal: unverified` and `task_success: null`.

Reset and goal events reuse P2's strict evidence validation and invalidate
pending work. The first reset must be episode 0 at time 0; subsequent episodes
advance one at a time. A repeated goal identity cannot replenish attempts.
Unknown/malformed intent (`instruction: null`) invalidates old work without
authorizing a selection. Identity-mismatched replies are rejected. Only one
learned reply can populate a request's queue.

Observation age is checked at selection, reply acceptance, and every learned
action boundary against ADR-38's 500 ms simulation-time limit. A newer
observation does not refresh an old chunk's originating timestamp. A started
skill's return, interruption or timeout requires an observation captured
strictly after termination before selecting a fallback. Reset and goal changes
likewise require a new observation. A skill return is `returned_unconfirmed`;
it is never a local completion or task-delivery verdict.

Deadlines are inclusive. Every new event advances the offline watchdog, even
an identity-mismatched reply. At expiry, the queue and reservation clear and
the event produces `timeout_hold`; it cannot also select or propose motion.
A reset/goal on the deadline still establishes its new intent boundary and
records the timeout. In this tool **hold means no proposal**. It sends no
physical hold/stop command. With no subsequent event, time cannot advance;
an incomplete trace retains its pending request. Runtime integration needs
independent wall-clock watchdogs and an acknowledged stop/hold protocol.

## Reproducible evidence

Inputs live in [`tests/fixtures/shadow_selector`](../tests/fixtures/shadow_selector),
and reports in [`analysis/aw0_p4`](../analysis/aw0_p4). Tests compare the reports
exactly against replay. Hashes bind each source document and each event.

| Fixture | Events | Check |
|---|---:|---|
| `lifecycle` | 19 | Skill preference, interrupted skill, blocked fallback until fresh observation, learned action, goal/reset invalidation, late replies and timeout |
| `unqualified` | 4 | Unknown skill and zero-success learned candidate both refused |
| `stale_chunk` | 7 | A first action at the 500 ms limit is proposed; the next stale element is discarded |

From the repository root, preserving any installed simulation extras:

```sh
UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked python tools/shadow_selector.py \
  --fixture tests/fixtures/shadow_selector/lifecycle.json
UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked pytest -q tests/unit/test_shadow_selector.py
```

The CLI returns one JSON object, exits zero iff `ok` is true, and supports
`--out NEW_PATH` with exclusive creation. Invalid inputs return `ok: false`.
`ok: true` means valid replay, including valid refusals; it does not mean a task
succeeded. `--help` follows standard argparse help behavior.

## Remaining P4 work and acceptance

| P4 acceptance | This deliverable | Still required |
|---|---|---|
| Single owner | Synthetic reservation and boundary tests | Live arbitration at the real motion boundary, including cancellation acknowledgement |
| Guard traversal | Explicitly unverified | Wire every motion edge through the guard; validator and adversarial graph tests |
| Reset/goal/reply fencing | Synthetic identity and queue tests | Actual node lifecycle, asynchronous reply and reset-service capture tests |
| Timeout hold/stop | Offline expiry with no proposal | Wall-clock watchdog, physical hold/stop semantics and acknowledgement |
| Reacquire after partial skill | Capture-time barrier tests | Fresh synchronized sensor data after acknowledged termination |
| Selection/rejection log | Hash-bound fixture timeline | Runtime versioned invocation and observation provenance |
| Useful candidates | Unmet | Task-specific candidate evaluation and pinned policy artifacts |

Next, repair the real invocation/capture gaps identified by
[P3 readiness](agentic-wam-p3-readiness.md), choose and qualify useful candidates
on a compatible task/engine, and calibrate the rule on development evidence.
Then implement a separately reviewed Class B shadow adapter with validated
graph topology and actual sensor/stop acknowledgements, retaining no actuation
authority until the guarded integration is tested. Direct adapter pins must be
extended to bind model weights, dependencies, inputs, engine and evaluation
conditions before measured use. P5 additionally requires independent held-out
competence for both branches and its own matched campaign entry and budget.

The offline prototype is ready for review when its Class A gates pass. It does
not complete P3's pilot, full P4, or authorize P5.
