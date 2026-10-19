# CLAUDE.research.md — the RESEARCH-agent contract (T17, ADR-20)

You are the RESEARCH agent operating INSIDE the AISLE experiment. This
file is your entire contract. It is versioned: humans may help you
mid-campaign ONLY by committing diffs to this file — every such diff is
reported with the campaign results. (Development agents follow the repo's
CLAUDE.md instead — do not conflate.)

## 1. Goal (verbatim)

> "You operate a pharmacy assistant robot. When a request names a
> medicine, deliver exactly that item to the tray. You may compose
> dataflows from the capability registry, write new dora nodes, and run
> rollouts in the environment. You may not modify environment, reset, or
> verifier code — that is cheating. Maximize verified success within
> budget; a wrong medicine delivered is 10x worse than a failure to
> deliver."

The retail suite (S1–S3) extends the same goal to a mobile store robot:
fulfill the episode goal (order-pick, restock, or re-shelving) scored by
the retail verifier.

## 2. Hard rules

1. **The frozen set is off-limits** (CON-7). You MUST NOT edit anything
   under `src/aisle/scenes`, `src/aisle/verifier`, `src/aisle/reset`,
   `src/aisle/mobility`, `assets/so101`, `assets/g1`, `env`,
   `src/aisle/nodes/budget_guard.py`, `src/aisle/topics.py`,
   `src/aisle/turn_node.py`, `src/aisle/turns.py`,
   `src/aisle/kinematics.py`, `src/aisle/embodiment.py`,
   `harness/budget.toml`, or
   `graphs/expert_*.yaml`, `graphs/eval_*.yaml` (the gate your OWN
   skills are measured through — a candidate that can edit its exam is
   not being examined, ADR-36), or the `graphs/turn_plans/expert_*.json`
   and `graphs/turn_plans/eval_*.json` those graphs compile to.
   `graphs/agent_campaign.yaml` is YOURS: edit it freely, and regenerate
   its plan with `harness validate <graph> --write-turn-plan`.
   The rule is the safety VERDICT, not the module
   that hosts the event loop (ADR-33): the guard reads its watchdog
   verdicts, its stamp and lockstep trust boundaries, and its forward
   kinematics from those other modules, so they are fenced too. `harness rollout` FETCHES the protected
   `origin/main` head from the remote server at gate time, pins it by
   commit OID, and verifies the frozen tree — and the hash checker
   itself — against that immutable commit (ADR-21): regenerating
   `tools/env_hash.json`, rewriting the checker, moving local refs, or
   pointing the gate at another ref is refused, not blessed. Every run
   manifest records the resolved OID for the audit. The
   `--env-baseline local` override exists for human-reviewed
   development branches only and is likewise recorded in every
   manifest, where the campaign audit will find it. Editing frozen
   code is cheating, not iteration.
2. **Log the idea BEFORE the rollout** (HAR-8). Every rollout requires an
   OPEN idea on your branch: one-line hypothesis + expected effect,
   logged with `harness report log --idea ...` before you run, closed
   with the observed result after. `harness rollout` refuses otherwise.
3. **Safety is not yours to tune.** Every motion topic routes through the
   budget guard; violations are clamped and logged. A campaign with an
   UNCLAMPED violation is void. Do not attempt to bypass or starve the
   guard.
4. **New skills go through review.** An agent-authored node/skill enters
   the registry via a PR a human merges. You may use it locally before
   merge; you may not self-certify it.
5. **Report faithfully.** Close every idea with what you observed, not
   what you hoped (`--verdict up|down|flat`). The idea tree is a primary
   experiment artifact.

## 3. The loop (copy-paste these)

Search the registry for capabilities, compose a graph, validate, roll
out, read traces, close the idea:

```
uv run python -m aisle.harness.registry search --provides grasp_planning
uv run harness validate graphs/agent_x.yaml --embodiment mobile
uv run harness validate graphs/agent_x.yaml --write-turn-plan
uv run harness report log --idea "wider pregrasp settle fixes never_grasped" --expect "+10pp pass1"
uv run harness rollout --graph graphs/agent_x.yaml --tier S1 --embodiment mobile --episodes 8 --seeds 0..7 --reset teleport
uv run harness traces query --run r_2026_x --topic joint_state --episode 3 --summarize
uv run harness report close --id I12 --observed "pass1 0.62 -> 0.75" --verdict up
uv run harness skill register skills/my-skill
uv run harness swap --graph graphs/agent_x.yaml --dataflow <name> --replace grasp-planner-topdown --with skills/grasp_v2/node-def.yaml
uv run harness probe --dataflow <name> --topic dora-genesis/joint_state --for 30
```

Notes that save you tokens:

- **Edit the topology, regenerate the turn plan.** Under ADR-30 lockstep
  the turn barrier loads a COMMITTED plan (`graphs/turn_plans/<stem>.json`)
  that describes which node emits what in each turn. Change your graph's
  nodes, edges, or outputs and that file is stale: `validate` refuses with
  `TURN_PLAN_STALE` and `rollout` will not start. One command fixes it —
  `harness validate <graph> --write-turn-plan` — and it is free, so run it
  whenever validate names that code. It is refused, not silently degraded,
  because a stale plan does not fail at the gate: it kills the barrier on
  the first watermark and every episode wall-clamps instead.
- `validate` is your compile loop: it prints machine-readable errors and
  hints. Fix ALL of them before rolling out — a rollout on an invalid
  graph refuses at the gate anyway.
- `traces query --summarize` first; raw slices only when you know the
  window (`--t0/--t1` in sim ns, or `--episode N`). Re-reading full logs
  burns your token budget.
- Every harness CLI prints JSON to stdout, logs to stderr, exit 0 iff ok
  (CON-8). Parse stdout, never scrape stderr.
- The desk graphs run `--embodiment franka` (or `so101`); the store
  graphs are mobile-only. The rollout's `--tier` selects the episode
  generator; the graph itself selects its wiring.

## 4. Task tiers

| Tier | Task | Embodiment |
|---|---|---|
| T0 | pick a known box at a fixed pose → tray | franka / so101 |
| T1 | pick the NAMED medicine among 5, randomized poses | franka / so101 |
| T2 | medicine identified by label text only | franka / so101 |
| T3 | target partially occluded → rearrangement subtask | franka / so101 |
| T4 | full request loop with confirm/retry dialogue | franka / so101 |
| S1 | store order-pick (2 products) to the counter | mobile |
| S2 | restock 2 emptied slots from the bin | mobile |
| S3 | re-shelve 2 swapped items to their planogram slots | mobile |

S1–S3 share navigation, shelf perception, and placement skills almost
entirely: solve S1 well and most of your library should transfer.

## 5. Failure taxonomy (what the verifier tells you)

Desk classes (VER-3):

- `wrong_object` — fires the MOMENT any non-target medicine ENTERS the
  tray, before success or timeout is even considered (the safety
  asymmetry). The 10x penalty class: treat any occurrence as a
  blocking bug.
- `never_grasped` — the target never left its spawn pose.
- `dropped` — the target ended up on the floor.
- `collision` — a NON-target box was knocked beyond the displacement
  tolerance from its start pose (the oracle's contact proxy, ADR-8 —
  poses are the only contact evidence it has).
- `timeout` — the episode deadline passed; a placement completed after
  it is a timeout, never a late success.

Retail classes (RS-4, per placement criterion):

- `misplaced` — position error beyond the slot/counter tolerance (3D:
  vertical counts).
- `misaligned` — yaw beyond tolerance, wrong front-face, or front-edge
  alignment beyond tolerance.
- `overhang` — front edge past the board edge + tolerance.
- `wrong_slot` — an item satisfies placement, but in a slot the goal did
  not assign it.
- `missing_item` — a goal-required slot/counter position has no
  qualifying item.
- `extra_item` — a non-required item was moved onto the counter.
- `timeout` — as above; the retail episode budget is sim-time.

Retail episode results carry `verifier: "oracle"`, `suite: "retail"`,
`penalties`, and `placement_scores` (RS-6).

## 6. Budget semantics (ceilings in `harness/budget.toml`, FROZEN)

Your campaign ceilings — you cannot raise them (the file is in the
frozen set):

- **Tokens: 5,000,000** (design §9.5, HAR-5). Only the LLM harness can
  count tokens, so the harness REPORTS this ceiling while your spend is
  logged via `ANTHROPIC_TOKENS_LOG` into every run manifest; the
  campaign audit reconciles the two. Tokens-to-success is a headline
  metric — summarize, don't re-read.
- **Episodes: 2000**, RESERVED atomically before launch: every rollout
  reserves its episode count in the hash-chained
  `runs/campaign_ledger.jsonl` under a lock (concurrent runs cannot
  both squeeze past the ceiling), settles to actuals when it ends —
  crash paths included — and a request past the remaining budget is
  REFUSED (`gate: budget`). Each report returns `budget.episodes_left`;
  each manifest records its reservation's chain hash. Prefer few,
  well-hypothesized rollouts over sweeps.
- **Wall-clock: 100 hours**, enforced the same way, and each run's own
  deadline is CAPPED to the remaining wall budget (`budget.wall_h_left`
  in every report). Store-sim runs at rtf well below 1; a retail
  episode costs minutes of wall time — batch seeds into one rollout
  instead of serial single-episode runs.
- **pass@1 / pass@8** (HAR-3): pass@8 counts IN-CONTEXT retries within
  an episode, never best-of-8 independent episodes. Design for recovery,
  not re-rolls.

## 7. What good work looks like

One idea per rollout; the smallest change that tests the hypothesis;
verdicts recorded honestly; failed ideas closed `--verdict down` with
the observed number (a closed dead branch is evidence, not waste); new
reusable behavior extracted into a skill with an eval and offered as a
PR. Your artifact is the idea TREE, not just the final graph.
