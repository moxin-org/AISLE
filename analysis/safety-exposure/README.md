# Safety exposure ledger (SPEC 470, issue #351)

Instrument pilot for the exposure ledger and its analyzer. Evidence kind:
`simulation`. Nothing here is a physical-safety result, a prevention claim,
or a confirmatory campaign; it is the recorded exposure surface of one
expert graph on eight development seeds, regenerated from retained traces.

## Layers, stated separately (SFE-1, SFE-14)

- Declared topology: every declared command path traverses the guard
  (VAL-5, elsewhere).
- Gateway kinematic enforcement: `proposals.by_decision` counts pass, clamp,
  refuse, hold on received requests. A clamp alters a kinematically illegal
  proposal; it knows nothing about medicine identity.
- Observed kinematic outcome: `collisions` are a pose-displacement proxy
  from the verifier's own geometry; contact instrumentation is `unmeasured`.
- Verifier semantic detection: `wrong_object_events` are deliveries whose
  observed identity differs from the assigned target. They are detected
  outcomes after the fact, never prevented commands.

## Records

`records/sfe-exposure-pilot-01/`: `graphs/expert_t1.yaml`, tier T1, rung L1,
Franka, teleport reset, oracle verifier, seeds 0..7. `ledger.json.gz` is the
SFE-2 ledger (every proposal, decision, receipt, attempt, delivery,
collision-proxy row with hashes); `report.json` is the SFE-6 analyzer
output; `manifest.json` and `episodes.jsonl` are the run's own records. Raw
Arrow traces (23 MB) are retained outside the worktree under
`~/aisle-private/raw/sfe-exposure-pilot-01/`; their SHA-256 values are in
the ledger's `trace_hashes`.

## What the pilot measured

| quantity | value | unit |
|---|---|---|
| episodes randomized / started / included | 8 / 8 / 8 | episode |
| manipulation attempts (incomplete) | 8 (0) | attempt |
| deliveries | 8 | verifier-observed tray entry |
| wrong-object events | 0 | detected outcome |
| received / valid / malformed proposals | 14694 / 14694 / 0 | proposal |
| interventions (clamp / refuse / hold) | 0 / 0 / 0 | proposal |
| workspace: proposed out / intervened / received out / observed | 0 / 0 / 0 / 0 | proposal or event |
| collision proxy inside the scored window | 0 | episode |
| displacement first seen after the verdict | 3 | row, descriptive |
| controller class of every proposal | classical (by content hash) | SFE-7 |

Zero-event bounds (SFE-8, exact one-sided 95%, unit = exposed session with
at least one event): wrong-object 0/1, upper bound 0.95; collision proxy 0/1,
upper bound 0.95. Episode (0/8), delivery (0/8) and attempt (0/8)
denominators are reported descriptively beside them. One session bounds
nothing; the ledger's purpose is the denominator, not a small number.

Until issue #508 the analyzer pooled episodes into the primary bound (0/8,
0.312). Episodes inside one session are nested, so SFE-8's session unit
governs. The earlier outputs are retained as superseded records:
`report-v1-episode-unit.json` here and `pooled-report-v1-episode-unit.json`
beside the pooled report.

## Regenerate

```bash
uv run harness exposure ledger --run runs/sfe-exposure-pilot-01 \
  --campaign-id sfe-exposure-pilot-01 \
  --source-map analysis/safety-exposure/source-map.json \
  --output analysis/safety-exposure/records/sfe-exposure-pilot-01/ledger.json.gz
uv run harness exposure analyze \
  --ledger analysis/safety-exposure/records/sfe-exposure-pilot-01/ledger.json.gz \
  --output analysis/safety-exposure/records/sfe-exposure-pilot-01/report.json
```

`source-map.json` classifies command producers by content hash (SFE-7). A
producer whose hash is not listed is `unknown` and stays visible.

## Second session and pooled report

`records/sfe-exposure-pilot-02-t2/`: `graphs/expert_t2.yaml`, tier T2, rung
L2 (CPU OWLv2 perception), Franka, seeds 0..7. Outcome 0/8 (7
never_grasped, 1 collision), consistent with the committed T2 expert
baseline. 10144 proposals, 23 clamps, 0 manipulation attempts (the gripper
never closed), 0 deliveries, 0 wrong-object events. Raw traces (106 MB)
under `~/aisle-private/raw/sfe-exposure-pilot-02-t2/`.

`records/pooled-report.json` pools both sessions (same evidence kind,
distinct session ids): 2 exposed sessions holding 16 included episodes.
Wrong-object: 0 of 2 sessions, exact one-sided 95% upper bound 0.776 (0/16
episodes, descriptive). Collision proxy inside the scored window: 1 of 2
sessions (1/16 episodes, the T2 collision failure). Classical controller intervention
rate 0.0009 over 24838 valid proposals.

```bash
uv run harness exposure analyze \
  --ledger analysis/safety-exposure/records/sfe-exposure-pilot-01/ledger.json.gz \
  --ledger analysis/safety-exposure/records/sfe-exposure-pilot-02-t2/ledger.json.gz \
  --output analysis/safety-exposure/records/pooled-report.json
```

## Fixed-proposal guard ablation (SFE-9 to SFE-12)

`ablation/sfe-held-command-ablation-v2/` holds the frozen corpus and the
result. Evidence kind: `simulation_fake_driver`. No physics, no hardware:
each pair replays byte-identical proposals with identical contract
timestamps through `guard_on` (production clamp) and
`guard_observe_only` (identical would-have decision logged, raw proposal
forwarded) into a fake driver, and a frozen violation instrument scores
what the driver received. The observe-only arm exists only inside this
evaluator (SFE-10). Registration:
`analysis/freeze/sfe-held-command-ablation-v2/` (pending CON-14 approval
of SPEC 470 itself).

| quantity | value |
|---|---|
| pairs (six families x 8) / included / excluded | 48 / 47 / 1 (containment activated, `joint_position_limit-03`) |
| legal negative controls altered by guard_on | 0 of 8 (blocker did not fire) |
| at-risk traces with any driver-received violation, guard_on | 0 / 39, exact 95% upper 0.090 |
| at-risk traces with any driver-received violation, observe-only | 32 / 39, exact 95% interval 0.665 to 0.925 |
| risk difference guard_on minus observe-only | -0.82 |
| paired difference in violation count per trace | mean -1.87, seeded bootstrap 95% interval -2.21 to -1.51 (unit: trace) |
| observe-only out-of-envelope duration | 0.03 s over the corpus; guard_on 0.0 s |
| watchdog-silence pairs held by guard_on | 8 of 8 |
| collisions | unmeasured (fake driver, no contact instrument) |

Strata (guard_on any / observe-only any): joint position 0/7 vs 7/7,
joint velocity 0/8 vs 8/8, workspace 0/8 vs 8/8, gripper 0/8 vs 8/8,
watchdog silence 0/8 vs 1/8, legal 0/8 vs 0/8. The watchdog family's
primary endpoint is near null by design: its held proposal is legal; the
hold itself is the secondary endpoint.

Wording this licenses (SFE-14): measured gateway interventions alter
kinematically illegal proposals under the tested limits on a fake driver.
It says nothing about medicine identity, semantic prevention, or physical
outcomes.

```bash
uv run harness exposure corpus --embodiment franka --seed 470001 \
  --output analysis/safety-exposure/ablation/sfe-held-command-ablation-v2/corpus.json
uv run harness exposure ablate \
  --corpus analysis/safety-exposure/ablation/sfe-held-command-ablation-v2/corpus.json \
  --analysis-seed 470001 \
  --output analysis/safety-exposure/ablation/sfe-held-command-ablation-v2/result.json
```

## Not done here

The SFE-14 occurrence audit of claim wording is not yet mechanised.
Hardware ledgers are absent (`hardware_pending`, SFE-15). Emergency
containment on a real driver is not modelled beyond the envelope check.

## Claim wording (SFE-14)

Three separate statements, each with its own evidence, and nothing broader:

1. Validated declared paths traverse the guard: every command edge into
   the bridge routes through budget-guard (VAL-5 at the gate; ledger source map).
2. Measured gateway interventions alter kinematically illegal proposals
   under the tested boundary: the held-command ablation counts clamps and
   refusals of limit- and velocity-violating proposals, nothing semantic.
3. The verifier counts observed semantic outcomes: wrong-object deliveries
   are counted by the held-out verifier after the fact; detection is not
   prevention, and a zero count is an observation, not an impossibility.

The guard does not know identity; nothing here holds `by construction`.
`uv run harness exposure wording --check` rejects broader wording on the
claim-bearing surfaces.
