# P3 draft acceptance feasibility

The public 12-chain L1 Genesis qualification on source
`44118c9f33620f0c070a423d6aed6ab4019a1e7e` completed with eight successful
two-goal chains and four failed chains (three collisions, one timeout).
All twelve initial goals succeeded. The retained evidence contains 24 oracle
verdicts, 32 guarded invocations and eight reconstructable recovery sequences.
This clears the declared development qualification screen. Measured P3
sessions remain unrun.

The controller-only tool `tools/p3_acceptance_feasibility.py` checks the
[readiness draft](agentic-wam-p3-readiness.md#matched-pilot-draft): six successful
chains out of eight and zero `wrong_object` oracle verdicts. Both independent
oracle verdicts must succeed for a chain to count. The scorer does not exempt
the initial goal: the frozen verifier already classifies the scripted delivery.
Missing, duplicate, unplanned, incomplete or inconsistent evidence produces an
unscorable result with null success and retained outcomes. Executor returns
cannot override the physical verdicts.

`score_session` implements only this arithmetic on facts already validated by
the controller. It does not attest candidate identity, live execution, isolation,
budgets, seed-role disjointness, retention or safety incidents outside the
verdict records. These gates must precede any measured use. Neither the module
nor its measurement inputs/results belong in the participant filesystem or
policy evidence RPC. No participant tool surface is changed by this addition.

## Reproduce the public calibration calculation

Use a unit/development environment with the repository's locked dependencies.
The archive path is local retained evidence, not a tracked repository fixture:

```sh
uv run python tools/p3_acceptance_feasibility.py \
  --archive runs/aw0-p3-display-recovery-retention-44118c9f3362 \
  --catalog-sha256 41ecaeaae2f1239bfe45c97eab4edd61fe0cd4d9cb9e88326697410b60b34ef7 \
  --protocol-sha256 ecfde27da802db1bc0472848ccd8700ffd882b8991befaafe0b5a7dc35b1769e
```

The tool verifies the externally pinned archive catalog and protocol, rebuilds
the measurement chains and public invocation journal from retained sources,
checks source/graph/environment/skill bindings and compares each recomputed
assessment with its retained counterpart. It consumes all twelve declared
calibration slots; a missing slot refuses the calculation. It launches no
simulator, participant or provider process and creates no hidden seeds.

Of all 495 overlapping eight-chain subsets, 201 satisfy the draft rule:

| Successful chains in subset | Number of subsets |
| --- | ---: |
| 4 | 70 |
| 5 | 224 |
| 6 | 168 |
| 7 | 32 |
| 8 | 1 |

This is a descriptive sensitivity calculation on a fixed public calibration
sample. The subsets are not independent sessions. The calculation establishes
neither an agent success probability nor session-level variation, power or a
history benefit. It gives no permission to reuse these seeds as hidden
acceptance cases or to change the draft threshold after pilot collection.

## Remaining P3 integration

The existing isolated mock can serve the same policy evidence to both arms and
run a generated mock worker. A task-specific live adapter still needs to bind
the submitted worker to actual T4 motion, guard traversal, public observation
access and controller-private acceptance. The existing matched launchers are
specific to the typed-versus-monolithic study; renaming their arms does not
implement this history-versus-trace pilot.

Before measured collection, finish that adapter and its mock acceptance,
freeze the matched protocol and disjoint seed roles, reconcile the remaining
resource envelope, and resolve the named provider/model and API deferral.
The USD 200 ceiling remains in force. The completed qualification approval is
consumed; this calculation does not authorize another physical batch.

Requirements exercised by this Class A tool/test change: CON-5/8 and HAR-4,
with TC-8 oracle outcomes. It adds no verifier, frozen graph, specification,
registration, candidate-competence or measured treatment claim.
