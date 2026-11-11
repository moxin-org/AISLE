# ADR-73 — Nexus is the default engine: evidence and what was not re-derived

Status: ACCEPTED 2026-11-11 by owner direction ("merge 609 and 610, keep
going"), completing ADR-71 section 2.
Trigger: the three ADR-67 conditions ADR-71 lists for the default flip.

## Decision

`DEFAULT_ENGINE` in `src/aisle/sim/__init__.py` is `nexus`. The harness CLI,
the rollout runner and the simulator bridge take their default from it, so a
graph that declares no engine and a run that names none execute on Nexus.
`--sim-engine genesis` and `--sim-engine rapier` stay available, and a graph
that declares `AISLE_SIM_ENGINE` still owns the choice (ADR-67).

## Conditions

1. Wheels inside the lock: ADR-70.
2. Engine-neutral wording for SPEC 020 / SPEC 030: #609, merged.
3. Baseline on Nexus, simulation Mac, native Metal, `--env-baseline local`:

| Check | Commit | Result |
|---|---|---|
| M0-1, T0 expert, 50 seeds | `ed70b59` | pass1 0.98 (49/50); seed 33 `collision` at 12.2 s |
| M0-2 replicate | `ed70b59` | pass1 0.98; zero per-seed flips, identical `t_end` |
| T1 expert, L1 rung, 50 seeds | `baef84c` | 46/50; `dropped` x2, `timeout`, `collision` |
| T1 expert, L2 rung, 50 seeds | `baef84c` | 37/50; `never_grasped` x11, `collision` x2 |
| Environment hash | `ed70b59` | `0d649385...`, unchanged by this flip |

`baef84c` is the #610 head; its motion sources are the ones on `ed70b59`.

## What was not re-derived

- The T1 rungs have no recorded 50-seed Genesis figure, so the T1 rows are a
  first measurement, not a pass against a bar.
- At L2 the expert never attempts ibuprofen: `detected-pose` refuses it under
  the identity-margin floor. The same episodes fail on Genesis (seeds 1, 6, 11
  of 0..11) and on `main` before #610. It is not an engine effect and is left
  as is. Seed 0 (amoxicillin) is refused on Nexus only.
- The VER-8 calibration block is covered on Nexus by the `sim` suite, not by
  a separate measurement.
- BND band calibration, FLT bank calibration and the CSE/pilot treatment
  parameters were derived on Genesis. BND v18, CSE v31 and pilot v17 bind the
  new sources and say so; they authorize no collection. A campaign on Nexus
  needs its own calibration first.
- The Genesis-fit constants in `physics.toml` are unchanged (ADR-67).
- Carry-phase collisions (seed 33 at T0, seeds 33 and 45 at T1) are open.

## Consequences

- The 2026-08-28 record stays a Genesis record. Results across engines are
  not comparable.
- The graph and acceptance suites launch the bridge without naming an engine
  and now run on Nexus.
- The #347 pilot sits behind this switch and stays owner-started.
