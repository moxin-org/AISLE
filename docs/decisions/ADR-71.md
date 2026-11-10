# ADR-71 — hardware readiness: platform order, Nexus as default engine, work order

Status: ACCEPTED 2026-11-10 (owner direction given the same day; agent-drafted).
Amends: ADR-66 section 3 (execution order) and the SO-101-first order in the
#356 notes. Touches no frozen path and no spec. Every spec or frozen-set
change it names is a separate `spec-change` / env-change PR under CON-14 and
CON-7.

## Context

No hypothesis or ablation has been measured since 2026-08-28. The #347 pilot
has been re-registered fifteen times without a session. Meanwhile the first
physical robot the project will reach is not the SO-101 that SPEC 520 and
ADR-66 assumed: collaborators bring a Unitree G1. PR #601 already carries a
G1 right-arm profile for the 29-DOF model. ADR-67, ADR-68 and ADR-70 put Nexus
and rapier behind the scene contract and inside the lock, but left both as
development-only engines.

The owner's goal for this phase is to prepare the repository for hardware
adaptation.

## Decision

### 1. Platform order

1. Unitree G1 EDU U2, 23 DOF (5 arm joints per side, 1 waist joint).
2. Unitree G1, 29 DOF.
3. SO-101.

PR #601 keeps its 29-DOF scope (`g1_29dof_mode_15_with_dex1_1`, 7 arm joints
plus the Dex1-1 gripper). The 23-DOF unit is a separate embodiment profile
stacked on #601: its own wire order, limits, guard envelope and IK. It does
not reuse the 29-DOF gate evidence.

Not yet known, and blocking the 23-DOF wire and grasp work: the unit's
`mode_machine` number and its mounted end effector. Until the owner supplies
them, 23-DOF work stops at a simulated reachability report.

This replaces ADR-66 section 3 items 5 and 6. ADR-66 section 4 (spec
moratorium) is lifted for the spec changes this ADR names and for no others.

### 2. Nexus becomes the default simulation engine

- `nexus` becomes the default for `--sim-engine` and `AISLE_SIM_ENGINE`.
  On macOS arm64 it resolves to native Metal.
- `rapier` stays as the reproducibility cross-check: same renderer, CPU
  physics.
- `genesis` stays selectable. The 2026-08-28 measured record remains a
  Genesis record and keeps that label; results across engines are not
  comparable (ADR-67).

The default flips only after the conditions ADR-67 set are met:

1. Wheels inside the lock: done (ADR-70).
2. A `spec-change` PR generalizes the Genesis wording of SPEC 020 and
   SPEC 030 (CON-14).
3. The frozen baseline is re-established on Nexus under human review
   (CON-7): M0-1 and M0-2 (50 seeds, pass1 >= 0.95, replicate), the T1
   expert per perception rung, the VER-8 calibration, the environment hash
   and the freeze registrations.

If M0-1 does not pass on Nexus, tuning is limited to
`src/aisle/sim/nexus_physics.toml`. The frozen scene, limits and verifier
thresholds are not changed to make an engine pass. A gate that still fails
after that is reported, not worked around.

First evidence, development only: a 5-seed T0 expert run on Nexus on the
simulation Mac on 2026-11-10 gave 4/5 with one `collision` (seed 4) at
1.21x realtime. That is below the M0-1 bar and is not gate evidence.

### 3. The #347 pilot

The pilot moves behind the engine switch, because its registration binds the
Genesis instrument set and its output would size a study on an engine that is
no longer the default. It stays owner-started: it spends a provider account.
H6 keeps its place after the pilot.

### 4. Work order

| # | Workstream | Done when |
|---|---|---|
| 0 | This ADR and its tracking issue | merged, issue open |
| 1 | Trustworthy CI: #598, the sim/graph/accept nightly, the red `main` run | unit suite green on the simulation Mac; nightly green and scheduled |
| 2 | The two `main` fixes carried in #601 (undeclared `plan_done`; held replan opening the gripper) | merged as their own PRs |
| 3 | Engine switch (section 2) | M0 gate passes on Nexus; default flipped |
| 4 | Finish #601 (29-DOF): gate on the new default, split into spec / env / feature PRs | gate passes; PRs merged by the owner |
| 5 | 23-DOF profile, stacked on 4 | reachability report; then sim smoke once the end effector is known |
| 6 | G1 hardware day-one kit: station manifest, calibration, halt drill, safety-case templates, loopback dry run | dry run passes without a robot |
| 7 | Housekeeping: #508, #519, README status, #359, root archives | issues closed; status dated |

### 5. Merge rule for this work

Agent-authored PRs in this program may be admin-merged once CI is green
(owner authorization, 2026-11-10). `spec-change` and frozen-set PRs still
wait for the owner (CON-14, CON-7); later work stacks on them. Nothing is
pushed to `main` directly, and PRs by other authors are not merged by the
agent.

## Consequences

- SPEC 520 (SO-101 phase gate) is no longer the first hardware gate. It is
  not edited here; the G1 gate is SPEC 550 in #601, and the 23-DOF unit needs
  a section there or a sibling spec.
- The Genesis-fit constants in `physics.toml` were never re-tuned for Nexus
  (ADR-67). The re-baseline may show that they need an engine-specific
  counterpart outside the frozen set.
- Stepping determinism on Nexus is unmeasured (ADR-67). The M0-2 replicate on
  Nexus is its first measurement.
- Until section 2 completes, Genesis remains the default and CLAUDE.md is
  unchanged.
