# ADR-72 — palm clearance in the top-down grasp plan

Status: PROPOSED — owner review required (CON-7: changes the frozen scene
configuration and the environment hash). Direction approved by the owner on
2026-11-10 ("option 1" on #604).
Trigger: M0-1 on Nexus measured 0.94 twice, with the same three episodes
failing (ADR-71, #604).

## Context

`grasp-planner-topdown` already picks the grip axis whose open fingertips stay
clear of same-level neighbours. That check models the gripper as two points.
The Panda hand is about 20 cm long along the grip axis and its underside sits
3.7 cm above the fingertip TCP.

Two of the three failing M0-1 episodes on Nexus (4 and 29) are one layout:
metformin is the target, it can only be gripped across its 3.5 cm face, and
cetirizine stands 9 cm away along that same axis. Cetirizine is 1.5 cm taller.
At the default 3.5 cm finger engagement the hand's underside is 1.3 cm below
cetirizine's top, inside the hand's footprint.

The contact happens on both engines. With identical arm paths, the neighbour
moved 1.25 cm on Genesis and 3.1 cm on Nexus; the collision threshold is 3 cm.
Genesis passed by margin, not by clearance. A real hand would touch the box
too.

## Decision

1. **The hand body is part of the plan.** `plan_grasp` raises a top-down grasp
   when a same-level neighbour inside the hand's footprint is taller than the
   hand's underside, until the palm clears it by `clear_m`.
2. **The lift is bounded.** At least `min_grip_m` of finger stays on the box.
   When the bound binds, the plan is the best available, not a guarantee.
3. **The geometry is scene configuration (SCN-2).** `[embodiment.franka.palm]`
   in `physics.toml`: `half_length_m` 0.102, `half_width_m` 0.032,
   `above_tcp_m` 0.037, `clear_m` 0.005, `min_grip_m` 0.020. A profile without
   the block keeps the previous plan, so `so101` and `mobile` are unchanged.
4. **Neighbour rows carry their height.** `neighbour_constraints` appends each
   neighbour's height from `meds.toml`; neighbours are taken to stand on the
   target's level, as the fingertip check already assumes. The `neighbours`
   topic payload is unchanged.
5. **`plan_done` is sent only where it is declared** (carried from #601,
   G1-F4): under lockstep an undeclared send killed `ik-trajectory` on
   `expert_t0`.

## Consequences

- Frozen-set change: `physics.toml` and `tools/env_hash.json` move. Successor
  registrations: CSE v30, pilot v16, BND v17, FLT v8, SFE v5. The monolithic
  treatment table is regenerated.
- The Genesis record of 2026-08-28 was measured with the previous plan. Any
  Genesis episode with this layout would now grasp 1.5 cm higher.
- A shallower grip holds the box by 2 cm instead of 3.5 cm in the affected
  layouts. Carry stability there is covered by the gate re-run, not by
  analysis.
- Not addressed: the third failing episode (33) is a different mechanism, a
  shelf pick that fails during the carry at 12.2 s. It is the known transit
  collision class (`analysis/transit_collisions/`).
- Not included: the held-replan gripper latch from #601 (G1-F6). It changes
  retry behaviour on every tier and has only been exercised on the G1 profile;
  it stays with #601.

## Evidence

Gate results on the PR head are recorded in the PR and on #604.
