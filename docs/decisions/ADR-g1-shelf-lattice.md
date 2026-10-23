# ADR-g1-shelf-lattice — G1 slot re-lattice, shoulder-roll planning bound, grasp/carry clearance

Status: PROPOSED — operator rulings 2026-10-22 (re-lattice) and 2026-10-23
(8 mm grasp/carry margin), recorded for owner review under CON-14/CON-7.
Lives on `feat/g1-support`; G1 scene configuration only (`physics.toml`
`[embodiment.g1_right]`), so it rides the env-change draft PR. Franka and
SO-101 are untouched.

## Problem

The first 50-seed M0-6 gate on `g1_right` (integration `e6363bb`) scored
pass1 0.76 against the 0.80 bar. All 12 failures targeted the slot
(0.28, 0.08), and every one of them logged `advance/close bailed at joint 1
err ~0.15`; the other four slots went 38/38. A Genesis hold probe of the
planned advance configuration showed `right_shoulder_roll_link` touching
`torso_link` from shoulder roll ~0.0 rad and stopping at +0.039, while the
plan asked for +0.198 (error -0.159, matching the log). The slot sits past
the robot's midline; the IK has no self-collision model and turned the
shoulder into the torso to reach it.

## Decisions

1. **Planning-only shoulder-roll bound.** `ik_shoulder_roll_max_rad =
   -0.05` (0.05 rad margin under the measured contact onset) caps the
   planner's joint-1 upper limit. The guard's `env/limits.toml` envelope
   and the real-robot limits are unchanged — this constrains which IK
   solutions the planner may pick, not what the arm is allowed to do.
2. **Re-latticed slots.** Under that bound, top-down grasps are only
   plannable in x 0.24–0.28, y −0.24..+0.02 (measured: y −0.26, y beyond
   +0.02, and x ≥ 0.32 fail pregrasp IK; (0.28, 0.08) and (0.24, 0.08) fail
   outright). The five slots now sit inside that band at the 0.065 m y
   pitch (≥ the 0.0625 m worst half-width pair + `min_separation`), x
   staggered 0.28 / 0.24: (0.28, −0.24), (0.24, −0.175), (0.28, −0.11),
   (0.24, −0.045), (0.28, 0.02). Alternatives considered and declined by
   the operator: unlocking waist yaw (8+2 wire, spec-change, days of work)
   and keeping the unreachable slot as a known limitation (gate ceiling
   ~0.76).
3. **Grasp/carry clearance 8 mm.** The 0.065 m pitch leaves the hand
   9.2–15 mm from a *worst-case* neighbour (the widest med assumed at every
   other slot) during advance/close/lift/transfer, under the F9 15 mm
   planning margin. A 10-episode physical smoke on the re-latticed scene
   (seeds 0–9) had no collision or drop and moved non-target boxes at most
   0.4 mm (traces of the final launch only). The arm-link sweep test now
   requires 8 mm in those four stages and keeps 15 mm everywhere else; the
   oracle's collision threshold (3 cm displacement) is unchanged.

Supporting planner changes made for the new lattice (all G1-only):
transfer detours through a front waypoint only when the straight chord
passes within `trajectory_transfer_chord_clear_m` (0.07) of another slot,
backing out at the staging height; the home chain's step waypoint is chosen
by whether its contained chain stays under the ceiling and off the shelf
slots; containment splits any hop that sweeps low over a slot, with slerped
midpoint orientation.

## Consequences

- The 15 mm → 8 mm relaxation is a planning-test margin, not a verifier
  or gate threshold. If a later gate or smoke shows non-target
  displacement, revisit this before anything else.
- Five reachable slots consume the whole band: any new slot, a wider med,
  or a larger `min_separation` will not fit without expanding the control
  scope (waist yaw) — a spec-level decision.
- Simulation evidence only; real G1 reach and self-collision margins are
  `hardware_pending` (ADR-g1-embodiment).
