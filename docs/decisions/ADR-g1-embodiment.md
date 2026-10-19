# ADR-g1-embodiment — Unitree G1 support, scope A: `g1_right` wire, mode-15 machine, held-DOF mechanism

Status: PROPOSED — draft for owner review under CON-14/CON-7. This work
lives on `task/g1-a` (off `g1/base`, to be fast-forwarded into
`feat/g1-support`); it MUST NOT merge to `main` until the owner rules on
both the TC-5 amendment (spec-change draft in this branch) and the
frozen-set widening (assets/g1 in `tools/env_hash.py`). Relationship to
ADR-66: ADR-66 sequences evidence (pilot tier before confirmatory);
nothing here claims a measured result — this is embodiment plumbing for a
lane the blueprint scopes, and it inherits ADR-66's rule that draft work
stays out of the record until ratified.

## Scope A: why right arm + right hand only (7+2)

The G1 campaign's first slice controls the right arm's seven revolute
joints plus the right Dex1-1's two prismatic fingers — the same 7+2 wire
shape as `franka`, so the bridge, verifier workspace checks, and topic
contract semantics transfer unchanged (`n_dof=9`, `names` metadata
disambiguates per TC-5). Wiring all 33 actuated DOFs now would grow the
action space the campaign does not study yet and couple this lane to
balance/locomotion decisions nobody has ratified. The cost of scope A is
the held-DOF mechanism below; the cost of full-body scope is every later
decision being load-bearing on an unreviewed contract. Scope A is the
recoverable choice.

## Why `g1_29dof_mode_15_with_dex1_1`

The official `unitree_ros` table (vendored as
`assets/g1/UPSTREAM_README.md`) pins the mode: mode_machine 15, hip gear
{22.5, 22.5}, new 5010 wrists, unlocked waist, 6·2 leg + 3 waist + 7·2
arm + 2·2 Dex1-1 finger DOFs = 29 body + 4 fingers = **33 actuated
DOFs** (`floating_base_joint` is not a DOF). mode 15 is the up-to-date
variant with the 5010 wrist and unlocked waist — matching the physical
lane robot — and `with_dex1_1` is the only official description carrying
the Dex1-1 gripper the campaign actually grasps with. Pinned at upstream
commit `5994d4f` (BSD-3-Clause), vendored verbatim with all 39 referenced
meshes (`assets/g1/PROVENANCE.toml`), mirroring ADR-27's SO-101
provenance discipline (CON-5, SCN-4).

## Held-DOF mechanism: `held_joint_names`, not `hold_unlisted_joints`

Two designs were on the table:

1. an explicit `held_joint_names` list in the profile, or
2. a boolean `hold_unlisted_joints = true` ("everything not in the wire
   is held").

We choose (1). Reasons:

- **Fail-loud coverage.** With a list, `profile_dof_indices` can enforce
  the partition — wire ∪ held must equal the robot's DOF set, disjoint —
  and reject both overlap and gaps at startup. A boolean cannot detect a
  gap: if a new URDF revision renames a joint, "hold unlisted" silently
  holds a DOF nobody audited, and the failure surfaces only as degraded
  sim behavior. Explicitness turns an asset drift into a hard error,
  which is the CON-5 instinct.
- **Symmetry with the existing contract.** `arm_joint_names` /
  `gripper_joint_names` are already explicit lists; the held set gets the
  same shape, the same resolution path (`robot.get_joint` by name), and
  the same "never trust parser order" rule.
- **The boolean's only advantage** — profiles never go stale when the
  robot model changes — is exactly the property we do NOT want here: a
  G1 wire must fail loudly when the embodiment changes underneath it.

Semantics (implemented in `src/aisle/embodiment.py`, frozen file — this
ADR is the CON-7 authorization trail):

- `profile_held_joint_names(profile)`: `()` for every profile that does
  not declare held joints — including legacy full-coverage wires
  (`franka`/`so101`/`mobile`), whose behavior and return values are
  unchanged.
- `profile_dof_indices(robot, profile)`: without `held_joint_names`, the
  legacy rule stands verbatim (wire alone must cover every DOF, coverage
  and uniqueness enforced). With `held_joint_names`: resolve both by
  name, reject overlap, reject under-coverage, return **wire indices
  only** — the wire shape on the bus does not change.
- `held_dof_indices(robot, profile)`: the held DOF indices, in declared
  name order; `()` when none.
- `from_wire_joint_order(values, indices, base=None)`: `base=None` is
  bit-identical to the old behavior. With `base` (a full-length native
  vector, held joints already at home), the result is a copy of `base`
  with only the wire positions overwritten — the bridge's command path
  for partial-coverage wires.
- `held_joint_names` without `arm_joint_names`/`gripper_joint_names` is a
  `ValueError` (there is no wire to hold complement to).

## mode_machine consistency

Any physics engine this description loads into MUST run mode 15 — the
same `mode_machine` number the upstream table assigns to
`g1_29dof_mode_15_with_dex1_1`. A mismatched mode (e.g. a 23dof or
lock-waist variant) changes the joint set under the wire contract and
the partition check will (by design) fail; that failure is the mechanism
working, not a bug to suppress. Downstream config (physics.toml, scene
bridge) is out of scope here and lands with the next board item.

## Tests

`tests/unit/test_g1_contract.py` (mirrors `test_so101_contract.py`):
PROVENANCE/license/mesh closure; right-arm order vs
`G1_RIGHT_ARM_JOINTS`; Dex1-1 prismatic limits [-0.02, 0.0245]; total 33;
robot name `g1_29dof_mode_15`; fake-DOF-indexer tests for shuffled
parser order, partition rejection, held-position preservation, wire
round-trip, legacy non-regression. `tests/unit/test_env_hash.py` gains
the G1 asset-mutation case (CON-7).

## Guard velocity envelope (G1-F1 addendum)

`env/limits.toml [embodiment.g1_right] qdot_max` does NOT take the URDF's
declared motor velocities. The URDF rates — 37 rad/s at the shoulder/elbow
and 27 rad/s at the wrist, effort 25/13.4 Nm — are the raw actuator
envelope of a 25 Nm-class humanoid arm; a guard that admitted them would
sanction wrist flicks near 4 m/s at the 0.42 m chain, which no table-top
retail scene needs and no co-located operator should stand near. The guard
enforces a conservative envelope instead: arm joints `min(URDF, 2.0)`
rad/s (≈ 0.84 m/s at full extension), fingers at the URDF's own 0.2 m/s
(the Dex1-1 is travel-limited and slow by construction). Tightening the
guard below the actuator envelope is safety margin, never a claim about
the hardware's capability; if a task ever needs more, the owner amends
this section with measured justification, not a URDF copy-through. The
URDF rates stay visible in `chain.qdot_max` for provenance.

## G1-F2 known limitation: cross-body far-column transfers

The staged planner solves the full pick/place ladder for the C1 near
columns (x=0.28 and the right-biased x=0.24 slots), verified end-to-end
(rise, transit descent, in-place wrist roll, advance, lift, retract,
free-yaw transfer with tray roll, lower, home). The cross-body far column
[0.24, +0.16] — the RIGHT arm reaching across the torso — stalls the
transfer-stage DLS in an elbow-family switch: warm starts plateau ~0.04 m
out, and the cold solution lands on a different branch (3.5 rad away).
Every remediation was measured and rejected by a safety gate: five
Cartesian routes (direct, flipped parity, side-column, near-column hop,
high arc) with free-yaw all stall; the joint-space sweep between the two
families clears the shelf footprint (min TCP z 0.999 over boxes topping at
~0.92) but flips the palm mid-sweep (worst rot[2,2] +0.85 vs the required
palm-down -0.90). The planner therefore fails LOUD there (plan.error set,
no waypoints execute) rather than emitting an unverified swing — the same
discipline the front-flip sweep enforces. Lifting this needs either a
measured transfer-route profile (like so101's trajectory_transfer_route)
or a smarter solver; tracked for the follow-up. Grasp-side reachability
of that column is NOT affected (direct IK solves it; the sim parity test
uses seed-0 cetirizine at [0.24, -0.16] and passes).

## G1-H-r2: kinematic carry latch — a sim aid, not real-grasp evidence

The smoke evidence (run 20261019-215328-9e8aeb, seed 1) shows the Dex1-1
fingers DO close onto the box (normalized grip plateaus at 0.726 against
the box surface — the physical contact stops them short of the 1.0
command) and the box lifts ~0.12 m, then slides out of the pinch in
transit and falls down the slot column. Genesis does not retain the
Dex1-1 pinch under motion — the exact precedent SO-101 hit (ADR-27 /
ADR-18 pattern). The decision: enable the bridge's kinematic carry latch
for g1_right (close 0.65, below the measured on-box plateau; release
0.20; max distance 0.15 m). The latch captures the nearest in-envelope
box in the hand frame from grip close and returns it to physics at open
— no target identity, no contact forces.

**This is a simulation aid only.** It is NOT evidence that the real
Dex1-1 can retain the box: real retention is a force-controlled pinch
whose characterization belongs to the hardware phase. Any real-robot
grasp claim must be re-established with measured finger forces.

## G1-H: cold-build budget — warmup, not budget inflation

Genesis caches CoACD convex decompositions and compiled kernels per
machine. Measured (outer-loop probe, lane1, metal backend): the G1's
FIRST scene build costs ~470 s (import+init ≈20 s, add_entity ≈280 s —
almost entirely per-link CoACD — scene.build ≈170 s of kernel compile),
while a warm build costs ~20 s and 100 steps ~3 s.

`rollout.py`'s `GENESIS_BUILD_BUDGET_S = 420` therefore fails a COLD G1
build (~470 s). The decision: **the budget stays**. The build budget
protects every embodiment's steady-state expectations; inflating it for
a one-time per-machine cost would loosen protection for every future
run. The one-time cost is paid deliberately instead:

- `tools/sim_warmup.py --embodiment g1_right` builds the scene once,
  reports per-stage wall times, and infers cache warmth (warm if the
  scene build is < 60 s — between the measured warm ~27 s and cold
  ~470 s). New machines run warmup before their first rollout; the M0-6
  gate and rollout docs say so.
- No relock of the budget, no silent retry-inside-rollout: a cold build
  inside rollout still fails loudly, which is the correct signal that
  warmup was skipped.
- G1-H-r1: the FIRST camera render pays a one-time shader/kernel compile
  too (measured 10.6 s/frame — the wall_clamp root cause of the first
  M0-6 gate run). sim_warmup therefore renders one frame per camera on
  the channels the bridge publishes (overhead rgb+depth+seg, wrist rgb);
  scene-build warmth alone does not cover render compilation.
