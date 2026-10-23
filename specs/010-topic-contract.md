# SPEC 010 — Driver topic contract v0

Status: STABLE after M0. Class C (CON-10) — changes need human review.
This contract is what makes sim→real a node swap. Hardware drivers in Phase 4
MUST honor it byte-for-byte.

## 1. Conventions

- TC-1: All angles radians (float32); all positions meters; all frames are the
  robot BASE frame unless a topic name says otherwise. Quaternions are (x,y,z,w).
- TC-2: Every output message MUST carry metadata keys: `sim_time_ns` (int),
  `env_id` (int, 0 in single-env mode), `seq` (per-topic monotonic int). In a
  BRG-1 lockstep simulation (ADR-30) every
  message participating in an open control turn additionally carries
  `turn_epoch` (int, incremented per bridge process start — an ADR-23
  relaunch must not alias the prior incarnation's turns) and its monotonic
  `turn_id` (UInt64), both preserved unchanged across derived outputs. Only
  the bridge opens turns; it opens turn zero at startup and the boot reset is
  consumed in it. Hardware and the explicit non-attesting free-run mode do
  not carry turn metadata.
- TC-3: Image topics carry metadata `h`, `w`, `enc` ("rgb8") and data as a flat
  `UInt8` Arrow array of length h*w*3. Consumers MUST NOT assume resolution.
- TC-4: Rates are contracts, not hints: producers MUST publish within ±20% of
  the declared rate under nominal load; consumers MUST tolerate jitter within
  that band. The ±20% band is measured in WALL-CLOCK time on real hardware.
  Under SIMULATION the deterministic scheduler advances sim time and the sim
  MAY run sub-realtime (T08 live: ~0.5x with rendering; the mobile
  guard↔bridge feedback cycle is slower still); conformance is then enforced
  against the SIM-TIME rate (±20%), and the wall-clock rate MUST only stay
  above a liveness floor of 0.5x the declared rate — enough to catch a grossly
  throttled sim. Sim wall thresholds tighten toward the full band as hardware
  fidelity lands.

## 2. Topics (producer → schema @ rate)

| Topic | Dir | Arrow schema | Rate | Notes |
|---|---|---|---|---|
| `rgb_overhead` | out | UInt8[h*w*3] | 30 Hz | TC-3 metadata |
| `rgb_wrist` | out | UInt8[h*w*3] | 30 Hz | attached to EE link |
| `depth_overhead` | out | Float32[h*w] | 15 Hz | meters; 0 = invalid |
| `seg_overhead` | out | Int32[h*w] | 15 Hz | per-pixel segmentation id; L1 ONLY (TC-9). 0 = background. Genesis renders int64; the bridge NARROWS to Int32 because ids are small (~21 in the desk scene) and the raw type doubles a 640x480 payload at 15 Hz — the wire type is the contract, so a passthrough of the raw array is a TC-1 violation. Ids are the SCENE's segmentation map, not entity indices — see TC-9 |
| `joint_state` | out | Float32[n_dof] | 100 Hz | meta `names: list[str]` |
| `gripper_state` | out | Float32[1] | 100 Hz | 0 open … 1 closed |
| `oracle_state` | out | Float32[n_obj*7] | 30 Hz | pos+quat per box, order = scene manifest; VERIFIER-ONLY (VAL-6) |
| `poses` | out | Float32[n_obj*7] | 15 Hz | pos+quat per box, order = scene manifest (SCN-1), identical layout and ordering contract to oracle_state; ground-truth for tier-T0 oracle perception; NON-privileged (VAL-6 governs oracle_state only); T1/T2 tier gating of this topic is a Phase-2 validator rule (issue #2 resolution) |
| `joint_cmd` | in | Float32[n_dof] | ≤100 Hz | position targets |
| `gripper_cmd` | in | Float32[1] | ≤30 Hz | |
| `episode_result` | out (verifier) | JSON utf8 | per episode | see §3 |
| `human_msg` | out (human-sim) | JSON utf8 | per event | T4 dialogue (ADR-32): scripted request/confirm_reply/correction — a pure function of the per-goal seed and the observed `robot_msg` sequence (CON-5); goal_id-correlated (TC-7); FORWARD edge under ADR-30 lockstep |
| `robot_msg` | out (task-state-machine) | JSON utf8 | per event | T4 dialogue (ADR-32): confirm/ack back to the human-sim; `turn_edge: episodic` (breaks the dialogue cycle, ADR-30 §1.3). In T4 the state machine MUST take its task from `human_msg`, not `episode_goal` — the goal (final corrected target) MUST reach only `verifier-*`/measurement-tap nodes, validator-enforced (DIALOGUE_GOAL_LEAK, ADR-32 §1); a completed confirm exchange MUST precede the first `target_request`, and dialogue corrections count in `episode_feedback.dialogue_corrections`, never as HAR-3 retries |

- TC-5: The bridge MUST publish `joint_state` and accept `joint_cmd` for the
  embodiment profiles (`franka` n_dof=7+2, `so101` n_dof=5+1, `g1_right`
  n_dof=7+2) with identical semantics; `names` metadata disambiguates. The
  SO-101 arm-joint order MUST match the official follower model:
  `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  followed by `gripper` (ADR-27). For `g1_right`, every G1 DOF outside the
  wire is held at home via `held_joint_names`: wire ∪ held MUST partition the
  robot's DOFs (ADR-g1-embodiment, PROPOSED).

## 3. Services and actions (dora patterns)

- TC-6: Reset is a dora SERVICE (request/reply via `request_id` metadata,
  per dora docs/patterns.md): request `reset` payload UInt32[2] = (seed, mode)
  where mode 0=teleport 1=behavioral; reply `reset_done` payload UInt32[1]=1
  with metadata `seed`, `mode`, `t_reset_ms`. The bridge MUST NOT publish
  observations between receiving `reset` and sending `reset_done`.
  A request the service REFUSES (unparseable payload, unknown mode — ADR-8)
  MUST be answered on `reset_refused`, payload UInt32[1]=0, with metadata
  `request_id`, `t_reset_ms`=0, `error`, plus `seed`/`mode` when the payload
  was well-formed enough to carry them. `reset_done` therefore means exactly
  one thing — THE EPISODE BOUNDARY PASSED, the sim was touched — and a
  consumer of it needs no discriminator to know that (ADR-34, issue #195).
  A refusal is a reply, not a boundary: its audience is the one requester,
  while the boundary is broadcast state. `reset_done` MUST NOT carry a
  payload of 0 — the refusal case that used to make it contradict this
  clause now leaves on its own topic (issue #194). A graph carrying a reset
  service MUST declare `reset_refused` AND route it to the requester: dora
  silently DROPS a `send_output` to an undeclared output, so an unrouted
  refusal hangs the requester rather than failing it (validator rule
  `REFUSAL_UNROUTED`). The requester MUST NOT begin an episode on a refusal
  — the scene was never reset, and every other episode-state consumer takes
  its boundary from `reset_done` alone, so advancing would leave them an
  episode behind (issue #209).
- TC-7: An episode is a dora ACTION (goal/feedback/result via `goal_id`):
  goal `episode_goal` = JSON `{tier, target_med, timeout_s, seed}`;
  feedback `episode_feedback` = JSON `{t, phase}` at ≥1 Hz;
  result `episode_result` = JSON:
  `{"status": "success"|"fail", "failure": null|"wrong_object"|"dropped"|
    "timeout"|"never_grasped"|"collision", "t_end": float, "seed": int,
    "goal_id": str, "verifier": "oracle"|"realistic"}`.
- TC-8: `episode_result.status == "success"` from the ORACLE verifier is the
  ONLY ground truth any metric may count. Realistic-verifier verdicts are
  recorded alongside for the fidelity metric, never substituted.
- TC-9: perception ladder — the rung is a CONTRACT, not a convention. The rung (`L0|L1|L2`) selects which pose source a graph may consume, and the bridge MUST publish only what the rung permits. **L0** — the graph may consume `poses` (non-privileged ground-truth pose, TC-2 layout); no segmentation is rendered. **L1** — the bridge MUST NOT publish `poses` and MUST publish `seg_overhead`; pose is ESTIMATED from segmentation plus `depth_overhead`. **L2** — the bridge MUST publish neither `poses` nor `seg_overhead`. Detection and class/instance identity MUST consume rendered `rgb_*` only; the pose estimator MAY pair an RGB detection with same-stamp `depth_overhead` solely for metric 3D reconstruction. Depth MUST NOT supply class or instance identity. Ordinary camera depth is a hardware-portable sensor observation, not simulator semantic ground truth. This implemented contract is ratified by issue #143 and is no longer deferred. An unrecognized rung MUST be refused rather than defaulted to L0: a silent fallback would hand ground-truth pose to a graph that asked not to have it and report the result under the rung it typo'd. `seg_overhead` and `depth_overhead` MUST come from ONE render pass and carry the SAME `sim_time_ns`, which requires them to be co-scheduled on identical ticks (both 15 Hz, TC-4): an L1 estimate masks the segmentation and indexes the depth, so a pair from two ticks measures a scene that never existed — the same rule BRG-2 already gives for the overhead rgb/depth pair, and the same defect class that reached both the trace recorder and the realistic verifier before being caught. A consumer that receives one without a matching stamp waits rather than pairing across ticks. `seg_overhead` is rendered ONLY at L1 because a segmentation pass costs an extra render on every overhead tick and BRG-2 already rate-limits depth for that reason: an L0 run's render budget is unchanged by this topic existing. Ids in `seg_overhead` are the simulator's own segmentation map (Genesis: `scene.segmentation_idx_dict`, seg id → (entity_idx, link_idx)) and are NOT entity indices — a consumer that masks on entity index silently selects other geometry, measured as robot links whose pixel counts were identical across scenes with different object layouts. The bridge therefore MUST publish the per-object id map in `bridge_info` so a consumer never guesses it (measured on the desk scene: entity 5, amoxicillin, is seg id 16). Rung selection MUST be declared IN THE GRAPH — the bridge node's `env` key `AISLE_PERCEPTION`, a field dora's dataflow schema already defines and passes to the node process — so the graph hash attests which pose source a result used, and it is never inherited from ambient environment. The bridge MUST also announce the rung in `bridge_info`, so a RECORDED run attests its rung without reference to the graph it came from. An L1 pose estimate carries its supporting mask size so a consumer can refuse a partially occluded object rather than accept a centroid biased toward the visible fragment (measured: 2.2 mm mean XY error over 20 objects, but 19.5 mm on the smallest mask).

## 4. Acceptance

- TC-A1 (`tests/accept/test_contract.py::test_schema_conformance`): run the
  bridge 10 s headless; every observed message validates against §2 schemas,
  TC-2 metadata present (including preserved `turn_id` in lockstep), rates
  within the TC-4 band (sim-time rate ±20% in simulation, wall-clock held
  above the 0.5x liveness floor). Cites TC-1..5.
- TC-A2 (`::test_reset_service`): 20 seeded resets; no observation interleaves
  reset→reset_done; identical seed twice ⇒ identical `oracle_state` first
  message (CON-5). Cites TC-6.
- TC-A3 (`::test_episode_action_lifecycle`): scripted trivial episode ends with
  a schema-valid `episode_result` carrying the goal's `goal_id`. Cites TC-7/8.
- TC-A4 (`tests/unit/test_cmd_coalescing.py::test_bridge_publishes_only_what_the_rung_permits`,
  `::test_segmentation_and_depth_are_co_scheduled_on_every_tick`,
  `::test_segmentation_id_map_uses_the_scene_map_not_entity_indices`,
  `::test_perception_rung_from_env_defaults_l0`, `::test_bridge_info_shape`) plus
  `tests/sim/test_l1_perception.py::test_l1_estimate_matches_genesis_ground_truth`
  and `::test_segmentation_render_shares_the_depth_stamp_and_pass`: the ladder's
  publish rules, the one-render pairing, and the id map — the sim pair pins the
  estimated pose against Genesis ground truth, because an L1 number is only a
  policy measurement if the estimator agrees with what L0 would have given.
  `tests/unit/test_l2_pose.py::test_l2_rgb_identity_uses_same_stamp_sensor_depth_only_for_metric_pose`,
  `tests/unit/test_validator.py::test_expert_t1_l2_is_good`, and
  `tests/graph/test_expert_graph.py::test_expert_t1_l2_episode_succeeds` pin L2's
  RGB-only identity path, same-stamp depth geometry, and absence of both
  `poses` and `seg_overhead` from the policy graph.
  Cites TC-9.
