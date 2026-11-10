# SPEC 030 — simulation bridge node (`dora-genesis`)

Status: DRAFT until M0. Module: `src/aisle/nodes/dora_genesis.py`. Implements SPEC 010 against SPEC 020. The node keeps its historical name `dora-genesis` on every engine (ADR-67): graphs, traces and manifests refer to it by that name.

- BRG-1: Exactly one bridge node owns the simulation scene per dataflow.
  An attested, acceptance, or campaign simulation MUST use run-to-quiescence
  lockstep (ADR-30). The bridge publishes the observations due for
  simulated state `S(k)` and then `sim_turn` (`sim_turn_u64`, UInt64[1]); its
  epoch-scoped monotonic value and every same-turn message's
  `turn_epoch`/`turn_id` metadata identify exactly one open control turn.
  Every watermark enumerates EVERY output port in parallel
  `closed_outputs: list[str]` and `emitted_counts: list[int]` metadata in
  lexical port order (count 0 is ordinary; an omitted port is a malformed
  watermark). A participant (ADR-30 §1.1: any node with a FORWARD-edge path
  to a bridge command/reset input) closes turn `k` after receiving every
  count declared by every forward upstream for turn `k` plus the counts its
  EPISODIC inputs' producers declared in turn `k-1` (ADR-30 §1.3 — reply/
  verdict back-edges deliver into the next turn, which is what makes TC-6/
  TC-7 loops terminate); transport arrival order is never treated as closure.
  The bridge does not apply commands or advance physics until every forward
  branch closes and exactly one `turn_commit` (`sim_turn_u64`) arrives for
  the turn. On a motion commit the bridge applies inputs in canonical order
  (`joint_cmd`, `gripper_cmd`, `base_cmd`), advances sim by `cfg.dt` exactly
  once, and opens the next turn. On a reset commit, reset takes priority,
  discards same-turn motion, injects the new state, advances no physics, and
  opens the next globally numbered turn at episode-relative step zero; the
  bridge opens turn zero at startup and the initial reset is consumed in it.
  Participants emit turn-stamped messages only from turn-edge/episodic
  handlers, never from wall-timer handlers (ADR-30 §1.4). Missing,
  duplicate, stale, future, cross-epoch, or unstamped commits/commands fail
  loudly; arrival order and WALL latency never select a simulated step. A
  WALL watchdog (budgeted per turn type — verdict-bearing turns carry their
  work's own budget, ADR-30 §1.5) may abort a hung turn but never advances
  or manufactures a commit. Free-running `dora/timer/millis/10` stepping is
  permitted only for reset-less bring-up or interactive visualization and
  MUST be recorded non-attesting; it is not eligible for acceptance or
  campaign metrics.
- BRG-2: Rendering is rate-limited independently of physics (TC table rates); a tick MUST NOT render all cameras every step. Target: ≥5x realtime physics with rendering on, single env, M3.
- BRG-3: `joint_cmd` applies PD position control. In lockstep, multiple
  same-turn commands are coalesced by the producer's monotonic `seq`, never by
  arrival order (document in metadata `dropped:int`); free-run commands
  arriving faster than 100 Hz are coalesced to latest.
- BRG-4: Reset service per TC-6. Behavioral mode (mode=1) delegates to SPEC 040 reset routine; teleport mode re-invokes `build_scene` state injection without process restart.
- BRG-5: `n_envs>1`: every output message carries `env_id` (TC-2); `joint_cmd` metadata `env_id` routes commands; missing env_id in multi-env mode is an ERROR event, not a default.
- BRG-6: The node MUST publish a `bridge_info` message once at startup: JSON `{contract:"v0", embodiment, n_dof, n_envs, genesis_version, platform, env_hash}` — rollout runner (SPEC 070) refuses on hash mismatch (CON-7). The payload also names the engine as `sim_engine` (`genesis`, `nexus` or `rapier`; an unknown name is refused, never defaulted), and `genesis_version` keeps its name while carrying the version of whichever engine ran (ADR-67).
- BRG-8: (realistic-verifier increment one; declarative pre-implementation like VER-8..14 — the implementing bridge PR upgrades it to RFC-2119 normative wording together with the updated BRG-6 exact-shape test, avoiding a spec/test conflict on main) the `bridge_info` payload additionally carries `calibration` — the VER-8 v1 calibration block (SPEC 040) as a nested JSON object, populated from the BUILT scene's realized camera state converted to the v1 conventions per VER-8 (post-DR-jitter poses, the same values the render path uses; on hardware, from the measured-calibration artifact VER-8 names). The block is static per graph run — camera jitter realizes once at build — so consumers correlate it by run; time-varying wrist geometry correlates via `joint_state` sim stamps per VER-8. The existing BRG-6 fields, cadence, and the runner's hash-refusal behavior are unchanged.
- BRG-7: Failure honesty: sim exceptions crash the node loudly (dora ERROR event); the bridge MUST NOT catch-and-continue physics errors.

Acceptance: TC-A1..A3 exercise this node; plus
`tests/graph/test_bridge_lockstep.py` runs the same stamped control sequence
under two deliberately different node-delay schedules and asserts identical
command-to-turn assignment, reset/turn stamps, and physics state through
CON-5's complete 1.0 sim-s comparison window (BRG-1); it also proves a hung,
duplicate, stale, future, or unstamped turn fails without advancing physics.
`tests/graph/test_bridge_minimal.py::test_headless_60s_no_drift` exercises the
explicit non-attesting bring-up mode (BRG-1,2 — sim_time vs wall time ratio
logged), `::test_multi_env_routing` covers BRG-5, and
`tests/unit/test_cmd_coalescing.py` covers BRG-3 with sim mocked. Realistic
increment one (env-change PR): a bridge acceptance test asserts the published
`calibration` block field-for-field against the built scene's realized camera
state (BRG-8 + VER-8), including CONCRETE numerics after the
Genesis-to-v1 conversion (overhead 640x480 at vertical FOV 55°: fx = fy =
240/tan(27.5°) ≈ 461.04, cx = 319.5, cy = 239.5) and the converted
OpenCV-frame quaternion for the jitter-consistent overhead pose; the BRG-6
exact-shape assertion is updated in the SAME change.
