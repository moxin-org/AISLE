"""Unit tests for the bridge's pure control-plane logic (SPEC 030 BRG-1,
BRG-2, BRG-3, BRG-5, BRG-6) — sim mocked, no dora or genesis imports
(CON-12)."""

import json

import numpy as np
import pytest

from aisle.nodes.dora_genesis import (
    BridgeConfig,
    CommandQueue,
    RateScheduler,
    ResetQuarantine,
    make_bridge_info,
    may_publish,
    parse_bridge_config,
    require_usable_segmentation_ids,
    reset_publish_topics,
    rung_topic_rates,
    segmentation_id_map,
)

pytestmark = pytest.mark.unit

# the production wrist mount (SCN-5 `wrist_rotation_xyzw`, 180 deg about
# X): tests pass it explicitly because build_calibration_v1 requires it —
# a default would let a nominal block describe a different camera than the
# published one and stage 0 would refuse every episode (#110 review)
WRIST_MOUNT = np.diag([1.0, -1.0, -1.0])


def test_coalesce_keeps_latest_and_counts_dropped():
    """BRG-3: commands arriving faster than the tick are coalesced to the
    latest per (kind, env), with the number of superseded commands
    documented as dropped:int."""
    queue = CommandQueue(n_envs=1)
    queue.push("joint", 0, [0.1])
    queue.push("joint", 0, [0.2])
    queue.push("joint", 0, [0.3])
    assert queue.drain() == [("joint", 0, [0.3], 2)]
    assert queue.drain() == []  # drained until new pushes


def test_arrival_order_preserved_across_kinds():
    """BRG-1: pending inputs are serviced in ARRIVAL order — a joint_cmd
    arriving after a gripper_cmd is applied after it (the last-arrived
    command owns overlapping dofs), and vice versa."""
    queue = CommandQueue(n_envs=1)
    queue.push("gripper", 0, [0.5])
    queue.push("joint", 0, [0.1])
    assert [kind for kind, *_ in queue.drain()] == ["gripper", "joint"]
    queue.push("joint", 0, [0.2])
    queue.push("gripper", 0, [0.6])
    assert [kind for kind, *_ in queue.drain()] == ["joint", "gripper"]
    # a re-arrival moves the kind to the back of the order
    queue.push("gripper", 0, [0.7])
    queue.push("joint", 0, [0.3])
    queue.push("gripper", 0, [0.8])
    assert [(k, p) for k, _, p, _ in queue.drain()] == [("joint", [0.3]), ("gripper", [0.8])]


def test_coalesce_routes_per_env():
    """BRG-5: coalescing is per (kind, env) — commands for different envs
    never supersede each other."""
    queue = CommandQueue(n_envs=2)
    queue.push("joint", 0, [0.1])
    queue.push("joint", 1, [0.9])
    queue.push("joint", 1, [1.0])
    assert queue.drain() == [("joint", 0, [0.1], 0), ("joint", 1, [1.0], 1)]


def test_missing_env_id_in_multi_env_is_error():
    """BRG-5: a command without env_id in multi-env mode raises — it is an
    ERROR event, never a defaulted route."""
    queue = CommandQueue(n_envs=2)
    with pytest.raises(ValueError, match="env_id"):
        queue.push("joint", None, [0.1])


def test_out_of_range_env_id_is_error():
    """BRG-5: env_id outside [0, n_envs) is rejected up front — -1 must not
    silently route to the last environment."""
    queue = CommandQueue(n_envs=2)
    with pytest.raises(ValueError, match="outside"):
        queue.push("joint", -1, [0.1])
    with pytest.raises(ValueError, match="outside"):
        queue.push("joint", 2, [0.1])


def test_single_env_defaults_env_id_zero():
    """TC-2: in single-env mode env_id defaults to 0."""
    queue = CommandQueue(n_envs=1)
    queue.push("gripper", None, [0.5])
    assert queue.drain() == [("gripper", 0, [0.5], 0)]


def test_rate_scheduler_hits_contract_rates():
    """BRG-2, TC-4: with a 100 Hz tick, each topic fires at its declared
    contract rate (within the tick's granularity) and camera topics are due
    on only a subset of ticks — never all cameras every step."""
    rates = {"joint_state": 100, "oracle_state": 30, "rgb_overhead": 30, "depth_overhead": 15}
    scheduler = RateScheduler(rates, dt=0.01)
    fired = {topic: 0 for topic in rates}
    render_ticks = 0
    for _ in range(100):  # one simulated second
        due = scheduler.due()
        for topic in due:
            fired[topic] += 1
        if any(t.startswith(("rgb", "depth")) for t in due):
            render_ticks += 1
    assert fired["joint_state"] == 100
    assert fired["oracle_state"] == 30
    assert fired["rgb_overhead"] == 30
    assert fired["depth_overhead"] == 15
    assert render_ticks < 100  # BRG-2: not every tick renders


def test_rate_scheduler_is_deterministic():
    """CON-5: two schedulers with identical config produce identical due
    sequences."""
    rates = {"a": 30, "b": 15}
    first = RateScheduler(rates, dt=0.01)
    second = RateScheduler(rates, dt=0.01)
    sequence_a = [first.due() for _ in range(50)]
    sequence_b = [second.due() for _ in range(50)]
    assert sequence_a == sequence_b


def test_bridge_info_shape():
    """BRG-6 + BRG-8: bridge_info carries exactly the contract fields as
    JSON, including the VER-8 v1 `calibration` block (SPEC 040) built
    from the REALIZED camera state — the verifier's stage 0 refuses
    without it, so it is part of the exact shape."""
    from aisle.verifier.calibration import build_calibration_v1

    calibration = build_calibration_v1(
        [0.55, 0.0, 1.20],
        [0.55, 0.0, 0.20],
        (640, 480),
        55.0,
        [0.0, 0.0, 0.05],
        (320, 240),
        70.0,
        WRIST_MOUNT,
    )
    info = json.loads(
        make_bridge_info(
            embodiment="franka",
            n_dof=9,
            n_envs=1,
            genesis_version="1.2.3",
            env_hash="a" * 64,
            # required, not defaulted: a caller that forgot to wire the cfg
            # flag must fail loudly, not attest "off" while free-running
            step_without_reset=False,
            calibration=calibration,
            perception="L0",
            segmentation_ids={},
            sim_backend="metal",
            sim_engine="genesis",
        )
    )
    assert info == {
        "contract": "v0",
        "embodiment": "franka",
        "n_dof": 9,
        "n_envs": 1,
        "genesis_version": "1.2.3",
        "platform": info["platform"],  # host-dependent, non-empty
        "env_hash": "a" * 64,
        # ADR-25: bring-up mode leaves an attestation footprint in traces
        "step_without_reset": False,
        # BRG-8: nested, verbatim — the verifier compares it field-wise
        "calibration": calibration,
        # TC-9: the rung is attested in the TRACE, not only in the graph
        "perception": "L0",
        "segmentation_ids": {},
        "sim_backend": "metal",
        # ADR-67: the engine that realized the scene
        "sim_engine": "genesis",
    }
    assert info["platform"]
    assert info["calibration"]["calibration_version"] == 1


def test_bridge_config_from_env():
    """BRG-1, CON-5: bridge configuration, including the rollout-attested
    Genesis backend, comes from node environment variables with safe defaults."""
    cfg = parse_bridge_config({})
    assert (cfg.seed, cfg.embodiment, cfg.n_envs, cfg.sim_backend) == (0, "franka", 1, None)
    cfg = parse_bridge_config(
        {
            "AISLE_SEED": "7",
            "AISLE_EMBODIMENT": "so101",
            "AISLE_N_ENVS": "4",
            "AISLE_SIM_BACKEND": "cuda",
        }
    )
    assert (cfg.seed, cfg.embodiment, cfg.n_envs, cfg.sim_backend) == (7, "so101", 4, "cuda")

    with pytest.raises(ValueError, match="simulation backend"):
        parse_bridge_config({"AISLE_SIM_BACKEND": "auto"})
    # ADR-67: the engine is graph-declared like the rung; unknown names and
    # backends the engine cannot run are refused, never defaulted
    assert parse_bridge_config({}).sim_engine == "nexus"  # ADR-71 default
    assert parse_bridge_config({"AISLE_SIM_ENGINE": "nexus"}).sim_engine == "nexus"
    nexus_cfg = parse_bridge_config({"AISLE_SIM_ENGINE": "nexus", "AISLE_SIM_BACKEND": "webgpu"})
    assert (nexus_cfg.sim_engine, nexus_cfg.sim_backend) == ("nexus", "webgpu")
    with pytest.raises(ValueError, match="simulation engine"):
        parse_bridge_config({"AISLE_SIM_ENGINE": "bullet"})
    with pytest.raises(ValueError, match="simulation backend"):
        parse_bridge_config({"AISLE_SIM_ENGINE": "genesis", "AISLE_SIM_BACKEND": "webgpu"})


def test_step_without_reset_defaults_off():
    """CON-5/ADR-25 (issue #71): the bridge holds at sim step 0 until the
    first reset by DEFAULT — the pre-reset free-run that made two attested
    expert_s1 runs diverge is bring-up-only, opted into explicitly."""
    assert parse_bridge_config({}).step_without_reset is False
    assert parse_bridge_config({"AISLE_STEP_WITHOUT_RESET": "1"}).step_without_reset is True
    assert parse_bridge_config({"AISLE_STEP_WITHOUT_RESET": "0"}).step_without_reset is False


def test_mobile_rejects_batched_envs():
    """SPEC 210 MOB-1/ADR-13 (PR #14 review): the kinematic base is single-
    env per bridge; mobile with n_envs > 1 is rejected at startup rather
    than mislabel one global base_pose under every env. Fixed-base profiles
    and single-env mobile pass."""
    from aisle.nodes.dora_genesis import require_single_env_for_mobile

    with pytest.raises(ValueError, match="batched envs"):
        require_single_env_for_mobile("mobile", 4)
    require_single_env_for_mobile("mobile", 1)  # ok
    require_single_env_for_mobile("franka", 8)  # fixed-base batching unaffected


def test_non_integral_env_id_is_error():
    """BRG-5: fractional and boolean env_id values are rejected, never
    silently coerced into a route (0.7 must not become env 0)."""
    queue = CommandQueue(n_envs=2)
    for bad in (0.7, 1.0, True, "1"):
        with pytest.raises(ValueError, match="env_id"):
            queue.push("joint", bad, [0.1])


def test_reset_clock_is_injected():
    """CON-5: the bridge's reset timing uses an injected clock — the main
    entrypoint takes it as a parameter defaulting to time.perf_counter,
    never calling a wall clock ad hoc inside the loop."""
    import inspect
    import time as time_module

    from aisle.nodes.dora_genesis import main

    parameter = inspect.signature(main).parameters["clock"]
    assert parameter.default is time_module.perf_counter


def test_reset_quarantine_holds_then_releases():
    """BRG-4: after arm() the quarantine reports active for exactly ticks
    holds (one consumed per tick), then releases so normal command
    application resumes — the window that drops the ended episode's stale
    joint_cmds so they cannot drive the just-homed arm off home."""
    q = ResetQuarantine(3)
    assert q.hold() is False  # not armed: commands apply normally
    q.arm()
    assert [q.hold() for _ in range(4)] == [True, True, True, False]
    # re-arming restarts the full window (a second reset mid-window)
    q.arm()
    assert q.hold() is True
    q.arm()
    assert sum(q.hold() for _ in range(5)) == 3  # exactly `ticks` holds


def test_reset_quarantine_zero_ticks_never_holds():
    """A zero-tick quarantine (settle disabled) never quarantines."""
    q = ResetQuarantine(0)
    q.arm()
    assert q.hold() is False


class TestPerEnvQuarantine:
    """BRG-5 fleet mode: quarantine is PER ENV — a reset of env k must
    not freeze the other envs' episodes."""

    def test_single_env_arm_holds_only_that_env(self):
        q = ResetQuarantine(2, n_envs=3)
        q.arm(1)
        assert q.held(1) and not q.held(0) and not q.held(2)
        q.tick()
        assert q.held(1)
        q.tick()
        assert not q.held(1) and not q.any_held()

    def test_arm_all_covers_every_env(self):
        q = ResetQuarantine(1, n_envs=2)
        q.arm(None)
        assert q.held(0) and q.held(1)
        q.tick()
        assert not q.any_held()

    def test_legacy_hold_advances_and_reports(self):
        """The single-env path's contract is unchanged: hold() True for
        exactly `ticks` calls after arm()."""
        q = ResetQuarantine(3)
        q.arm()
        assert [q.hold() for _ in range(5)] == [True, True, True, False, False]


class TestPerEnvDrain:
    """BRG-5 fleet mode: a per-env reset drains only its own env's
    pending commands."""

    def test_env_scoped_drain_leaves_other_envs(self):
        from aisle.nodes.dora_genesis import CommandQueue as CommandCoalescer

        c = CommandCoalescer(3)
        c.push("joint_cmd", 0, "a")
        c.push("joint_cmd", 1, "b")
        c.push("gripper_cmd", 1, "c")
        c.push("joint_cmd", 2, "d")
        drained = c.drain(1)
        assert [(k, e, p) for k, e, p, _ in drained] == [
            ("joint_cmd", 1, "b"),
            ("gripper_cmd", 1, "c"),
        ]
        rest = c.drain()
        assert [(k, e, p) for k, e, p, _ in rest] == [
            ("joint_cmd", 0, "a"),
            ("joint_cmd", 2, "d"),
        ]


def test_perception_rung_from_env_defaults_l0():
    """TC-9: the rung comes from the graph node's env, defaults to L0, and is
    read case-insensitively. An UNRECOGNIZED rung refuses rather than falling
    back to L0 — a silent fallback would publish ground-truth pose to a graph
    that asked not to have it and report the result under the rung it typo'd."""
    assert parse_bridge_config({}).perception == "L0"
    assert parse_bridge_config({"AISLE_PERCEPTION": "l1"}).perception == "L1"
    assert parse_bridge_config({"AISLE_PERCEPTION": " L2 "}).perception == "L2"
    with pytest.raises(ValueError, match="unknown perception rung"):
        parse_bridge_config({"AISLE_PERCEPTION": "L3"})
    # PRESENT but empty is an UNKNOWN rung, not an absent one. A trailing
    # `or "L0"` mapped both onto the default and defeated this refusal — the
    # same blank-value hole VAL-8 had on the validator side. Only a MISSING key
    # defaults.
    # `AISLE_PERCEPTION:` with no value parses from YAML as None, and
    # `AISLE_PERCEPTION: ""` is the empty string. Both are a graph DECLARING a
    # rung, so both must refuse rather than inherit L0. Each narrower version of
    # this check let one more shape through: `or "L0"` let "" and "   " past,
    # `is None` let YAML null past. Only an ABSENT key defaults.
    for blank in ("", "   ", "\t", None):
        with pytest.raises(ValueError, match="unknown perception rung"):
            parse_bridge_config({"AISLE_PERCEPTION": blank})


def test_bridge_publishes_only_what_the_rung_permits():
    """TC-9: the bridge does not PUBLISH a topic the rung forbids — the other
    half of VAL-8, which only rejects a graph that consumes one. The validator
    can be bypassed (instrumented run copies, hand-edited graphs); a topic that
    is never on the wire cannot be consumed by accident."""
    l0 = rung_topic_rates("L0", is_mobile=False)
    assert "poses" in l0 and "seg_overhead" not in l0
    l1 = rung_topic_rates("L1", is_mobile=False)
    assert "poses" not in l1 and l1["seg_overhead"] == 15
    l2 = rung_topic_rates("L2", is_mobile=False)
    assert "poses" not in l2 and "seg_overhead" not in l2
    # the rung is orthogonal to embodiment: mobile still adds its base topics
    assert {"base_pose", "base_scan"} <= set(rung_topic_rates("L1", is_mobile=True))
    # and the rung never disturbs the topics it says nothing about
    for rung in ("L0", "L1", "L2"):
        assert {"rgb_overhead", "depth_overhead", "joint_state", "oracle_state"} <= set(
            rung_topic_rates(rung, is_mobile=False)
        )


def test_segmentation_and_depth_are_co_scheduled_on_every_tick():
    """TC-9, CON-5: `seg_overhead` and `depth_overhead` MUST be due on the
    same ticks, because an L1 estimate masks the segmentation and indexes the
    depth. If they could fire independently the estimator would either pair
    across ticks (measuring a scene that never existed) or wait forever."""
    scheduler = RateScheduler(rung_topic_rates("L1", is_mobile=False), dt=0.01)
    seg_ticks, depth_ticks = [], []
    for tick in range(1000):
        due = scheduler.due()
        if "seg_overhead" in due:
            seg_ticks.append(tick)
        if "depth_overhead" in due:
            depth_ticks.append(tick)
    assert seg_ticks == depth_ticks
    assert len(seg_ticks) == 150  # 15 Hz over 10 s (TC-4)


def test_segmentation_id_map_uses_the_scene_map_not_entity_indices():
    """TC-9: the ids in `seg_overhead` are genesis's own segmentation map, NOT
    entity indices. These numbers are MEASURED from the desk scene (seed 3):
    entity 5 is amoxicillin and its seg id is 16. A consumer that masked on
    the entity index would select other geometry — observed as robot links
    whose pixel counts were identical across different object layouts."""
    idx_dict = {0: -1, 1: (0, 0), 5: (4, 4), 6: (4, 5), 16: (5, 16), 17: (6, 17)}
    ids = segmentation_id_map(idx_dict, {"amoxicillin": 5, "ibuprofen": 6})
    assert ids == {"amoxicillin": [16], "ibuprofen": [17]}
    # a multi-link entity contributes every one of its ids, sorted (CON-5)
    assert segmentation_id_map({7: (4, 7), 5: (4, 4)}, {"robot": 4}) == {"robot": [5, 7]}
    # an entity with no rendered geometry maps to an empty list, never a guess
    assert segmentation_id_map(idx_dict, {"tray": 99}) == {"tray": []}


def test_reset_path_publishes_only_what_the_rung_permits():
    """TC-9: the reset path publishes DIRECTLY, off the scheduler, so gating
    only the scheduler is not enough. `publish` gates on the rung's topic set
    itself, which is what makes every direct call safe by construction.

    Before this, `publish("poses")` after each reset put ground-truth pose on
    an L1 wire once per episode — at the freshest possible moment, and into the
    trace the recorder keeps. `rung_topic_rates` was correct and the bug was
    entirely in the path that bypassed it, which is why the rung test above
    passed while the bridge leaked."""
    from aisle.nodes.dora_genesis import RESET_PUBLISH

    # oracle_state is verifier-only at every rung: VAL-6 is the rule, ADR-28
    # records why the ladder does not widen to it
    assert "oracle_state" in RESET_PUBLISH
    assert "oracle_state" in rung_topic_rates("L1", is_mobile=False)
    # `poses` is published on the reset path at L0 and must be filtered out
    # above it — the gate is membership in the rung's own topic set
    assert "poses" in RESET_PUBLISH
    permitted = {rung: set(rung_topic_rates(rung, is_mobile=False)) for rung in ("L0", "L1", "L2")}
    assert {t for t in RESET_PUBLISH if t in permitted["L0"]} == {"oracle_state", "poses"}
    assert {t for t in RESET_PUBLISH if t in permitted["L1"]} == {"oracle_state"}
    assert {t for t in RESET_PUBLISH if t in permitted["L2"]} == {"oracle_state"}


def test_segmentation_id_map_handles_every_genesis_segmentation_level():
    """TC-9: the seg_key shape depends on `VisOptions.segmentation_level`, read
    from genesis's own construction — `geom` -> (entity, link, geom), `link`
    (the default) -> (entity, link), `entity` -> a BARE int. The entity index
    is first in every shape, so all three resolve.

    An earlier version required a tuple, so at `segmentation_level="entity"`
    every object mapped to an empty id list, the L1 estimator refused every
    pose, and the episode died on a timeout with one stderr line as the clue.
    A silent downgrade to "refuse everything" is the worst of the three."""
    entity_idx = {"amoxicillin": 5}
    # link level (genesis default), as measured on the desk scene
    assert segmentation_id_map({0: -1, 16: (5, 16)}, entity_idx) == {"amoxicillin": [16]}
    # geom level
    assert segmentation_id_map({0: -1, 16: (5, 16, 2)}, entity_idx) == {"amoxicillin": [16]}
    # entity level: a bare int, not a tuple
    assert segmentation_id_map({0: -1, 16: 5}, entity_idx) == {"amoxicillin": [16]}
    # background (-1) is never an entity, at any level
    assert segmentation_id_map({0: -1}, {"bg": -1}) == {"bg": []}
    # numpy integers are ints too: genesis hands back its own scalar types
    assert segmentation_id_map({16: np.int64(5)}, entity_idx) == {"amoxicillin": [16]}


def test_bridge_info_carries_the_l1_id_map():
    """TC-9/BRG-6: at L1 the id map travels in bridge_info, so a consumer never
    derives genesis's numbering. Pinned separately from the L0 shape test
    because the L1 path through the JSON was otherwise unexercised."""
    info = json.loads(
        make_bridge_info(
            embodiment="franka",
            n_dof=9,
            n_envs=1,
            genesis_version="1.2.3",
            env_hash="a" * 64,
            step_without_reset=False,
            calibration={"calibration_version": 1},
            perception="L1",
            segmentation_ids={"amoxicillin": [16], "ibuprofen": [17]},
            sim_backend="cuda",
            sim_engine="genesis",
        )
    )
    assert info["perception"] == "L1"
    assert info["segmentation_ids"] == {"amoxicillin": [16], "ibuprofen": [17]}
    assert info["sim_backend"] == "cuda"


def test_publish_gate_blocks_forbidden_topics_including_direct_calls():
    """TC-9: the gate `publish` actually consults, bound directly.

    The first version of this test re-derived the gate from RESET_PUBLISH and
    rung_topic_rates and stayed GREEN when the guard inside publish() was
    deleted — it asserted set arithmetic over two module constants, not
    behaviour. This binds `may_publish`, the predicate publish() calls, so
    removing the guard fails here."""
    l0, l1 = rung_topic_rates("L0", is_mobile=False), rung_topic_rates("L1", is_mobile=False)
    # the reset path's direct publishes, as the reset path computes them. This is
    # the leak that actually happened, so it is asserted on the function the
    # bridge calls rather than on set arithmetic over two constants.
    assert reset_publish_topics(l0) == ("oracle_state", "poses")
    assert reset_publish_topics(l1) == ("oracle_state",)
    assert reset_publish_topics(rung_topic_rates("L2", is_mobile=False)) == ("oracle_state",)
    assert may_publish("poses", l0) is True
    assert may_publish("poses", l1) is False
    # oracle_state survives every rung (VAL-6 keeps it verifier-only; ADR-28)
    assert may_publish("oracle_state", l0) is True
    assert may_publish("oracle_state", l1) is True
    # and the L1-only topic is gated the other way
    assert may_publish("seg_overhead", l1) is True
    assert may_publish("seg_overhead", l0) is False


def test_publish_is_wired_to_the_gate_not_to_an_inline_check():
    """TC-9: `publish` must route through `may_publish`, so the predicate the
    test above pins is the one the bridge uses. Checked structurally because
    publish() is a closure inside main() and needs dora + genesis to call."""
    import inspect

    from aisle.nodes import dora_genesis

    # publish() is a closure inside main() needing dora + genesis to call, so
    # these are STRUCTURAL checks. Matched on function names only: an assertion
    # carrying the argument list would break on a harmless rename.
    source = inspect.getsource(dora_genesis.main)
    assert "may_publish(" in source, "publish() bypasses the TC-9 gate"
    # the reset path must go through the FILTER, not the raw tuple. Replacing
    # `reset_publish_topics(topic_rates)` with `RESET_PUBLISH` is precisely how
    # the original leak looked, and the behavioural test above cannot see a call
    # site that stopped calling it.
    assert "for topic in reset_publish_topics(topic_rates):" in source
    assert "for topic in RESET_PUBLISH:" not in source
    # the extracted config/id-map refusals must still be CALLED. Their own
    # behavioural tests pass whether or not main() invokes them, so removing a
    # call site is the regression those tests cannot see.
    assert "require_supported_perception(cfg)" in source
    assert "require_usable_segmentation_ids(segmentation_ids, cfg.perception)" in source

    # WHAT THIS DOES NOT COVER, stated rather than implied: INVERTING the guard
    # inside publish() (`if may_publish(...): return`) passes every unit test
    # here, because reaching that line needs the dora runtime. It is not a
    # silent failure though -- every topic is in `topic_rates`, so an inverted
    # guard drops ALL of them and any graph- or sim-marked test fails on the
    # first missing observation. The leak that actually happened is prevented by
    # `reset_publish_topics`, which IS bound behaviourally above.


def test_bridge_info_requires_the_rung_rather_than_defaulting_it():
    """TC-9/BRG-8/ADR-67: `perception`, `segmentation_ids` and `sim_engine` are
    REQUIRED arguments. A defaulted rung would attest "L0" in the trace for a
    run that executed L1 — the recorded-vs-actual divergence the rung refusal
    and the env scrub exist to prevent, and one no test can catch because the
    default is a valid value. A defaulted engine attests "genesis" for a run
    that executed on Nexus for exactly the same reason. Same discipline the
    docstring already argues for calibration."""
    import inspect

    sig = inspect.signature(make_bridge_info)
    for name in ("perception", "segmentation_ids", "sim_engine"):
        assert sig.parameters[name].default is inspect.Parameter.empty, name


def test_seg_and_depth_publish_order_is_the_one_the_consumer_needs():
    """TC-9: the L1 estimator buffers only the DEPTH side and drops a seg frame
    whose partner has not arrived, so depth must be published before seg on a
    shared tick. That currently holds because TOPIC_RATES lists depth_overhead
    first and RateScheduler preserves insertion order — an incidental property
    worth an assertion, since flipping the two would make the estimator publish
    no pose at all while every topic looked healthy."""
    order = list(rung_topic_rates("L1", is_mobile=False))
    assert order.index("depth_overhead") < order.index("seg_overhead")


def test_store_scene_refuses_the_l1_rung():
    """TC-9: the store keys graspables by item id (`slot#k`) while the L1
    estimator asks by med name, so an L1 store run would announce a well-formed
    id map and then refuse every pose, dying on a timeout that scores as a
    POLICY failure. Refused at config time, loudly, instead."""
    from aisle.nodes.dora_genesis import require_supported_perception

    store_l1 = BridgeConfig(seed=0, embodiment="mobile", n_envs=1, scene="store", perception="L1")
    with pytest.raises(ValueError, match="not supported for the store scene"):
        require_supported_perception(store_l1)
    # the desk scene at L1 is the supported case, and the store at L0 is
    # untouched — the refusal is about the id-map namespace, not about L1
    require_supported_perception(
        BridgeConfig(seed=0, embodiment="franka", n_envs=1, perception="L1")
    )
    require_supported_perception(
        BridgeConfig(seed=0, embodiment="mobile", n_envs=1, scene="store", perception="L0")
    )
    # L2 is SERVED since detected-pose landed (idea I7): RGB alone supplies
    # identity and same-stamp sensor depth supplies metric geometry. The desk
    # scene must pass. The STORE at L2 stays refused for the same
    # namespace reason as store+L1: the detector vocabulary is the desk med
    # list, so every estimate would refuse.
    require_supported_perception(
        BridgeConfig(seed=0, embodiment="franka", n_envs=1, perception="L2")
    )
    with pytest.raises(ValueError, match="rung L2 is not supported for the store"):
        require_supported_perception(
            BridgeConfig(seed=0, embodiment="mobile", n_envs=1, scene="store", perception="L2")
        )


def test_l1_refuses_an_unusable_segmentation_id_map():
    """TC-9: at L1 the id map is load-bearing, so BOTH an empty and a partial
    map are refused at startup. The partial-only check was vacuously satisfied
    by an empty map, so bridge_info announced `"segmentation_ids": {}` at L1 —
    attested-looking and unusable, and every estimate would then refuse one
    stderr line at a time while the episode died on a timeout."""
    require_usable_segmentation_ids({"amoxicillin": [16]}, "L1")  # usable
    with pytest.raises(ValueError, match="no segmentation ids resolved"):
        require_usable_segmentation_ids({}, "L1")
    with pytest.raises(ValueError, match="amoxicillin"):
        require_usable_segmentation_ids({"amoxicillin": [], "ibuprofen": [17]}, "L1")
    # below L1 there is no map to require: L0 publishes ground truth instead
    require_usable_segmentation_ids({}, "L0")


# -- ADR-67: timing sidecar ---------------------------------------------------


def test_step_timer_flushes_windows_with_engine_gpu_time(tmp_path):
    """ADR-67 / BRG-6 (recorded-vs-actual): the bridge's timing sidecar
    carries per-window step wall time, render wall time, sim seconds and the
    engine's GPU time when reported, so a run's performance is comparable
    across engines without a new topic (SPEC 010 untouched)."""
    from pathlib import Path

    from aisle.nodes.dora_genesis import StepTimer, timing_sidecar_path

    path = tmp_path / "sim_timing.jsonl"
    timer = StepTimer("nexus", path, flush_every=3, clock=lambda: 0.0)
    timer.record_step(0.004, 10_000_000, {"gpu_ms": 3.0}, tick_s=0.010)
    timer.record_render(0.002)
    timer.record_step(0.006, 20_000_000, None, tick_s=0.011)
    timer.record_step(0.008, 30_000_000, {"gpu_ms": 5.0})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["engine"] == "nexus" and row["steps"] == 3 and row["frames"] == 1
    assert row["step_ms_mean"] == pytest.approx(6.0) and row["step_ms_max"] == pytest.approx(8.0)
    # the third step gave no tick time, so its step time stands in for it
    assert row["tick_ms_mean"] == pytest.approx((10.0 + 11.0 + 8.0) / 3)
    assert row["render_ms_mean"] == pytest.approx(2.0)
    assert row["gpu_ms_mean"] == pytest.approx(4.0)  # over the steps that reported gpu time
    assert row["sim_window_s"] == pytest.approx(0.02) and row["sim_time_s"] == pytest.approx(0.03)
    # a window without renders or gpu reports leaves those null, never zero
    timer.record_step(0.001, 40_000_000)
    assert timer.flush(40_000_000)["render_ms_mean"] is None
    assert timer.rows[-1]["gpu_ms_mean"] is None
    # no results path (hand-launched graph): no sidecar
    assert timing_sidecar_path({}) is None
    assert timing_sidecar_path({"AISLE_RESULTS": "/runs/x/episodes.jsonl"}) == Path(
        "/runs/x/sim_timing.jsonl"
    )


def test_debug_view_is_off_by_default_and_parses_presets():
    """ADR-67 operator tooling: AISLE_DEBUG_VIEW adds a camera that feeds
    debug_view.mp4 only (no topic, SPEC 010 untouched): unset means none,
    `side` looks at the shelf front from the tray side, an explicit
    `eye;lookat` pair is honored, and a malformed value is refused."""
    from aisle.nodes.dora_genesis import parse_debug_view
    from aisle.scenes.pharmacy import load_physics, resolve_layout

    layout = resolve_layout(load_physics(), "franka")
    assert parse_debug_view({}, layout) is None
    assert parse_debug_view({"AISLE_DEBUG_VIEW": ""}, layout) is None
    eye, lookat = parse_debug_view({"AISLE_DEBUG_VIEW": "side"}, layout)
    shelf_front_x = layout["shelf"]["pos"][0] - layout["shelf"]["level_size"][0] / 2
    assert eye[0] < shelf_front_x < lookat[0]  # outside the shelf, looking into it
    assert eye[1] < layout["tray"]["pos"][1]  # from beyond the tray side
    assert eye[2] > lookat[2] > 0.0
    assert parse_debug_view({"AISLE_DEBUG_VIEW": "1,-1,0.5;0.5,0,0.3"}, layout) == (
        [1.0, -1.0, 0.5],
        [0.5, 0.0, 0.3],
    )
    with pytest.raises(ValueError, match="AISLE_DEBUG_VIEW"):
        parse_debug_view({"AISLE_DEBUG_VIEW": "sideways"}, layout)
    with pytest.raises(ValueError, match="AISLE_DEBUG_VIEW"):
        parse_debug_view({"AISLE_DEBUG_VIEW": "1,2;3,4"}, layout)
