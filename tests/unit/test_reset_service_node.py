"""Node-level tests for the reset service's reply contract (SPEC 040 RST-1/2).

Issue #192 moved every episode-state consumer off the bridge and onto this
node's `reset_done` output, on the premise that the service answers on EVERY
route: it relays the bridge on teleport, replies directly on behavioral
success, and forwards a teleport on behavioral exhaustion.

That premise had no test. `route_reset` (the pure dispatcher) was unit-tested
and the graph wiring is checked statically, but nothing exercised the node
itself — so "the service always replies" was an assertion in a commit
message. If it is ever false, every consumer in every graph now waits
forever on a boundary that never comes, which is a strictly worse failure
than the one #192 fixed.

dora is faked (CON-12 keeps the import inside `main()`), and the behavioral
runtime is stubbed so no model load or sim is needed.
"""

import sys
import types

import numpy as np
import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit

TELEPORT, BEHAVIORAL = 0, 1


class FakeNode:
    def __init__(self, events):
        self._events = events
        self.sent: list[tuple[str, list, dict]] = []

    def __iter__(self):
        return iter(self._events)

    def send_output(self, topic, value, metadata=None):
        self.sent.append((topic, list(value.to_pylist()), metadata or {}))


class StubRuntime:
    """The BehavioralRuntime surface `service.main()` actually touches.

    `succeed_after` joint_state events flips the outcome, mirroring the real
    runtime settling asynchronously rather than inside `start()`.
    """

    def __init__(
        self, *, succeed_after: int = 1, outcome: str = "success", start_raises=None, **_kwargs
    ):
        self.start_raises = start_raises
        self.bridge_info_raises = _kwargs.pop("bridge_info_raises", None)
        self.calibration = None
        self.succeed_after = succeed_after
        self.final_outcome = outcome
        self.seed = 0
        self.request_meta: dict = {}
        self.attempts = 0
        self.outcome = None
        self._started = False
        self._ticks = 0

    @property
    def active(self) -> bool:
        return self._started

    def start(self, seed: int, request_meta: dict) -> None:
        if self.start_raises is not None:
            # partially started, THEN failed — the shape that leaves a live
            # half-initialised runtime behind if the service keeps it
            self._started = True
            raise self.start_raises
        self.seed = seed
        self.request_meta = dict(request_meta)
        self.attempts = 1
        self.outcome = None
        self._started = True
        self._ticks = 0

    def on_bridge_info(self, info):
        if getattr(self, "bridge_info_raises", None) is not None:
            raise self.bridge_info_raises
        self.calibration = info.get("calibration")

    def on_rgb(self, rgb):  # pragma: no cover
        pass

    def on_depth(self, depth):  # pragma: no cover
        pass

    def on_joint_state(self, qpos):
        self._ticks += 1
        if self._ticks >= self.succeed_after:
            self.outcome = self.final_outcome
        return None, None


def _inp(topic, value, metadata=None):
    return {"type": "INPUT", "id": topic, "value": value, "metadata": metadata or {}}


def reset_request(seed: int, mode: int, request_id: str = "req-1", **extra) -> dict:
    return _inp(
        "reset",
        pa.array(np.array([seed, mode], dtype=np.uint32)),
        {"request_id": request_id, **extra},
    )


def bridge_reply(**metadata) -> dict:
    """What dora-genesis sends back after a teleport."""
    return _inp("reset_done", pa.array(np.array([1], dtype=np.uint32)), metadata)


def joint_state(sim_ns: int = 1_000_000) -> dict:
    return _inp(
        "joint_state",
        pa.array(np.zeros(9, dtype=np.float32)),
        {"sim_time_ns": sim_ns},
    )


def run_service(
    events, monkeypatch, *, runtime_load_ticks: int = 0, build_raises=None, **runtime_kwargs
) -> FakeNode:
    node = FakeNode(events)
    fake_dora = types.ModuleType("dora")
    fake_dora.Node = lambda: node
    monkeypatch.setitem(sys.modules, "dora", fake_dora)

    import aisle.reset.runtime as runtime_mod
    import aisle.verifier.models as models_mod
    from aisle.scenes import pharmacy

    # CON-5: injected so t_reset_ms is reproducible. Each call advances 0.25 s,
    # so an elapsed measurement is a deterministic multiple rather than a race.
    ticks = iter(range(10_000))

    def clock():
        return next(ticks) * 0.25

    builds = []

    def build_runtime(**kw):
        builds.append(1)
        # `runtime_load_ticks` stands in for the ~2 s Owlv2 load real
        # construction pays. Without it the fake builds instantly and NO test
        # can tell whether the reset clock starts above or below get_runtime()
        # — which is exactly how that bug shipped (round-2 review).
        for _ in range(runtime_load_ticks):
            next(ticks)
        if build_raises is not None:
            # what a missing/corrupt model cache looks like from here: the
            # real construction does load_physics/load_meds/load_pinned, and
            # load_pinned is an Owlv2 from_pretrained (issue #206)
            raise build_raises
        return StubRuntime(**kw, **runtime_kwargs)

    monkeypatch.setattr(runtime_mod, "BehavioralRuntime", build_runtime)
    monkeypatch.setattr(models_mod, "load_pinned", lambda *_a, **_k: object())
    monkeypatch.setattr(pharmacy, "load_meds", lambda *_a, **_k: {})
    monkeypatch.setattr(pharmacy, "resolve_layout", lambda *_a, **_k: {})

    from aisle.reset.service import main

    main(clock=clock)
    node.builds = len(builds)
    return node


def replies(node: FakeNode) -> list[dict]:
    return [m for topic, _, m in node.sent if topic == "reset_done"]


def refusals(node: FakeNode) -> list[dict]:
    """ADR-34: refusals answer on their own topic, never on the boundary."""
    return [m for topic, _, m in node.sent if topic == "reset_refused"]


def forwards(node: FakeNode) -> list[list]:
    return [v for topic, v, _ in node.sent if topic == "bridge_reset"]


def test_teleport_is_forwarded_and_its_reply_relayed(monkeypatch):
    """RST-1: the teleport route reaches the bridge, and the bridge's reply
    comes back out on the SERVICE's output — which is the only edge
    episode-state consumers listen to since issue #192."""
    node = run_service(
        [reset_request(7, TELEPORT), bridge_reply(sim_time_ns=555, env_id=2, request_id="req-1")],
        monkeypatch,
    )
    assert forwards(node) == [[7, TELEPORT]]
    assert len(replies(node)) == 1
    relayed = replies(node)[0]
    # the bridge's own routing/timing keys survive the relay
    assert relayed["sim_time_ns"] == 555
    assert relayed["env_id"] == 2


def test_behavioral_success_replies_without_touching_the_bridge(monkeypatch):
    """RST-2 + issue #192, the route that motivated the whole fix: a
    successful behavioral reset never forwards to the bridge, so
    `dora-genesis/reset_done` cannot fire. The service must answer itself,
    or every consumer waits forever."""
    node = run_service(
        [reset_request(4, BEHAVIORAL, env_id=3), joint_state(sim_ns=9_000)],
        monkeypatch,
    )
    assert forwards(node) == [], "behavioral success must not teleport"
    assert len(replies(node)) == 1, "the service did not answer a behavioral reset"
    reply = replies(node)[0]
    assert reply["env_id"] == 3, "the guard slices its per-env boundary on this (BRG-5)"
    assert reply["sim_time_ns"] == 9_000, "post-motion sim time (BRG-4 parity)"
    assert reply["fallback"] is False


def test_behavioral_exhaustion_falls_back_and_still_answers_exactly_once(monkeypatch):
    """The third route. Exhaustion forwards a TELEPORT and lets the bridge's
    reply be relayed — so there must be exactly ONE reset_done, not two.
    Consumers advance an episode epoch on each reply (issue #179), and a
    double-fire would desync nav's epoch from its producer's and make nav
    refuse every goal of the new episode."""
    node = run_service(
        [
            reset_request(5, BEHAVIORAL),
            joint_state(),
            bridge_reply(sim_time_ns=42, request_id="req-1"),
        ],
        monkeypatch,
        outcome="exhausted",
    )
    assert forwards(node) == [[5, TELEPORT]], "exhaustion must fall back to a teleport"
    assert len(replies(node)) == 1, f"expected exactly one boundary, got {replies(node)}"


def test_a_refused_request_still_answers_on_its_own_topic(monkeypatch):
    """ADR-8/TC-6/ADR-34: a malformed payload is refused loudly — it must
    still produce a reply (a silent refusal hangs the requester), and since
    issue #195 that reply leaves on `reset_refused`, NOT on the boundary."""
    node = run_service(
        [_inp("reset", pa.array(np.array([1, 2, 3], dtype=np.uint32)), {"request_id": "req-2"})],
        monkeypatch,
    )
    assert forwards(node) == []
    assert len(refusals(node)) == 1
    assert "error" in refusals(node)[0]
    assert refusals(node)[0]["request_id"] == "req-2"  # TC-6 correlation survives the move
    assert replies(node) == [], "a refusal reached the episode boundary topic (ADR-34)"


def test_the_boundary_topic_never_carries_a_refusal(monkeypatch):
    """The single guarantee ADR-34 buys, pinned on its own.

    `budget_guard` and `label_reader` each used to filter `metadata["error"]`
    off `reset_done`; ADR-34 DELETED both filters, so from here on the
    guard re-references velocity/hold state to home on every reply it hears,
    and the OCR session clears its read barrier on every one. That is only
    safe while this holds. Every refusable shape in one test, because the
    cost of a leak is silent: a clamp against a false origin permits a
    larger real jump than the limit allows."""
    for payload, meta in (
        (pa.array(np.array([1, 2, 3], dtype=np.uint32)), {"request_id": "wrong-shape"}),
        (pa.array(["not-a-number"]), {"request_id": "non-numeric"}),
        (pa.array(np.array([4, 9], dtype=np.uint32)), {"request_id": "unknown-mode"}),
    ):
        node = run_service([_inp("reset", payload, meta)], monkeypatch)
        assert len(refusals(node)) == 1, (payload, refusals(node))
        assert replies(node) == [], f"{meta['request_id']} refusal rode the boundary topic"
        assert forwards(node) == [], f"{meta['request_id']} was forwarded to the bridge"


def test_the_episode_epoch_is_monotonic_across_mixed_routes(monkeypatch):
    """CON-5 + issue #179: consumers use `reset_done`'s seq as the episode
    epoch, and nav refuses goals stamped with a stale one. Because every
    reply now comes from this single node, the seq must advance by one per
    boundary regardless of which route produced it — a per-route counter
    would restart and make epochs ambiguous."""
    node = run_service(
        [
            reset_request(1, TELEPORT),
            bridge_reply(sim_time_ns=1),
            reset_request(2, BEHAVIORAL),
            joint_state(sim_ns=2),
            reset_request(3, TELEPORT),
            bridge_reply(sim_time_ns=3),
        ],
        monkeypatch,
    )
    seqs = [m["seq"] for m in replies(node)]
    assert seqs == [1, 2, 3], f"episode epochs are not monotonic across routes: {seqs}"


def test_a_refusal_does_not_advance_the_episode_epoch(monkeypatch):
    """CON-5 + issue #179, the case the sibling test cannot see: a refusal
    is not a boundary, so it must not consume a boundary sequence number.

    While refusals shared `reset_done` they also shared its counter, so a
    refused request between two real resets made the epoch jump 1 -> 3.
    Consumers read that seq as the episode epoch and nav REFUSES goals
    stamped with a stale one, so the gap is not cosmetic. ADR-34 gives
    `reset_refused` its own sequence; this pins that it did."""
    node = run_service(
        [
            reset_request(1, TELEPORT),
            bridge_reply(sim_time_ns=1),
            _inp("reset", pa.array(np.array([9, 9, 9], dtype=np.uint32)), {"request_id": "bad1"}),
            reset_request(2, TELEPORT),
            bridge_reply(sim_time_ns=2),
            _inp("reset", pa.array(np.array([8, 8, 8], dtype=np.uint32)), {"request_id": "bad2"}),
        ],
        monkeypatch,
    )
    assert [m["seq"] for m in replies(node)] == [1, 2], (
        f"a refusal consumed an episode epoch: {[m['seq'] for m in replies(node)]}"
    )
    # TWO refusals: with one, `seq_refused = 1` (a constant) is
    # indistinguishable from a counter, and TC-2 wants per-topic MONOTONIC,
    # not per-topic constant (round-2 review).
    assert [m["seq"] for m in refusals(node)] == [1, 2], (
        f"reset_refused's sequence does not advance: {[m['seq'] for m in refusals(node)]}"
    )


def test_a_request_without_request_id_is_dropped_with_no_reply(monkeypatch):
    """TC-6: with no request_id there is nothing to correlate a reply TO, so
    the service drops it. Pinned deliberately — it is the ONE path that
    answers nothing, and after #192 that means every consumer misses this
    boundary. Only `rollout-client` issues resets and `make_sender` always
    stamps a request_id, so it is unreachable in the real graphs; this test
    exists so the assumption is written down rather than assumed."""
    node = run_service(
        [_inp("reset", pa.array(np.array([1, TELEPORT], dtype=np.uint32)), {})], monkeypatch
    )
    assert replies(node) == []
    assert forwards(node) == []


def _run_guard(events, monkeypatch) -> FakeNode:
    """Drive budget_guard.main() over `events` on the mobile profile."""
    node = FakeNode(events)
    fake_dora = types.ModuleType("dora")
    fake_dora.Node = lambda: node
    monkeypatch.setitem(sys.modules, "dora", fake_dora)
    monkeypatch.setenv("AISLE_EMBODIMENT", "mobile")
    monkeypatch.setenv("AISLE_SCENE", "store")

    from aisle.nodes.budget_guard import main as guard_main

    guard_main(clock=lambda: 100.0)
    return node


def _base_cmd(v: float) -> dict:
    return _inp("base_cmd", pa.array(np.array([v, 0.0], dtype=np.float32)), {"env_id": 0})


def _boundary(**metadata) -> dict:
    return _inp("reset_done", pa.array(np.array([1], dtype=np.uint32)), {"env_id": 0, **metadata})


def test_a_real_boundary_retracts_a_latched_base_command(monkeypatch):
    """The control for the test below: on a genuine boundary the guard
    emits an explicit zero so a pre-reset command still in flight to the
    bridge cannot re-latch unwatched (MOB-3, PR #156 review). This is the
    observable that tells the two branches apart."""
    node = _run_guard([_base_cmd(0.4), _boundary()], monkeypatch)
    safes = [v for topic, v, _ in node.sent if topic == "base_cmd_safe"]
    assert safes and safes[0][0] > 0.0, f"the base command never latched: {safes}"
    assert safes[-1] == [0.0, 0.0], f"a real boundary did not retract the latch: {safes}"


def test_the_guard_now_trusts_every_reply_on_the_boundary_topic(monkeypatch):
    """ADR-34 (issue #195) replaced the guard's `metadata["error"]` filter
    with a structural guarantee, and this pins the consequence honestly: the
    guard no longer inspects reply metadata at all, so a reply carrying an
    `error` key IS treated as a boundary here.

    That is safe ONLY because refusals no longer ride this topic — the
    service publishes them on `reset_refused`, which
    `test_the_boundary_topic_never_carries_a_refusal` pins at the node and
    `test_refusals_ride_their_own_topic_and_reach_only_the_requester` pins
    in every graph. If either of those goes red, the hazard BG-2/ADR-8
    described is live again: velocity and hold state re-referenced to a home
    the robot is not at, clamping the next command against a false origin.
    Read this test with those two, never alone."""
    node = _run_guard([_base_cmd(0.4), _boundary(error="stale key from somewhere")], monkeypatch)
    safes = [v for topic, v, _ in node.sent if topic == "base_cmd_safe"]
    assert safes and safes[0][0] > 0.0, f"the base command never latched: {safes}"
    assert safes[-1] == [0.0, 0.0], (
        f"the guard still branches on reply metadata; ADR-34 deleted that filter: {safes}"
    )


def test_a_non_numeric_payload_is_refused_not_fatal(monkeypatch):
    """BG-3/TC-6 (issue #192 review): the service is now the single boundary
    authority for every consumer in every graph, and dora does not restart
    nodes — so an unparseable request must produce a refusal reply, never an
    exception out of the event loop. `except ValueError` alone let a
    non-numeric payload through, because that raises TypeError."""
    node = run_service(
        [_inp("reset", pa.array(["not-a-number"]), {"request_id": "req-3"})],
        monkeypatch,
    )
    assert len(refusals(node)) == 1, "a malformed request killed the boundary authority"
    assert "error" in refusals(node)[0]
    assert forwards(node) == []


def test_the_behavioral_reply_is_tc6_complete(monkeypatch):
    """TC-6 (issue #194): the reply carries `seed`, `mode`, `t_reset_ms`. The
    behavioral-success route carried NONE of the three — and it is the route
    that can take three real motion attempts, so it is exactly where RST-1's
    <2 s budget most needs to be auditable. `t_reset_ms` measured from the
    request, not from a teleport that never happened."""
    node = run_service(
        [reset_request(9, BEHAVIORAL), joint_state(sim_ns=7_000)],
        monkeypatch,
    )
    assert len(replies(node)) == 1
    reply = replies(node)[0]
    assert reply["seed"] == 9
    assert reply["mode"] == BEHAVIORAL
    # EXACT, not >=: the injected clock advances 0.25 s per read, so one read
    # separates this request from its reply. `>= 250` was satisfied by any two
    # distinct reads and so pinned only "a clock was read twice" — it survived
    # moving the anchor out of the request branch entirely (round-2 review).
    assert reply["t_reset_ms"] == 250, reply["t_reset_ms"]


def test_t_reset_ms_measures_THIS_request_not_the_node_lifetime(monkeypatch):
    """CON-5/RST-1: the anchor is per-request. A node-lifetime anchor passes
    a single-request test and then reports cumulative time since startup on
    every later reset — an unbounded OVERCOUNT of exactly the <2 s budget the
    key exists to audit. Two requests in one stream is the observable that
    tells the two apart: both must read one tick, not 250 then 750."""
    node = run_service(
        [
            reset_request(1, BEHAVIORAL, request_id="a"),
            joint_state(sim_ns=1_000),
            reset_request(2, BEHAVIORAL, request_id="b"),
            joint_state(sim_ns=2_000),
        ],
        monkeypatch,
    )
    assert [r["request_id"] for r in replies(node)] == ["a", "b"]
    assert [r["t_reset_ms"] for r in replies(node)] == [250, 250], replies(node)


def test_t_reset_ms_includes_the_model_load_the_first_request_pays(monkeypatch):
    """RST-1 (round-2 review): the first behavioral request builds the
    runtime, and building it loads Owlv2 — ~2 s against a <2 s budget. The
    `bridge_info` pre-warm moves that cost out of the request only when
    bridge_info arrives first, which nothing enforces. Starting the clock
    below `get_runtime()` would hide the single largest component of the
    number, so the reset that breaches RST-1 would report as compliant.

    Here the load costs one clock tick, so an honest measurement reads TWO:
    the load plus the attempt."""
    node = run_service(
        [reset_request(3, BEHAVIORAL), joint_state()],
        monkeypatch,
        runtime_load_ticks=1,
    )
    assert replies(node)[0]["t_reset_ms"] == 500, replies(node)[0]


def test_the_two_pre_existing_routes_still_report_the_timing_key(monkeypatch):
    """A regression control for the sibling behavioral assertions, NOT
    coverage of the fix: both routes here are untouched by issue #194 — the
    teleport relay copies the bridge's metadata verbatim and the refusal's 0
    comes from `refusal_reply_metadata`. Named for what it pins, because an
    earlier revision claimed "every route" while omitting both behavioral
    routes, and the exhaustion route still reports only the fallback
    teleport's duration (open on #194)."""
    relay = run_service(
        [reset_request(1, TELEPORT), bridge_reply(sim_time_ns=1, t_reset_ms=1234, seed=1, mode=0)],
        monkeypatch,
    )
    assert replies(relay)[0]["t_reset_ms"] == 1234

    refused = run_service(
        [_inp("reset", pa.array(np.array([1, 2, 3], dtype=np.uint32)), {"request_id": "r"})],
        monkeypatch,
    )
    assert len(refusals(refused)) == 1
    assert refusals(refused)[0]["t_reset_ms"] == 0


def test_the_fallback_reply_carries_the_behavioral_audit_trail(monkeypatch):
    """RST-2 (issue #205): "fall back to teleport with `fallback:true` in
    reply metadata". On the wire it was not there.

    `fallback_teleport` attaches `fallback`/`behavioral_attempts` to the
    FORWARD, and the bridge builds its reply metadata fresh — echoing only
    `request_id` (dora_genesis `_metadata(...)`). So after three real motion
    attempts the reply consumers see was byte-for-byte a plain teleport, and
    nothing downstream of the graph could tell the two apart. A6 audited its
    fallbacks from the node's STDERR, which is why this stayed invisible.

    The service is the right place to fix it: it already relays the bridge's
    reply on its own output (issue #192) and it is the only node that knows
    the behavioral context. No bridge change, no cross-node contract."""
    node = run_service(
        [
            reset_request(5, BEHAVIORAL, request_id="req-9"),
            joint_state(),
            bridge_reply(sim_time_ns=42, request_id="req-9", t_reset_ms=12, seed=5, mode=0),
        ],
        monkeypatch,
        outcome="exhausted",
    )
    assert len(replies(node)) == 1, replies(node)
    reply = replies(node)[0]
    assert reply["fallback"] is True, "RST-2's fallback flag is still missing from the reply"
    assert reply["behavioral_attempts"] == 1, reply
    assert reply["sim_time_ns"] == 42, "the bridge's post-teleport stamp must survive the merge"


def test_the_fallback_t_reset_ms_covers_the_whole_reset(monkeypatch):
    """RST-1 (issue #194): the exhaustion route's `t_reset_ms` came from the
    bridge, so it measured only the fallback teleport and silently excluded
    the up-to-three behavioral attempts that preceded it — on the one route
    where the reset is most expensive, against a <2 s budget.

    The injected clock advances 0.25 s per read, so the whole reset spans
    strictly more than the bridge's own 12 ms."""
    node = run_service(
        [
            reset_request(5, BEHAVIORAL, request_id="req-9"),
            joint_state(),
            bridge_reply(sim_time_ns=42, request_id="req-9", t_reset_ms=12, seed=5, mode=0),
        ],
        monkeypatch,
        outcome="exhausted",
    )
    assert replies(node)[0]["t_reset_ms"] >= 250, (
        f"t_reset_ms {replies(node)[0]['t_reset_ms']} is the teleport alone, not the reset"
    )


def test_the_fallback_reply_reports_the_REQUESTED_mode(monkeypatch):
    """TC-6 is silent on whether `mode` is the mode requested or the one
    executed, and the routes disagreed: the refusal echoes the request, the
    fallback reported the bridge's 0. Interpretation recorded in ADR-35 —
    `mode` is what was ASKED FOR on every route, and `fallback` is what says
    a teleport actually ran. That keeps one meaning per key and loses
    nothing: both facts are still recoverable."""
    node = run_service(
        [
            reset_request(5, BEHAVIORAL, request_id="req-9"),
            joint_state(),
            bridge_reply(sim_time_ns=42, request_id="req-9", t_reset_ms=12, seed=5, mode=0),
        ],
        monkeypatch,
        outcome="exhausted",
    )
    assert replies(node)[0]["mode"] == BEHAVIORAL, replies(node)[0]


def test_a_plain_teleport_relay_is_untouched_by_the_fallback_merge(monkeypatch):
    """The control. The merge must fire ONLY for a reply that answers a
    fallback this service forwarded — an ordinary teleport carries the
    bridge's own timing and mode, and inventing a behavioral audit trail on
    it would be a different lie from the one being fixed."""
    node = run_service(
        [reset_request(7, TELEPORT, request_id="t-1"), bridge_reply(request_id="t-1", mode=0)],
        monkeypatch,
    )
    reply = replies(node)[0]
    assert "fallback" not in reply and "behavioral_attempts" not in reply, reply
    assert reply["mode"] == TELEPORT


def test_the_fallback_merge_is_correlated_not_positional(monkeypatch):
    """TC-6 correlates request and reply on `request_id`, and the merge must
    too: it decorates the reply to THIS fallback, not whichever reply
    happens to arrive next.

    Without the check the first bridge reply to show up gets stamped
    `fallback: true` with another request's attempt count — a fabricated
    audit trail, which is a worse failure than the missing one being fixed
    here. Caught by mutation: replacing the correlation with a bare
    `if pending_fallback:` left every other test in this file green."""
    node = run_service(
        [
            reset_request(5, BEHAVIORAL, request_id="mine"),
            joint_state(),
            # a reply that answers something else entirely
            bridge_reply(sim_time_ns=7, request_id="someone-elses", t_reset_ms=3, seed=1, mode=0),
            bridge_reply(sim_time_ns=42, request_id="mine", t_reset_ms=12, seed=5, mode=0),
        ],
        monkeypatch,
        outcome="exhausted",
    )
    stray, mine = replies(node)
    assert "fallback" not in stray, f"a stray reply was dressed up as this fallback: {stray}"
    assert stray["mode"] == TELEPORT and stray["t_reset_ms"] == 3, stray
    assert mine["fallback"] is True and mine["behavioral_attempts"] == 1, mine


def test_a_runtime_that_cannot_be_built_refuses_instead_of_killing_the_node(monkeypatch):
    """Issue #206: `get_runtime()` does file I/O and an Owlv2
    `from_pretrained`, and it sat OUTSIDE the try/except that guards the
    payload parse. An OSError from a missing model cache therefore escaped
    `for event in node:` and killed the node.

    This node's death has no recovery: since issue #192 every consumer in
    every graph takes its episode boundary from here, and dora does not
    restart nodes. So a first-run cache problem stranded the whole run —
    strictly worse than the failure #192 was fixed to prevent.

    The request is refused (ADR-34: on `reset_refused`), the error says what
    broke, and the service keeps serving."""
    node = run_service(
        [reset_request(4, BEHAVIORAL, request_id="boom")],
        monkeypatch,
        build_raises=OSError("model cache missing"),
    )
    assert len(refusals(node)) == 1, "the node died instead of refusing"
    refusal = refusals(node)[0]
    assert refusal["request_id"] == "boom"
    assert "model cache missing" in refusal["error"], refusal
    assert replies(node) == [], "an infrastructure failure was reported as a boundary"


def test_a_failed_runtime_does_not_end_the_service(monkeypatch):
    """The property that matters more than the refusal itself: teleport needs
    no runtime at all, so a behavioral request that cannot build one must not
    take the teleport route down with it."""
    node = run_service(
        [
            reset_request(4, BEHAVIORAL, request_id="boom"),
            reset_request(5, TELEPORT, request_id="after"),
            bridge_reply(sim_time_ns=99, request_id="after"),
        ],
        monkeypatch,
        build_raises=OSError("model cache missing"),
    )
    assert len(refusals(node)) == 1
    assert forwards(node) == [[5, TELEPORT]], (
        "the teleport after the failure never reached the bridge"
    )
    assert len(replies(node)) == 1 and replies(node)[0]["sim_time_ns"] == 99


def test_the_bridge_info_prewarm_cannot_kill_the_node_either(monkeypatch):
    """The OTHER unguarded construction path, which issue #206 does not name:
    `bridge_info` builds the runtime early so the model load never lands
    inside a reset. That is precisely where a bad cache surfaces FIRST, and
    it called `get_runtime()` bare. No reply is owed to a broadcast input —
    the requirement is only that the service survives and still resets."""
    node = run_service(
        [
            _inp("bridge_info", pa.array(['{"calibration": {}}']), {}),
            reset_request(6, TELEPORT, request_id="after"),
            bridge_reply(sim_time_ns=7, request_id="after"),
        ],
        monkeypatch,
        build_raises=OSError("model cache missing"),
    )
    assert forwards(node) == [[6, TELEPORT]], "a pre-warm failure killed the boundary authority"
    assert len(replies(node)) == 1


def test_a_runtime_that_fails_mid_start_is_discarded_not_reused(monkeypatch):
    """Issue #206, the half-started case: the build can SUCCEED and
    `rt.start()` fail during planning, which leaves a cached runtime that
    already believes it is active.

    Keeping it means the next `joint_state` drives a half-initialised
    runtime — streaming commands through the guard, or settling an outcome,
    for a reset that never really began. The service drops it so the next
    behavioral request builds a fresh one."""
    node = run_service(
        [
            reset_request(4, BEHAVIORAL, request_id="boom"),
            joint_state(sim_ns=5_000),
            joint_state(sim_ns=6_000),
        ],
        monkeypatch,
        start_raises=RuntimeError("planner blew up"),
    )
    assert len(refusals(node)) == 1, refusals(node)
    commands = [topic for topic, _, _ in node.sent if topic.startswith("reset_")]
    assert commands == ["reset_refused"], f"a discarded runtime still drove the arm: {commands}"
    assert replies(node) == [], "a half-started runtime produced an episode boundary"


def test_a_failed_prewarm_keeps_the_runtime_it_already_built(monkeypatch):
    """RST-2 (cross-review of #223): the pre-warm guard must NOT discard a
    runtime that built.

    `bridge_info` is emitted ONCE, before the bridge's event loop, and
    `on_bridge_info` is the only writer of `calibration`. Dropping the
    runtime there loses calibration permanently, and every later behavioral
    reset then burns all three attempts on `calibration is None` and reports
    `fallback: true, behavioral_attempts: 3` — the silent A6 void that
    `harness validate` refuses unwired graphs to prevent, and the exact lie
    the behavioral branch refuses to write. A failed BUILD leaves `runtime`
    unset anyway, so the retry path is unaffected.

    Observable: the service builds the runtime once, not twice."""
    node = run_service(
        [
            _inp("bridge_info", pa.array(['{"calibration": {}}']), {}),
            reset_request(3, BEHAVIORAL, request_id="after"),
            joint_state(sim_ns=1_000),
        ],
        monkeypatch,
        bridge_info_raises=KeyError("calibration"),
    )
    assert node.builds == 1, (
        f"the pre-warm failure discarded a runtime that had already built ({node.builds} builds) "
        "— calibration is written once and cannot be recovered"
    )


def test_behavioral_reset_refused_for_g1_right(monkeypatch):
    """RST-2, TC-6, ADR-34, G1-C2: with AISLE_EMBODIMENT=g1_right a
    behavioral request is REFUSED on reset_refused (never reset_done) with
    a reason that names the embodiment — and the behavioral runtime is
    never built, so no Franka limits can leak into a G1 graph."""
    monkeypatch.setenv("AISLE_EMBODIMENT", "g1_right")
    node = run_service([reset_request(7, BEHAVIORAL)], monkeypatch)
    assert forwards(node) == []  # nothing reached the bridge
    assert replies(node) == []  # ADR-34: no boundary was emitted
    [refusal] = refusals(node)
    assert "g1_right" in refusal["error"]
    assert "teleport" in refusal["error"]
    assert refusal["t_reset_ms"] == 0
    assert node.builds == 0  # the runtime never started building


def test_behavioral_reset_still_runs_for_franka(monkeypatch):
    """RST-2 regression: the embodiment gate opens exactly for franka —
    the default environment keeps the behavioral route it always had."""
    monkeypatch.setenv("AISLE_EMBODIMENT", "franka")
    node = run_service([reset_request(7, BEHAVIORAL), joint_state(), joint_state()], monkeypatch)
    assert node.builds == 1
    assert forwards(node) == []  # still in the attempt loop, no fallback yet
