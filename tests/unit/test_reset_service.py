"""Unit tests for the reset dispatcher (SPEC 040 RST-1, RST-2) — the pure
routing function, no dora (CON-12)."""

import numpy as np
import pytest

from aisle.reset.service import refusal_reply_metadata, route_reset, stamp

pytestmark = pytest.mark.unit


def test_teleport_routes_to_bridge():
    """RST-1: mode 0 (teleport) dispatches to the bridge, which owns state
    injection; the <2 s completion budget is measured live in acceptance A2
    through this dispatcher."""
    assert route_reset(0) == "bridge"


def test_behavioral_routes_to_the_attempt_loop():
    """RST-2: behavioral mode dispatches to the attempt loop (whose
    exhaustion ALSO ends at the bridge, carrying fallback metadata — the
    loop never hangs on a reset)."""
    assert route_reset(1) == "behavioral"


class TestBehavioralAttemptLoop:
    """RST-2 semantics: retry <=3, then teleport fallback with
    fallback: true in the reply metadata."""

    def test_attempts_exhaust_into_fallback(self):
        from aisle.reset.behavioral import MAX_ATTEMPTS, BehavioralReset

        calls = []

        def failing():
            calls.append(1)
            return False

        outcome = BehavioralReset(attempt=failing).run()
        assert outcome.fallback is True
        assert outcome.attempts == MAX_ATTEMPTS == 3  # the spec's <=3
        assert len(calls) == 3

    def test_success_stops_the_loop_without_fallback(self):
        from aisle.reset.behavioral import BehavioralReset

        results = iter([False, True])
        outcome = BehavioralReset(attempt=lambda: next(results)).run()
        assert outcome.fallback is False
        assert outcome.attempts == 2

    def test_no_motion_strategy_always_falls_back(self):
        """PR-1 production wiring: no motion capability exists yet, so a
        behavioral request must deterministically reach the teleport
        fallback (never hang, never pretend)."""
        from aisle.reset.behavioral import BehavioralReset, no_motion_available

        outcome = BehavioralReset(attempt=no_motion_available).run()
        assert outcome.fallback is True

    def test_reply_metadata_carries_the_audit_trail(self):
        from aisle.reset.behavioral import (
            BehavioralOutcome,
            behavioral_reply_metadata,
        )

        meta = behavioral_reply_metadata(
            {"request_id": "req-3"}, BehavioralOutcome(fallback=True, attempts=3)
        )
        assert meta == {"request_id": "req-3", "fallback": True, "behavioral_attempts": 3}


def test_unknown_mode_is_rejected():
    """RST-1: the reset request schema admits only modes 0 and 1; anything
    else is an explicit error, not a default."""
    with pytest.raises(ValueError, match="reset mode"):
        route_reset(2)


def test_stamp_adds_tc2_keys_and_service_seq():
    """TC-2 (PR review): every service output carries sim_time_ns, env_id,
    and the service's OWN per-topic monotonic seq; upstream values for the
    first two are preserved when present."""
    assert stamp({}, 3) == {"sim_time_ns": 0, "env_id": 0, "seq": 3}
    stamped = stamp({"sim_time_ns": 42, "env_id": 1, "seq": 999, "request_id": "r"}, 4)
    assert stamped == {"sim_time_ns": 42, "env_id": 1, "seq": 4, "request_id": "r"}


def test_refusal_reply_metadata_is_tc6_complete():
    """TC-6 (PR review): a refusal reply echoes request_id, carries
    seed/mode when the payload was well-formed, t_reset_ms=0 (the sim was
    never touched), and the error."""
    payload = np.array([7, 1], dtype=np.uint32)
    meta = refusal_reply_metadata({"request_id": "req-9"}, payload, "behavioral is Phase 2")
    assert meta == {
        "request_id": "req-9",
        "t_reset_ms": 0,
        "error": "behavioral is Phase 2",
        "seed": 7,
        "mode": 1,
    }
    malformed = refusal_reply_metadata({"request_id": "req-10"}, np.array([], np.uint32), "bad")
    assert "seed" not in malformed and malformed["request_id"] == "req-10"


def test_reply_metadata_carries_env_id_on_every_route():
    """BRG-5 + issue #192: episode-state consumers now take the boundary
    from the SERVICE's reset_done rather than the bridge's, and the guard
    slices its per-env boundary on `env_id`. A reply without one defaults to
    env 0 (see `stamp` above) and would clear only that env's state, so the
    behavioral and refusal routes must carry the same routing key the
    bridge's teleport reply does — or the modes cut different envs."""
    from aisle.reset.behavioral import BehavioralOutcome, behavioral_reply_metadata

    request = {"request_id": "req-1", "env_id": 3}
    behavioral = behavioral_reply_metadata(request, BehavioralOutcome(fallback=False, attempts=2))
    assert behavioral["env_id"] == 3

    refusal = refusal_reply_metadata(request, np.array([7, 1], dtype=np.uint32), "nope")
    assert refusal["env_id"] == 3

    # a request WITHOUT env_id must not invent one — stamp()'s default is
    # the single-env behaviour every legacy caller already relies on
    assert "env_id" not in behavioral_reply_metadata(
        {"request_id": "r"}, BehavioralOutcome(fallback=True, attempts=1)
    )
    assert "env_id" not in refusal_reply_metadata(
        {"request_id": "r"}, np.array([], np.uint32), "bad"
    )


def test_behavioral_gate_is_franka_only():
    """RST-2, TC-6, G1-C2: the behavioral motion chain is Franka-verified
    (reset/motion.py pins Franka limits and a 7-dof home). g1_right and
    so101 must gate CLOSED — driving them with Franka limits would command
    wrong bounds and a wrong-length home; teleport (mode 0) stays open."""
    from aisle.reset.service import behavioral_reset_supported

    assert behavioral_reset_supported("franka") is True
    assert behavioral_reset_supported("so101") is False
    assert behavioral_reset_supported("g1_right") is False
    assert behavioral_reset_supported("humanoid") is False
