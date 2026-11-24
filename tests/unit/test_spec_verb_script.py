"""spec-verb-script: one SPEC request at a time, each matched to its reply."""

from __future__ import annotations

import json

from aisle.nodes.spec_verb_script import MAX_NOT_READY, VerbScript


def run(script):
    sent = []
    verbs = VerbScript(script, lambda topic, value: sent.append(json.loads(value.to_pylist()[0])))
    return verbs, sent


def test_script_waits_for_reset_and_sends_one_request_at_a_time():
    """TC-6: each request carries its own request_id and the next verb waits for
    the matching reply; nothing is sent before the first reset_done."""
    verbs, sent = run([["a", {}, "0"], ["b", {"x": 1}, "LIMIT_VIOLATION"]])
    verbs.on_turn(0.0)
    assert sent == []
    verbs.on_reset_done()
    verbs.on_turn(0.1)
    verbs.on_turn(0.2)
    assert [s["verb"] for s in sent] == ["a"]
    verbs.on_response({"request_id": "stale", "code": "0"}, 0.3)
    assert verbs.pending is not None
    verbs.on_response({"request_id": sent[0]["request_id"], "code": "0"}, 0.3)
    verbs.on_turn(0.4)
    verbs.on_response({"request_id": sent[1]["request_id"], "code": "LIMIT_VIOLATION"}, 0.5)
    assert verbs.finished and [r["ok"] for r in verbs.results] == [True, True]
    assert sent[1]["params"] == {"x": 1} and sent[0]["request_id"] != sent[1]["request_id"]


def test_not_ready_is_retried_under_one_result_entry():
    """TC-6: a NOT_READY reply re-sends the same verb with a fresh request_id."""
    verbs, sent = run([["a", {}, "0"]])
    verbs.on_reset_done()
    verbs.on_turn(0.0)
    verbs.on_response({"request_id": sent[0]["request_id"], "code": "NOT_READY"}, 0.1)
    verbs.on_turn(0.2)
    assert len(sent) == 2 and sent[0]["request_id"] != sent[1]["request_id"]
    verbs.on_response({"request_id": sent[1]["request_id"], "code": "0"}, 0.3)
    assert verbs.finished and len(verbs.results) == 1 and verbs.results[0]["ok"]
    assert MAX_NOT_READY > 0


def test_uncorrelated_response_after_the_script_finished_is_ignored():
    """TC-6: a reply with no request_id is not the answer to a request never sent."""
    verbs, sent = run([["a", {}, "0"]])
    verbs.on_reset_done()
    verbs.on_turn(0.0)
    verbs.on_response({"request_id": sent[0]["request_id"], "code": "0"}, 0.1)
    verbs.on_response({"code": "INVALID_PARAMS"}, 0.2)
    assert verbs.finished and len(verbs.results) == 1
