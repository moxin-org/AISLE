"""P4 offline interface tests; synthetic decisions never establish task success."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import shadow_selector as shadow  # noqa: E402

pytestmark = pytest.mark.unit
SKILL = "ik-transfer-v2"
LEARNED = "vla-policy-smolvla"


def event(kind, stamp, *, episode=0, goal="g1", **fields):
    row = dict(id=f"e{stamp}", kind=kind, sim_time_ns=stamp, env_id=0, episode=episode)
    if kind != "reset":
        row["goal_revision"] = goal
    if kind == "goal":
        row["instruction"] = "move the requested object"
    return row | fields


def prefix():
    return [event("reset", 0), event("goal", 1), event("observation", 2, obs_ns=2)]


def select(stamp=3, *, applicable=None, **fields):
    return (
        event(
            "select",
            stamp,
            stage="transfer",
            remaining_budget_ns=200,
            applicable=[SKILL, LEARNED] if applicable is None else applicable,
            parameters={SKILL: {"grasp_pose": [0.1, 0.2, 0.3, 0, 0, 0, 1]}, LEARNED: {}},
        )
        | fields
    )


def reply(stamp=4, *, request="e3", candidate=LEARNED, **fields):
    return (
        event(
            "reply",
            stamp,
            request_id=request,
            candidate=candidate,
            version=shadow.adapter_pin(candidate),
            chunk=[{"joints": [0.1] * 7, "gripper": 0.2}, {"joints": [0.2] * 7, "gripper": 0.3}]
            if candidate == LEARNED
            else None,
        )
        | fields
    )


def document(events):
    return dict(
        format="aisle_shadow_events",
        version=1,
        provenance="synthetic",
        candidates=[
            dict(
                id=name,
                version=shadow.adapter_pin(name),
                stages=["transfer"],
                reliability={"successes": 8, "trials": 10},
                max_attempts=2,
                timeout_ns=100,
            )
            for name in (SKILL, LEARNED)
        ],
        records=events,
    )


def run(events, mutate=None):
    doc = document(events)
    if mutate:
        mutate(doc)
    return shadow.replay(doc)


def test_fixed_rule_and_single_owner():
    """CON-5; ADR-38: deterministic skill preference, no interleaved ownership."""
    doc = document(prefix() + [select(), event("boundary", 4), select(5)])
    result = shadow.replay(doc)
    assert result == shadow.replay(copy.deepcopy(doc))
    assert result["timeline"][3]["selected"] == SKILL
    assert result["timeline"][4]["proposal"]["kind"] == "skill_invocation"
    assert result["timeline"][4]["proposal"]["goal_revision"] == "g1"
    assert result["timeline"][5]["reason"] == "controller_busy"
    assert result["timeline"][5]["owner"] == "e3"
    assert result["actuation_authority"] is False
    assert result["guard_traversal"] == "unverified"
    assert result["task_success"] is None


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("stage", "grasp", "stage_mismatch"),
        ("applicable", [], "not_applicable"),
        ("remaining_budget_ns", 99, "insufficient_budget"),
        ("parameters", {SKILL: {}, LEARNED: {"python": "exec"}}, "invalid_parameters"),
    ],
)
def test_candidate_filters(field, value, reason):
    """CON-5/8; P4: every candidate rejection has a reason."""
    row = run(prefix() + [select(**{field: value})])["timeline"][-1]
    assert row["reason"] == "no_eligible_candidate"
    assert all(reason in item["reasons"] for item in row["candidates"])


@pytest.mark.parametrize("reliability", [None, {"successes": 0, "trials": 8}])
def test_unknown_or_incompetent_candidates_hold(reliability):
    """CON-5; P4: unknown or zero success cannot qualify a fallback."""
    result = run(
        prefix() + [select()],
        lambda d: [c.update(reliability=reliability) for c in d["candidates"]],
    )
    assert result["timeline"][-1]["reason"] == "no_eligible_candidate"


def test_attempts_are_consumed_and_goal_scoped():
    """TC-7; P4: selections consume attempts, new goals start new accounting."""
    rows = prefix() + [
        select(),
        event("interrupt", 4, request_id="e3"),
        select(5),
        event("interrupt", 6, request_id="e5"),
        select(7),
        event("goal", 8, goal="g2"),
        event("observation", 9, goal="g2", obs_ns=9),
        select(10, goal_revision="g2"),
    ]
    trace = run(rows)["timeline"]
    assert trace[7]["selected"] == LEARNED
    assert "attempts_exhausted" in trace[7]["candidates"][0]["reasons"]
    assert trace[-1]["selected"] == SKILL


def test_learned_chunks_are_sequential_and_replies_single_use():
    """ADR-38; TC-7: only the selected request can supply a chunk, once."""
    result = run(
        prefix()
        + [
            select(applicable=[LEARNED]),
            reply(),
            reply(5),
            event("boundary", 6),
            event("boundary", 7),
            event("boundary", 8),
        ]
    )
    trace = result["timeline"]
    assert trace[5]["reason"] == "duplicate_reply"
    assert trace[6]["proposal"]["action"]["joints"] == [0.1] * 7
    assert trace[7]["proposal"]["action"]["joints"] == [0.2] * 7
    assert trace[7]["owner"] is None
    assert trace[8]["reason"] == "idle_hold"


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "old"},
        {"candidate": SKILL, "version": "old"},
        {"version": "old"},
        {"episode": 9},
        {"goal_revision": "old"},
    ],
)
def test_async_reply_fencing(change):
    """TC-6/7; ADR-38: late/misattributed replies cannot refill the queue."""
    trace = run(prefix() + [select(applicable=[LEARNED]), reply(**change), event("boundary", 5)])[
        "timeline"
    ]
    assert trace[-2]["decision"] == "rejected"
    assert trace[-1]["reason"] == "awaiting_reply"
    assert "proposal" not in trace[-1]


@pytest.mark.parametrize("invalidation", ["reset", "goal"])
def test_reset_and_goal_flush_pending_work(invalidation):
    """TC-6/7; ADR-38: an old reply cannot revive a cancelled request."""
    update = (
        event("reset", 5, episode=1) if invalidation == "reset" else event("goal", 5, goal="g2")
    )
    trace = run(prefix() + [select(applicable=[LEARNED]), reply(), update, reply(6)])["timeline"]
    assert trace[-2]["cancelled_request"] == "e3"
    assert trace[-2]["owner"] is None
    assert trace[-1]["decision"] == "rejected"
    assert "proposal" not in trace[-1]


def test_reused_goal_is_not_an_attempt_reset():
    """TC-7: replaying the same goal identity cannot replenish attempts."""
    trace = run(prefix() + [select(), event("goal", 4)])["timeline"]
    assert trace[-1]["reason"] == "reused_goal_revision"
    assert trace[-1]["owner"] == "e3"


@pytest.mark.parametrize("terminal", ["interrupt", "reply", "timeout"])
def test_skill_end_requires_observation_reacquisition(terminal):
    """ADR-38; P4: partial execution cannot fall back using pre-action observations."""
    stamp = 103 if terminal == "timeout" else 5
    end = (
        event("boundary", stamp)
        if terminal == "timeout"
        else reply(stamp, candidate=SKILL)
        if terminal == "reply"
        else event("interrupt", stamp, request_id="e3")
    )
    trace = run(
        prefix()
        + [
            select(),
            event("boundary", 4),
            end,
            select(stamp + 1, applicable=[LEARNED]),
            event("observation", stamp + 2, obs_ns=4),
            select(stamp + 3, applicable=[LEARNED]),
            event("observation", stamp + 4, obs_ns=stamp + 4),
            select(stamp + 5, applicable=[LEARNED]),
        ]
    )["timeline"]
    assert trace[6]["reason"] == "observation_required"
    assert trace[8]["reason"] == "observation_required"
    assert trace[-1]["selected"] == LEARNED
    assert trace[5].get("task_success") is None


def test_timeout_clears_queue_and_late_reply_holds():
    """ADR-38; P4: deadline is inclusive and never emits remaining commands."""
    trace = run(
        prefix() + [select(applicable=[LEARNED]), reply(), event("boundary", 103), reply(104)]
    )["timeline"]
    assert trace[-2]["reason"] == "timeout_hold"
    assert trace[-2]["owner"] is None
    assert trace[-1]["reason"] == "no_pending_request"


def test_observation_age_is_rechecked_at_each_chunk_boundary():
    """ADR-38 rule 4: queue acceptance does not make later elements fresh."""
    stale = shadow.VLA_STALE_NS
    doc = document(
        prefix()
        + [
            select(remaining_budget_ns=stale * 3, applicable=[LEARNED]),
            reply(),
            event("boundary", stale + 2),
            event("boundary", stale + 3),
        ]
    )
    for candidate in doc["candidates"]:
        candidate["timeout_ns"] = stale * 2
    trace = shadow.replay(doc)["timeline"]
    assert "proposal" in trace[-2]
    assert trace[-1]["reason"] == "stale_observation_hold"
    assert trace[-1]["owner"] is None


def test_future_and_pre_goal_observations_are_rejected():
    """CON-5; TC-6/7: observations must belong to the current intent interval."""
    trace = run(
        prefix()[:2]
        + [event("observation", 2, obs_ns=3), event("observation", 3, obs_ns=1), select(4)]
    )["timeline"]
    assert all(row["decision"] == "rejected" for row in trace[2:4])
    assert trace[-1]["reason"] == "observation_required"


@pytest.mark.parametrize(
    "chunk",
    [
        [],
        [[1] * 8],
        [{"joints": [1] * 6, "gripper": 0}],
        [{"joints": [True] * 7, "gripper": 0}],
        [{"joints": [0] * 7, "gripper": 2}],
    ],
)
def test_malformed_chunks_cannot_create_proposals(chunk):
    """CON-8; TC-1: only finite, typed seven-joint/ranged-gripper chunks enter the queue."""
    trace = run(
        prefix() + [select(applicable=[LEARNED]), reply(chunk=chunk), event("boundary", 5)]
    )["timeline"]
    assert trace[-2]["reason"] == "invalid_chunk"
    assert "proposal" not in trace[-1]


def test_nonfinite_input_is_rejected_before_replay():
    """CON-8: non-JSON numeric values invalidate the entire source artifact."""
    with pytest.raises(ValueError, match="JSON compliant"):
        run(
            prefix()
            + [
                select(applicable=[LEARNED]),
                reply(chunk=[{"joints": [float("nan")] * 7, "gripper": 0}]),
            ]
        )


def test_preemption_flushes_old_chunk_and_fences_same_goal_reply():
    """ADR-38: explicit cancellation replaces the chunk without interleaving."""
    trace = run(
        prefix()
        + [
            select(applicable=[LEARNED]),
            reply(),
            event("boundary", 5),
            event("interrupt", 6, request_id="e3"),
            select(7, applicable=[LEARNED]),
            reply(8),
            event("boundary", 9),
            reply(10, request="e7"),
            event("boundary", 11),
        ]
    )["timeline"]
    assert trace[8]["reason"] == "wrong_request"
    assert trace[9]["reason"] == "awaiting_reply"
    assert trace[-1]["proposal"]["action"]["joints"] == [0.1] * 7


def test_wrong_episode_reply_cannot_extend_timeout():
    """TC-6; P4: even invalid arrivals advance the offline watchdog."""
    trace = run(prefix() + [select(applicable=[LEARNED]), reply(103, episode=99)])["timeline"]
    assert trace[-1]["reason"] == "timeout_hold"
    assert trace[-1]["owner"] is None


def test_stale_selection_and_late_initial_chunk_hold():
    """ADR-38: staleness applies to selection and first inference arrival."""
    stale = shadow.VLA_STALE_NS + 3
    assert run(prefix() + [select(stale)])["timeline"][-1]["reason"] == "stale_observation_hold"
    doc = document(
        prefix() + [select(remaining_budget_ns=stale * 2, applicable=[LEARNED]), reply(stale)]
    )
    for c in doc["candidates"]:
        c["timeout_ns"] = stale * 2
    assert shadow.replay(doc)["timeline"][-1]["reason"] == "stale_observation_hold"


def test_expired_pending_work_is_flushed_before_new_goal():
    """TC-7; P4: a goal arriving on a deadline still invalidates the old intent."""
    trace = run(
        prefix()
        + [
            select(),
            event("goal", 103, goal="g2"),
            event("observation", 104, goal="g2", obs_ns=104),
            select(105, goal_revision="g2"),
        ]
    )["timeline"]
    assert trace[4]["timed_out_request"] == "e3"
    assert trace[-1]["selected"] == SKILL


def test_skill_return_is_not_a_delivery_verdict_or_guard_attestation():
    """TC-8; BG-1: synthetic returns establish neither success nor guard traversal."""
    result = run(prefix() + [select(), event("boundary", 4), reply(5, candidate=SKILL)])
    assert result["timeline"][-1]["reason"] == "returned_unconfirmed"
    assert result["task_success"] is None
    assert result["guard_traversal"] == "unverified"


def test_replay_does_not_mutate_input_and_candidate_order_is_irrelevant():
    """CON-5: fixed priority is independent of caller candidate ordering."""
    doc = document(prefix() + [select()])
    saved = copy.deepcopy(doc)
    first = shadow.replay(doc)
    assert doc == saved
    doc["candidates"].reverse()
    assert shadow.replay(doc)["timeline"] == first["timeline"]


def test_duplicate_event_is_idempotent_but_conflicting_identity_fails():
    """CON-5; TC-7: duplicate delivery never repeats a shadow command."""
    rows = prefix() + [select(), event("boundary", 4)]
    trace = run(rows + [copy.deepcopy(rows[-1])])["timeline"]
    assert trace[-1]["reason"] == "duplicate_event"
    assert "proposal" not in trace[-1]
    with pytest.raises(ValueError, match="identity"):
        run(rows + [rows[-1] | {"kind": "select"}])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(provenance="retained_dataset"),
        lambda d: d["candidates"][0].update(id="arbitrary-python"),
        lambda d: d["candidates"][0].update(version="unbound"),
        lambda d: d["candidates"][0].update(timeout_ns=True),
        lambda d: d["candidates"][0].update(reliability={"successes": 2, "trials": 1}),
        lambda d: d["records"][0].update(env_id=1),
        lambda d: d["records"][1].update(oracle_success=True),
    ],
)
def test_invalid_inputs_are_rejected(mutate):
    """CON-5/8; P4: explicit synthetic provenance and pinned allowlist only."""
    with pytest.raises(ValueError):
        run(prefix(), mutate)


def test_cli_json_and_no_overwrite(tmp_path, capsys):
    """CON-8: success and errors are single JSON results; never overwrite evidence."""
    source, out = tmp_path / "fixture.json", tmp_path / "out.json"
    source.write_text(json.dumps(document(prefix() + [select()])))
    assert shadow.main(["--fixture", str(source), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    before = out.read_bytes()
    assert shadow.main(["--fixture", str(source), "--out", str(out)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert out.read_bytes() == before
    assert shadow.main([]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


@pytest.mark.parametrize("name", ["lifecycle", "unqualified", "stale_chunk"])
def test_committed_synthetic_evidence_replays_exactly(name):
    """CON-5: committed reports reproduce from explicitly synthetic inputs."""
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root / f"tests/fixtures/shadow_selector/{name}.json").read_text())
    report = json.loads((root / f"analysis/aw0_p4/{name}.json").read_text())
    assert shadow.replay(source) == report
