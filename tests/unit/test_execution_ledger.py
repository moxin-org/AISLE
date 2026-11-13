"""P2 offline semantics; all invocation evidence here is explicitly synthetic."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import execution_ledger as ledger  # noqa: E402

pytestmark = pytest.mark.unit

SKILL = {"name": "fixture-reach", "version": "sha256:fixture-v1"}


def event(kind, stamp, *, invocation="a", episode=0, goal="g1", env=0, **payload):
    record = {
        "id": f"{env}:{episode}:{kind}:{stamp}:{invocation}",
        "kind": kind,
        "sim_time_ns": stamp,
        "env_id": env,
        "episode": episode,
        "goal_revision": goal,
        "invocation_id": invocation,
        "skill": SKILL,
        **payload,
    }
    if kind in {"reset", "goal"}:
        record.pop("invocation_id")
        record.pop("skill")
    if kind == "reset":
        record.pop("goal_revision")
    if kind == "goal":
        record.setdefault("instruction", "reach the local joint target")
    if kind == "proposed":
        record.setdefault("parameters", {})
        record.setdefault(
            "completion",
            {
                "kind": "joint_position",
                "names": ["joint_a", "joint_b"],
                "target": [1.0, 2.0],
                "tolerance": 0.01,
            },
        )
    if kind == "observed":
        record.setdefault("positions", [1.0, 2.0])
        record.setdefault("names", ["joint_a", "joint_b"])
        record.setdefault("units", "rad")
        record.setdefault("frame", "joint")
    return record


def document(records):
    return {
        "format": "aisle_execution_events",
        "version": 1,
        "provenance": "synthetic",
        "records": records,
    }


def prefix():
    return [event("reset", 0), event("goal", 1), event("proposed", 2), event("started", 3)]


def entries(report):
    return report["active"][0]["entries"]


def test_partial_execution_recovery_and_evidence_linked_completion():
    """CON-5, HAR-4/6: interrupted work stays interrupted; recovery has its own identity."""
    records = prefix() + [
        event("interrupted", 4),
        event("observed", 5),
        event("proposed", 6, invocation="recovery"),
        event("started", 7, invocation="recovery"),
        event("returned", 8, invocation="recovery"),
        event("observed", 9, invocation="recovery"),
    ]
    report = ledger.replay_events(document(records))
    a, b = entries(report)
    assert a["status"] == "interrupted"
    assert b["status"] == "completed"
    assert b["establishes"] == "joint_configuration_within_tolerance"
    assert b["delivery_completion"] == "unknown"
    assert b["evidence"][-1]["row"] == 9
    assert report["timeline"][5]["decision"] == "rejected"
    assert report == ledger.replay_events(document(records))


@pytest.mark.parametrize("tail", [[], [event("returned", 4)]])
def test_plan_and_return_do_not_establish_completion(tail):
    """HAR-4/6: execution reports alone cannot establish a physical result."""
    report = ledger.replay_events(document(prefix() + tail))
    assert entries(report)[0]["status"] == ("unconfirmed" if tail else "started")
    assert entries(report)[0]["delivery_completion"] == "unknown"


def test_duplicate_delivery_is_idempotent_and_conflicting_id_refused():
    """CON-5: replayed events cannot cause repeated execution-state transitions."""
    doc = document(prefix() + [event("observed", 4)])
    expected = ledger.replay_events(doc)
    # Evidence identities use the first source row, so appended duplicates preserve state.
    duplicate = copy.deepcopy(doc)
    duplicate["records"].append(copy.deepcopy(doc["records"][-1]))
    actual = ledger.replay_events(duplicate)
    assert actual["active"] == expected["active"]
    assert actual["duplicates"] == 1
    duplicate["records"][-1]["positions"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="conflicting event"):
        ledger.replay_events(duplicate)


def test_reset_clears_history_and_rejects_delayed_old_episode():
    """CON-5, HAR-6: reset separates episodes, including delayed old replies."""
    report = ledger.replay_events(
        document(
            prefix()
            + [
                event("reset", 10, episode=1),
                event("goal", 11, episode=1, goal="g2"),
                event("observed", 12),
            ]
        )
    )
    assert report["histories"][0]["entries"][0]["status"] == "interrupted"
    assert report["active"][0]["episode"] == 1
    assert entries(report) == []
    assert report["timeline"][-1]["reason"] == "stale_episode"


def test_goal_change_invalidates_intent_and_rejects_delayed_reply():
    """HAR-6: late replies and repeated revisions cannot revive an obsolete goal."""
    report = ledger.replay_events(
        document(
            prefix()
            + [
                event("goal", 4, goal="g2"),
                event("observed", 5),
                event("goal", 6, goal="g1"),
                event("proposed", 7, invocation="b", goal="g2"),
            ]
        )
    )
    assert entries(report)[0]["status"] == "interrupted"
    assert not entries(report)[0]["intent_current"]
    assert entries(report)[1]["goal_revision"] == "g2"
    assert report["active"][0]["goal_revision"] == "g2"
    assert [e["reason"] for e in report["timeline"][-3:-1]] == ["stale_goal", "reused_goal"]


def test_failed_execution_cannot_be_completed_by_late_observation():
    """HAR-4: a local failure remains distinct from scorer and recovery outcomes."""
    report = ledger.replay_events(document(prefix() + [event("failed", 4), event("observed", 5)]))
    assert entries(report)[0]["status"] == "failed"
    assert entries(report)[0]["establishes"] == "execution_reported_failure"


@pytest.mark.parametrize(
    "change",
    [
        {"positions": [0.0, 0.0]},
        {"positions": [1.0]},
        {"units": "deg"},
        {"frame": "world"},
        {"names": ["joint_b", "joint_a"]},
    ],
)
def test_observation_must_match_declared_predicate(change):
    """HAR-6: measured state must satisfy the declared local condition, without projection."""
    report = ledger.replay_events(document(prefix() + [event("observed", 4, **change)]))
    assert entries(report)[0]["status"] != "completed"


def test_observation_before_start_and_unknown_skill_remain_unconfirmed():
    """HAR-4: absent invocation/version/start evidence is never inferred from sensors."""
    records = prefix()[:3] + [event("observed", 3)]
    report = ledger.replay_events(document(records))
    assert entries(report)[0]["status"] == "proposed"
    records[2]["skill"] = {"name": None, "version": None}
    report = ledger.replay_events(document(records[:3]))
    assert entries(report)[0]["status"] == "unconfirmed"
    assert entries(report)[0]["reason"] == "missing_skill_identity"


def test_wrong_skill_version_cannot_update_an_invocation():
    """HAR-4: lifecycle evidence is bound to the invoked skill version."""
    record = event("observed", 4, skill={"name": "fixture-reach", "version": "other"})
    report = ledger.replay_events(document(prefix() + [record]))
    assert report["timeline"][-1]["reason"] == "skill_mismatch"
    assert entries(report)[0]["status"] == "started"


def test_environment_isolation_and_old_timestamp_rejection():
    """HAR-6: environment and simulation-time boundaries hold across arrivals."""
    report = ledger.replay_events(
        document(
            prefix()
            + [
                event("reset", 0, env=1),
                event("goal", 1, env=1),
                event("observed", 4, env=1),
                event("returned", 6),
                event("observed", 5),
            ]
        )
    )
    assert report["active"][1]["entries"] == []
    assert entries(report)[0]["status"] == "unconfirmed"
    assert report["timeline"][-1]["reason"] == "noncausal_event"


def test_same_timestamp_completion_and_goal_change_is_ambiguous():
    """CON-5, HAR-6: no invented total ordering at cross-producer timestamps."""
    with pytest.raises(ValueError, match="ambiguous timestamp"):
        ledger.replay_events(
            document(prefix() + [event("observed", 4), event("goal", 4, goal="g2")])
        )


@pytest.mark.parametrize("extra", [{"kind": "verifier_result"}, {"oracle_result": "success"}])
def test_privileged_or_unrecognized_evidence_is_refused(extra):
    """VAL-6, HAR-4: held-out scorer data cannot enter the policy ledger."""
    row = event("observed", 4)
    row.update(extra)
    with pytest.raises(ValueError):
        ledger.replay_events(document(prefix() + [row]))


@pytest.mark.parametrize(
    "field,value",
    [("sim_time_ns", -1), ("episode", True), ("env_id", None), ("positions", [float("nan"), 2])],
)
def test_malformed_evidence_is_refused(field, value):
    """CON-8, HAR-4: invalid identities and nonfinite sensor evidence fail closed."""
    with pytest.raises(ValueError):
        ledger.replay_events(document(prefix() + [event("observed", 4, **{field: value})]))


def test_bounded_replay_refuses_overflow_without_eviction():
    """CON-5/8: bounded history refuses capacity overflow rather than forgetting IDs."""
    with pytest.raises(ValueError, match="event limit"):
        ledger.replay_events(document(prefix()), max_events=3)
    with pytest.raises(ValueError, match="invocation limit"):
        ledger.replay_events(
            document(prefix() + [event("proposed", 4, invocation="b")]), max_invocations=1
        )


def test_delayed_matching_observation_cannot_override_newer_mismatch():
    """CON-5, HAR-6: stale sensor evidence cannot confirm after a newer mismatch."""
    report = ledger.replay_events(
        document(
            prefix()
            + [
                event("observed", 6, positions=[0.0, 0.0]),
                event("observed", 5),
            ]
        )
    )
    assert entries(report)[0]["status"] == "started"
    assert report["timeline"][-1]["reason"] == "noncausal_event"


def test_reset_invalidates_intent_at_its_evidence_timestamp():
    """HAR-6: interruption timestamps identify the evidence that invalidated intent."""
    report = ledger.replay_events(document(prefix() + [event("reset", 10, episode=1)]))
    assert report["histories"][0]["entries"][0]["updated_ns"] == 10


def test_fixture_must_be_explicitly_synthetic():
    """HAR-4: hand-authored events cannot claim to be retained execution evidence."""
    doc = document(prefix())
    doc["provenance"] = "retained"
    with pytest.raises(ValueError, match="synthetic"):
        ledger.replay_events(doc)


def test_cli_emits_single_json_and_refuses_overwriting_input(tmp_path, capsys):
    """CON-8: CLI success/error are machine readable and never overwrite evidence."""
    path = tmp_path / "events.json"
    path.write_text(json.dumps(document(prefix())))
    assert ledger.main(["--fixture", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert ledger.main(["--fixture", str(path), "--out", str(path)]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_p1_adapter_keeps_unknown_spans_and_scoring_separate(tmp_path):
    """HAR-4/6, VAL-6: replay source requests, never infer skills from outcome labels."""
    import demo_dataset as ds
    from test_demo_dataset import recorded, topic

    run = recorded.__wrapped__(tmp_path)
    # A correction after the last command still invalidates intent.
    topic(
        run,
        "task__target_request",
        [1, 80, 101],
        texts=[
            json.dumps({"instruction": text}) for text in ["first", "corrected", "second episode"]
        ],
    )
    out = tmp_path / "dataset"
    ds.export_dataset(run, out)
    report = ledger.replay_dataset(out)
    assert report["provenance"] == "retained_dataset"
    assert report["coverage"]["episodes"] == 2
    assert report["coverage"]["identified_invocations"] == 0
    assert all(not h["entries"] for h in report["histories"] + report["active"])
    assert report["histories"][0]["instruction"] == "corrected"
    assert "success" not in json.dumps(report)
    assert "ibuprofen" not in json.dumps(report)
    assert report == ledger.replay_dataset(out)
    # Scorer outcomes may change while policy-facing execution history does not.
    outcomes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    for row in outcomes:
        row["status"] = "success" if row["status"] == "fail" else "fail"
    (run / "episodes.jsonl").write_text("\n".join(map(json.dumps, outcomes)))
    other = tmp_path / "other"
    ds.export_dataset(run, other)
    alternative = ledger.replay_dataset(other)
    for key in ("active", "histories", "timeline", "coverage"):
        assert report[key] == alternative[key]
    (out / "source/episodes.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        ledger.replay_dataset(out)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("success_recovery", ["interrupted", "completed"]),
        ("failure", ["failed", "unconfirmed"]),
    ],
)
def test_committed_replay_fixtures_are_reconstructable(name, expected):
    """CON-5, HAR-4: every fixture history resolves to exact evidence rows and hashes."""
    root = Path(__file__).resolve().parents[2]
    path = root / "tests/fixtures/execution_ledger" / f"{name}.json"
    doc = json.loads(path.read_text())
    original = copy.deepcopy(doc)
    report = ledger.replay_events(doc)
    assert [e["status"] for e in entries(report)] == expected
    assert report == ledger.replay_events(doc)
    assert doc == original
    for entry in entries(report):
        assert entry["delivery_completion"] == "unknown"
        for ref in entry["evidence"]:
            assert ledger.digest(doc["records"][ref["row"]]) == ref["sha256"]


def test_malformed_goal_clears_intent_and_unknown_predicate_cannot_complete():
    """HAR-6, VAL-6: unknown policy instructions/conditions do not borrow scorer truth."""
    records = prefix()
    records[2]["completion"] = None
    report = ledger.replay_events(document(records + [event("observed", 4)]))
    assert report["timeline"][-1]["reason"] == "unknown_completion_condition"
    report = ledger.replay_events(
        document(
            records
            + [
                event("goal", 4, goal="invalid", instruction=None),
                event("proposed", 5, goal="invalid", invocation="b"),
            ]
        )
    )
    assert entries(report)[0]["status"] == "interrupted"
    assert report["active"][0]["instruction"] is None
    assert report["timeline"][-1]["reason"] == "unknown_goal"


def test_excessively_large_numeric_evidence_is_a_validation_error():
    """CON-8: malformed numeric data is refused instead of crashing the CLI."""
    with pytest.raises(ValueError):
        ledger.replay_events(document(prefix() + [event("observed", 4, positions=[10**500, 2])]))
