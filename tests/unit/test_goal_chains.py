"""P3 recovery capture: goal verdicts belong to physical reset windows (HAR-4/6)."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from aisle.harness import trace_recorder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import goal_chains as chains  # noqa: E402

pytestmark = pytest.mark.unit


def message(endpoint, stamp, payload, *, seq=0, **metadata):
    return dict(
        type="INPUT",
        id=endpoint,
        value=pa.array([json.dumps(payload)] if isinstance(payload, dict) else payload),
        metadata=dict(sim_time_ns=stamp, env_id=0, seq=seq, **metadata),
    )


def events():
    initial = dict(
        goal_id="g1", status="success", failure=None, seed=4, verifier="oracle", t_end=1.0
    )
    recovery = dict(
        goal_id="g2", status="fail", failure="not_returned", seed=4, verifier="oracle", t_end=2.0
    )
    return [
        message("reset__reset_done", 0, [1], request_id="reset-0"),
        message(
            "client__episode_goal", 1, dict(seed=4, reset_sim_ns=0, target_med="a"), goal_id="g1"
        ),
        message("oracle__episode_result", 10, initial, goal_id="g1"),
        message(
            "client__episode_goal",
            11,
            dict(seed=4, reset_sim_ns=10, target_med="b", expects_return=True),
            seq=1,
            goal_id="g2",
        ),
        message("oracle__episode_result", 20, recovery, seq=1, goal_id="g2"),
        message("reset__reset_done", 30, [1], seq=1, request_id="reset-1"),
    ]


def capture(tmp_path, monkeypatch, messages=None):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("AISLE_TRACE_DIR", str(run / "traces"))
    # This fixture has no images; a video import must not add sim extras to units.
    imageio = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(v2=imageio))
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio)
    monkeypatch.setitem(
        sys.modules, "dora", SimpleNamespace(Node=lambda: iter(messages or events()))
    )
    monkeypatch.setattr(trace_recorder.signal, "signal", lambda *a: None)
    trace_recorder.main()
    (run / "manifest.json").write_text(
        json.dumps(dict(tier="T4", sim_engine="genesis", evidence_kind="synthetic_test"))
    )
    verdicts = [
        json.loads(e["value"][0].as_py())
        for e in (messages or events())
        if e["id"].endswith("__episode_result")
    ]
    (run / "episodes.jsonl").write_text(
        "".join(
            json.dumps(dict(episode=0, **r, **({"recovery": True} if i else {}))) + "\n"
            for i, r in enumerate(verdicts)
        )
    )
    return run


def test_recorder_preserves_goal_identity_bound_to_original_arrow_row(tmp_path, monkeypatch):
    """HAR-4; TC-7: capture retains goal metadata omitted by the numeric trace schema."""
    run = capture(tmp_path, monkeypatch)
    identities = [
        json.loads(line) for line in (run / "traces/identities.jsonl").read_text().splitlines()
    ]
    goals = [r for r in identities if r["endpoint"] == "client__episode_goal"]
    assert [r["metadata"]["goal_id"] for r in goals] == ["g1", "g2"]
    assert [r["row"] for r in goals] == [0, 1]
    assert goals[0]["trace_row_sha256"] != goals[1]["trace_row_sha256"]
    assert all(r["format"] == "aisle_trace_identity" and r["version"] == 1 for r in identities)


def test_round_trip_keeps_two_goals_one_reset_window_and_all_verdicts(tmp_path, monkeypatch):
    """HAR-4/6; TC-6/7: recovery is not another reset or a replacement outcome."""
    run = capture(tmp_path, monkeypatch)
    out = tmp_path / "chains"
    chains.export_chains(run, out)
    data = chains.read_chains(out)
    assert len(data["episodes"]) == 2  # closed chain and explicit final open reset
    first = data["episodes"][0]
    assert first["window_ns"] == [0, 30]
    assert [g["goal_id"] for g in first["goals"]] == ["g1", "g2"]
    assert [g["outcome"]["status"] for g in first["goals"]] == ["success", "fail"]
    assert first["goals"][1]["goal_reset_sim_ns"] == 10
    assert first["goals"][1]["physical_reset_ns"] == 0
    assert data["episodes"][1]["goals"] == []
    assert data["manifest"]["policy_visible"] is False
    assert data["manifest"]["coverage"]["verdict_records"] == 2
    assert data["manifest"]["coverage"]["unscored_goals"] == 0
    assert (out / "source/episodes.jsonl").read_bytes() == (run / "episodes.jsonl").read_bytes()
    rebuilt = tmp_path / "rebuilt"
    chains.export_chains(out / "source", rebuilt)
    assert chains.read_chains(rebuilt) == data


def test_p1_duplicate_episode_rejection_is_unchanged(tmp_path, monkeypatch):
    """CON-13; HAR-4: the P3 companion cannot weaken P1's one-outcome format."""
    import demo_dataset

    run = capture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate episode identity"):
        demo_dataset.export_dataset(run, tmp_path / "p1")


@pytest.mark.parametrize("fault", ["missing", "wrong_hash", "wrong_goal", "duplicate", "truncated"])
def test_identity_sidecar_must_bind_every_goal(tmp_path, monkeypatch, fault):
    """HAR-4; TC-7: missing or altered capture identities cannot be guessed from goal order."""
    run = capture(tmp_path, monkeypatch)
    path = run / "traces/identities.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(r for r in rows if r["endpoint"] == "client__episode_goal")
    if fault == "missing":
        rows.remove(target)
    elif fault == "wrong_hash":
        target["trace_row_sha256"] = "0" * 64
    elif fault == "wrong_goal":
        target["metadata"]["goal_id"] = "other"
    elif fault == "duplicate":
        rows.append(copy.deepcopy(target))
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + ("{" if fault == "truncated" else "")
    )
    with pytest.raises(ValueError):
        chains.export_chains(run, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_goal",
        "wrong_reset",
        "missing_recovery_marker",
        "wrong_episode",
        "wrong_verdict",
        "wrong_seed",
    ],
)
def test_conflicting_recovery_evidence_is_refused(tmp_path, monkeypatch, fault):
    """HAR-4/6; TC-6/7: explicit recovery evidence must agree with sources and physical window."""
    messages = events()
    if fault in {"duplicate_goal", "wrong_reset", "wrong_seed"}:
        goal = json.loads(messages[3]["value"][0].as_py())
        if fault == "duplicate_goal":
            messages[3]["metadata"]["goal_id"] = "g1"
        else:
            goal["reset_sim_ns" if fault == "wrong_reset" else "seed"] = 99
            messages[3]["value"] = pa.array([json.dumps(goal)])
    run = capture(tmp_path, monkeypatch, messages)
    path = run / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "missing_recovery_marker":
        records[1].pop("recovery")
    elif fault == "wrong_episode":
        records[1]["episode"] = 1
    elif fault == "wrong_verdict":
        records[1]["status"] = "success"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        chains.export_chains(run, tmp_path / "bad")


def test_unfinished_goal_stays_unscored(tmp_path, monkeypatch):
    """HAR-4: an incomplete recovery retains its goal and null outcome, never zero success."""
    run = capture(tmp_path, monkeypatch, events()[:4])
    chains.export_chains(run, tmp_path / "chains")
    data = chains.read_chains(tmp_path / "chains")
    assert data["episodes"][0]["window_ns"] == [0, None]
    assert data["episodes"][0]["goals"][1]["outcome"] is None
    assert data["manifest"]["coverage"]["unscored_goals"] == 1


def test_first_goal_without_any_verdict_stays_unscored(tmp_path, monkeypatch):
    """HAR-4: stopping before the first verdict must retain the attempted goal."""
    run = capture(tmp_path, monkeypatch, events()[:2])
    chains.export_chains(run, tmp_path / "chains")
    data = chains.read_chains(tmp_path / "chains")
    assert data["manifest"]["endpoints"]["episode_result"] is None
    assert data["manifest"]["coverage"]["unscored_goals"] == 1
    assert data["episodes"][0]["goals"][0]["outcome"] is None


def test_native_realistic_verdict_uses_goal_and_rollout_seed_binding(tmp_path, monkeypatch):
    """HAR-4; TC-7: realistic verdicts omit seed; source goal and rollout record bind it."""
    from aisle.nodes.verifier_realistic import episode_result

    messages = events()
    for index, goal, success, failure in ((2, "g1", True, None), (4, "g2", False, "unverified")):
        messages[index]["id"] = "realistic__episode_result"
        messages[index]["value"] = pa.array(
            [json.dumps(episode_result(goal, success, failure, 1.0))]
        )
    run = capture(tmp_path, monkeypatch, messages)
    path = run / "episodes.jsonl"
    path.write_text(
        "".join(
            json.dumps(json.loads(line) | {"seed": 4}) + "\n"
            for line in path.read_text().splitlines()
        )
    )
    chains.export_chains(run, tmp_path / "chains")
    goals = chains.read_chains(tmp_path / "chains")["episodes"][0]["goals"]
    assert all(g["outcome"]["seed"] == 4 for g in goals)
    assert all("seed" not in g["verdict"] for g in goals)


@pytest.mark.parametrize(
    "field,value", [("seed", True), ("status", "unknown"), ("verifier", "invented")]
)
def test_malformed_verdicts_do_not_become_complete_records(tmp_path, monkeypatch, field, value):
    """TC-8; HAR-4: a malformed scorer record is an instrument error."""
    messages = events()
    verdict = json.loads(messages[2]["value"][0].as_py())
    verdict[field] = value
    messages[2]["value"] = pa.array([json.dumps(verdict)])
    run = capture(tmp_path, monkeypatch, messages)
    with pytest.raises(ValueError):
        chains.export_chains(run, tmp_path / "bad")


def test_rehashed_derived_outcome_still_must_match_original_source(tmp_path, monkeypatch):
    """HAR-4/6: recomputing an artifact hash cannot promote a failed recovery."""
    run = capture(tmp_path, monkeypatch)
    out = tmp_path / "chains"
    chains.export_chains(run, out)
    path = out / "episodes.json"
    data = json.loads(path.read_text())
    data[0]["goals"][1]["outcome"]["status"] = "success"
    path.write_text(json.dumps(data))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["artifacts_sha256"]["episodes.json"] = chains.sha256(path)
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source"):
        chains.read_chains(out)


def test_cli_single_json_and_no_overwrite(tmp_path, monkeypatch, capsys):
    """CON-8: CLI succeeds only with a validated new output; failures are JSON."""
    run = capture(tmp_path, monkeypatch)
    out = tmp_path / "chains"
    args = ["--run", str(run), "--out", str(out)]
    assert chains.main(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert chains.main(args) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
