"""P3's controller owns treatment selection; no provider is invoked here."""

import json
import sys
from pathlib import Path

import pytest
from test_p3_evidence import run_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from p3_evidence import digest, export_policy
from p3_mock_session import EvidenceSession, main, run_mock

pytestmark = pytest.mark.unit


@pytest.fixture
def bundle(tmp_path):
    target = tmp_path / "public"
    export_policy(run_fixture(tmp_path), target)
    return target


def session(bundle, tmp_path, arm="trace", name="s1", **kwargs):
    return EvidenceSession(
        bundle=bundle,
        manifest_sha256=digest(bundle / "manifest.json"),
        arm=arm,
        session_id=name,
        output=tmp_path / name,
        **kwargs,
    )


def test_common_raw_calls_equal_and_trace_history_refusal_is_retained(bundle, tmp_path):
    """HAR-4/6; TC-8: equal raw evidence, history only for the controller-assigned arm."""
    with (
        session(bundle, tmp_path, "trace", "trace") as trace,
        session(bundle, tmp_path, "history", "history") as history,
    ):
        for operation, args in [
            ("journal", {"limit": 2}),
            ("artifacts", {}),
            ("raw", {"endpoint": "ik-trajectory/joint_cmd"}),
            ("artifact", {"path": "traces/identities.jsonl", "limit": 20}),
        ]:
            request = {"operation": operation, "arguments": args}
            assert trace.call(request) == history.call(request)
        assert trace.call({"operation": "history", "arguments": {}})["ok"] is False
        result = history.call({"operation": "history", "arguments": {}})
        assert result["ok"] and result["invocations"][0]["delivery_completion"] == "unknown"
    events = [json.loads(x) for x in (tmp_path / "trace/events.jsonl").read_text().splitlines()]
    assert len(events) == 10 and events[-1]["response"]["ok"] is False
    assert all("SECRET_" not in json.dumps(e) for e in events)


@pytest.mark.parametrize(
    "extra", [{"arm": "history"}, {"session_id": "peer"}, {"bundle": "/private/scorer"}]
)
def test_participant_cannot_override_controller_identity(bundle, tmp_path, extra):
    """TC-8; HAR-6: request fields cannot select another treatment, session or evidence."""
    with session(bundle, tmp_path) as controller:
        result = controller.call({"operation": "journal", "arguments": {}, **extra})
        assert not result["ok"]
        result = controller.call({"operation": "journal", "arguments": extra})
        assert not result["ok"]


def test_bundle_replacement_and_scorer_paths_fail_closed(bundle, tmp_path):
    """CON-5; TC-8: each response binds the controller's original manifest, not a replacement."""
    with session(bundle, tmp_path) as controller:
        assert not controller.call(
            {"operation": "artifact", "arguments": {"path": "../run/episodes.jsonl"}}
        )["ok"]
        manifest = bundle / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        assert not controller.call({"operation": "journal", "arguments": {}})["ok"]


def test_refusals_consume_call_budget_and_closed_sessions_cannot_resume(bundle, tmp_path):
    """HAR-4; CON-8: refused requests stay in the denominator and cannot reset the budget."""
    with session(bundle, tmp_path, max_calls=1) as controller:
        assert not controller.call({"operation": "history", "arguments": {}})["ok"]
        with pytest.raises(ValueError, match="budget"):
            controller.call({"operation": "journal", "arguments": {}})
    with pytest.raises(ValueError, match="closed"):
        controller.call({"operation": "journal", "arguments": {}})
    with pytest.raises(FileExistsError):
        session(bundle, tmp_path)
    summary = json.loads((tmp_path / "s1/summary.json").read_text())
    assert summary["calls"] == 1 and summary["refused_calls"] == 1
    events = [json.loads(x) for x in (tmp_path / "s1/events.jsonl").read_text().splitlines()]
    assert events[-1] == {"kind": "stop", "reason": "call_budget_exhausted"}
    assert summary["budget_stopped"] is True
    assert summary["provider_calls"] == 0 and summary["measured_pilot"] is False


def test_mock_run_retains_all_planned_slots_and_never_dispatches_network(
    bundle, tmp_path, monkeypatch
):
    """HAR-4; CON-8: synthetic paired checks retain errors and cannot become pilot measurements."""
    import socket

    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network used"))
    result = run_mock(bundle, tmp_path / "mock", blocks=2)
    assert result["ok"] and result["mock_sessions"] == 4
    assert result["measured_sessions"] == 0 and result["provider_calls"] == 0
    assert result["provider_spend_dollars"] == 0 and result["hard_dollar_ceiling"] == 200
    assert result["p3_complete"] is False and result["p4_on_hold"] is True
    assert len(result["slots"]) == 4
    assert all(s["status"] == "mock_checks_passed" for s in result["slots"])


def test_mock_run_keeps_failed_and_unstarted_slots(bundle, tmp_path, monkeypatch):
    """HAR-4: a fixture failure is retained and never replaced by a new mock session."""
    monkeypatch.setattr(
        EvidenceSession, "call", lambda *a: (_ for _ in ()).throw(ValueError("fixture"))
    )
    result = run_mock(bundle, tmp_path / "mock", blocks=2)
    assert result["ok"] is False
    assert [s["status"] for s in result["slots"]] == [
        "mock_failure",
        "unstarted",
        "unstarted",
        "unstarted",
    ]


def test_cli_json_and_no_live_mode(bundle, tmp_path, capsys):
    """CON-8: CLI emits a synthetic-only result, with no live provider option."""
    assert main(["--bundle", str(bundle), "--out", str(tmp_path / "cli"), "--blocks", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["measured_sessions"] == 0


@pytest.mark.parametrize("payload", [[], "history", None, 1, {"operation": "journal"}])
def test_malformed_requests_are_retained_refusals(bundle, tmp_path, payload):
    """CON-8; HAR-4: malformed participant requests cannot crash or bypass retention."""
    with session(bundle, tmp_path) as controller:
        assert controller.call(payload) == {"ok": False, "error": "evidence_request_refused"}
    events = (tmp_path / "s1/events.jsonl").read_text().splitlines()
    assert len(events) == 2
