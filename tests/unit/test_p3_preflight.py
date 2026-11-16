"""Qualification preparation must exercise the launch admission gates."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from aisle.harness.ideas import close_idea, log_idea

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import p3_preflight as preflight

pytestmark = pytest.mark.unit


@pytest.fixture
def plan(tmp_path, monkeypatch):
    graph = tmp_path / "graph.yaml"
    graph.write_text("nodes: []\n")
    value = dict(
        status="prepared_not_authorized",
        instrument_worktree=str(tmp_path),
        instrument_commit="a" * 40,
        instrument_branch="codex/test",
        graph="graph.yaml",
        graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),
        protected_environment_baseline="b" * 40,
        env_fingerprint="c" * 64,
        env_hash="d" * 64,
        engine="genesis",
        library_hashes={},
        skill_sources={},
        assessment_sources={},
    )
    monkeypatch.setattr(preflight, "git_state", lambda root: ("a" * 40, "codex/test", False))
    monkeypatch.setattr(preflight, "check_runtime", lambda: {"ok": True})
    return value


def test_missing_closed_and_other_branch_ideas_refuse_before_gate_or_freeze(plan, monkeypatch):
    """HAR-2/8: an environment-only precheck cannot admit an absent branch idea."""
    monkeypatch.setattr(preflight, "run_gates", lambda *a, **k: pytest.fail("gate called"))
    root = Path(plan["instrument_worktree"])
    assert preflight.check(plan)["gate"] == "idea"
    log_idea(root, "other", "unrelated", "t0", "a" * 40)
    assert preflight.check(plan)["gate"] == "idea"
    log_idea(root, "codex/test", "test", "t0", "a" * 40)
    close_idea(root, "codex/test", "I1", "closed", "flat", "t1")
    assert preflight.check(plan)["gate"] == "idea"
    assert sorted(p.name for p in (root / "runs").iterdir()) == ["ideas"]


def test_open_idea_still_runs_real_launch_contract_with_one_graph_snapshot(plan, monkeypatch):
    """HAR-2; CON-5: preflight delegates all gates with the pinned graph and baseline."""
    root = Path(plan["instrument_worktree"])
    log_idea(root, "codex/test", "test", "t0", "a" * 40)
    seen = []

    def gates(*args, **kwargs):
        seen.append((args, kwargs))
        return dict(
            ok=True,
            idea="I1",
            env_attested=True,
            env_fingerprint=plan["env_fingerprint"],
            env_hash=plan["env_hash"],
        )

    monkeypatch.setattr(preflight, "run_gates", gates)
    result = preflight.check(plan)
    assert result["ok"] and result["simulation_launched"] is False
    assert result["collection_authorized"] is False
    assert seen[0][1]["no_idea_gate"] is False
    assert seen[0][1]["episodes"] == 1
    assert seen[0][1]["env_baseline"] == plan["protected_environment_baseline"]
    assert seen[0][1]["graph_snapshot"] == b"nodes: []\n"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "consumed_closed_instrument_failure_no_retry"),
        ("instrument_commit", "e" * 40),
        ("graph_sha256", "f" * 64),
        ("instrument_branch", "other"),
    ],
)
def test_closed_protocol_and_source_drift_cannot_pass(plan, field, value):
    """CON-5; HAR-2: preparation cannot reuse a consumed protocol or changed source."""
    log_idea(Path(plan["instrument_worktree"]), "codex/test", "test", "t0", "a" * 40)
    with pytest.raises(ValueError):
        preflight.check(plan | {field: value})


def test_later_launch_refusal_and_fingerprint_drift_remain_refusals(plan, monkeypatch):
    """HAR-2: an OPEN idea cannot mask budget, runtime, or attestation failures."""
    log_idea(Path(plan["instrument_worktree"]), "codex/test", "test", "t0", "a" * 40)
    monkeypatch.setattr(preflight, "run_gates", lambda *a, **k: dict(ok=False, gate="budget"))
    assert preflight.check(plan)["gate"] == "budget"
    monkeypatch.setattr(
        preflight,
        "run_gates",
        lambda *a, **k: dict(
            ok=True,
            idea="I1",
            env_attested=True,
            env_fingerprint="wrong",
            env_hash=plan["env_hash"],
        ),
    )
    assert preflight.check(plan)["gate"] == "attestation"


def test_cli_refusal_is_json_and_nonzero(tmp_path, capsys):
    """CON-8: malformed preparation input fails without creating a batch."""
    proposal = tmp_path / "bad.json"
    proposal.write_text("{}")
    assert preflight.main(["--proposal", str(proposal)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_refuses_malformed_source_maps(plan, tmp_path, capsys):
    """CON-8: invalid JSON shapes are structured refusals, never tracebacks."""
    proposal = tmp_path / "bad-map.json"
    proposal.write_text(json.dumps(plan | {"library_hashes": []}))
    assert preflight.main(["--proposal", str(proposal)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
