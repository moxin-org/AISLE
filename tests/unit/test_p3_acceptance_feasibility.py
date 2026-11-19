"""Private P3 draft acceptance arithmetic and retained calibration sensitivity."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from p3_acceptance_feasibility import (
    calibration_feasibility,
    main,
    read_calibration,
    score_session,
    verify_archive,
)
from p3_evidence import digest

pytestmark = pytest.mark.unit


def records(successes=6, count=8):
    result = []
    for i in range(count):
        seed = 300 + 4 * i
        verdicts = [
            dict(seed=seed, goal_id="initial", verifier="oracle", status="success", failure=None),
            dict(
                seed=seed,
                goal_id="recovery",
                verifier="oracle",
                status="success" if i < successes else "fail",
                failure=None if i < successes else "collision",
            ),
        ]
        result.append(
            dict(
                seed=seed,
                status="complete",
                success=i < successes,
                goal_ids=["initial", "recovery"],
                verdicts=verdicts,
                recovery_sequence=i < successes,
            )
        )
    return result


@pytest.mark.parametrize("successes,expected", [(5, False), (6, True), (8, True)])
def test_fixed_eight_chain_denominator(successes, expected):
    """CON-5; TC-8: the draft threshold is six complete two-goal successes out of eight."""
    rows = records(successes)
    result = score_session(rows, [r["seed"] for r in rows])
    assert result["success"] is expected
    assert result["successful_chains"] == successes
    assert result["outcomes"] == rows


@pytest.mark.parametrize("goal", [0, 1])
def test_no_wrong_object_exemption_for_either_goal(goal):
    """TC-8: initial scripted delivery is classified by the oracle, never waived here."""
    rows = records(8)
    rows[0]["verdicts"][goal].update(status="fail", failure="wrong_object")
    rows[0]["success"] = False
    result = score_session(rows, [r["seed"] for r in rows])
    assert result["successful_chains"] == 7
    assert result["wrong_object_events"] == 1
    assert result["success"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0]),
        lambda rows: rows[0].update(seed=999),
        lambda rows: rows[0].update(status="instrument_failure", error="capture lost"),
        lambda rows: rows[0].update(verdicts=[]),
        lambda rows: rows[0].update(success=False),
        lambda rows: rows[0]["verdicts"][0].update(verifier="realistic"),
        lambda rows: rows[0]["verdicts"][0].update(seed=304),
        lambda rows: rows[0]["verdicts"][0].update(goal_id="recovery"),
        lambda rows: rows[0]["verdicts"][0].update(status="success", failure="collision"),
        lambda rows: rows[0]["verdicts"][0].update(status="fail", failure=None),
        lambda rows: rows[0].update(seed=True),
    ],
)
def test_unusable_evidence_is_not_a_negative_trial(mutation):
    """HAR-4; TC-8: missing, duplicated or malformed slots retain null scores and outcomes."""
    rows = records()
    seeds = [r["seed"] for r in rows]
    mutation(rows)
    result = score_session(rows, seeds)
    assert result["status"] == "unscorable"
    assert result["success"] is result["successful_chains"] is None
    assert result["outcomes"] == rows
    assert result["errors"]


@pytest.mark.parametrize("seeds", [list(range(7)), [300] * 8, [True] + list(range(7))])
def test_invalid_plan_is_refused(seeds):
    """CON-5: the denominator cannot be adjusted to the evidence available."""
    with pytest.raises(ValueError, match="eight"):
        score_session([], seeds)


def test_exhaustive_subsets_are_not_independent_sessions():
    """CON-5; TC-8: deterministic sensitivity retains public calibration roles and all slots."""
    rows = records(8, 12)
    seeds = [r["seed"] for r in rows]
    result = calibration_feasibility(rows, seeds)
    assert result["subset_count"] == 495
    assert result["passing_subsets"] == 201
    assert result["successful_chain_histogram"] == {4: 70, 5: 224, 6: 168, 7: 32, 8: 1}
    assert result["measured_sessions"] == 0
    assert result["session_integration_complete"] is False
    assert result == calibration_feasibility(list(reversed(rows)), seeds)
    rows[0]["status"] = "instrument_failure"
    with pytest.raises(ValueError, match="complete"):
        calibration_feasibility(rows, seeds)


def archive_fixture(tmp_path):
    (tmp_path / "batch").mkdir()
    (tmp_path / "batch/protocol.json").write_text("{}")
    catalog = dict(
        files=[
            dict(
                path="batch/protocol.json", bytes=2, sha256=digest(tmp_path / "batch/protocol.json")
            )
        ]
    )
    (tmp_path / "archive-catalog.json").write_text(json.dumps(catalog))
    return digest(tmp_path / "archive-catalog.json")


def test_archive_requires_external_pin_and_preserves_hash_binding(tmp_path):
    """CON-5; HAR-4: an internally rewritten catalog cannot replace the retained external pin."""
    pin = archive_fixture(tmp_path)
    assert verify_archive(tmp_path, pin) == {"batch/protocol.json"}
    with pytest.raises(ValueError, match="catalog"):
        verify_archive(tmp_path, "0" * 64)
    (tmp_path / "batch/protocol.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact"):
        verify_archive(tmp_path, pin)


@pytest.mark.parametrize("name", ["../outside", "/outside", "batch/../protocol.json"])
def test_catalog_rejects_unsafe_paths_before_read(tmp_path, name):
    """HAR-4: retained evidence references cannot traverse outside the archive."""
    (tmp_path / "archive-catalog.json").write_text(
        json.dumps(dict(files=[dict(path=name, bytes=0, sha256="0" * 64)]))
    )
    with pytest.raises(ValueError, match="path"):
        verify_archive(tmp_path, digest(tmp_path / "archive-catalog.json"))


def test_json_cli_failure_does_not_emit_a_score(tmp_path, capsys):
    """CON-8: missing evidence produces a single JSON refusal with a nonzero exit."""
    assert (
        main(
            [
                "--archive",
                str(tmp_path),
                "--catalog-sha256",
                "0" * 64,
                "--protocol-sha256",
                "0" * 64,
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert "feasibility" not in result


def partial_archive(tmp_path):
    from p3_calibration_evidence import assess_run
    from test_p3_calibration_evidence import calibration_run_fixture

    run, pins = calibration_run_fixture(tmp_path / "fixture")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["env_hash"] = "d" * 64
    (run / "manifest.json").write_text(json.dumps(manifest))
    archive = tmp_path / "archive"
    destination = archive / "runs/fixture-seed200"
    destination.parent.mkdir(parents=True)
    run.rename(destination)
    assess_run(destination, archive / "batch/seed-200/evidence", **pins)
    protocol = dict(
        **{k: v for k, v in pins.items() if k != "seed"},
        format="aisle_p3_calibration_plan",
        seed_role="public calibration only; never pilot or confirmatory holdout",
        seeds=list(range(200, 248, 4)),
        engine="genesis",
        perception="L1",
        env_hash="d" * 64,
        run_id_prefix="fixture",
    )
    (archive / "batch/protocol.json").write_text(json.dumps(protocol))
    return archive


def seal_archive(archive):
    entries = [
        dict(path=p.relative_to(archive).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
        for p in sorted(archive.rglob("*"))
        if p.is_file() and p.name != "archive-catalog.json"
    ]
    (archive / "archive-catalog.json").write_text(json.dumps(dict(files=entries)))
    return digest(archive / "archive-catalog.json"), digest(archive / "batch/protocol.json")


def test_source_reader_requires_all_planned_chains(tmp_path):
    """HAR-4; TC-8: a real source-bound chain is verified but cannot replace eleven absent slots."""
    archive = partial_archive(tmp_path)
    pins = seal_archive(archive)
    # The first real Arrow/journal chain passes; the next planned slot is absent.
    with pytest.raises(ValueError, match="required evidence absent"):
        read_calibration(archive, *pins)


@pytest.mark.parametrize("target", ["assessment", "source", "protocol"])
def test_pinned_but_inconsistent_evidence_is_rejected(tmp_path, target):
    """CON-5; HAR-4; TC-8: hashes do not replace recomputation or cross-source identity checks."""
    archive = partial_archive(tmp_path)
    if target == "assessment":
        path = archive / "batch/seed-200/evidence/assessment.json"
        data = json.loads(path.read_text())
        data["success"] = True
        message = "assessment differs"
    elif target == "source":
        path = archive / "runs/fixture-seed200/manifest.json"
        data = json.loads(path.read_text())
        data["git_sha"] = "f" * 40
        message = "source manifest mismatch"
    else:
        path = archive / "batch/protocol.json"
        data = json.loads(path.read_text())
        data["env_fingerprint"] = "e" * 64
        message = "provenance mismatch"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        read_calibration(archive, *seal_archive(archive))


def test_json_cli_success_still_reports_zero_sessions(monkeypatch, capsys):
    """CON-8: a successful descriptive calculation cannot masquerade as measured sessions."""
    rows = records(8, 12)
    monkeypatch.setattr(
        "p3_acceptance_feasibility.read_calibration",
        lambda *args: (rows, [r["seed"] for r in rows]),
    )
    assert (
        main(["--archive", "unused", "--catalog-sha256", "a" * 64, "--protocol-sha256", "b" * 64])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["feasibility"]["measured_sessions"] == 0
