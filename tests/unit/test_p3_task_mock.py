"""Generated T4 task worker over real framed pipes; OS confinement is acceptance-tested."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_p3_evidence import run_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from p3_evidence import digest, export_policy
from p3_isolated_mock import capture_generated_source, supervise_client
from p3_mock_session import EvidenceSession
from p3_task_mock import WORKER_SOURCE, ScriptedTaskTransport, main, participant_source

from aisle.harness.p3_task_boundary import TaskSession

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arm", ["history", "trace"])
def test_generated_worker_controls_public_t4_sequence_through_host(tmp_path, arm):
    """TC-6/7/8; HAR-4: real worker pipes retain public requests and correlated completions."""
    bundle = tmp_path / "bundle"
    export_policy(run_fixture(tmp_path), bundle)
    transport = ScriptedTaskTransport()
    with EvidenceSession(
        bundle=bundle,
        manifest_sha256=digest(bundle / "manifest.json"),
        arm=arm,
        session_id="unit-worker",
        output=tmp_path / "evidence",
    ) as evidence:
        with TaskSession(
            evidence=evidence,
            transport=transport,
            output=tmp_path / "task",
            source_sha256="a" * 64,
            medicines=["a", "b"],
        ) as service:
            with (tmp_path / "stderr").open("wb") as stderr:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", "-c", WORKER_SOURCE.decode()],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    start_new_session=True,
                )
                result = supervise_client(process, service, tmp_path / "rpc", timeout_s=5)
            assert result["ok"], (result, (tmp_path / "stderr").read_text())
    assert [r["topic"] for r in transport.sent] == [
        "robot_msg",
        "target_request",
        "robot_msg",
        "return_request",
        "target_request",
    ]
    log = [json.loads(line) for line in (tmp_path / "task/events.jsonl").read_text().splitlines()]
    public = [row["response"] for row in log if row["kind"] == "response"]
    assert "PRIVATE-SENTINEL" not in json.dumps(public)
    assert "physical_completion" in json.dumps(public)
    assert all("episode_result" not in json.dumps(r) for r in public)
    summary = json.loads((tmp_path / "task/summary.json").read_text())
    assert summary["motion_requests"] == 3 and summary["input_exhausted"]
    assert summary["measured_sessions"] == 0


def test_participant_captures_task_worker_source_without_host_execution(tmp_path):
    """TRT-6/9; HAR-4: the participant produces a distinct worker artifact for later capture."""
    bundle = tmp_path / "bundle"
    export_policy(run_fixture(tmp_path), bundle)
    home = tmp_path / "home"
    home.mkdir()
    with EvidenceSession(
        bundle=bundle,
        manifest_sha256=digest(bundle / "manifest.json"),
        arm="trace",
        session_id="unit-participant",
        output=tmp_path / "evidence",
    ) as evidence:
        with (tmp_path / "stderr").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", participant_source().decode()],
                cwd=home,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
            result = supervise_client(process, evidence, tmp_path / "rpc", timeout_s=5)
        assert result["ok"], result
    assert process.poll() == 0
    assert capture_generated_source(home) == WORKER_SOURCE


def test_mock_cli_refuses_invalid_bundle_as_json(tmp_path, capsys):
    """CON-8: a bad evidence binding cannot start a process or yield a scored result."""
    assert main(["--bundle", str(tmp_path / "missing"), "--out", str(tmp_path / "out")]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False and result["measured_sessions"] == 0
    assert not (tmp_path / "out").exists()
