"""Actual P3 participant/generated-worker confinement with synthetic T4 transport."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from p3_evidence import digest, export_policy
from p3_isolated_mock import launch_client
from p3_task_mock import ScriptedTaskTransport, run_mock
from test_p3_evidence import run_fixture

from aisle.harness.matched_runtime import capture_runtime
from aisle.harness.p3_task_boundary import TaskSession

pytestmark = [
    pytest.mark.accept,
    pytest.mark.skipif(sys.platform != "darwin", reason="actual macOS confinement"),
]


def test_isolated_generated_workers_dispatch_t4_requests_with_parity(tmp_path):
    """TRT-5/6/7/9/10; TC-6/7/8: source-bound isolated workers get equal public task authority."""
    bundle = tmp_path / "bundle"
    export_policy(run_fixture(tmp_path), bundle)
    result = run_mock(bundle, tmp_path / "paired")
    assert result["ok"], result
    assert len(result["slots"]) == 4
    assert result["measured_sessions"] == result["provider_calls"] == 0
    assert result["guard_traversal_verified"] is False
    for slot in result["slots"]:
        controller = tmp_path / "paired/private" / slot["id"]
        launch = json.loads((controller / "result.json").read_text())
        assert launch["confinement_verified"] and launch["ok"]
        assert json.loads((controller / "attestation.json").read_text())["capability_pass"]
        if slot["role"] == "generated-worker":
            task = json.loads((controller / "task/summary.json").read_text())
            assert task["source_sha256"] == slot["source_sha256"]
            assert task["ok"] and task["motion_requests"] == 3


def test_real_task_worker_cannot_finish_before_consuming_inputs(tmp_path):
    """TRT-9/10: source execution success alone cannot certify a completed task exchange."""
    bundle = tmp_path / "bundle"
    export_policy(run_fixture(tmp_path), bundle)
    source = (
        b'import sys\nb=b\'{"kind":"finished"}\'\n'
        b'sys.stdout.buffer.write(len(b).to_bytes(4,"big")+b)\n'
        b"sys.stdout.buffer.flush()\n"
    )

    def factory(evidence, path, source_hash):
        return TaskSession(
            evidence=evidence,
            transport=ScriptedTaskTransport(),
            output=path,
            source_sha256=source_hash,
            medicines=["a", "b"],
        )

    result = launch_client(
        source=source,
        workspace=tmp_path / "workspace",
        controller=tmp_path / "controller",
        bundle=bundle,
        manifest_sha256=digest(bundle / "manifest.json"),
        arm="trace",
        session_id="early-worker",
        runtime=capture_runtime([Path(sys.base_prefix).resolve()]),
        hidden_roots=(Path(__file__).resolve().parents[2],),
        service_factory=factory,
    )
    assert not result["ok"] and result["client"]["ok"]
    task = json.loads((tmp_path / "controller/task/summary.json").read_text())
    assert not task["ok"] and not task["input_exhausted"]
    assert "exhaustion" in task["error"]
