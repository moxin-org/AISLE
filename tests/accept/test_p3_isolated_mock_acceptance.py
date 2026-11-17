"""Local macOS capability evidence only; no provider, simulator or scored pilot."""

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from p3_evidence import digest, export_policy
from p3_isolated_mock import MOCK_SOURCE, launch_client, run_mock
from test_p3_evidence import run_fixture

from aisle.harness.matched_runtime import capture_runtime

pytestmark = [
    pytest.mark.accept,
    pytest.mark.skipif(sys.platform != "darwin", reason="requires actual macOS confinement"),
]


def test_real_isolated_pair_and_generated_workers(tmp_path):
    """HAR-4/6; TRT-5/7/9: both roles have fresh authority and identical raw access."""
    bundle = tmp_path / "policy"
    export_policy(run_fixture(tmp_path), bundle)
    result = run_mock(bundle, tmp_path / "paired", blocks=1)
    assert result["ok"], result
    assert len(result["slots"]) == 4
    assert len({slot["source_sha256"] for slot in result["slots"]}) == 1
    assert result["measured_sessions"] == result["provider_calls"] == 0
    assert result["p3_complete"] is False and result["p4_on_hold"] is True
    for slot in result["slots"]:
        private = tmp_path / "paired/private" / slot["id"]
        launch = json.loads((private / "result.json").read_text())
        assert launch["confinement_verified"] and launch["client"]["ok"]
        audit = json.loads((private / "attestation.json").read_text())
        assert audit["capability_pass"]
        events = [
            json.loads(line)
            for line in (private / "evidence/events.jsonl").read_text().splitlines()
        ]
        operations = [row["request"]["operation"] for row in events if row["kind"] == "request"]
        assert operations == ["journal", "raw", "artifacts", "artifact", "history", "history"]


def test_real_client_denies_private_files_network_fork_and_ambient_secrets(tmp_path, monkeypatch):
    """TRT-5/6/7; TC-8: actual client denies scorer/peer/credentials and extra processes."""
    private = tmp_path / "hidden"
    private.mkdir()
    secret = private / "synthetic-scorer-peer-credential.txt"
    secret.write_text("SYNTHETIC-PRIVATE")
    monkeypatch.setenv("P3_SYNTHETIC_CREDENTIAL", "DO-NOT-INHERIT")
    bundle = tmp_path / "policy"
    export_policy(run_fixture(tmp_path), bundle)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        # Positive controls: the test controller can read and reach both fixtures.
        assert secret.read_text() == "SYNTHETIC-PRIVATE"
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            connection, _ = server.accept()
            connection.close()
        denied = [
            str(secret),
            str(bundle / "manifest.json"),
            str(Path(__file__).resolve().parents[2] / "tools/p3_mock_session.py"),
        ]
        probe = f"""
import os,socket
from pathlib import Path
assert "P3_SYNTHETIC_CREDENTIAL" not in os.environ
assert "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
for name in {denied!r}:
    try: Path(name).read_bytes()
    except PermissionError: pass
    else: raise RuntimeError("private read allowed")
try: socket.create_connection(("127.0.0.1",{port}),timeout=0.2)
except PermissionError: pass
else: raise RuntimeError("network allowed")
try: child=os.fork()
except OSError: pass
else:
    if child==0: os._exit(0)
    os.waitpid(child,0)
    raise RuntimeError("fork allowed")
"""
        result = launch_client(
            source=probe.encode() + MOCK_SOURCE,
            workspace=tmp_path / "workspace",
            controller=tmp_path / "controller",
            bundle=bundle,
            manifest_sha256=digest(bundle / "manifest.json"),
            arm="trace",
            session_id="denial-fixture",
            runtime=capture_runtime([Path(sys.base_prefix).resolve()]),
            hidden_roots=(private, Path(__file__).resolve().parents[2]),
        )
    assert result["ok"], result
