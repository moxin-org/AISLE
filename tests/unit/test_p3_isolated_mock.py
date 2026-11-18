"""P3 mock transport tests use real pipes, not a claim of OS confinement."""

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_p3_evidence import run_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from p3_evidence import digest, export_policy
from p3_mock_session import EvidenceSession

pytestmark = pytest.mark.unit

CLIENT = """
import json,sys
def exchange(value):
    data=json.dumps(value).encode()
    sys.stdout.buffer.write(len(data).to_bytes(4,'big')+data)
    sys.stdout.buffer.flush()
    n=int.from_bytes(sys.stdin.buffer.read(4),'big')
    return json.loads(sys.stdin.buffer.read(n))
def finish():
    data=b'{"kind":"finished"}'
    sys.stdout.buffer.write(len(data).to_bytes(4,'big')+data)
    sys.stdout.buffer.flush()
"""


@pytest.fixture
def evidence(tmp_path):
    bundle = tmp_path / "bundle"
    export_policy(run_fixture(tmp_path), bundle)
    return bundle


def run_child(tmp_path, evidence, source, *, arm="trace", timeout=0.8, calls=10):
    from p3_isolated_mock import supervise_client

    with EvidenceSession(
        bundle=evidence,
        manifest_sha256=digest(evidence / "manifest.json"),
        arm=arm,
        session_id="fixture",
        output=tmp_path / "evidence-log",
        max_calls=calls,
    ) as session:
        with (tmp_path / "stderr").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", CLIENT + source],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
            result = supervise_client(process, session, tmp_path / "rpc", timeout_s=timeout)
    assert process.poll() is not None
    assert json.loads((tmp_path / "rpc/result.json").read_text()) == result
    return result


@pytest.mark.parametrize("arm", ["history", "trace"])
def test_controller_owns_history_grant_and_retains_pipe_bytes(tmp_path, evidence, arm):
    """HAR-4/6; TC-8: the child cannot choose its grant; wire evidence is retained."""
    result = run_child(
        tmp_path,
        evidence,
        "reply=exchange({'kind':'call','request':{'operation':'history','arguments':{}}})\n"
        f"assert reply['ok'] is {arm == 'history'}\nfinish()\n",
        arm=arm,
    )
    assert result["ok"] and result["requests"] == 1
    assert result["provider_calls"] == 0 and result["measured_sessions"] == 0
    assert len(list((tmp_path / "rpc").glob("*.frame"))) == 3


@pytest.mark.parametrize(
    "source",
    [
        "sys.stdout.buffer.write((100000).to_bytes(4,'big'));sys.stdout.buffer.flush()",
        "sys.stdout.buffer.write(b'\\0\\0');sys.stdout.buffer.flush()",
        "finish();sys.stdout.buffer.write(b'bad');sys.stdout.buffer.flush()",
        "import time;time.sleep(10)",
        "exchange({'kind':'call','request':{'operation':'journal','arguments':{}}})\n"
        "exchange({'kind':'call','request':{'operation':'journal','arguments':{}}})\nfinish()",
    ],
)
def test_invalid_partial_trailing_timeout_and_budget_stop_retained(tmp_path, evidence, source):
    """HAR-4; CON-8: bad or stalled children terminate with retained failure evidence."""
    result = run_child(tmp_path, evidence, source, timeout=0.15, calls=1)
    assert not result["ok"] and result["error"]
    assert (tmp_path / "rpc/messages.jsonl").exists()


def test_json_frame_rejects_nonfinite_and_duplicate_keys():
    """CON-8: the child cannot send ambiguous/non-JSON values into the controller."""
    from p3_isolated_mock import receive_frame

    for payload in (b'{"x":NaN}', b'{"x":1,"x":2}', b"[]"):
        with pytest.raises(ValueError):
            receive_frame(io.BytesIO(len(payload).to_bytes(4, "big") + payload))


def test_generated_source_capture_refuses_links_and_large_files(tmp_path):
    """TRT-6/9: generated source is captured as bounded regular bytes, never followed."""
    from p3_isolated_mock import capture_generated_source

    home = tmp_path / "home"
    home.mkdir()
    source = home / "worker.py"
    source.write_text("print('fixture')\n")
    assert capture_generated_source(home) == source.read_bytes()
    source.unlink()
    source.symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError):
        capture_generated_source(home)
    source.unlink()
    source.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError):
        capture_generated_source(home)


def test_confinement_failure_prevents_client_spawn(tmp_path, evidence, monkeypatch):
    """TRT-5: no client starts when the required OS capability audit fails."""
    import p3_isolated_mock as tool

    calls = []

    def refuse(*args, **kwargs):
        raise RuntimeError("synthetic adapter refusal")

    # Admission failure is injected here; actual profile compilation and OS
    # authority are covered by the macOS acceptance tests, not Linux host paths.
    monkeypatch.setattr(
        tool, "compile_macos_profile", lambda policy: SimpleNamespace(text="synthetic-profile")
    )
    monkeypatch.setattr(tool, "run_macos_capability_audit", refuse)
    monkeypatch.setattr(tool, "spawn_isolated_process", lambda *a, **k: calls.append(a))
    result = tool.launch_client(
        source=b"pass\n",
        workspace=tmp_path / "work",
        controller=tmp_path / "controller",
        bundle=evidence,
        manifest_sha256=digest(evidence / "manifest.json"),
        arm="trace",
        session_id="fixture",
        runtime={"trees": {str(Path(sys.base_prefix).resolve()): {}}},
        hidden_roots=(),
    )
    assert not result["ok"] and not result["confinement_verified"]
    assert not calls and "adapter refusal" in result["error"]


def test_bound_evidence_drift_prevents_any_os_probe(tmp_path, evidence, monkeypatch):
    """CON-5; TC-8: all roles bind the same manifest before admission or source execution."""
    import p3_isolated_mock as tool

    monkeypatch.setattr(
        tool, "run_macos_capability_audit", lambda *a, **k: pytest.fail("audit used")
    )
    with pytest.raises(ValueError, match="evidence"):
        tool.launch_client(
            source=b"pass\n",
            workspace=tmp_path / "work",
            controller=tmp_path / "controller",
            bundle=evidence,
            manifest_sha256="0" * 64,
            arm="trace",
            session_id="fixture",
            runtime={"trees": {}},
            hidden_roots=(),
        )


def test_source_changed_during_admission_cannot_be_rebound(tmp_path, evidence, monkeypatch):
    """CON-5; TRT-9: preflight cannot silently adopt different participant source."""
    import p3_isolated_mock as tool

    def changed(policy, **kwargs):
        script = policy.visible_roots[0] / "client.py"
        script.chmod(0o644)
        script.write_bytes(b"raise RuntimeError('substituted')\n")
        return {}

    monkeypatch.setattr(
        tool, "compile_macos_profile", lambda policy: SimpleNamespace(text="synthetic-profile")
    )
    monkeypatch.setattr(tool, "run_macos_capability_audit", changed)
    monkeypatch.setattr(tool, "verify_runtime", lambda record: None)
    monkeypatch.setattr(tool, "wrap_verified_command", lambda *a, **k: pytest.fail("wrapped drift"))
    result = tool.launch_client(
        source=b"pass\n",
        workspace=tmp_path / "work",
        controller=tmp_path / "controller",
        bundle=evidence,
        manifest_sha256=digest(evidence / "manifest.json"),
        arm="trace",
        session_id="fixture",
        runtime={"trees": {str(Path(sys.base_prefix).resolve()): {}}},
        hidden_roots=(),
    )
    assert not result["ok"] and "source changed" in result["error"]


def test_cli_invalid_scope_emits_one_json_failure(tmp_path, capsys):
    """CON-8: malformed mock scope fails without creating output or launching clients."""
    from p3_isolated_mock import main

    output = tmp_path / "output"
    assert main(["--bundle", str(tmp_path), "--out", str(output), "--blocks", "0"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False and result["provider_calls"] == 0
    assert not output.exists()
