"""Unit tests for tools/g1_runtime.py's install-provenance reader (G1-D,
ADR-g1-driver).

The Unitree SDK2 lives outside the uv lock (hardware-only, cyclonedds pin),
so `uv.lock` says nothing about it and the receipt is gitignored: these
tests pin the reader the run manifest records and its CON-8 CLI contract.
No network, no install — receipt read/write and command behavior only."""

import json
import sys
from pathlib import Path

import pytest
from cli_helpers import REPO_ROOT, run_tool

pytestmark = pytest.mark.unit

RECEIPT_NAME = ".g1-runtime-receipt.json"


def runtime_module():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import g1_runtime

    return g1_runtime


def write_receipt(root: Path, payload: dict) -> Path:
    path = root / RECEIPT_NAME
    path.write_text(json.dumps(payload))
    return path


def test_pins_are_explicit_and_receipt_reader_carries_provenance(tmp_path):
    """G1-D/CON-5: the SDK2 commit and cyclonedds pin are named constants,
    and a run manifest can carry exactly what was installed."""
    module = runtime_module()
    assert module.SDK2_COMMIT == "814556d15970dd2ecf1c9984e845ca02ab07e206"
    assert module.CYCLONEDDS_PIN == "0.10.2"
    payload = {
        "schema_version": 1,
        "sdk2_commit": module.SDK2_COMMIT,
        "sdk2_repo": module.SDK2_REPO,
        "version": "1.0.1",
        "cyclonedds": "0.10.2",
        "platform": "darwin-arm64",
    }
    path = write_receipt(tmp_path, payload)

    result = module.read_receipt(tmp_path)

    assert result == {
        "installed": True,
        "path": str(path),
        "receipt": payload,
        "problem": None,
    }


def test_read_receipt_reports_absence_and_corruption_as_facts(tmp_path):
    """CON-8: never-installed or unparsable provenance is a reported
    problem, never an exception escaping into the rollout."""
    module = runtime_module()

    missing = module.read_receipt(tmp_path)
    assert missing["installed"] is False and missing["receipt"] is None
    assert "no g1 runtime receipt" in missing["problem"]

    (tmp_path / RECEIPT_NAME).write_text("{not json")
    broken = module.read_receipt(tmp_path)
    assert broken["installed"] is False and "unreadable" in broken["problem"]

    (tmp_path / RECEIPT_NAME).write_text('["a list"]')
    wrong_shape = module.read_receipt(tmp_path)
    assert wrong_shape["installed"] is False and "malformed" in wrong_shape["problem"]


def test_receipt_cli_emits_json_and_exits_on_the_verdict():
    """CON-8: JSON on stdout, logs to stderr, exit 0 iff a usable receipt
    exists — the same contract every AISLE CLI holds to."""
    proc = run_tool("g1_runtime.py", "receipt")
    report = json.loads(proc.stdout)
    assert report["ok"] is report["installed"]
    assert proc.returncode == (0 if report["installed"] else 1)
    assert report["path"].endswith(RECEIPT_NAME)


def test_gitignore_covers_the_gitignored_receipt():
    """G1-D: the receipt is deliberately untracked (out-of-lock tooling,
    machine-local like the ADR-67 wheel receipts) — pin that it stays
    ignored so a stray install never dirties a commit."""
    text = (REPO_ROOT / ".gitignore").read_text()
    assert RECEIPT_NAME in text
