"""Unit tests for tools/sim_warmup.py (CON-8, CON-5).

Shape-only: the CLI contract (JSON on stdout, exit codes, arg handling)
is testable without genesis — the tool itself refuses to import genesis
until it builds, and these tests never pass --embodiment (they exercise
the argument surface and the ok:false path with an unavailable scene
module). No scene is ever built here.
"""

import json
import subprocess

import pytest
from cli_helpers import REPO_ROOT, run_tool

pytestmark = pytest.mark.unit


def run_warmup(*args: str) -> subprocess.CompletedProcess:
    return run_tool("sim_warmup.py", *args)


def test_cli_help_exits_zero_without_genesis():
    """CON-8: --help is pure argparse; no sim import happens."""
    proc = run_warmup("--help")
    assert proc.returncode == 0
    assert "--embodiment" in proc.stdout


def test_cli_reports_json_error_and_nonzero_when_scene_cannot_build(capsys):
    """CON-8: a build failure (here: an unknown embodiment) is JSON
    ok:false on stdout with a named error, exit 1 — never a bare
    traceback, and stdout stays machine-parseable."""
    proc = run_warmup("--embodiment", "humanoid")
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["ok"] is False
    assert "humanoid" in report["error"]
    assert report["embodiment"] == "humanoid"


def test_warmth_threshold_matches_the_measured_lane_numbers():
    """CON-5 provenance: the warm/cold threshold (60 s) sits between the
    measured warm scene build (~27 s, C1 evidence) and the cold build
    (~470 s, outer-loop probe) — generous enough for CI variance, tight
    enough that a cold build never reports warm."""
    source = (REPO_ROOT / "tools" / "sim_warmup.py").read_text()
    assert 'warm = stages["scene_build_s"] < 60.0' in source
