"""The REAL sim_warmup build test (G1-H).

Moved from tests/unit per the 01:15 outer-loop review: this case builds
the scene (warm ~20 s, cold ~8 min) — sim-class cost, never `pytest -m
unit`. The unit file keeps only the shape cases that never build.
"""

import importlib.util
import json

import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None, reason="sim extra not installed"
    ),
]


def test_cli_is_json_on_stdout_exit_zero_on_ok():
    """CON-8: the default invocation reports JSON with ok:true, exit 0 —
    the one real warmup build, also serving as the pre-rollout cache
    check on this machine."""
    from cli_helpers import run_tool

    proc = run_tool("sim_warmup.py", "--embodiment", "g1_right")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["embodiment"] == "g1_right"
    assert report["stages"]["n_dofs"] == 33
    assert isinstance(report["cache"], bool)
