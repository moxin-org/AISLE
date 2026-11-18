"""CON-3/8, TC-7: JIT development launches source, never a native fallback."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "tools/aisle-jit"
pytestmark = pytest.mark.unit


@pytest.fixture
def environment(tmp_path):
    runner = tmp_path / "runner with spaces"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "print(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd(),"
        "'root':os.environ.get('AISLE_JIT_ROOT'),'cache':os.environ.get('RELAX_CACHE_DIR'),"
        "'std':os.environ.get('RELAX_STD'),"
        "'launcher':os.environ.get('AISLE_JIT_LAUNCHER')}))\n"
    )
    runner.chmod(0o755)
    return dict(
        os.environ, AISLE_JIT_RUNNER=str(runner), AISLE_JIT_CACHE_ROOT=str(tmp_path / "cache")
    )


def run(args, environment, cwd=None):
    return subprocess.run(
        [str(LAUNCHER), *args],
        env=environment,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_launches_source_and_preserves_argument_boundaries(environment):
    """CON-3/8: application arguments are argv, never shell-evaluated text."""
    result = run(["nano", "run", "model", "--model", "/tmp/model with spaces.urdf"], environment)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["args"] == [
        "run",
        "--allow-c",
        "--ahead=minimal",
        "aisle-nano",
        "--no-default-features",
        "--bin",
        "aisle-nano-dev",
        "--",
        "model",
        "--model",
        "/tmp/model with spaces.urdf",
    ]
    assert value["cwd"] == str(ROOT / "native/nano")
    assert value["root"] == str(ROOT)
    assert value["launcher"] == str(LAUNCHER)
    assert value["cache"].endswith("/cache/nano")
    assert "--unattended" not in value["args"]  # live edits must remain enabled


def test_checks_and_live_tests_use_relax(environment):
    """CON-8: checking and integration tests do not substitute cargo binaries."""
    result = run(["octos", "check"], environment)
    assert result.returncode == 0
    assert json.loads(result.stdout)["args"] == [
        "check",
        "--allow-c",
        "aisle-octos",
        "--no-default-features",
        "--bin",
        "aisle-octos",
    ]
    result = run(["nano", "test-live", "--test", "skills"], environment)
    assert result.returncode == 0
    assert json.loads(result.stdout)["args"] == [
        "test",
        "--allow-c",
        "--live",
        "aisle-nano",
        "--no-default-features",
        "--test",
        "skills",
    ]


def test_missing_runner_is_error_without_building(environment, tmp_path):
    """CON-8: an unavailable runner never starts a native application build."""
    environment["AISLE_JIT_RUNNER"] = str(tmp_path / "missing")
    result = run(["nano", "run", "--help"], environment)
    assert result.returncode != 0
    assert json.loads(result.stdout)["ok"] is False


def test_bad_component_and_mode_fail(environment):
    """CON-8: mistyped development commands fail with a structured error."""
    for args in [["unknown", "run"], ["nano", "unknown"]]:
        result = run(args, environment)
        assert result.returncode != 0
        assert json.loads(result.stdout)["ok"] is False


def test_dora_mode_prepares_private_std_and_keeps_transport(environment, tmp_path):
    """CON-3/5: Dora uses JIT plus the upstream std fix, without changing installed std."""
    source = tmp_path / "installed std"
    (source / "core/slice").mkdir(parents=True)
    original = """impl<T> [T] {
    pub fn element_offset(&self, element: &T) -> Option<usize> {
        let offset = byte_offset / size;
        if offset < self.len() { Some(offset) } else { None }
    }
    #[track_caller]
    pub fn swap(&mut self, a: usize, b: usize) {
        super::swap(self, a, b)
    }
}
"""
    path = source / "core/slice/inherent.rs"
    path.write_text(original)
    environment["AISLE_JIT_STD_SOURCE"] = str(source)
    result = run(["nano", "dora-run", "node", "guard"], environment)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    assert value["args"] == [
        "run",
        "--allow-c",
        "--ahead=minimal",
        "aisle-nano",
        "--bin",
        "aisle-nano",
        "--",
        "node",
        "guard",
    ]
    assert path.read_text() == original
    assert Path(value["std"]) != source
    assert "as_mut_ptr_range" in (Path(value["std"]) / "core/slice/inherent.rs").read_text()
    checked = run(["octos", "dora-test", "--test", "dora_flow"], environment)
    assert checked.returncode == 0
    assert "--no-default-features" not in json.loads(checked.stdout)["args"]
    environment["AISLE_JIT_CACHE_ROOT"] = "relative cache"
    relative = run(["nano", "dora-run", "node", "guard"], environment, cwd=tmp_path)
    assert relative.returncode == 0, relative.stdout + relative.stderr
    value = json.loads(relative.stdout)
    expected = (tmp_path / "relative cache").resolve()
    assert Path(value["std"]).parent == expected
    assert Path(value["cache"]) == expected / "nano"


def test_dora_mode_cannot_copy_relax_as_native_rollout(environment):
    """CON-5/8: current_exe() under JIT is not a frozen Nano binary."""
    result = run(["nano", "dora-run", "rollout"], environment)
    assert result.returncode != 0
    assert json.loads(result.stdout)["ok"] is False
