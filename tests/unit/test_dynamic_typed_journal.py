"""MON-12/MON-13: ordinary typed runs retain real child preparation and refusals.

Outer participant admission is an explicit engineering fixture. Validation and
worker declarations use actual macOS capability probes; no simulation is claimed.
"""

import copy
import hashlib
import importlib
import json
import shutil
import sys
import venv
from pathlib import Path

import pytest
from test_monolith_worker_launch import _worker_interpreter
from test_typed_run_prepare import _controller
from test_typed_validation_snapshot import ROOT

pytestmark = pytest.mark.unit


@pytest.mark.skipif(sys.platform != "darwin", reason="actual worker capability")
def test_ordinary_typed_child_retains_preparation_and_gate_refusal(tmp_path):
    """MON-12: reserved run, validated snapshot, actual child and refusal remain auditable."""
    from aisle.harness.matched_evidence import audit_tool_journal
    from aisle.harness.matched_runtime import capture_runtime
    from aisle.harness.matched_session import admit_pair
    from aisle.harness.treatment_confinement import SANDBOX_EXEC, MacOSPolicy, compile_macos_profile
    from aisle.harness.worker_capability import audit_worker_capability

    controller, views, output, validation = _controller(tmp_path)
    shutil.copytree(
        ROOT / "src/aisle",
        controller.root / "src/aisle",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    # This fixture exercises the retained env-hash refusal.  Give the rollout
    # subprocess an importable engine sentinel so the earlier availability
    # gate does not replace the refusal under test when the optional wheel is
    # absent.  Keep it out of the sealed interpreter runtime below.
    (controller.root / "src/genesis.py").write_text("__version__ = 'test-sentinel'\n")
    packages = tmp_path / "runtime-packages"
    packages.mkdir()
    for name in (
        "yaml",
        "jsonschema",
        "jsonschema_specifications",
        "referencing",
        "rpds",
        "attrs",
        "attr",
        "typing_extensions",
        "numpy",
        "pyarrow",
    ):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if name == "typing_extensions" and exc.name == name and sys.version_info >= (3, 13):
                continue
            raise
        source = Path(module.__file__)
        if hasattr(module, "__path__"):
            shutil.copytree(
                source.parent, packages / name, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copyfile(source, packages / source.name)
    runtime_env = tmp_path / "runtime-env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime_env)
    direct_python, runtime_root = _worker_interpreter()
    # venv writes `home` from the unresolved base executable. uv-managed
    # interpreters are reached through a minor-version symlink
    # (cpython-3.13-... -> cpython-3.13.5-...), which lies outside the
    # sandbox's runtime read roots (those are resolved paths), so the confined
    # validator could not find its standard library (#598).
    config = runtime_env / "pyvenv.cfg"
    config.write_text(
        "".join(
            f"home = {direct_python.parent}\n" if line.startswith("home = ") else line
            for line in config.read_text().splitlines(keepends=True)
        )
    )
    python = runtime_env / "bin/python"
    python.unlink()
    python.symlink_to(direct_python)
    site = next((runtime_env / "lib").glob("python*/site-packages"))
    site.rmdir()
    site.symlink_to(packages, target_is_directory=True)
    runtime = capture_runtime((runtime_root, runtime_env, packages))
    validation = copy.deepcopy(validation)
    validation["python"] = str(python)
    validation["python_sha256"] = hashlib.sha256(python.read_bytes()).hexdigest()
    private = tmp_path / "actual-validator-audit"
    private.mkdir()
    policy = MacOSPolicy(
        **{
            key: value if key == "network_policy" else tuple(Path(p) for p in value)
            for key, value in validation["policy"].items()
        }
    )
    from dataclasses import replace

    policy = replace(
        policy,
        runtime_read_roots=tuple(Path(p) for p in runtime["trees"]),
        allowed_executables=(direct_python,),
        hidden_roots=(*policy.hidden_roots, private),
    )
    validation["policy"] = policy.canonical_dict()
    Path(validation["profile_path"]).write_text(compile_macos_profile(policy).text)
    validation["attestation"] = audit_worker_capability(
        policy=policy,
        profile_path=validation["profile_path"],
        python=python,
        environment=validation["environment"],
        environment_record=validation["environment_record"],
        output=private / "capability",
    )
    assert validation["attestation"]["capability_pass"]
    run_bindings = copy.deepcopy(controller.plan["run_controller"])
    for binding in run_bindings.values():
        binding["python"] = str(python)
        binding["python_sha256"] = hashlib.sha256(python.read_bytes()).hexdigest()
    candidates = copy.deepcopy(controller.plan["arms"])
    for candidate in candidates.values():
        candidate.pop("immutable_id")
        candidate["repository"].pop("visible_files")
        for binary in candidate["runtime_binaries"]:
            if binary["name"] == "harness-python":
                binary["sha256"] = validation["python_sha256"]
        candidate["confinement"]["adapter_binary_sha256"] = hashlib.sha256(
            SANDBOX_EXEC.read_bytes()
        ).hexdigest()
    controller.plan = admit_pair(
        controller.root,
        candidates,
        views,
        confinement=controller.plan["confinement_bindings"],
        ambient=controller.plan["ambient_bindings"],
        development=controller.plan["development"],
        tool_runtime=runtime,
        typed_validation=validation,
        run_controller=run_bindings,
    )
    controller.worker_preparations = [
        {
            "provider": {
                "allocation_root": str(tmp_path / "workers"),
                "timeout_s": 5,
                "max_calls": 1000,
            }
        }
    ]
    result = controller.run()
    assert result["preparation"]["validation"]["ok"], result
    assert result["reservation"] == {"runs": 1, "episodes": 1}
    assert result["process"]["rc"] == 1, result
    assert result["result"].get("refused"), result
    assert result["result"]["refused"]["gate"] == "env_hash", result
    provider = output / "tool-000001/run-controller/typed-provider/launch-0/result.json"
    assert json.loads(provider.read_text())["ok"], result
    audit = audit_tool_journal(
        output,
        session_id=controller.session_id,
        plan_id=controller.plan["immutable_id"],
        arm="typed",
        development=controller.plan["development"],
    )
    assert audit["ok"], audit["error"]

    def reaudit():
        return audit_tool_journal(
            output,
            session_id=controller.session_id,
            plan_id=controller.plan["immutable_id"],
            arm="typed",
            development=controller.plan["development"],
        )

    extra = provider.parent / "extra.txt"
    extra.write_text("not indexed")
    assert "provider evidence inventory differs" in reaudit()["error"]
    extra.unlink()
    original = provider.read_bytes()
    receipt = json.loads(original)
    receipt["snapshot_id"] = "different-snapshot"
    provider.write_text(json.dumps(receipt))
    assert "provider receipt differs from sealed run" in reaudit()["error"]
    provider.write_bytes(original)
    assert reaudit()["ok"]
