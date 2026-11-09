"""The run records which engine realization produced it (ADR-67, ADR-70, CON-5).

`env_hash` is engine neutral by construction: the frozen set does not contain
`src/aisle/sim`, so two runs on different engines, or on the same engine with
different solver settings, hash identically. The gate therefore asks the
trusted checker for the engine digest as well and carries it into the manifest.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aisle.harness import rollout as rollout_module

pytestmark = pytest.mark.unit

ENGINE_FACTS = {
    "engine": "nexus",
    "sim_engine_hash": "b" * 64,
    "n_files": 3,
}


@pytest.fixture
def captured_hash_cmd(monkeypatch):
    """Answer the trusted env_hash checker with a canned report, keeping the
    argv it was called with. Every other subprocess call is refused so the
    test cannot silently exercise something else. The engine wheels ride the
    sim extra, which the unit tier does not install (ADR-70)."""
    monkeypatch.setattr("aisle.sim.engine_available", lambda engine: True)
    seen: dict = {}
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and str(cmd[-1]).endswith("env_hash.py"):
            raise AssertionError(f"unexpected env_hash invocation shape: {cmd}")
        if isinstance(cmd, list) and any(str(part).endswith("env_hash.py") for part in cmd):
            seen["cmd"] = [str(part) for part in cmd]
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"ok": True, "env_hash": "a" * 64, "sim": ENGINE_FACTS}),
                stderr="",
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(rollout_module.subprocess, "run", fake_run)
    return seen


def _gates(root: Path, engine: str) -> dict:
    return rollout_module.run_gates(
        root=root,
        graph=root / "graphs" / "expert_t0.yaml",
        branch="feat/test",
        no_idea_gate=True,
        env_baseline="local",
        sim_engine=engine,
    )


def test_gate_asks_the_checker_for_the_engine_digest(captured_hash_cmd):
    """ADR-67: the engine name reaches the trusted checker, so the digest it
    reports is the one for the engine this run will actually launch. Without
    it the checker would report the default engine's digest for a Nexus run,
    which is the recorded-vs-actual divergence the digest exists to close."""
    root = Path(__file__).resolve().parents[2]
    gates = _gates(root, "nexus")
    assert gates["ok"] is True, gates
    cmd = captured_hash_cmd["cmd"]
    assert "--sim-engine" in cmd
    assert cmd[cmd.index("--sim-engine") + 1] == "nexus"


def test_gate_carries_the_engine_digest_into_the_manifest_facts(captured_hash_cmd):
    """CON-5/ADR-67: the engine digest is a fact of the run, recorded verbatim
    beside the engine-neutral env_hash, which cannot tell engines apart."""
    root = Path(__file__).resolve().parents[2]
    gates = _gates(root, "nexus")
    assert gates["sim_engine_build"] == ENGINE_FACTS
    assert gates["sim_engine_build"]["sim_engine_hash"] != gates["env_hash"]


def test_gate_needs_no_receipt_for_a_locked_engine(monkeypatch):
    """CON-5/ADR-70: the engine wheels come from the lock, so the real trusted
    checker passes a Nexus run without any build receipt and reports the
    engine digest; the dist attestation covers the wheel's provenance."""
    monkeypatch.setattr("aisle.sim.engine_available", lambda engine: True)
    root = Path(__file__).resolve().parents[2]
    gates = _gates(root, "nexus")
    assert gates["ok"] is True, gates
    assert gates["sim_engine_build"]["engine"] == "nexus"
    assert "build" not in gates["sim_engine_build"]
