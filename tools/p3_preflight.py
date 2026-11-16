"""Check P3 launch prerequisites without freezing a batch or starting a rollout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from pathlib import Path

from aisle.harness.ideas import open_ideas
from aisle.harness.rollout import run_gates


def git_state(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return (
        git("rev-parse", "HEAD"),
        git("branch", "--show-current"),
        bool(git("status", "--porcelain", "--untracked-files=no")),
    )


def check_runtime():
    """The repo permits the paired Dora 1.0.1 release for development runs."""
    binary = shutil.which("dora")
    if binary is None:
        raise ValueError("Dora CLI is missing")
    version = subprocess.check_output([binary, "--version"], text=True, timeout=10).strip()
    api = importlib.metadata.version("dora-rs")
    if api != "1.0.1" or re.search(r"\b1\.0\.1\b", version) is None:
        raise ValueError("Dora CLI/Python API do not match the permitted 1.0.1 release")
    return {
        "ok": True,
        "cli": version,
        "python_api": api,
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
    }


def check(plan):
    if plan["status"] not in {"prepared_not_authorized", "approved_for_preflight"}:
        raise ValueError("only a new, unconsumed proposal can be preflighted")
    root = Path(plan["instrument_worktree"]).resolve()
    revision, branch, dirty = git_state(root)
    if dirty or revision != plan["instrument_commit"] or branch != plan["instrument_branch"]:
        raise ValueError("instrument source or branch differs from proposal")
    if not re.fullmatch(r"[0-9a-f]{40}", plan["protected_environment_baseline"]):
        raise ValueError("protected baseline must be pinned; no local override")
    for group in ("library_hashes", "skill_sources", "assessment_sources"):
        if type(plan[group]) is not dict:
            raise ValueError("source hashes must be maps")
        for relative, expected in plan[group].items():
            path = root / relative
            if (
                not path.resolve().is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("source pin differs: " + relative)
    graph = root / plan["graph"]
    if not graph.resolve().is_relative_to(root):
        raise ValueError("graph escapes instrument tree")
    snapshot = graph.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != plan["graph_sha256"]:
        raise ValueError("graph pin differs")
    if not open_ideas(root, branch):
        return {
            "ok": False,
            "gate": "idea",
            "detail": "no OPEN idea for pinned branch (HAR-8)",
            "simulation_launched": False,
            "collection_authorized": False,
        }
    runtime = check_runtime()
    gates = run_gates(
        root,
        graph,
        branch,
        no_idea_gate=False,
        embodiment="franka",
        env_baseline=plan["protected_environment_baseline"],
        episodes=1,
        sim_extra="sim",
        graph_snapshot=snapshot,
        sim_engine=plan["engine"],
    )
    if gates["ok"] and (
        not gates.get("env_attested")
        or gates.get("env_fingerprint") != plan["env_fingerprint"]
        or gates.get("env_hash") != plan["env_hash"]
    ):
        gates = {"ok": False, "gate": "attestation", "detail": "proposal environment differs"}
    return {
        **gates,
        "runtime": runtime,
        "instrument_commit": revision,
        "instrument_branch": branch,
        "graph_sha256": plan["graph_sha256"],
        "simulation_launched": False,
        "collection_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = check(json.loads(args.proposal.read_text()))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        result = {
            "ok": False,
            "error": str(exc),
            "simulation_launched": False,
            "collection_authorized": False,
        }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
