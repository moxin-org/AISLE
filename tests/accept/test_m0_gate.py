"""SPEC 090 acceptance: the M0 gate.

M0-1/M0-2 share one module-scoped 50-episode rollout (the spec's M0-2 is
"re-running M0-1", so the second full run happens inside the M0-2 test).
The run directories are kept under runs/ as evidence for the ADR-M0
sign-off (M0-6). M0-4 is enforced by tools/trace_check.py --strict in CI;
M0-6 is a human act, not a test.
"""

import importlib.util
import shutil
import subprocess
import sys
import uuid

import pytest
from accept_helpers import REPO_ROOT, run_harness

pytestmark = [
    pytest.mark.accept,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None or shutil.which("dora") is None,
        reason="sim extra or dora CLI not installed",
    ),
]

GRAPH = REPO_ROOT / "graphs" / "expert_t0.yaml"
THRESHOLDS = REPO_ROOT / "src" / "aisle" / "verifier" / "thresholds.toml"
SO101_URDF = REPO_ROOT / "assets" / "so101" / "so101.urdf"


def rollout_50(run_id: str, timeout: float = 4 * 3600) -> dict:
    """One spec-literal M0-1 invocation, bracketed by an ideas-log entry
    so the HAR-2 gate is exercised for real."""
    code, logged = run_harness("report", "log", "--idea", f"M0 gate run {run_id}", timeout=60)
    assert code == 0, logged
    try:
        code, report = run_harness(
            "rollout",
            "--graph",
            str(GRAPH),
            "--tier",
            "T0",
            "--episodes",
            "50",
            "--seeds",
            "0..49",
            "--reset",
            "teleport",
            "--run-id",
            run_id,
            # ADR-21: dev branches carry legitimate frozen diffs; logged
            "--env-baseline",
            "local",
            timeout=timeout,
        )
        assert code == 0, report
        assert report["ok"] is True and len(report["episodes"]) == 50
        return report
    finally:
        run_harness(
            "report",
            "close",
            "--id",
            logged["id"],
            "--observed",
            f"m0 run {run_id} finished",
            "--verdict",
            "flat",
            timeout=60,
        )


@pytest.fixture(scope="module")
def m0_first_run() -> dict:
    return rollout_50(f"m0-1-{uuid.uuid4().hex[:6]}")


def test_m0_1_pass1_at_least_95(m0_first_run):
    """SPEC 090 M0-1 (HAR-1, SCN-3, BRG-2, VER-4, TC-2): 50 episodes over
    seeds 0..49 on graphs/expert_t0.yaml report pass1 >= 0.95 on
    macOS-arm64."""
    assert m0_first_run["pass1"] >= 0.95, m0_first_run["failures"]


def _episode_goal_bytes_sans_timing(run_id: str) -> list[bytes]:
    """The run's RAW episode_goal payload bytes with only the timing-layer
    reset_sim_ns field textually excised — CON-5 layer (a) promises the
    seed-derived goal content BIT-identically (PR #88 re-review: parsed-
    object equality would forgive key order, whitespace, and numeric-
    representation drift), while layers (b)/(d) own the timing."""
    import re

    import pyarrow.ipc as ipc

    path = REPO_ROOT / "runs" / run_id / "traces" / "rollout-client__episode_goal.arrow"
    out = []
    for row in ipc.open_stream(path).read_all().to_pylist():
        stripped, n = re.subn(r',\s*"reset_sim_ns":\s*\d+', "", row["text"])
        assert n == 1, f"goal payload missing reset_sim_ns: {row['text'][:120]}"
        out.append(stripped.encode())
    return out


def test_m0_2_rerun_reproduces_the_gate_and_goal_content(m0_first_run):
    """SPEC 090 M0-2 (CON-5 layers (a)/(d), ADR-26): the rerun MUST
    independently satisfy M0-1's pass1 >= 0.95 — bounding the rate
    difference at <= 0.04 by construction; non-rejection by a
    significance test is NOT equivalence and does not gate (PR #88
    review: 48/50 vs 40/50 gives Fisher p = 0.028, which must fail).
    Episode goals MUST be bit-identical after stripping reset_sim_ns
    (layer (a)). The replicate evidence — both success counts, the
    per-seed flip set, and the Fisher p as context — is persisted to
    runs/<rerun-id>/m0_2_replicate.json (the MUST-log requirement;
    pytest capture hides passing-test stdout)."""
    import json as _json

    from aisle.harness.stats import fisher_exact_two_sided

    rerun_id = f"m0-2-{uuid.uuid4().hex[:6]}"
    rerun = rollout_50(rerun_id)

    # layer (a): seed-derived goal content bit-identical across the pair
    first_goals = _episode_goal_bytes_sans_timing(m0_first_run["run_id"])
    rerun_goals = _episode_goal_bytes_sans_timing(rerun_id)
    assert first_goals == rerun_goals, "seed-derived episode goals differ (CON-5 layer (a))"

    # seed-list integrity BEFORE the dict conversion: a duplicate seed
    # would silently collapse and let two identically malformed 50-row
    # runs pass with n < 50 (PR #88 re-review)
    for label, report in (("first", m0_first_run), ("rerun", rerun)):
        seeds = [e["seed"] for e in report["episodes"]]
        assert sorted(seeds) == list(range(50)), f"{label}: seed list is not exactly 0..49"

    first = {e["seed"]: e["status"] for e in m0_first_run["episodes"]}
    second = {e["seed"]: e["status"] for e in rerun["episodes"]}
    assert first.keys() == second.keys()
    flips = {s: (first[s], second[s]) for s in sorted(first) if first[s] != second[s]}
    k1 = sum(1 for s in first.values() if s == "success")
    k2 = sum(1 for s in second.values() if s == "success")
    evidence = {
        "first_run": m0_first_run["run_id"],
        "rerun": rerun_id,
        "success_counts": [k1, k2],
        "n": len(first),
        "per_seed_flips": {str(k): v for k, v in flips.items()},
        "fisher_two_sided_p_context_only": fisher_exact_two_sided(k1, len(first), k2, len(second)),
    }
    evidence_path = REPO_ROOT / "runs" / rerun_id / "m0_2_replicate.json"
    evidence_path.write_text(_json.dumps(evidence, indent=1) + "\n")
    assert evidence_path.exists()

    # the GATE: the rerun independently re-satisfies M0-1's threshold
    assert rerun["pass1"] >= 0.95, (
        f"rerun pass1 {rerun['pass1']} < 0.95 — the replicate does not re-satisfy "
        f"M0-1; evidence at {evidence_path}; flips: {flips}"
    )


def test_m0_3_mutated_frozen_file_refuses_rollout():
    """SPEC 090 M0-3 (CON-7, HAR-2): tools/env_hash.py --check passes on
    the committed hash, and a single mutated byte in
    verifier/thresholds.toml makes `harness rollout` refuse at the
    env-hash gate. The file is restored byte-for-byte afterwards."""
    check = subprocess.run(
        [sys.executable, "tools/env_hash.py", "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert check.returncode == 0, check.stdout + check.stderr

    original = THRESHOLDS.read_bytes()
    try:
        THRESHOLDS.write_bytes(original + b"# mutated byte\n")
        code, report = run_harness(
            "rollout",
            "--graph",
            str(GRAPH),
            "--tier",
            "T0",
            "--episodes",
            "1",
            "--seeds",
            "0",
            "--reset",
            "teleport",
            "--no-idea-gate",
            "--env-baseline",
            "local",
            timeout=300,
        )
        assert code != 0
        assert report["ok"] is False
        assert report["refused"]["gate"] == "env_hash", report
    finally:
        THRESHOLDS.write_bytes(original)


def _so101_ready() -> bool:
    """M0-5 starts only when the official asset and BOTH motion/safety
    manifests declare SO-101; partial readiness must not spend 50 episodes."""
    manifests = [
        (REPO_ROOT / "registry" / "manifests" / name).read_text()
        for name in ("ik-trajectory.yaml", "budget-guard.yaml")
    ]
    return SO101_URDF.exists() and all("so101" in manifest for manifest in manifests)


def _g1_ready() -> bool:
    """M0-6 starts only when the vendored asset and BOTH motion/safety
    manifests declare g1_right (G1-F2's scope); partial readiness must not
    spend 50 episodes on an embodiment the gate would refuse anyway."""
    manifests = [
        (REPO_ROOT / "registry" / "manifests" / name).read_text()
        for name in ("ik-trajectory.yaml", "budget-guard.yaml", "grasp-planner-topdown.yaml")
    ]
    g1_urdf = REPO_ROOT / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf"
    return g1_urdf.exists() and all("g1_right" in manifest for manifest in manifests)


@pytest.mark.skipif(
    not _g1_ready(),
    reason="g1_right blocked: G1-F2 (ik/grasp/guard g1_right kinematics) not merged; "
    "the HAR-2 gate refuses the swap (EMBODIMENT_MISMATCH) until then",
)
def test_m0_6_g1_right_profile_swap_pass1_at_least_80():
    """SPEC 090 M0-6 (TC-5, SCN-4): the same T0 graph with --embodiment
    g1_right (profile swap only, zero YAML edits) reaches pass1 >= 0.80.
    The bridge serves the 9-DOF wire while 24 held DOFs stay at standing
    home (G1-C1/G1-C2)."""
    run_id = f"m0-6-{uuid.uuid4().hex[:6]}"
    code, logged = run_harness("report", "log", "--idea", f"M0-6 g1_right {run_id}", timeout=60)
    assert code == 0, logged
    try:
        code, report = run_harness(
            "rollout",
            "--graph",
            str(GRAPH),
            "--tier",
            "T0",
            "--embodiment",
            "g1_right",
            "--episodes",
            "50",
            "--seeds",
            "0..49",
            "--reset",
            "teleport",
            # g1_right's frozen-set changes (assets/g1, env_hash.py) live on
            # feat/g1-support, not yet on protected main — the trusted
            # origin/main baseline refuses the gate tooling itself. local is
            # the documented dev override (rollout.py, HAR-2).
            "--env-baseline",
            "local",
            "--run-id",
            run_id,
            timeout=4 * 3600,
        )
        assert code == 0, report
        assert report["pass1"] >= 0.80, report["failures"]
    finally:
        run_harness(
            "report",
            "close",
            "--id",
            logged["id"],
            "--observed",
            "m0-6 finished",
            "--verdict",
            "flat",
            timeout=60,
        )


@pytest.mark.skipif(
    not _so101_ready(),
    reason="so101 blocked: asset acquisition (ADR-6) and/or both motion manifests' "
    "so101 support; the HAR-2 gate refuses the swap "
    "(EMBODIMENT_MISMATCH) until both land",
)
def test_m0_5_so101_profile_swap_pass1_at_least_80():
    """SPEC 090 M0-5 (TC-5, SCN-4): the same T0 graph with --embodiment
    so101 (profile swap only, zero YAML edits) reaches pass1 >= 0.80."""
    run_id = f"m0-5-{uuid.uuid4().hex[:6]}"
    code, logged = run_harness("report", "log", "--idea", f"M0-5 so101 {run_id}", timeout=60)
    assert code == 0, logged
    try:
        code, report = run_harness(
            "rollout",
            "--graph",
            str(GRAPH),
            "--tier",
            "T0",
            "--embodiment",
            "so101",
            "--episodes",
            "50",
            "--seeds",
            "0..49",
            "--reset",
            "teleport",
            "--run-id",
            run_id,
            timeout=4 * 3600,
        )
        assert code == 0, report
        assert report["pass1"] >= 0.80, report["failures"]
    finally:
        run_harness(
            "report",
            "close",
            "--id",
            logged["id"],
            "--observed",
            "m0-5 finished",
            "--verdict",
            "flat",
            timeout=60,
        )
