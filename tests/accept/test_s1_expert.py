"""SPEC 200 acceptance: the scripted S1 expert completes a fixed-seed
order-picking episode end-to-end THROUGH THE PUBLIC HARNESS PATH (RS-6:
`harness rollout --tier S1` is the entry point, PR #21 — not a bare
`dora run`). Marker `accept`: full dora graph with genesis."""

import importlib.util
import shutil
import uuid
from pathlib import Path

import pytest
from accept_helpers import run_harness

pytestmark = [
    pytest.mark.accept,
    pytest.mark.skipif(
        importlib.util.find_spec("genesis") is None or shutil.which("dora") is None,
        reason="sim extra or dora CLI not installed",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH = REPO_ROOT / "graphs" / "expert_s1.yaml"


def test_scripted_order_pick():
    """RS-6/RS-7 (integration): `harness rollout --graph graphs/expert_s1.yaml
    --tier S1 --embodiment mobile` verbatim — gates, instrumentation, store
    bridge, guard (arm+base), waypoint-nav, order-reader, task-planner,
    s1-expert, verifier-retail — completes the seed-1 order (1 amoxicillin +
    1 omeprazole, both L1-sourceable, ADR-18) with a success episode_result
    carrying the RS-6 fields."""
    run_id = f"s1-gate-{uuid.uuid4().hex[:6]}"
    code, logged = run_harness("report", "log", "--idea", f"S1 gate run {run_id}", timeout=60)
    assert code == 0, logged
    try:
        # Genesis on Metal (2026-10-06): ~1m40s build and a ~4 min episode
        # (steady rtf ~0.7), ~7 min in all; ADR-18's first green run took 28:39.
        # The harness's own deadline is 420 + 2100 s (tier_budgets); the
        # outer timeout only backstops a hung harness process.
        code, report = run_harness(
            "rollout",
            "--graph",
            str(GRAPH),
            "--tier",
            "S1",
            "--embodiment",
            "mobile",
            "--episodes",
            "1",
            "--seeds",
            "1",
            "--reset",
            "teleport",
            "--run-id",
            run_id,
            # ADR-21: acceptance runs on dev branches whose frozen files
            # legitimately differ from origin/main (env-change PRs); the
            # override is logged in the run manifest
            "--env-baseline",
            "local",
            timeout=2700,
        )
        assert code == 0, report
        assert report["ok"] is True, report
        assert len(report["episodes"]) == 1, report
        record = report["episodes"][0]
        assert record["status"] == "success", record
        # RS-6: the episode record carries the retail scoring fields
        assert record["success"] is True
        assert record["penalties"] == []
        # TC-7/TC-8: the discriminator stays in the closed set; retail
        # identity rides the additive suite field (#22, option 1)
        assert record["verifier"] == "oracle"
        assert record["suite"] == "retail"
        assert record["seed"] == 1
        assert report["pass1"] == 1.0
        # HAR-4: the retail rollout records the overhead video and traces
        # like every other rollout — a green run with videos: [] means the
        # graph dropped the rgb_overhead stream (PR #21)
        assert any("overhead" in v for v in report["videos"]), report["videos"]
        assert (REPO_ROOT / report["traces_dir"] / "dora-genesis__rgb_overhead.arrow").exists()
    finally:
        run_harness(
            "report",
            "close",
            "--id",
            logged["id"],
            "--observed",
            f"s1 gate run {run_id} finished",
            "--verdict",
            "flat",
            timeout=60,
        )
