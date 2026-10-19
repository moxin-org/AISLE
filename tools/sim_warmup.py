#!/usr/bin/env python3
"""sim_warmup: pre-build a simulation scene so cold caches never eat the
rollout budget (CON-8, CON-5).

Genesis caches CoACD convex decompositions and compiled kernels per
machine. The G1's FIRST build takes ~470 s — beyond rollout.py's 420 s
GENESIS_BUILD_BUDGET_S — while a warm (cached) build takes ~20 s. This
tool pays the one-time cost deliberately: it builds the scene once and
reports per-stage wall times and whether the cache was already warm.

The budget itself is NOT raised (ADR-g1-embodiment): warmup, not budget
inflation, is the answer to a one-time cost.

CLI: JSON on stdout, logs to stderr, exit 0 iff ok (CON-8).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G1_CACHE_SENTINEL = ROOT / ".genesis" / "cache"  # informational only


def _stage_times(embodiment: str, seed: int, headless: bool) -> dict:
    """Build the scene once, timing import/init, add_entity, and build.

    Genesis does not expose stage hooks, so the stages are bounded by the
    calls that bracket them; the total is the contract the rollout budget
    actually sees. Cache-warmth is inferred from the observed add_entity
    wall time (cold ~280 s vs warm ~9 s on the measured lane machine).
    """
    import genesis as gs

    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    gs.init(backend=gs.metal if platform.system() == "Darwin" else gs.cpu, logging_level="error")
    stages["init_s"] = round(time.perf_counter() - t0, 2)

    from aisle.scenes.pharmacy import SceneCfg, build_scene

    t0 = time.perf_counter()
    handle = build_scene(
        seed=seed,
        embodiment=embodiment,
        headless=headless,
        cfg=SceneCfg(),
    )
    stages["scene_build_s"] = round(time.perf_counter() - t0, 2)
    stages["n_dofs"] = int(handle.robot.n_dofs)
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", default="g1_right")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-viewer", action="store_true")
    args = parser.parse_args()

    report: dict = {"embodiment": args.embodiment, "seed": args.seed}
    try:
        stages = _stage_times(args.embodiment, args.seed, headless=not args.show_viewer)
    except Exception as exc:  # noqa: BLE001 - the CLI's ok:false contract
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(report))
        return 1
    # Warmth inference: a warm add_entity+build is under ~60 s; cold is
    # ~450 s. The threshold is deliberately generous (CI variance) — the
    # absolute times are reported either way.
    warm = stages["scene_build_s"] < 60.0
    report.update(
        ok=True,
        stages=stages,
        cache=warm,
        note=(
            "cache warm"
            if warm
            else "cache COLD: this was the one-time CoACD/kernel build; "
            "re-run to confirm warm times before rollout"
        ),
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
