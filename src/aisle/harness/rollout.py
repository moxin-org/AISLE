"""Rollout runner (SPEC 070 HAR-1..5).

`harness rollout --graph G --tier T --episodes N --seeds a..b --reset
teleport` gates (HAR-2: env hash, validation, open idea), instruments the
graph with a trace recorder, drives it via the rollout-client's env
config, and writes runs/<run_id>/ with the manifest (HAR-4), per-episode
results, Arrow traces, and the overhead video. pass@8 follows HAR-3's
in-context-retry semantics: the episode records carry a retries count
(0 while the task-state-machine runs single-attempt, ADR-10) — pass@8 is
NEVER computed as best-of-8 independent episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as platform_module
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from aisle.harness.ideas import open_ideas
from aisle.harness.reaper import reap_orphans
from aisle.harness.validate import validate

# every declared node/output endpoint is traced (HAR-4); image topics
# record payload-free rows — pixels live in the mp4 (ADR-11), and seg
# rows carry #131 provenance text
# hard-won budgets (see ADR-11): the per-episode verifier budget in SIM
# seconds; the one-off genesis build and per-episode WALL budgets; and
# the stall detector's thresholds (pre-data = the build produces no
# traces; post-data = a dead bridge freezes the stream)
EPISODE_TIMEOUT_S = 60
GENESIS_BUILD_BUDGET_S = 420
# ADR-67: Nexus assembles the same scene in seconds (no kernel compilation),
# so a wedged first episode must clamp in a minute rather than seven. Sized
# well above the measured build to cover wheel import and GPU adapter start.
NEXUS_BUILD_BUDGET_S = 60
PER_EPISODE_BUDGET_S = 150
PRE_DATA_STALL_S = 600
# The pre-data grace exists because the scene build emits no traces; on Nexus
# that window is seconds, so two minutes already covers a slow adapter start.
NEXUS_PRE_DATA_STALL_S = 120
STALL_S = 180
# Operator progress cadence on stderr while a run waits on dora (HAR-1). CON-8
# keeps stdout JSON-only, so this never touches stdout. AISLE_ROLLOUT_PROGRESS_S
# overrides it; 0 disables.
PROGRESS_INTERVAL_S = 15.0
# retail tiers (RS-6, ADR-18): store-sim rtf ~0.1 on the dev machine — the
# fixed-seed S1 episode runs ~101.5 sim s / ~25 wall min plus a ~2.5 min
# store build (first green run: 28:39 total). The desk budgets above would
# kill a HEALTHY retail episode at 60 sim s / 150 wall s (PR #21)
RETAIL_EPISODE_TIMEOUT_S = 600
RETAIL_PER_EPISODE_BUDGET_S = 2100
# T2's scan tour reads up to six candidate faces before the grasp (~8-10
# sim s per read cycle + ~30 s grasp): the desk 60 s cap made the tier
# structurally impossible — the first acceptance probe timed out MID-TOUR
# on every non-trivial episode (run 20260811-161222-dda648)
T2_EPISODE_TIMEOUT_S = 150
T2_PER_EPISODE_BUDGET_S = 400


def tier_budgets(tier: str) -> tuple[int, int]:
    """(episode timeout in SIM seconds, per-episode WALL budget in seconds)
    for a tier: `harness rollout` is the public path for EVERY tier (HAR-1,
    RS-6), so retail tiers get nightly-suite-scale budgets, the T2 scan
    tour gets room to visit every candidate, and the other desk tiers
    keep the tight ADR-11 ones."""
    if tier in ("S1", "S2", "S3"):
        return RETAIL_EPISODE_TIMEOUT_S, RETAIL_PER_EPISODE_BUDGET_S
    if tier in ("T2", "T3"):
        return T2_EPISODE_TIMEOUT_S, T2_PER_EPISODE_BUDGET_S
    return EPISODE_TIMEOUT_S, PER_EPISODE_BUDGET_S


def build_budget_s(engine: str) -> int:
    """The default scene-build wall grace for an engine (ADR-67, HAR-1).

    Genesis compiles kernels for minutes, which is what the 420 s default was
    sized for; Nexus builds the same scene in seconds, and rapier builds the
    same Nexus scene to render through (ADR-68)."""
    return NEXUS_BUILD_BUDGET_S if engine in ("nexus", "rapier") else GENESIS_BUILD_BUDGET_S


def pre_data_stall_s(engine: str) -> int:
    """How long a launch may produce no traces at all before it counts as
    stalled (ADR-67). Same reasoning as `build_budget_s`: the silent window is
    the scene build, so it is engine-sized rather than Genesis-sized."""
    return NEXUS_PRE_DATA_STALL_S if engine in ("nexus", "rapier") else PRE_DATA_STALL_S


# A7 wall-budget sizing (issue #160 item 6): in `--verifier realistic`
# nothing ends an episode early — the reset/goal are downstream of the
# verifier's own verdict, which it renders at sim-budget expiry — so EVERY
# episode, successes included, runs the full sim budget and is then judged.
# The ADR-23 per-episode wall clamp must therefore cover full-sim-budget
# wall time at a pessimistic rtf PLUS the judge, or healthy A7 runs trip
# the clamp and the relaunch machinery kicks in for nothing.
A7_WALL_PER_SIM = 3  # wall seconds per sim second: covers rtf >= 0.33 (observed ~0.5)
A7_JUDGE_BUDGET_S = 30  # judge_frames is 3-5 s; margin for queue drain


def a7_per_episode_budget_s(episode_timeout_s: int, per_episode_budget_s: int) -> int:
    """The A7 per-episode wall budget: never below the tier's, never below
    the full-sim-budget episode A7 structurally produces."""
    return max(per_episode_budget_s, episode_timeout_s * A7_WALL_PER_SIM + A7_JUDGE_BUDGET_S)


def resolve_budgets(
    tier: str, verifier: str, per_episode_wall_override_s: int | None = None
) -> tuple[int, int]:
    """(episode timeout in SIM seconds, per-episode WALL budget) for a run.

    The verifier is part of the budget, not just the tier: in A7
    (`--verifier realistic`) nothing ends an episode early, so every one
    runs its full sim budget and is judged at expiry. Kept as its own pure
    function because the wiring — not the arithmetic — was the part with no
    test: deleting the A7 branch at the call site left the whole unit suite
    green (PR #177 review)."""
    episode_timeout_s, per_episode_budget_s = tier_budgets(tier)
    if verifier == "realistic":
        if tier in ("S1", "S2", "S3"):
            # A7_WALL_PER_SIM encodes a DESK rtf. Retail runs ~0.07 (the
            # measured 101.5 sim s in ~25 wall min above), so a full
            # 600 sim-s A7 episode costs ~8900 wall s against the tier's
            # 2100 s clamp: the max() below silently no-ops and EVERY
            # episode clamps at ~24%, relaunches, and clamps again — a
            # scored 0.0 dressed up as a budget. Refuse until a retail A7
            # budget is measured and re-budgeted under ADR-21 (PR #177
            # review); `--verifier both` is unaffected, it never changes
            # control flow.
            raise ValueError(
                f"tier {tier} with --verifier realistic (A7) has no measured wall budget: "
                f"a full {episode_timeout_s} sim-s episode at the documented retail rtf "
                f"needs roughly 4x the tier's {per_episode_budget_s} s per-episode clamp, "
                "so every episode would wall_clamp. Use --verifier both, or land a "
                "measured retail A7 budget with an ADR-21 re-budget."
            )
        per_episode_budget_s = a7_per_episode_budget_s(episode_timeout_s, per_episode_budget_s)
    if per_episode_wall_override_s is not None:
        # lockstep VLA eval (ADR-38 amendment): in-turn inference freezes
        # sim time while wall time runs, so the wall clamp must scale with
        # model latency; SIM budgets are the tier's, untouched
        per_episode_budget_s = int(per_episode_wall_override_s)
    return episode_timeout_s, per_episode_budget_s


def parse_seed_range(spec: str) -> list[int]:
    """'0..49' -> [0..49]; '3' -> [3]; '1,4,7' -> [1, 4, 7]."""
    if ".." in spec:
        a, b = spec.split("..", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def compute_metrics(episodes: list[dict]) -> dict:
    """HAR-1/HAR-3: pass1 counts first-attempt successes; pass8 counts an
    episode as passed if it succeeded within <=8 IN-CONTEXT retries
    (retries field, 0 today) — never best-of-8 independent episodes."""
    n = len(episodes)
    pass1 = sum(1 for e in episodes if e["status"] == "success" and e.get("retries", 0) == 0)
    pass8 = sum(1 for e in episodes if e["status"] == "success" and e.get("retries", 0) <= 8)
    failures: dict[str, int] = {}
    for e in episodes:
        if e["status"] != "success":
            reason = e.get("failure") or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
    return {
        "pass1": pass1 / n if n else 0.0,
        "pass8": pass8 / n if n else 0.0,
        "failures": failures,
    }


def progress_interval_s(env: dict | None = None) -> float:
    """The stderr progress cadence: AISLE_ROLLOUT_PROGRESS_S (seconds, 0 or a
    negative value disables), else PROGRESS_INTERVAL_S. Malformed values are
    refused loudly rather than silently muting the report."""
    raw = (os.environ if env is None else env).get("AISLE_ROLLOUT_PROGRESS_S")
    if raw is None or not str(raw).strip():
        return PROGRESS_INTERVAL_S
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"AISLE_ROLLOUT_PROGRESS_S must be a number of seconds, got {raw!r}"
        ) from exc


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def progress_report(
    *,
    now: float,
    started: float,
    deadline: float,
    lines: int,
    episodes: int,
    seeds: list[int],
    last_line_t: float,
    episode_grace_s: float,
    traces_size: int,
    last_record: dict | None,
    relaunches: int = 0,
    timing: dict | None = None,
) -> str:
    """One human-readable progress line for stderr while the run waits on
    dora (HAR-1 operator feedback). Before the first trace byte the scene is
    still building; afterwards it names the running episode, how long it has
    run against its per-episode wall budget, the last verdict, and the run
    deadline, so an operator can tell a long build from a wedged episode
    without reading the traces."""
    elapsed = now - started
    head = f"[rollout +{_hms(elapsed)}]"
    if lines >= episodes:
        phase = f"all {episodes} episodes recorded, waiting for the client to exit"
    else:
        seed = seeds[lines] if lines < len(seeds) else None
        seed_txt = f" (seed {seed})" if seed is not None else ""
        running = now - last_line_t
        if traces_size <= 0:
            phase = (
                f"episode {lines + 1}/{episodes}{seed_txt}: building the scene, "
                f"no traces yet ({_hms(running)})"
            )
        else:
            phase = (
                f"episode {lines + 1}/{episodes}{seed_txt} running {_hms(running)} "
                f"of {_hms(episode_grace_s)} wall budget, traces {traces_size / 1e6:.1f} MB"
            )
    parts = [head, phase]
    if last_record:
        status = last_record.get("status", "?")
        failure = last_record.get("failure")
        verdict = status if not failure else f"{status} ({failure})"
        t_end = last_record.get("t_end")
        t_txt = f" at t={float(t_end):.1f}s" if isinstance(t_end, (int, float)) else ""
        episode, seed = last_record.get("episode", "?"), last_record.get("seed", "?")
        parts.append(f"last: episode {episode} seed {seed} {verdict}{t_txt}")
    if relaunches:
        parts.append(f"relaunches {relaunches}")
    phrase = timing_phrase(timing or {})
    if phrase:
        parts.append(phrase)
    parts.append(f"run deadline in {_hms(deadline - now)}")
    return f"{parts[0]} " + "; ".join(parts[1:])


def summarize_timing(rows: list[dict]) -> dict:
    """Aggregate the bridge's timing sidecar (`sim_timing.jsonl`, one row per
    window of steps) into run-level means, weighted by step and frame counts:
    physics wall time per step, engine GPU time per step where the engine
    reports it, render wall time per frame, and the real-time factor (sim
    seconds per wall second of bridge ticks). A row's tick already spans the
    step, the renders and the publishes, so render time is added only for
    older rows that carry no tick. Empty when there is nothing to aggregate."""
    steps = sum(int(r.get("steps", 0)) for r in rows)
    frames = sum(int(r.get("frames", 0)) for r in rows)
    if steps <= 0:
        return {}
    step_wall = sum(float(r.get("step_ms_mean") or 0.0) * int(r.get("steps", 0)) for r in rows)
    tick_wall = sum(
        float(r.get("tick_ms_mean") or r.get("step_ms_mean") or 0.0) * int(r.get("steps", 0))
        for r in rows
    )
    gpu_rows = [r for r in rows if r.get("gpu_ms_mean") is not None]
    gpu_steps = sum(int(r.get("steps", 0)) for r in gpu_rows)
    gpu_wall = sum(float(r["gpu_ms_mean"]) * int(r.get("steps", 0)) for r in gpu_rows)
    render_wall = sum(float(r.get("render_ms_mean") or 0.0) * int(r.get("frames", 0)) for r in rows)
    sim_s = sum(float(r.get("sim_window_s") or 0.0) for r in rows)
    wall_s = (
        sum(
            float(r["tick_ms_mean"]) * int(r.get("steps", 0))
            if r.get("tick_ms_mean") is not None
            else float(r.get("step_ms_mean") or 0.0) * int(r.get("steps", 0))
            + float(r.get("render_ms_mean") or 0.0) * int(r.get("frames", 0))
            for r in rows
        )
        / 1000.0
    )
    out = {
        "engine": rows[-1].get("engine"),
        "steps": steps,
        "frames": frames,
        "step_ms_mean": step_wall / steps,
        "step_ms_max": max(float(r.get("step_ms_max") or 0.0) for r in rows),
        "tick_ms_mean": tick_wall / steps,
        "render_ms_mean": (render_wall / frames) if frames else None,
        "gpu_ms_mean": (gpu_wall / gpu_steps) if gpu_steps else None,
        "rtf": (sim_s / wall_s) if wall_s > 0 else None,
    }
    return out


def read_timing_rows(run_dir: Path) -> list[dict]:
    """The bridge's timing sidecar rows under `run_dir` (every launch's), or
    an empty list when the bridge wrote none."""
    rows: list[dict] = []
    for path in sorted(run_dir.rglob("sim_timing.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def timing_phrase(summary: dict) -> str:
    """The performance clause of a progress line: wall time of the physics
    step call and of the whole tick (step plus the state reads that sync the
    GPU), GPU time per step when the engine reports it, render time per frame,
    and the real-time factor."""
    if not summary:
        return ""
    parts = [f"physics {summary['step_ms_mean']:.1f} ms/step"]
    if summary.get("tick_ms_mean") is not None:
        parts.append(f"tick {summary['tick_ms_mean']:.1f} ms")
    if summary.get("gpu_ms_mean") is not None:
        parts.append(f"gpu {summary['gpu_ms_mean']:.1f} ms")
    if summary.get("render_ms_mean") is not None:
        parts.append(f"render {summary['render_ms_mean']:.1f} ms/frame")
    if summary.get("rtf") is not None:
        parts.append(f"rtf {summary['rtf']:.2f}x")
    return ", ".join(parts)


def _last_record(raw: bytes) -> dict | None:
    """The last parseable episode row of a results file, or None."""
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    return None


def sim_device_for(engine: str, backend: str) -> str:
    """The live device an (engine, backend) pair runs on (CON-5, ADR-67).

    Derived from the resolved pair rather than from the host OS: Genesis's
    Metal backend runs through torch MPS, while Nexus names its GPU adapter by
    the backend itself, so a Nexus webgpu or metal run is never attested
    ``cpu`` just because the host is not a Mac."""
    if backend == "cpu":
        return "cpu"
    if engine == "genesis" and backend == "metal":
        return "mps"
    return backend


def resolve_sim_identity(sim_extra: str, sim_engine: str = "genesis") -> dict:
    """Resolve the requested lock extra to a fail-closed backend/device.

    CON-5 requires this live hardware fact to ride with the run identity;
    the portable ``sim`` selection deliberately never probes into CUDA.
    sim_engine (ADR-67) picks whose backend table answers; an engine whose
    package is not installed refuses here, before anything launches.
    """
    from aisle.sim import engine_available, normalize_engine, select_sim_backend

    try:
        sim_engine = normalize_engine(sim_engine)
    except ValueError as exc:
        return {"ok": False, "gate": "sim_engine", "detail": str(exc)}
    if not engine_available(sim_engine):
        return {
            "ok": False,
            "gate": "sim_engine",
            "detail": f"simulation engine {sim_engine!r} is not installed in this environment",
        }

    system = platform_module.system()
    cuda_available = False
    # only the CUDA probe names a concrete device; every other pair is derived
    # from the resolved backend below
    device = None
    if sim_extra == "cuda":
        if system != "Linux":
            return {
                "ok": False,
                "gate": "sim_backend",
                "detail": "the locked CUDA simulation extra is supported only on Linux",
            }
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                index = int(torch.cuda.current_device())
                device = f"cuda:{index}:{torch.cuda.get_device_name(index)}"
        except (ImportError, RuntimeError) as exc:
            return {
                "ok": False,
                "gate": "sim_backend",
                "detail": f"cannot inspect the requested CUDA device: {exc}",
            }
    try:
        backend = select_sim_backend(sim_engine, sim_extra, system, cuda_available)
    except ValueError as exc:
        return {"ok": False, "gate": "sim_backend", "detail": str(exc)}
    return {
        "ok": True,
        "sim_extra": sim_extra,
        "sim_engine": sim_engine,
        "sim_backend": backend,
        "sim_device": device or sim_device_for(sim_engine, backend),
    }


def load_campaign_budget(root: Path) -> dict:
    """ADR-21: the campaign ceilings from harness/budget.toml (FROZEN — a
    research agent must not raise its own budget)."""
    import tomllib

    with open(root / "harness" / "budget.toml", "rb") as f:
        return tomllib.load(f)["campaign"]


_LEDGER = "runs/campaign_ledger.jsonl"


def _entry_hash(prev_hash: str, entry: dict) -> str:
    """Tamper-evident chain (ADR-21 round 3): each ledger entry hashes its
    predecessor's hash + its own canonical content — an edited or dropped
    line breaks every hash after it, and each run manifest records its
    entries' hashes so the audit can cross-verify."""
    canon = json.dumps({k: entry[k] for k in sorted(entry) if k != "hash"}, sort_keys=True)
    return hashlib.sha256((prev_hash + canon).encode()).hexdigest()


def budget_ledger(root: Path) -> list[dict]:
    path = root / _LEDGER
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_ledger(root: Path) -> bool:
    """True iff every entry's chain hash verifies."""
    prev = ""
    for entry in budget_ledger(root):
        if entry.get("hash") != _entry_hash(prev, entry):
            return False
        prev = entry["hash"]
    return True


def _ledger_spend(entries: list[dict]) -> tuple[int, float]:
    """(episodes, wall_s) spent: settled runs at their actuals, UNSETTLED
    reservations at their reserved episodes (a crashed run stays charged —
    conservative accounting, ADR-21 round 3)."""
    settled = {e["run_id"]: e for e in entries if e.get("kind") == "settle"}
    episodes = wall = 0.0
    for e in entries:
        if e.get("kind") == "settle":
            episodes += int(e.get("episodes", 0))
            wall += float(e.get("wall_s", 0.0))
        elif e.get("kind") == "reserve" and e["run_id"] not in settled:
            episodes += int(e.get("episodes", 0))
    return int(episodes), wall


def budget_remaining(root: Path) -> dict:
    """Episodes/wall remaining under the campaign ceilings; the token
    ceiling is reported for the external accountant (HAR-5: only the LLM
    harness can count tokens — manifests carry the log for the audit)."""
    ceilings = load_campaign_budget(root)
    spent_eps, spent_wall = _ledger_spend(budget_ledger(root))
    return {
        "episodes_left": int(ceilings["episodes"]) - spent_eps,
        "wall_h_left": round(float(ceilings["wall_h"]) - spent_wall / 3600.0, 3),
        "tokens_ceiling": int(ceilings["tokens"]),
    }


class _LedgerLock:
    """O_EXCL lockfile: reserve is check-then-append under one lock, so
    concurrent rollouts cannot both pass a nearly-exhausted ceiling."""

    def __init__(self, root: Path, timeout_s: float = 10.0):
        self.path = root / "runs" / "campaign_ledger.lock"
        self.timeout_s = timeout_s

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"campaign ledger locked: {self.path}") from None
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _append_ledger(root: Path, entry: dict) -> str:
    entries = budget_ledger(root)
    prev = entries[-1]["hash"] if entries else ""
    entry["hash"] = _entry_hash(prev, entry)
    path = root / _LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["hash"]


def reserve_budget(root: Path, run_id: str, episodes: int) -> dict:
    """ADR-21 round 3: atomically CHECK AND RESERVE before launch. Returns
    {"ok": True, "entry": <hash>, "remaining": {...}} or a gate-shaped
    refusal. The reservation charges the episode ceiling immediately; an
    interrupted run therefore never runs off-ledger."""
    try:
        with _LedgerLock(root):
            remaining = budget_remaining(root)
            if remaining["episodes_left"] < episodes:
                return {
                    "ok": False,
                    "gate": "budget",
                    "detail": f"campaign episode budget exhausted: "
                    f"{remaining['episodes_left']} left, {episodes} requested "
                    f"(harness/budget.toml, ADR-21)",
                }
            if remaining["wall_h_left"] <= 0.0:
                return {
                    "ok": False,
                    "gate": "budget",
                    "detail": "campaign wall-clock budget exhausted (harness/budget.toml, ADR-21)",
                }
            entry_hash = _append_ledger(
                root, {"kind": "reserve", "run_id": run_id, "episodes": episodes}
            )
            return {"ok": True, "entry": entry_hash, "remaining": remaining}
    except TimeoutError as stuck:
        return {"ok": False, "gate": "budget", "detail": str(stuck)}


def settle_budget(root: Path, run_id: str, episodes: int, wall_s: float) -> str:
    """The reconciliation entry (written in `finally`): actual episodes and
    wall seconds supersede the reservation in the accounting."""
    with _LedgerLock(root):
        return _append_ledger(
            root,
            {"kind": "settle", "run_id": run_id, "episodes": episodes, "wall_s": round(wall_s, 1)},
        )


_COMMIT_OID = re.compile(r"^[0-9a-f]{40}$")


def resolve_trusted_baseline(
    root: Path, baseline: str = "origin/main"
) -> tuple[str | None, str | None]:
    """(commit OID, error): fetch refs/heads/main FROM THE REMOTE SERVER
    and resolve either its head or a campaign-pinned ancestor (ADR-21
    round 3, issue #91). Trust still comes from the protected server:
    arbitrary local OIDs and movable refs are refused. The returned OID
    is content-addressed, so its blobs cannot change beneath a campaign.
    Fail-closed: no remote, no trusted gate."""
    fetch = subprocess.run(
        ["git", "fetch", "--quiet", "origin", "refs/heads/main"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if fetch.returncode != 0:
        return None, f"cannot fetch origin main: {fetch.stderr.strip() or 'no remote?'}"
    head = subprocess.run(
        ["git", "rev-parse", "FETCH_HEAD"], capture_output=True, text=True, cwd=root
    )
    if head.returncode != 0:
        return None, "cannot resolve FETCH_HEAD after fetch"
    server_head = head.stdout.strip()
    if baseline == "origin/main":
        return server_head, None

    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{baseline}^{{commit}}"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if commit.returncode != 0:
        return None, f"cannot resolve campaign baseline OID {baseline}"
    oid = commit.stdout.strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", oid, server_head],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if ancestry.returncode == 1:
        return None, f"campaign baseline {baseline} is not in protected origin/main history"
    if ancestry.returncode != 0:
        return None, "cannot verify campaign baseline against protected origin/main history"
    return oid, None


def run_gates(
    root: Path,
    graph: Path,
    branch: str,
    no_idea_gate: bool,
    embodiment: str = "franka",
    env_baseline: str = "origin/main",
    episodes: int = 0,
    sim_extra: str = "sim",
    graph_snapshot: bytes | None = None,
    typed_stage: tuple[Path, dict] | None = None,
    sim_engine: str = "genesis",
) -> dict:
    """HAR-2: refuse on env-hash mismatch (TRUSTED baseline by default,
    ADR-21: the baseline commit is fetched from the remote SERVER and
    pinned by OID — a research agent cannot satisfy it by moving local
    refs; a campaign may retain a full OID only while it remains in the
    protected main history; --env-baseline local is the dev override —
    humans only; logged), on validation failure, on a missing OPEN idea
    (unless --no-idea-gate — humans only; logged), and — for trusted runs
    only — on an exhausted campaign budget (episodes/wall reserved
    atomically by rollout(); tokens are audited externally per HAR-5).
    Local-override runs are exempt from budget refusal and never charge
    the ledger: the budget meters the campaign, not development."""
    if env_baseline not in ("origin/main", "local") and not _COMMIT_OID.fullmatch(env_baseline):
        return {
            "ok": False,
            "gate": "env_hash",
            "detail": f"unknown baseline {env_baseline!r}: only protected "
            "'origin/main', a full campaign-pinned main-history OID, or the "
            "logged dev override 'local' are accepted (ADR-21, issue #91)",
        }
    sim_identity = resolve_sim_identity(sim_extra, sim_engine)
    if not sim_identity["ok"]:
        return sim_identity
    baseline_oid = None
    hash_cmd = [sys.executable, str(root / "tools" / "env_hash.py"), "--check", "--root", str(root)]
    # ADR-24: rollouts need the sim extra — declare the selection so the
    # trusted checker attests THIS environment shape (HAR-2)
    hash_cmd += ["--extras", sim_extra]
    # ADR-67: the frozen set is engine neutral, so the engine digest and the
    # engine's own build provenance ride beside env_hash rather than in it
    hash_cmd += ["--sim-engine", sim_identity["sim_engine"]]
    if env_baseline != "local":
        baseline_oid, err = resolve_trusted_baseline(root, env_baseline)
        if err:
            return {"ok": False, "gate": "env_hash", "detail": err}
        hash_cmd += ["--baseline", baseline_oid]
    hash_proc = subprocess.run(hash_cmd, capture_output=True, text=True)
    if hash_proc.returncode != 0:
        return {"ok": False, "gate": "env_hash", "detail": hash_proc.stdout.strip()}
    hash_report = json.loads(hash_proc.stdout)
    env_hash = hash_report["env_hash"]
    dist = hash_report.get("dist")
    if env_baseline != "local":
        # ADR-24 D2/D3: trusted runs REFUSE on missing or failed
        # attestation — record-by-convention is not a gate
        if not isinstance(dist, dict):
            return {
                "ok": False,
                "gate": "dist",
                "detail": "DIST_DRIFT: no attestation evidence from the trusted "
                "checker (stale env_hash.py at the baseline?)",
            }
        if not dist.get("attested"):
            return {
                "ok": False,
                "gate": "dist",
                "detail": "DIST_DRIFT: " + "; ".join(dist.get("problems") or ["unattested"]),
            }
    remaining = budget_remaining(root)
    if env_baseline != "local":
        if episodes > 0 and remaining["episodes_left"] < episodes:
            return {
                "ok": False,
                "gate": "budget",
                "detail": f"campaign episode budget exhausted: {remaining['episodes_left']} left, "
                f"{episodes} requested (harness/budget.toml, ADR-21)",
            }
        if remaining["wall_h_left"] <= 0.0:
            return {
                "ok": False,
                "gate": "budget",
                "detail": "campaign wall-clock budget exhausted (harness/budget.toml, ADR-21)",
            }
    # validate against the embodiment that will actually run (M0-5): a
    # graph whose nodes do not support it must refuse HERE, not crash
    # hours into the rollout
    registry_root = root
    if typed_stage is None:
        validation = validate(
            graph,
            root,
            embodiment,
            allow_unproven=False,
            graph_snapshot=graph_snapshot,
        )
    else:
        from aisle.harness.typed_graph_stage import validation_for_rollout_gates

        try:
            validation, registry_root = validation_for_rollout_gates(
                *typed_stage,
                authored_bytes=graph.read_bytes() if graph_snapshot is None else graph_snapshot,
                graph=graph,
                controller_root=root,
                embodiment=embodiment,
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return {
                "ok": False,
                "gate": "typed_stage",
                "detail": str(exc),
                "infrastructure_invalid": True,
            }
    if not validation["ok"]:
        return {"ok": False, "gate": "validate", "detail": validation["errors"]}
    if env_baseline != "local":
        from aisle.harness.registry import load_manifests
        from aisle.harness.validate import _clock_topology, load_graph

        gate_nodes, _ = load_graph(graph, graph_snapshot)
        manifest_list, manifest_errors = load_manifests(registry_root)
        manifests = {manifest["id"]: manifest for _, manifest in manifest_list}
        topology = _clock_topology(gate_nodes or [], manifests) if not manifest_errors else {}
        if topology.get("bridge_ids") and not topology.get("barrier_ids"):
            return {
                "ok": False,
                "gate": "lockstep",
                "detail": "attesting simulation refused: ADR-30 requires a validated "
                "turn barrier; use --env-baseline local for non-attesting free-run bring-up",
            }
    gates = {
        **sim_identity,
        "env_hash": env_hash,
        "env_baseline": env_baseline,
        # the resolved immutable identity (ADR-21 round 3): the audit
        # re-verifies blobs at this OID, not at whatever a ref says later
        "env_baseline_oid": baseline_oid,
        # ADR-24: CON-5's fifth component + the attestation verdict (facts
        # from the trusted checker; local runs record honestly, trusted
        # runs were already refused above on failure)
        "env_fingerprint": (dist or {}).get("env_fingerprint"),
        "env_attested": bool((dist or {}).get("attested")),
        "dist_problems": (dist or {}).get("problems") or [],
        "dist_inventory": (dist or {}).get("inventory"),
        # ADR-67: {"engine", "sim_engine_hash", "n_files"}, the digest of the
        # engine realization package; the engine wheel itself is identified
        # by the lock like every other dist (ADR-70)
        "sim_engine_build": hash_report.get("sim"),
        "budget": remaining,
    }
    if typed_stage is not None:
        gates["typed_stage_id"] = typed_stage[1]["immutable_id"]
    if no_idea_gate:
        return {"ok": True, **gates, "idea": None, "no_idea_gate": True}
    ideas = open_ideas(root, branch)
    if not ideas:
        return {
            "ok": False,
            "gate": "idea",
            "detail": f"no OPEN idea for branch {branch!r} (HAR-8); "
            "log one with `harness report log --idea ...`",
        }
    return {"ok": True, **gates, "idea": ideas[-1]["id"], "no_idea_gate": False}


# camera queue depth for the realistic verifier (issue #120): 400 covers two
# back-to-back 3-5 s judges at 30 Hz without dropping — dropping voids the
# router's arrival proof. HONEST worst-case transient memory if all three
# queues fill (review: an earlier comment said ~400 MB, ~2.3x low): 400 x
# ~0.9 MB rgb8 640x480 + 400 x ~1.2 MB depth f32 + 400 x ~0.23 MB wrist
# 320x240 ~= 930 MB of dora shared memory, held only across stacked judges;
# steady-state the queues sit near-empty. The unit test imports this.
CAMERA_QUEUE_DEPTH = 400


def realistic_verifier_node(root: Path, run_dir: Path, doc: dict, timeout_s: float) -> dict:
    """The VER-5 judge as a live node (increment 1b).

    In `--verifier both` it is a SIDECAR: nothing consumes its
    `episode_result` (the loop advances on the oracle's), and what the
    comparison needs is the VER-14 sidecar `judge_frames` writes to the run
    dir. In `--verifier realistic` (A7) the caller rewires the loop's
    consumers onto this node's verdict.

    ORACLE-FREE: subscribes to camera frames, joint_state, bridge_info,
    episode_goal, and the client's own `reset` request — never
    `oracle_state`, and a test asserts that; A7's whole premise is that
    this verdict never saw privileged state. The reset request carries
    only episode-boundary TIMING (which the next goal encodes anyway),
    no verdict content (issue #120)."""
    producers = {
        topic: f"{node['id']}/{topic}"
        for node in doc["nodes"]
        for topic in (node.get("outputs") or [])
    }
    wanted = (
        "bridge_info",
        "episode_goal",
        # the client's reset REQUEST ends the running episode before any
        # RST-2 reset motion enters the frame window (issue #120) — the
        # client's own action signal, not the oracle's verdict, so the
        # oracle-free premise holds (episode_goal timing already encodes
        # the same episode boundary)
        "reset",
        "joint_state",
        "rgb_overhead",
        "depth_overhead",
        "rgb_wrist",
    )
    return {
        "id": "verifier-realistic",
        "path": str((root / "src" / "aisle" / "nodes" / "verifier_realistic.py").resolve()),
        # joint_state is STATE, not a command stream: latest-wins, the same
        # reasoning graphs/expert_t0.yaml gives for the executor. With a deep
        # queue the node reads STALE poses after falling behind during a
        # judge, and a wrist ROI composed from one describes a different arm
        # than the pixels show (VER-8). CAMERA queues must survive a 3-5 s
        # judge WITHOUT dropping (issue #120): the router's arrival proof
        # (a later stamp means no earlier frames remain on that stream)
        # only holds if the queue never overflows, and 100 deep is ~3.3 s
        # at 30 Hz. A third back-to-back judge can still overflow silently
        # — the residual, noted in issue #120's follow-ups.
        "inputs": {
            topic: {
                "source": producers[topic],
                "queue_size": (
                    1
                    if topic == "joint_state"
                    else CAMERA_QUEUE_DEPTH
                    if topic in ("rgb_overhead", "depth_overhead", "rgb_wrist")
                    else 100
                ),
            }
            for topic in wanted
            if topic in producers
        },
        "outputs": ["episode_result"],
        "env": {
            "AISLE_RESULTS_DIR": str(run_dir.resolve()),
            "AISLE_TIMEOUT_S": str(timeout_s),
        },
    }


class EngineConflict(RuntimeError):
    """The graph declares a physics engine the run asked to override (ADR-67).

    Its own type so the refusal reports the `sim_engine` gate rather than the
    `perception` one the other instrumentation failure carries."""


def instrumented_graph(
    graph: Path,
    root: Path,
    run_dir: Path,
    trace_dir: Path | None = None,
    name: str = "graph.yaml",
    verifier: str = "oracle",
    episode_timeout_s: float = 60.0,
    sim_backend: str | None = None,
    graph_snapshot: bytes | None = None,
    typed_stage: tuple[Path, dict] | None = None,
    work_launch: int | None = None,
    sim_engine: str | None = None,
) -> Path:
    """The input graph plus a trace-recorder node (HAR-4) with absolutized
    node paths, written under the run dir (dora's cwd becomes the run dir,
    which also scopes orphan cleanup). trace_dir/name vary per wall-clamp
    relaunch (ADR-23): the recorder opens its Arrow/video files in write
    mode, so a relaunch pointed at the SAME dir would truncate the prior
    launch's evidence (PR #58 review)."""
    from aisle.harness.registry import load_manifests
    from aisle.harness.validate import FORBIDDEN_BY_RUNG, graph_perception_rung

    text = (
        graph_snapshot.decode("utf-8")
        if graph_snapshot is not None
        else graph.read_bytes().decode("utf-8")
    )
    registry_root = root
    if typed_stage is not None:
        from aisle.harness.typed_graph_stage import transport_for_instrumentation

        text, registry_root = transport_for_instrumentation(
            *typed_stage, authored_bytes=text.encode("utf-8"), graph=graph, controller_root=root
        )
    doc = yaml.safe_load(text)
    for node in doc["nodes"]:
        node["path"] = str((graph.parent / node["path"]).resolve())
    # issue #128 (TC-9): the recorder subscribes to declared endpoints, but a
    # bridge output the graph's rung FORBIDS is filtered out — an L1 trace
    # then cannot carry the NON-privileged ground-truth pose endpoint,
    # instead of relying on the bridge's runtime restraint. (oracle_state
    # remains recorded at every rung: it is the verifier's privileged input,
    # governed by VAL-6/ADR-28, not by the rung.) FAIL CLOSED on unreadable
    # rungs (PR #135 round-2 review). Registry state is deliberately re-read
    # at launch and every wall-clamp relaunch, hours after the HAR-2 gate: a
    # registry broken in between must refuse loudly. The graph content comes
    # from rollout's one capture, so relaunches cannot silently change graphs.
    manifest_list, manifest_errors = load_manifests(registry_root)
    rung, bridge_ids, rung_errors = graph_perception_rung(
        doc["nodes"], {} if manifest_errors else {m["id"]: m for _, m in manifest_list}
    )
    if rung_errors:
        raise RuntimeError(
            "perception rung unresolvable at instrumentation time (TC-9): "
            + "; ".join(e["detail"] for e in rung_errors)
        )
    # MON-13: accounting is controller-owned, including its output paths.
    if any(
        key.startswith("AISLE_SIM_WORK_")
        for node in doc["nodes"]
        for key in (node.get("env") or {})
    ):
        raise RuntimeError("authored simulator accounting configuration is forbidden")
    if work_launch is not None:
        from aisle.harness.simulator_work import launch_binding

        bridges = [node for node in doc["nodes"] if node["id"] in bridge_ids]
        trusted_bridge = str((root / "src/aisle/nodes/dora_genesis.py").resolve())
        if len(bridges) != 1 or bridges[0]["path"] != trusted_bridge:
            raise RuntimeError("simulator accounting requires one trusted bridge")
        binding = launch_binding(
            run_dir / "simulator-work", run_id=run_dir.name, launch=work_launch
        )
        bridges[0]["env"] = {**(bridges[0].get("env") or {}), **binding}
    if sim_backend is not None:
        for node in doc["nodes"]:
            if node["id"] in bridge_ids:
                node["env"] = {**(node.get("env") or {}), "AISLE_SIM_BACKEND": sim_backend}
    if sim_engine is not None:
        for node in doc["nodes"]:
            if node["id"] not in bridge_ids:
                continue
            env = node.get("env") or {}
            # ADR-67: the engine is DECLARED on the bridge, like the rung, so
            # the graph hash attests which physics produced a result. A
            # declaration this runner would overwrite is refused (engine_check
            # catches it before launch; this is the same rule at the last
            # writer).
            declared = str(env.get("AISLE_SIM_ENGINE") or "").strip().lower()
            if "AISLE_SIM_ENGINE" in env and declared != sim_engine:
                raise EngineConflict(
                    f"{node['id']!r} declares AISLE_SIM_ENGINE={env['AISLE_SIM_ENGINE']!r}, "
                    f"but the run requested engine {sim_engine!r} (ADR-67)"
                )
            node["env"] = {**env, "AISLE_SIM_ENGINE": sim_engine}
    forbidden = FORBIDDEN_BY_RUNG.get(rung, ())
    # HAR-4: EVERY declared endpoint, keyed <producer>__<topic> so two
    # producers of the same topic name (e.g. reset_done from both the
    # bridge and the reset service) stay distinct endpoints
    inputs = {
        f"{node['id']}__{topic}": {"source": f"{node['id']}/{topic}", "queue_size": 100}
        for node in doc["nodes"]
        for topic in (node.get("outputs") or [])
        if not (node["id"] in bridge_ids and topic in forbidden)
    }
    doc["nodes"].append(
        {
            "id": "trace-recorder",
            # .resolve(): a relative root (`--root .`) with dora cwd = the
            # run dir otherwise kills the dataflow at startup, zero episodes
            "path": str((root / "src" / "aisle" / "harness" / "trace_recorder.py").resolve()),
            "inputs": inputs,
            # frame capture is declared IN THE GRAPH, not inherited from the
            # runner's process env: the graph hash then attests whether a run
            # recorded replayable frames (the same reasoning as ADR-25's
            # bring-up scrub, in the opposite direction)
            "env": {
                "AISLE_TRACE_DIR": str(trace_dir if trace_dir else run_dir / "traces"),
                "AISLE_FRAME_CAPTURE_PERIOD_S": os.environ.get("AISLE_FRAME_CAPTURE_PERIOD_S", "0"),
            },
        }
    )
    if verifier in ("both", "realistic"):
        realistic = realistic_verifier_node(root, run_dir, doc, episode_timeout_s)
        doc["nodes"].append(realistic)
        # the recorder's inputs were computed BEFORE this node existed, so its
        # verdicts were absent from every trace — add them explicitly rather
        # than leaving the node's own output the one unrecorded endpoint
        recorder = next(n for n in doc["nodes"] if n["id"] == "trace-recorder")
        for topic in realistic["outputs"]:
            recorder["inputs"][f"{realistic['id']}__{topic}"] = {
                "source": f"{realistic['id']}/{topic}",
                "queue_size": 100,
            }
    if verifier == "realistic":
        # A7 mode (Phase-2 DoD): the LOOP advances on the realistic verdict —
        # both episode_result consumers rewire to the sidecar — while the
        # ORACLE stays in the graph, held out for scoring: its endpoint
        # remains recorded above, so the analysis compares what the loop
        # believed against ground truth. `both` never rewires (its design is
        # judging WITHOUT perturbing control flow).
        for node in doc["nodes"]:
            src = (node.get("inputs") or {}).get("episode_result", {})
            if isinstance(src, dict) and src.get("source") == "verifier-oracle/episode_result":
                src["source"] = "verifier-realistic/episode_result"
        # ADR-30 §1.5: in A7 the live judge is causal — its verdict drives
        # the next reset — so it joins the closed turn and receives the
        # longer verdict-bearing watchdog budget.  In `both` it remains a
        # non-causal measurement tap and deliberately stays asynchronous.
        realistic = next(n for n in doc["nodes"] if n["id"] == "verifier-realistic")
        realistic["inputs"]["turn"] = {
            "source": "turn-barrier/turn",
            "queue_size": 4,
            "queue_policy": "backpressure",
        }
        realistic["outputs"].append("turn_done")
        realistic["env"].update(
            {
                "AISLE_LOCKSTEP": "1",
                "AISLE_TURN_NODE": "verifier-realistic",
                "AISLE_TURN_OUTPUTS": "episode_result,turn_done",
            }
        )
        barrier = next(n for n in doc["nodes"] if n["id"] == "turn-barrier")
        done_index = 1 + max(
            [
                int(port.rpartition("_")[2])
                for port in barrier["inputs"]
                if port.startswith("done_") and port.rpartition("_")[2].isdigit()
            ],
            default=-1,
        )
        barrier["inputs"][f"done_{done_index}"] = {
            "source": "verifier-realistic/turn_done",
            "queue_size": 4,
            "queue_policy": "backpressure",
        }

        # The oracle remains a held-out scoring tap.  It must continue
        # processing stamped observations, not buffer forever waiting for a
        # turn the A7 causal topology no longer schedules for it.
        oracle = next(n for n in doc["nodes"] if n["id"] == "verifier-oracle")
        oracle["inputs"].pop("turn", None)
        oracle_env = oracle.get("env") or {}
        for key in ("AISLE_LOCKSTEP", "AISLE_TURN_NODE", "AISLE_TURN_OUTPUTS"):
            oracle_env.pop(key, None)
        oracle["env"] = oracle_env

        from aisle.harness.validate import compile_turn_plan

        manifests = {m["id"]: m for _, m in manifest_list}
        plan = compile_turn_plan(doc["nodes"], manifests)
        plan_path = run_dir / f"{Path(name).stem}-turn-plan.json"
        plan_path.write_text(
            json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        barrier["env"]["AISLE_TURN_PLAN"] = str(plan_path.resolve())
    out_path = run_dir / name
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out_path


# settings that MUST come from the graph (where the graph hash attests them)
# or from the runner itself, and never from the ambient process environment
SCRUBBED_ENV = (
    "AISLE_SIM_WORK_PATH",
    "AISLE_SIM_WORK_RUN_ID",
    "AISLE_SIM_WORK_LAUNCH",
    # ADR-30: bridge incarnations are assigned by this runner.  Ambient
    # values could alias a dead process after a wall-clamp relaunch.
    "AISLE_TURN_EPOCH",
    # ADR-23: the runner's own relaunch offset. rollout() sets it AFTER
    # this scrub on a relaunch, so the override still lands; what this
    # closes is the fleet path (cli.py) and tools/h4_iteration.py, which
    # inject only AISLE_SEEDS/AISLE_RESULTS and would otherwise inherit a
    # stale developer-shell value and renumber every episode (PR #177
    # review — the same class as AISLE_TARGET_MEDS below).
    "AISLE_EPISODE_BASE",
    # HAR-1: which med each episode targets. The rollout runner never sets
    # this, and no graph declares it, so an ambient developer-shell value
    # was the ONLY way it could arrive — silently re-targeting every
    # episode of a measured run while git_sha/env_hash/graph_hash all
    # attest clean (PR #178 review). Scrubbed, the client falls back to
    # its deterministic seed-derived default. Graph tests that launch the
    # client directly via `dora run` still set it; they do not go through
    # this runner.
    "AISLE_TARGET_MEDS",
    # ADR-25 (issue #71): the bridge's bring-up opt-out
    "AISLE_STEP_WITHOUT_RESET",
    # T2: the label toggle changes the SCENE'S PIXELS -- graph-declared for
    # the same attestation reason as the rung below
    "AISLE_LABELS",
    # T2: the color permutation changes which box LOOKS like the target
    "AISLE_SHUFFLE_COLORS",
    # T2: the tier switches the state machine into the scan tour; ambient
    # leakage would run a tour on a T0/T1 graph with no reader wired
    "AISLE_TASK_TIER",
    # HAR-3: the retry budget changes pass@8 semantics — graph-attested
    "AISLE_MAX_RETRIES",
    # T3: the occlusion layout changes the SCENE — graph-attested
    "AISLE_OCCLUSION",
    # TC-9: the perception rung. The bridge reads it via parse_bridge_config(
    # os.environ), so an ambient AISLE_PERCEPTION=L1 would set the rung of a
    # run whose graph never declared one — and the validator, which sees only
    # graph YAML, could never detect the divergence. Scrubbed with the variable
    # that introduced the same hazard rather than after the first bad run.
    "AISLE_PERCEPTION",
    # ADR-67: which physics engine realizes the scene. The bridge reads it via
    # select_engine(os.environ), and `harness fleet` builds its child env from
    # this scrub, so an ambient AISLE_SIM_ENGINE=nexus in an operator's shell
    # silently swapped the physics of a fleet run while every hash attested
    # clean. Each runner that launches a sim now sets it back explicitly,
    # AFTER the scrub, so the choice is the runner's.
    "AISLE_SIM_ENGINE",
    # CON-5/BRG-6: the engine's backend, resolved by the gate from the
    # attested dependency extra. Ambient leakage would attest one backend and
    # run another, which is the same hazard as the engine above.
    "AISLE_SIM_BACKEND",
)


def scrub_bringup_env(env: dict) -> dict:
    """ADR-25 (issue #71) + TC-9: settings the graph must own must never reach a
    measured rollout from the ambient environment — an ambient
    AISLE_STEP_WITHOUT_RESET=1 would silently restore the pre-reset startup
    race, and an ambient AISLE_PERCEPTION would silently set the perception
    rung, while git_sha/env_hash/graph_hash all attest clean. Graph-YAML env
    stays visible in the graph hash; the process environment does not, so it is
    scrubbed here."""
    return {k: v for k, v in env.items() if k not in SCRUBBED_ENV}


def _dora_runtime_log(run_dir: Path, stream: str, relaunch: int = 0) -> Path:
    """Return the per-launch sink for one dora process stream."""
    suffix = f".relaunch-{relaunch}" if relaunch else ""
    return run_dir / f"dora.{stream}{suffix}.log"


def dora_stdout_log(run_dir: Path, relaunch: int = 0) -> Path:
    """Where a launch's dora daemon log goes (issue #201).

    The pinned dora tracing subscriber writes its INFO/WARN stream to stdout.
    A regular file preserves that scheduler, startup, and exit evidence without
    the bounded-buffer deadlock risk of an unread PIPE.  Each wall-clamp
    relaunch gets a distinct sink so the wedged launch's diagnostics survive.
    """
    return _dora_runtime_log(run_dir, "stdout", relaunch)


def dora_stderr_log(run_dir: Path, relaunch: int = 0) -> Path:
    """Where a launch's dora error stream goes (issue #183).

    Per LAUNCH, not per run. `Popen` opens this in write mode, so pointing
    every wall-clamp relaunch (ADR-23) at one path meant each relaunch
    truncated the stderr of the launch that had just wedged — deleting the
    one file you would read to explain why the clamp fired. Traces already
    get this isolation (`traces/relaunch-N/`) for exactly the same reason;
    stderr did not.

    Launch 1 keeps the bare name so existing habits and docs still find it;
    relaunches sort next to it. Kept at the run root rather than inside
    `traces/relaunch-N/` because this is the dora runtime's own log, not
    recorded evidence, and `traces/` is the recorder's namespace."""
    return _dora_runtime_log(run_dir, "stderr", relaunch)


def _spawn_dora(exec_graph: Path, run_dir: Path, env: dict, relaunch: int = 0) -> subprocess.Popen:
    return subprocess.Popen(
        ["dora", "run", str(exec_graph), "--uv"],
        # cwd = the run dir: dora spawns nodes with this cwd, which is what
        # the orphan reaper filters on — with cwd=root the filter matched
        # nothing and leaked nodes raced the cleanup (T09 smoke)
        cwd=run_dir,
        # Dora's Python nodes do not inherit the controller's -B flag.
        # Keep bytecode and Numba's independent JIT cache out of the runtime.
        env={
            **env,
            "PYTHONDONTWRITEBYTECODE": "1",
            "NUMBA_CACHE_DIR": str(run_dir / ".cache" / "numba"),
        },
        # The pinned dora tracing subscriber emits the daemon INFO/WARN stream
        # on stdout (issue #201). A file has no PIPE capacity ceiling, preserves
        # the scheduler/startup/exit diagnostics, and cannot block because the
        # parent forgot to drain it. Per-launch paths also stop a wall-clamp
        # relaunch from erasing the wedged launch's evidence (issue #183).
        stdout=dora_stdout_log(run_dir, relaunch).open("w"),
        # Top-level eyre errors and panics still land on dora's own stderr.
        # Per-NODE stderr is separate and already persisted by dora as
        # out/<dataflow>/log_<node>.jsonl rows with "stream":"stderr".
        stderr=dora_stderr_log(run_dir, relaunch).open("w"),
        text=True,
        start_new_session=True,
    )


def await_realistic_sidecar(run_dir: Path, expected: int, timeout_s: float = 45.0) -> int:
    """Wait (bounded) for the realistic verifier's VER-14 sidecar to carry
    one record per episode before teardown, and return how many it has.

    The node judges an episode once its end (reset request, next goal, or
    sim-budget expiry) is bounded AND the camera streams prove its frames
    arrived (issue #120); the LAST episode has no next boundary, so its
    only chance is the teardown flush — and `judge_frames` takes seconds,
    which it loses to teardown. Observed: 2 records for 3 episodes, twice.
    Waiting here is the fix that does not require the node to win a race,
    and a bounded wait cannot hang a run: on timeout the count is simply
    short and the caller reports it.

    Counts DISTINCT goal_ids, not lines (issue #160 item 5): a duplicate
    goal_id — the relaunch-numbering bug this issue fixes, or any future
    re-judge — must not mask a missing episode."""
    sidecar = run_dir / "verifier_stages.jsonl"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if sidecar.exists() and len(_sidecar_goal_ids(sidecar)) >= expected:
            return expected
        time.sleep(1.0)
    return len(_sidecar_goal_ids(sidecar)) if sidecar.exists() else 0


def _sidecar_goal_ids(sidecar: Path) -> set[str]:
    """DISTINCT goal_ids among the sidecar's parseable records; malformed
    lines count for nothing here — fidelity.load_sidecar refuses them
    later with a real error, and an unparseable line must not satisfy the
    per-episode wait."""
    ids: set[str] = set()
    for line in sidecar.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        goal_id = record.get("goal_id") if isinstance(record, dict) else None
        if isinstance(goal_id, str) and goal_id:
            ids.add(goal_id)
    return ids


def _terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _guard_divergence_or_none(run_dir: Path) -> dict | None:
    """#267: the proposed-vs-executed summary for this run, or None.

    Best-effort by construction. The statistic informs a fine-tuning label
    decision; it is not part of the run's verdict, so no failure here may cost
    the manifest that carries the verdict."""
    try:
        from aisle.harness.guard_divergence import summary_for_run

        return summary_for_run(run_dir)
    except Exception:  # noqa: BLE001 — evidence extra, never fatal to the run
        return None


def _graph_hash(graph: Path) -> str:
    return hashlib.sha256(graph.read_bytes()).hexdigest()


def perception_check(
    root: Path,
    graph: Path,
    requested: str | None,
    graph_snapshot: bytes | None = None,
) -> dict:
    """HAR-1/TC-9: the graph DECLARES the perception rung, where the graph
    hash attests it, and rollout scrubs ambient AISLE_PERCEPTION for the same
    reason — so --perception cannot inject a rung. It ASSERTS one: a mismatch
    is refused before any gate, budget reservation, or launch, because the
    run would measure a different rung than the caller asked for. Returns
    {"ok": True, "rung": <declared>} when the assertion holds (or none was
    made); the declared rung is what rollout records in the run manifest."""
    from aisle.harness.registry import load_manifests
    from aisle.harness.validate import graph_perception_rung, load_graph

    nodes, _ = load_graph(graph, graph_snapshot)
    manifest_list, manifest_errors = load_manifests(root)
    if nodes is None or manifest_errors:
        # run_gates' validate owns the full structured report for a broken
        # graph or registry; the flag only refuses what it cannot assert
        if requested is None:
            return {"ok": True, "rung": None}
        return {
            "ok": False,
            "gate": "perception",
            "detail": f"cannot assert --perception {requested}: {graph} or the registry "
            "does not load, so the declared rung is unreadable (run `harness validate`)",
        }
    rung, _, rung_errors = graph_perception_rung(nodes, {m["id"]: m for _, m in manifest_list})
    if rung_errors:
        # TC-9's refuse-don't-guess rule: an unreadable rung matches nothing —
        # never compare the request against the strictest-assumed fallback
        if requested is None:
            return {"ok": True, "rung": None}
        return {
            "ok": False,
            "gate": "perception",
            "detail": "; ".join(e["detail"] for e in rung_errors),
        }
    if requested is not None and requested != rung:
        return {
            "ok": False,
            "gate": "perception",
            "detail": f"--perception {requested} asserted, but the graph declares rung {rung} "
            "(TC-9: the rung rides the graph, where the graph hash attests it)",
            "hint": f"run a graph whose sim bridge declares AISLE_PERCEPTION: {requested}",
        }
    return {"ok": True, "rung": rung}


def engine_check(
    root: Path,
    graph: Path,
    requested: str | None,
    graph_snapshot: bytes | None = None,
) -> dict:
    """ADR-67: the physics engine rides the graph's sim bridge, exactly as the
    perception rung does (TC-9), so the graph hash attests which engine
    produced a result. `--sim-engine` therefore ASSERTS the declaration rather
    than replacing it: a conflict is refused before any gate, budget
    reservation, or launch, a graph that declares nothing takes the request
    (default genesis), and a request of nothing takes the declaration.
    Returns {"ok": True, "engine": <resolved>, "declared": <or None>}."""
    from aisle.harness.registry import load_manifests
    from aisle.harness.validate import graph_perception_rung, load_graph
    from aisle.sim import DEFAULT_ENGINE, normalize_engine

    try:
        requested = None if requested is None else normalize_engine(requested)
    except ValueError as exc:
        return {"ok": False, "gate": "sim_engine", "detail": str(exc)}
    fallback = {"ok": True, "engine": requested or DEFAULT_ENGINE, "declared": None}
    nodes, _ = load_graph(graph, graph_snapshot)
    manifest_list, manifest_errors = load_manifests(root)
    if nodes is None or manifest_errors:
        # run_gates' validate owns the verdict on a broken graph or registry;
        # this check only refuses what it can actually read
        return fallback
    _, bridge_ids, _ = graph_perception_rung(nodes, {m["id"]: m for _, m in manifest_list})
    by_id = {node["id"]: node for node in nodes}
    declared: set[str] = set()
    for bridge_id in sorted(bridge_ids):
        env = by_id[bridge_id].get("env") or {}
        if "AISLE_SIM_ENGINE" not in env:
            continue
        try:
            declared.add(normalize_engine(str(env.get("AISLE_SIM_ENGINE") or "")))
        except ValueError as exc:
            return {
                "ok": False,
                "gate": "sim_engine",
                "detail": f"{bridge_id!r} declares an unusable AISLE_SIM_ENGINE: {exc}",
            }
    if len(declared) > 1:
        return {
            "ok": False,
            "gate": "sim_engine",
            "detail": f"conflicting engines {sorted(declared)} declared across the sim "
            f"bridges {sorted(bridge_ids)} (ADR-67): a graph with two engines attests neither",
        }
    if not declared:
        return fallback
    engine = declared.pop()
    if requested is not None and requested != engine:
        return {
            "ok": False,
            "gate": "sim_engine",
            "detail": f"--sim-engine {requested} asserted, but the graph declares engine "
            f"{engine} (ADR-67: the engine rides the graph, where the graph hash attests it)",
            "hint": f"run a graph whose sim bridge declares AISLE_SIM_ENGINE: {requested}",
        }
    return {"ok": True, "engine": engine, "declared": engine}


def rollout(
    root: Path,
    graph: Path,
    tier: str,
    episodes: int,
    seeds: list[int],
    reset_mode: str,
    verifier: str,
    run_id: str,
    branch: str,
    no_idea_gate: bool,
    timeout_s: float | None = None,
    embodiment: str = "franka",
    env_baseline: str = "origin/main",
    perception: str | None = None,
    sim_extra: str = "sim",
    per_episode_wall_s: int | None = None,
    typed_stage_factory=None,
    record_simulator_work: bool = False,
    sim_engine: str | None = None,
) -> dict:
    """HAR-1: the full run. Returns the report dict (CON-8: caller emits).

    sim_engine ASSERTS the engine the graph's bridge declares (ADR-67); None
    means no assertion, so the declaration wins and an undeclared graph runs
    the default engine. The scene-build wall grace the first episode of every
    launch gets (and that each launch adds to the run budget) is
    engine-derived (`build_budget_s`), since Genesis builds in minutes and
    Nexus in seconds."""
    # A relative root (`--root .`) must be pinned to THIS process's cwd:
    # dora runs with cwd = the run dir, so relative AISLE_RESULTS /
    # AISLE_TRACE_DIR strings would resolve to a nested runs/<id>/runs/<id>/
    # tree the stall watcher never sees (T18 live shakeout)
    root = root.resolve()
    if reset_mode not in ("teleport", "behavioral"):
        return {"ok": False, "error": f"unknown reset mode {reset_mode!r}"}
    if verifier not in ("oracle", "both", "realistic"):
        return {"ok": False, "error": f"unknown verifier {verifier!r}"}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        return {"ok": False, "error": f"unsafe run_id {run_id!r}"}
    if (root / "runs" / run_id).exists():
        return {"ok": False, "error": f"run_id {run_id!r} already exists; refusing to overwrite"}
    # Issue #128 / HAR-2: one read is the run's immutable authored graph.
    # Rung detection, validation, instrumentation, relaunches, and the
    # authored hash all consume these exact bytes. The original path remains
    # provenance for VAL-2 relative-path identity checks.
    try:
        graph_snapshot = graph.read_bytes()
    except OSError as exc:
        return {
            "ok": False,
            "refused": {
                "gate": "validate",
                "detail": [
                    {
                        "code": "GRAPH_INVALID",
                        "node": str(graph),
                        "detail": f"cannot read graph: {exc}",
                        "hint": "pass a readable UTF-8 dataflow YAML path",
                    }
                ],
            },
        }
    typed_stage = None
    typed_stages = []
    typed_stage_error = None
    typed_postflight = []
    typed_artifacts = []
    registry_root = root
    if typed_stage_factory is not None:
        from aisle.harness.typed_graph_stage import select_rollout_stage

        try:
            typed_stage, registry_root = select_rollout_stage(
                typed_stage_factory,
                0,
                typed_stages,
                authored_bytes=graph_snapshot,
                graph=graph,
                controller_root=root,
                embodiment=embodiment,
            )
        except RuntimeError as exc:
            return {
                "ok": False,
                "infrastructure_invalid": True,
                "refused": {"gate": "typed_stage", "detail": str(exc)},
            }
    perception_gate = perception_check(registry_root, graph, perception, graph_snapshot)
    if not perception_gate["ok"]:
        return {"ok": False, "refused": perception_gate}
    engine_gate = engine_check(root, graph, sim_engine, graph_snapshot)
    if not engine_gate["ok"]:
        return {"ok": False, "refused": engine_gate}
    sim_engine = engine_gate["engine"]
    authored_graph_hash = hashlib.sha256(graph_snapshot).hexdigest()
    gates = run_gates(
        root,
        graph,
        branch,
        no_idea_gate,
        embodiment,
        env_baseline,
        episodes,
        sim_extra,
        graph_snapshot=graph_snapshot,
        **({"typed_stage": typed_stage} if typed_stage is not None else {}),
        sim_engine=sim_engine,
    )
    if not gates["ok"]:
        return {
            "ok": False,
            "refused": gates,
            **({"infrastructure_invalid": True} if gates.get("infrastructure_invalid") else {}),
        }
    if reset_mode == "behavioral":
        # RST-2 (issue #196): refuse rather than silently degrade. Most
        # graphs wire the reset node with {reset, reset_done} only; on those,
        # mode 1 is accepted, burns all three attempts on a frame that never
        # arrives, and falls back to teleport every episode. The run then
        # reports fallback:true and measures nothing, with no error.
        #
        # BELOW run_gates deliberately: validate owns the verdict on a
        # malformed graph (GRAPH_INVALID), so a `nodes:`-less or wrong-shaped
        # file is diagnosed as broken rather than as "cannot serve a
        # behavioral reset" — the diagnosis must not depend on --reset.
        # ABOVE the ledger: a refusal costs no episodes.
        from aisle.harness.validate import behavioral_reset_gaps, load_graph

        gaps = behavioral_reset_gaps(load_graph(graph, graph_snapshot)[0] or [])
        if gaps:
            return {
                "ok": False,
                "refused": {
                    "gate": "reset_mode",
                    "detail": [
                        {
                            "code": "BEHAVIORAL_RESET_UNSUPPORTED",
                            "node": "reset",
                            "detail": "graph cannot serve a behavioral reset",
                            "missing": gaps,
                            "hint": "use --reset teleport, or a graph whose reset node is wired "
                            "for RST-2 (see graphs/expert_t1_behavioral.yaml)",
                        }
                    ],
                },
            }
    # ADR-21 round 3: RESERVE atomically before launch (trusted runs only;
    # the ledger meters the campaign, not development) — concurrent
    # rollouts contend under the ledger lock and an interrupted run stays
    # charged at its reservation until settled
    reservation = None
    if env_baseline != "local":
        reservation = reserve_budget(root, run_id, episodes)
        if not reservation["ok"]:
            return {"ok": False, "refused": reservation}

    seeds = (seeds * ((episodes + len(seeds) - 1) // len(seeds)))[:episodes]
    run_dir = root / "runs" / run_id
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    work_directory = run_dir / "simulator-work"
    if record_simulator_work:
        work_directory.mkdir(exist_ok=False)
    authored_snapshot_path = run_dir / "authored-graph.yaml"
    authored_snapshot_path.write_bytes(graph_snapshot)
    # budgets before the graph: the realistic verifier node needs the episode
    # SIM timeout declared in the graph, since it ends episodes on its own
    # budget rather than waiting for the oracle (VER-5, increment 1b)
    try:
        episode_timeout_s, per_episode_budget_s = resolve_budgets(
            tier, verifier, per_episode_wall_override_s=per_episode_wall_s
        )
    except ValueError as exc:
        # refuse the unbudgeted tier/verifier combination the same way the
        # gates refuse: loudly, before anything launches (PR #177 review)
        return {"ok": False, "error": str(exc)}
    try:
        exec_graph = instrumented_graph(
            graph,
            root,
            run_dir,
            verifier=verifier,
            episode_timeout_s=episode_timeout_s,
            sim_backend=gates["sim_backend"],
            **({"work_launch": 0} if record_simulator_work else {}),
            graph_snapshot=graph_snapshot,
            **({"typed_stage": typed_stage} if typed_stage is not None else {}),
            sim_engine=gates["sim_engine"],
        )
    except EngineConflict as exc:
        return {"ok": False, "refused": {"gate": "sim_engine", "detail": str(exc)}}
    except RuntimeError as exc:
        return {
            "ok": False,
            "refused": {"gate": "perception", "detail": str(exc)},
            **({"infrastructure_invalid": True} if typed_stage is not None else {}),
        }
    # issue #128: attest what actually RAN, not only what was authored — one
    # hash per launch (a wall-clamp relaunch writes a new exec copy whose
    # trace_dir env differs, so its hash differs)
    exec_graph_hashes = [_graph_hash(exec_graph)]
    results_path = run_dir / "episodes.jsonl"

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root
    ).stdout.strip()
    env_hash = gates["env_hash"]

    env = scrub_bringup_env(
        {
            **os.environ,
            "AISLE_SEEDS": ",".join(str(s) for s in seeds),
            "AISLE_TURN_EPOCH": "1",
            # the caller-selected tier propagates to the graph (HAR-1): the
            # rollout client stamps it into every goal, and the SELECTED graph
            # determines its tier-specific wiring
            "AISLE_TIER": tier,
            # M0-5: the embodiment profile swap rides on env, zero YAML edits
            "AISLE_EMBODIMENT": embodiment,
            "AISLE_TIMEOUT_S": str(episode_timeout_s),
            "AISLE_RESULTS": str(results_path),
            # RST-2: the client stamps every reset request with the mode
            "AISLE_RESET_MODE": reset_mode,
        }
    )
    # CON-5 / ADR-67: applied AFTER the scrub, which now strips both from the
    # ambient environment. The gate resolved them from the explicitly selected
    # extra and the graph's declaration; a developer shell never may.
    env["AISLE_SIM_BACKEND"] = gates["sim_backend"]
    env["AISLE_SIM_ENGINE"] = gates["sim_engine"]
    started = time.monotonic()
    build_grace_s = build_budget_s(gates["sim_engine"])
    stall_grace_s = pre_data_stall_s(gates["sim_engine"])
    run_budget_s = timeout_s or (build_grace_s + per_episode_budget_s * episodes)
    if env_baseline != "local":
        # ADR-21 round 3: the run is CAPPED to the campaign's remaining
        # wall budget — a single long rollout cannot blow through the
        # ceiling it passed at the gate
        run_budget_s = min(run_budget_s, gates["budget"]["wall_h_left"] * 3600.0)
    # the campaign wall cap also bounds relaunch deadline extensions
    hard_cap_s = gates["budget"]["wall_h_left"] * 3600.0 if env_baseline != "local" else None
    deadline = started + run_budget_s

    def retain_work():
        if not record_simulator_work:
            return
        from aisle.harness.simulator_work import summarize_launches

        summary = summarize_launches(
            work_directory, run_id=run_id, expected_graph_hashes=exec_graph_hashes
        )
        (run_dir / "simulator-work-summary.json").write_text(json.dumps(summary, indent=2))

    def spawn_launch(index):
        if record_simulator_work:
            from aisle.harness.simulator_work import reserve_launch

            reserve_launch(
                work_directory,
                run_id=run_id,
                launch=index,
                graph_sha256=exec_graph_hashes[index],
            )
        return _spawn_dora(exec_graph, run_dir, env, **({"relaunch": index} if index else {}))

    proc = None
    pending_typed_stage = typed_stage

    def audit_stopped_stage():
        nonlocal pending_typed_stage, typed_stage_error
        if pending_typed_stage is None:
            return True
        from aisle.harness.typed_graph_audit import audit_graph_stage, retain_graph_stage

        stage_path, stage_record = pending_typed_stage
        launch_index = len(typed_postflight)
        try:
            audit = audit_graph_stage(stage_path, stage_record)
        except Exception as exc:
            audit = {
                "ok": False,
                "stage_id": stage_record["immutable_id"],
                "errors": [{"node": None, "error": str(exc)}],
                "process_tree_verified": False,
                "confirmatory_ready": False,
            }
        artifact_path = f"typed-artifacts-{launch_index}"
        try:
            collection = retain_graph_stage(stage_path, run_dir / artifact_path, audit)
        except Exception as exc:
            collection = {"ok": False, "errors": [str(exc)]}
        typed_artifacts.append(
            {"launch": launch_index, "path": artifact_path, "report": collection}
        )
        typed_postflight.append(audit)
        (run_dir / f"typed-postflight-{launch_index}.json").write_text(json.dumps(audit, indent=1))
        pending_typed_stage = None
        if audit.get("ok") is not True or collection.get("ok") is not True:
            typed_stage_error = typed_stage_error or {
                "launch": launch_index,
                "error": "typed worker postflight or evidence collection failed",
                "audit_errors": audit.get("errors", []),
                "collection_errors": collection.get("errors", []),
            }
            return False
        return True

    episode_records: list[dict] = []
    stalled = False
    clamped_seeds: list[int] = []
    relaunches = 0
    last_size = -1
    last_growth = time.monotonic()
    current_traces = traces_dir
    lines_at_launch = 0
    count_met_t = None  # inc-2 chained-recovery grace (see below)
    lines_at_count = 0
    last_lines = 0
    last_line_t = time.monotonic()
    progress_every = progress_interval_s()
    last_progress_t = time.monotonic()
    try:
        proc = spawn_launch(0)
        while time.monotonic() < deadline:
            raw = results_path.read_bytes() if results_path.exists() else b""
            # T4 inc-2: recovery records ("recovery": true) are EXTRA
            # rows chained inside an episode -- they must not satisfy
            # the completion count (measured: a 3-seed run stopped at 3
            # rows with seed 4's recovery and seed 8 unrun). And the
            # count alone must not break MID-CHAIN: the last seed's
            # recovery lands after its goal-1 row (measured again on
            # r3: seed 8's recovery cut off). Once the count is met,
            # wait for the client to exit or the file to go quiet.
            total = raw.count(b"\n")
            lines = total - raw.count(b'"recovery": true')
            if lines >= episodes:
                if count_met_t is None:
                    count_met_t = time.monotonic()
                    lines_at_count = total
                if total > lines_at_count:
                    lines_at_count = total
                    count_met_t = time.monotonic()
                if proc.poll() is not None or (
                    typed_stage is None and time.monotonic() - count_met_t > 90.0
                ):
                    break
                if typed_stage is not None:
                    # Typed hosts still verify runtime integrity and write their
                    # terminal records after authored workers finish. Episode
                    # silence is not a completed host postflight. The enclosing
                    # run deadline continues to bound this wait.
                    time.sleep(2.0)
                    continue
            if lines != last_lines:
                last_lines = lines
                last_line_t = time.monotonic()
            if proc.poll() is not None:
                break
            # liveness: a dead bridge leaves `dora run` alive but the trace
            # stream frozen (a NaN crash burned 17 idle minutes in the T09
            # diag run) — bail once traces stop growing. Before the FIRST
            # data the genesis build is running (minutes, no traces yet),
            # so the pre-data grace is much longer (an early fire killed
            # the building bridge at 180 s)
            # the stall signal watches the CURRENT launch's dir only:
            # keyed on the whole tree, a relaunch build inherits the prior
            # launch's nonzero total and gets the short post-data grace —
            # falsely stall-killed mid-build (PR #58 self-review)
            size = sum(f.stat().st_size for f in current_traces.glob("*") if f.is_file())
            if size != last_size:
                last_size = size
                last_growth = time.monotonic()
            elif time.monotonic() - last_growth > (stall_grace_s if last_size <= 0 else STALL_S):
                stalled = True
                break
            grace = per_episode_budget_s + (build_grace_s if lines == lines_at_launch else 0)
            now = time.monotonic()
            if progress_every > 0 and now - last_progress_t >= progress_every:
                last_progress_t = now
                print(
                    progress_report(
                        now=now,
                        started=started,
                        deadline=deadline,
                        lines=lines,
                        episodes=episodes,
                        seeds=seeds,
                        last_line_t=last_line_t,
                        episode_grace_s=grace,
                        traces_size=size,
                        last_record=_last_record(raw),
                        relaunches=relaunches,
                        timing=summarize_timing(read_timing_rows(run_dir)[-3:]),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            # per-episode WALL clamp (ADR-23, W/S2 holdout wedge): traces
            # kept GROWING while one episode ran 4 h, so the stall detector
            # above never fired and the wedged episode ate the whole run
            # window, masking every remaining seed. An episode gets the
            # tier's per-episode wall budget (+ build grace when it is the
            # first of a launch); past that it is killed, recorded as a
            # synthetic wall_clamp failure, and the run RELAUNCHES with the
            # remaining seeds so they still get scored.
            if time.monotonic() - last_line_t > grace:
                _terminate(proc)
                # stale nodes from the killed launch are CONCURRENT WRITERS
                # to results/traces (dora-rs/dora#2856) — reap before any
                # relaunch, not only in the finally (PR #58 review)
                reap_orphans(run_dir)
                postflight_ok = audit_stopped_stage()
                seed = seeds[lines] if lines < len(seeds) else None
                with open(results_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "episode": lines,
                                "seed": seed,
                                # the SAME goal_id the client would have
                                # minted for this attempt: without it
                                # fidelity.load_oracle_results refuses the
                                # whole run ("episode has no goal_id"), so
                                # a relaunched A7/both run still lost its
                                # VER-6 comparison — the refusal had just
                                # moved from load_sidecar (PR #177 review)
                                "goal_id": f"ep-{lines:04d}",
                                "status": "fail",
                                "failure": "wall_clamp",
                                "success": False,
                                "synthetic": True,
                                # TC-8: the oracle is the only ground truth
                                # a metric may count; this row is the
                                # harness's own synthesized outcome
                                "verifier": "oracle",
                            }
                        )
                        + "\n"
                    )
                if seed is not None:
                    clamped_seeds.append(seed)
                remaining = seeds[lines + 1 :]
                if not remaining or not postflight_ok:
                    break
                relaunches += 1
                # the relaunched client continues the RUN-GLOBAL episode
                # numbering (issue #160 item 5): `lines` rows exist before
                # the synthetic clamp record, so the next episode is
                # lines + 1 — a restart at ep-0000 would duplicate goal_ids
                # and fidelity.load_sidecar refuses duplicates, losing the
                # whole VER-6 comparison of a relaunched A7/both run
                env = {
                    **env,
                    "AISLE_SEEDS": ",".join(str(s) for s in remaining),
                    "AISLE_EPISODE_BASE": str(lines + 1),
                    # ADR-30: a relaunched bridge is a new process epoch.
                    # Turn ids restart at zero, so the monotonically
                    # increasing epoch prevents delayed traffic from the
                    # reaped process aliasing the replacement.
                    "AISLE_TURN_EPOCH": str(relaunches + 1),
                }
                # fresh trace dir + graph per launch: the recorder truncates
                # on open, and prior evidence must survive (HAR-4)
                relaunch_traces = traces_dir / f"relaunch-{relaunches}"
                relaunch_traces.mkdir(parents=True, exist_ok=True)
                current_traces = relaunch_traces
                try:
                    if typed_stage_factory is not None:
                        typed_stage, _ = select_rollout_stage(
                            typed_stage_factory,
                            relaunches,
                            typed_stages,
                            authored_bytes=graph_snapshot,
                            graph=graph,
                            controller_root=root,
                            embodiment=embodiment,
                        )
                    exec_graph = instrumented_graph(
                        graph,
                        root,
                        run_dir,
                        trace_dir=relaunch_traces,
                        name=f"graph-r{relaunches}.yaml",
                        verifier=verifier,
                        episode_timeout_s=episode_timeout_s,
                        sim_backend=gates["sim_backend"],
                        **({"work_launch": relaunches} if record_simulator_work else {}),
                        graph_snapshot=graph_snapshot,
                        **({"typed_stage": typed_stage} if typed_stage is not None else {}),
                        sim_engine=gates["sim_engine"],
                    )
                except RuntimeError as exc:
                    # fail closed mid-run too: a registry broken since the
                    # gate must not relaunch with an unfiltered recorder —
                    # remaining seeds are lost LOUDLY (they stay short in
                    # the episode count) rather than recorded unattested
                    if typed_stage_factory is not None:
                        typed_stage_error = {"launch": relaunches, "error": str(exc)}
                    print(f"relaunch refused: {exc}", file=sys.stderr)
                    break
                exec_graph_hashes.append(_graph_hash(exec_graph))
                # each relaunch pays a fresh build: extend the deadline by
                # the build grace (still bounded by the campaign wall cap),
                # else consecutive wedges cut the tail seeds (PR #58 review)
                deadline += build_grace_s
                if hard_cap_s is not None:
                    deadline = min(deadline, started + hard_cap_s)
                pending_typed_stage = typed_stage
                proc = spawn_launch(relaunches)
                lines_at_launch = last_lines = lines + 1
                last_line_t = time.monotonic()
                last_size = -1
                last_growth = time.monotonic()
                continue
            time.sleep(2.0)
    finally:
        from contextlib import ExitStack

        # Every cleanup is attempted, even when an earlier cleanup fails.
        # Register in reverse order: stop, reap, audit, then retain accounting.
        with ExitStack() as cleanup:
            cleanup.callback(retain_work)
            cleanup.callback(audit_stopped_stage)
            cleanup.callback(reap_orphans, run_dir)
            if proc is not None:
                cleanup.callback(_terminate, proc)
            # let the realistic judge finish the LAST episode before teardown
            # (it judges on the next goal, and the last episode has none)
            if verifier in ("both", "realistic"):
                await_realistic_sidecar(run_dir, episodes)
            # ADR-21 round 3: reconcile the reservation with actuals no matter
            # how the run ended — crash paths settle too. Count from the RESULTS
            # FILE, not episode_records: that list is parsed after this
            # try/finally, so it is always [] here and every settle recorded 0
            # episodes — the ceiling never decremented (found by the first real
            # trusted campaign run; wall clamps' synthetic records count, they
            # consumed attempts)
            if reservation is not None:
                actual_episodes = (
                    sum(1 for line in results_path.read_text().splitlines() if line.strip())
                    if results_path.exists()
                    else 0
                )
                settle_budget(root, run_id, actual_episodes, time.monotonic() - started)
        if verifier in ("both", "realistic"):
            # count AFTER teardown, not before: the node can still land the
            # last record during the SIGTERM grace, and reporting the
            # pre-teardown count said "2/3" for a run that ended with 3/3
            judged = await_realistic_sidecar(run_dir, episodes, timeout_s=0.0)
            if judged < episodes:
                print(f"realistic sidecar has {judged}/{episodes} records", file=sys.stderr)

    if results_path.exists():
        episode_records = [
            json.loads(line) for line in results_path.read_text().splitlines() if line.strip()
        ]
    # ADR-24 D2 (PR #69 review F1): a trusted run is attested only if the
    # environment ALSO verifies after execution — against the gate-time
    # inventory, via the self-verified checker. Dev runs (local baseline)
    # skip the ~2-min audit per D4 and record post_run_audit: null.
    post_run_audit = None
    env_attested = gates.get("env_attested")
    if env_baseline != "local" and gates.get("dist_inventory"):
        inventory_path = run_dir / "gate_inventory.json"
        inventory_path.write_text(json.dumps(gates["dist_inventory"]))
        audit_cmd = [
            sys.executable,
            str(root / "tools" / "env_hash.py"),
            "--verify-records",
            "--expected",
            str(inventory_path),
            "--root",
            str(root),
        ]
        if gates.get("env_baseline_oid"):
            audit_cmd += ["--baseline", gates["env_baseline_oid"]]
        audit_proc = subprocess.run(audit_cmd, capture_output=True, text=True)
        try:
            post_run_audit = json.loads(audit_proc.stdout)
        except json.JSONDecodeError:
            post_run_audit = {"ok": False, "problems": ["audit produced no parseable report"]}
        env_attested = bool(env_attested) and bool(post_run_audit.get("ok"))
    elif env_baseline != "local":
        # trusted run without an inventory: never mark attested
        env_attested = False
    wall_s = time.monotonic() - started
    metrics = compute_metrics(episode_records)
    videos = sorted(str(p.relative_to(root)) for p in traces_dir.rglob("*.mp4"))
    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "env_hash": env_hash,
        "platform": platform_module.platform(),
        "graph": str(graph),
        # Issue #128: graph_hash and graph_snapshot attest the one AUTHORED
        # capture; exec_graph_hashes attest each derived copy that actually ran.
        "graph_hash": authored_graph_hash,
        "graph_snapshot": str(authored_snapshot_path.relative_to(root)),
        "exec_graph_hashes": exec_graph_hashes,
        "tier": tier,
        # TC-9: the run attests which pose source produced it, read from the
        # graph (never from the flag or ambient env — both cannot inject)
        "perception": perception_gate["rung"],
        "embodiment": embodiment,
        # CON-5 / SCN-7: dependency selection and the resolved live device
        # distinguish CPU/Metal/CUDA physics evidence with no hidden probe.
        "sim_extra": gates["sim_extra"],
        "sim_engine": gates["sim_engine"],
        "sim_backend": gates["sim_backend"],
        "sim_device": gates["sim_device"],
        # ADR-67: env_hash cannot tell two engines (or two solver settings)
        # apart, so the engine digest rides here
        "sim_engine_build": gates.get("sim_engine_build"),
        # ADR-67: the bridge's step/render/GPU timing, aggregated over the run,
        # so engines are compared on the same graph and seeds
        "sim_timing": summarize_timing(read_timing_rows(run_dir)),
        # the first-episode build grace this run clamped against (HAR-1)
        "build_grace_s": build_grace_s,
        "seeds": seeds,
        "reset": reset_mode,
        "verifier": verifier,
        "idea": gates.get("idea"),
        "no_idea_gate": gates.get("no_idea_gate", False),
        # ADR-21: which frozen-set baseline validated this run — "local" is
        # the logged dev override, auditable in every manifest — and the
        # RESOLVED immutable commit OID plus this run's ledger reservation
        # hash, so the audit can re-verify both
        "env_baseline": gates.get("env_baseline"),
        "env_baseline_oid": gates.get("env_baseline_oid"),
        # ADR-24 (HAR-4): the CON-5 fifth component + attestation verdict
        "env_fingerprint": gates.get("env_fingerprint"),
        "env_attested": env_attested,
        "dist_problems": gates.get("dist_problems"),
        "post_run_audit": post_run_audit,
        "budget_reservation": (reservation or {}).get("entry"),
        # HAR-5: best-effort token accounting
        "tokens_log": os.environ.get("ANTHROPIC_TOKENS_LOG"),
        # ADR-23: per-episode wall clamp — which seeds were cut and how
        # many relaunches carried the remaining seeds
        "wall_clamped": clamped_seeds,
        "relaunches": relaunches,
        # #267: how often the EXECUTED action differed from the PROPOSED one.
        # Persisted rather than left recomputable: the traces it derives from
        # live under gitignored runs/, so a statistic that is only "available
        # in principle" is the #245/#266 failure one artifact over. Never
        # fatal — a missing or truncated trace loses the statistic, not the
        # run record it accompanies.
        "guard_divergence": _guard_divergence_or_none(run_dir),
    }
    if typed_stage_factory is not None:
        manifest["campaign_purpose"] = "expert_parity"
        manifest["typed_stages"] = typed_stages
        manifest["typed_stage_error"] = typed_stage_error
        manifest["typed_postflight"] = typed_postflight
        manifest["typed_artifacts"] = typed_artifacts
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {
        "ok": len(episode_records) >= episodes and typed_stage_error is None,
        **({"campaign_purpose": "expert_parity"} if typed_stage_factory is not None else {}),
        **({"infrastructure_invalid": True} if typed_stage_error is not None else {}),
        "stalled": stalled,
        "run_id": run_id,
        **metrics,
        "episodes": episode_records,
        "traces_dir": str(traces_dir.relative_to(root)),
        "videos": videos,
        "durations": {
            "wall_s": round(wall_s, 1),
            "sim_s": round(sum(e.get("t_end", 0.0) for e in episode_records), 1),
        },
        "budget": budget_remaining(root),
    }
