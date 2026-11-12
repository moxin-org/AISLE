"""Read-only P0 inventory and alignment audit. Findings are not dataset certification.

JSON to stdout (CON-8). No simulator, model, training, or external service is used.
Video decoding is optional; a matching frame count does not prove pixel identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from pathlib import Path

from aisle.harness.traces import episode_window, query

SOURCE_FILES = (
    "tools/audit_skill_data.py",
    "tools/demo_export.py",
    "tools/demo_dataset.py",
    "tools/finetune_smolvla.py",
    "src/aisle/harness/traces.py",
    "src/aisle/harness/trace_recorder.py",
    "src/aisle/harness/skill.py",
    "src/aisle/nodes/ik_trajectory.py",
    "skills/ik-transfer-v2/ik_transfer_v2.py",
    "skills/ik-transfer-v2/skill.yaml",
    "skills/ik-transfer-v2/eval.yaml",
    "graphs/eval_ik_transfer_v2.yaml",
)


def loader_probe(run: Path, record: dict) -> dict:
    """Keep P0 diagnostics readable after retiring the unsafe unversioned loader.

    The committed P0 report retains its original probe. Current callers use the
    P1 exporter/manifest for coverage; do not mislabel legacy behavior as current.
    """
    try:
        requests = query(run, "target_request", episode=record["episode"]).get("text", [])
    except FileNotFoundError:
        requests = []
    return {
        "status": "legacy_probe_retired_use_versioned_demo_export",
        "target_request_examples": requests[:1],
    }


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def skill_referenced(run: Path, skill: str) -> bool:
    # A textual reference is a search hit, NOT proof that a stage executed.
    return any(
        skill in p.read_text()
        for p in [run / "manifest.json", run / "graph.yaml", run / "authored-graph.yaml"]
        if p.is_file()
    )


def inventory(roots: list[Path], skill: str) -> list[dict]:
    result = []
    for root in roots:
        files = sorted(root.rglob("episodes.jsonl")) if root.is_dir() else []
        result.append(
            {
                "root": str(root),
                "exists": root.exists(),
                "episode_files": len(files),
                "skill_runs": [
                    {
                        "run": str(p.parent),
                        "episodes_sha256": sha256(p),
                        "arrow_files": len(list((p.parent / "traces").glob("*.arrow"))),
                        "video_exists": (p.parent / "traces/overhead.mp4").is_file(),
                    }
                    for p in files
                    if skill_referenced(p.parent, skill)
                ],
            }
        )
    return result


def video_summary(path: Path, indices: set[int]) -> dict:
    if not path.is_file():
        return {"status": "missing"}
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    samples, count = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count in indices:
                samples.append(
                    {
                        "index": count,
                        "shape": list(frame.shape),
                        "decoded_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    }
                )
            count += 1
    finally:
        capture.release()
    return {
        "status": "decoded",
        "container_frames": declared,
        "decoded_frames": count,
        "fps": fps,
        "samples": samples,
        "pixel_source_identity": "unverified_without_capture_mapping",
    }


def inspect_run(run: Path, episodes: list[int], decode_video=True, skill="ik-transfer-v2") -> dict:
    records = [json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines() if x]
    manifest = json.loads((run / "manifest.json").read_text())
    frames = query(run, "rgb_overhead")["sim_time_ns"]
    if any(a > b for a, b in zip(frames, frames[1:], strict=False)):
        raise ValueError("nonmonotonic camera timestamps: index audit requires ordered frames")
    output, sample_indices = [], set()
    for episode in episodes:
        rec = next(r for r in records if r["episode"] == episode)
        start, end = episode_window(run, episode)
        commands = query(run, "joint_cmd_safe", episode=episode)
        states = query(run, "joint_state", episode=episode)
        grips = query(run, "gripper_cmd_safe", episode=episode)
        times = commands["sim_time_ns"]
        state_times, grip_times = states["sim_time_ns"], grips["sim_time_ns"]
        for stamps in (times, state_times, grip_times):
            if any(a > b for a, b in zip(stamps, stamps[1:], strict=False)):
                raise ValueError("nonmonotonic control timestamps")
        matches = [bisect_right(frames, stamp) - 1 for stamp in times]
        kept = [i for i, f in enumerate(matches) if f >= 0]
        selected = sorted({0, len(times) - 1}) if times else []
        examples = []
        for i in selected:
            f = matches[i]
            if f >= 0:
                sample_indices.add(f)
            examples.append(
                {
                    "command_row_in_episode": i,
                    "command_ns": times[i],
                    "frame_index_assumed_by_exporter": f,
                    "frame_ns": frames[f] if f >= 0 else None,
                    "state_row": bisect_right(state_times, times[i]) - 1,
                    "gripper_row": bisect_right(grip_times, times[i]) - 1,
                }
            )
        output.append(
            {
                "episode": episode,
                "recorded_status": rec.get("status"),
                "recorded_failure": rec.get("failure"),
                "seed": rec.get("seed"),
                "window_ns": [start, end],
                "commands": len(times),
                "command_widths": sorted({len(x) for x in commands["data"] if x is not None}),
                "state_widths": sorted({len(x) for x in states["data"] if x is not None}),
                "commands_without_episode_frame": sum(f < 0 or frames[f] < start for f in matches),
                "legacy_cross_episode_frame_pairs": sum(
                    f >= 0 and frames[f] < start for f in matches
                ),
                "legacy_command_index_mismatches": sum(i != j for j, i in enumerate(kept)),
                "commands_without_prior_state": sum(
                    bisect_right(state_times, t) == 0 for t in times
                ),
                "commands_without_prior_gripper": sum(
                    bisect_right(grip_times, t) == 0 for t in times
                ),
                "episode_record_has_target_med": "target_med" in rec,
                "legacy_exporter_keeps_status": rec.get("status") == "success",
                "examples": examples,
                "loader_probe": loader_probe(run, rec),
            }
        )
    paths = [
        run / n for n in ("manifest.json", "episodes.jsonl", "graph.yaml", "authored-graph.yaml")
    ]
    paths += sorted((run / "traces").glob("*.arrow"))
    paths += [run / "traces/overhead.mp4"]
    video = (
        video_summary(run / "traces/overhead.mp4", sample_indices)
        if decode_video
        else {"status": "not_requested"}
    )
    return {
        "run": str(run),
        "requested_skill_referenced": skill_referenced(run, skill),
        "provenance": {
            k: manifest.get(k)
            for k in (
                "git_sha",
                "env_hash",
                "env_fingerprint",
                "env_attested",
                "env_baseline",
                "post_run_audit",
                "graph_hash",
                "sim_engine",
                "embodiment",
                "perception",
            )
        },
        "authored_graph_hash_matches": (
            sha256(run / "authored-graph.yaml") == manifest.get("graph_hash")
            if (run / "authored-graph.yaml").is_file()
            else None
        ),
        "artifacts_sha256": {str(p.relative_to(run)): sha256(p) for p in paths if p.is_file()},
        "camera_rows": len(frames),
        "duplicate_camera_timestamps": len(frames) - len(set(frames)),
        "video": video,
        "raw_capture_files": len(list((run / "traces/frames").rglob("*.npz"))),
        "episodes": output,
        "limitations": [
            "Episode status is the retained verifier verdict, not independent re-verification.",
            "Skill reference is not execution attribution; "
            "source and stage identity need separate evidence.",
            "Trace queries retain complete Arrow batches; "
            "this audit does not certify truncated tails.",
            "Matching video counts does not establish image identity "
            "or same-timestamp event ordering.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", default="0,1")
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--skill", default="ik-transfer-v2")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = {
            "ok": True,
            "evidence_kind": "read_only_audit_not_certification",
            "current_source_sha256": {
                name: sha256(Path(__file__).resolve().parents[1] / name) for name in SOURCE_FILES
            },
            "inventory": inventory(args.search_root, args.skill),
            "audit": inspect_run(
                args.run,
                [int(x) for x in args.episodes.split(",")],
                not args.skip_video,
                args.skill,
            ),
        }
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        StopIteration,
        ImportError,
    ) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
