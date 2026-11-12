"""Offline P1 dataset contract. No runtime instrumentation or model imports.

Source rows are immutable identities. Strictly earlier joins avoid inventing an
ordering between independent producers at equal simulation timestamps. Raw RGB
captures can establish timestamp identity for a single environment/camera;
legacy video ordinals cannot. Exports retain diagnostics and every outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

FORMAT_VERSION = 1
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_CONTRACT = {
    "embodiment": "so101",
    "state": {"names": JOINTS, "units": ["rad"] * 6, "frame": "joint"},
    "forwarded": {
        "names": JOINTS,
        "units": ["rad"] * 6,
        "frame": "joint",
        "semantics": "position_target",
    },
    "gripper": {
        "names": ["gripper"],
        "units": ["normalized_closed_fraction"],
        "frame": "joint",
        "semantics": "position_target",
    },
    "training_action": "first_five_forwarded_joints_plus_gripper",
}
TOPICS = (
    "reset_done",
    "rgb_overhead",
    "joint_state",
    "joint_cmd_safe",
    "gripper_cmd_safe",
    "joint_cmd",
    "gripper_cmd",
    "target_request",
    "episode_goal",
)
CHANNEL_TOPICS = {
    "forwarded": "joint_cmd_safe",
    "proposed": "joint_cmd",
    "proposed_gripper": "gripper_cmd",
    "state": "joint_state",
    "gripper": "gripper_cmd_safe",
    "frame": "rgb_overhead",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def split_for(family: str) -> str:
    """Group before sampling; unknown families share ONE conservative group."""
    if not isinstance(family, str) or not family:
        raise ValueError("task_family must be a nonempty string")
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def strict_stream(path: Path, root: Path) -> list[dict]:
    import pyarrow as pa

    from aisle.harness.trace_recorder import TRACE_SCHEMA

    # Arrow permits EOF without EOS; for a closed dataset that is incomplete.
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8))
        if stream.read() != b"\xff\xff\xff\xff\x00\x00\x00\x00":
            raise ValueError(f"incomplete Arrow stream: {path.name}")
    try:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_stream(source)
            if not reader.schema.equals(TRACE_SCHEMA, check_metadata=False):
                raise ValueError(f"unsupported Arrow schema: {path.name}")
            rows = reader.read_all().to_pylist()
            if source.tell() != path.stat().st_size:
                raise ValueError(f"trailing Arrow bytes: {path.name}")
    except pa.ArrowException as exc:
        raise ValueError(f"invalid Arrow stream: {path.name}: {exc}") from exc
    producer, sep, topic = path.stem.partition("__")
    previous = {}
    for i, row in enumerate(rows):
        stamp, env, seq = (row[k] for k in ("sim_time_ns", "env_id", "seq"))
        if stamp is None or stamp < 0 or env is None or seq is None:
            raise ValueError(f"missing/invalid timestamp or identity: {path.name} row {i}")
        if env in previous and (stamp <= previous[env][0] or seq <= previous[env][1]):
            raise ValueError(f"duplicate/nonmonotonic timestamp or sequence: {path.name} row {i}")
        previous[env] = (stamp, seq)
        row.update(
            source=str(path.relative_to(root)),
            producer=producer if sep else None,
            topic=topic if sep else producer,
            row=i,
        )
    return rows


def resolve_stream(run: Path, topic: str, endpoints: dict) -> Path | None:
    if topic in endpoints:
        name = endpoints[topic]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(f"__{topic}.arrow")
        ):
            raise ValueError(f"invalid endpoint for {topic}")
        path = run / "traces" / name
        if not path.is_file():
            raise ValueError(f"missing selected endpoint: {name}")
        return path
    choices = sorted((run / "traces").glob(f"*__{topic}.arrow"))
    if topic == "reset_done" and run / "traces/reset__reset_done.arrow" in choices:
        return run / "traces/reset__reset_done.arrow"
    legacy = run / "traces" / f"{topic}.arrow"
    if legacy.exists():
        choices.append(legacy)
    if len(choices) > 1:
        raise ValueError(f"ambiguous producers for {topic}; supply --endpoints")
    return choices[0] if choices else None


def inspect_video_frames(capture, declared: int, camera_rows: int) -> dict:
    count, shape, changed = 0, None, False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current = list(frame.shape)
        if shape is not None and current != shape:
            changed = True
        shape = shape or current
        count += 1
    return {
        "status": "resolution_changed"
        if changed
        else "decoded"
        if count == declared == camera_rows and count > 0
        else "incomplete",
        "decoded_frames": count,
        "container_frames": declared,
        "camera_rows": camera_rows,
        "shape": shape,
        "mapping": "unverified_ordinal",
    }


def inspect_video(path: Path, count: int) -> dict:
    if not path.exists():
        return {"status": "missing", "mapping": "unknown"}
    try:
        import cv2
    except ImportError:
        return {"status": "decoder_unavailable", "mapping": "unverified_ordinal"}
    cap = cv2.VideoCapture(str(path))
    try:
        return inspect_video_frames(cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), count)
    finally:
        cap.release()


def pixel_index(root: Path, frames: list[dict], single_camera: bool = True) -> dict[int, dict]:
    import numpy as np

    result, shape = {}, None
    single_env = len({r["env_id"] for r in frames}) == 1
    for frame in frames:
        path = root / "source/traces/frames/overhead" / f"{frame['sim_time_ns']:020d}.npz"
        pixel = {"status": "unverified_ordinal", "path": None}
        if path.exists() and single_env and single_camera:
            try:
                with np.load(path, allow_pickle=False) as data:
                    rgb = data["rgb"]
                    if (
                        rgb.dtype != np.uint8
                        or rgb.ndim != 3
                        or rgb.shape[2] != 3
                        or min(rgb.shape) == 0
                    ):
                        raise ValueError("invalid RGB shape/dtype")
                    current = list(rgb.shape)
                    pixel = {
                        "status": "raw_timestamp_identity",
                        "path": str(path.relative_to(root)),
                        "shape": current,
                        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    }
                    if shape is not None and shape != current:
                        pixel["status"] = "resolution_changed"
                    shape = shape or current
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                pixel = {"status": "decode_failed", "path": str(path.relative_to(root))}
        elif path.exists():
            pixel["status"] = "ambiguous_environment_or_camera"
        result[frame["row"]] = pixel
    return result


def validate_contract(contract: dict | None, embodiment: str | None) -> None:
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    if contract.get("embodiment") != embodiment:
        raise ValueError("contract embodiment conflicts with source manifest")
    for key in ("state", "forwarded", "gripper"):
        channel = contract.get(key, {})
        if not isinstance(channel, dict):
            raise ValueError(f"contract channel must be a JSON object: {key}")
        names, units = channel.get("names"), channel.get("units")
        if (
            not isinstance(names, list)
            or not names
            or len(set(names)) != len(names)
            or not all(isinstance(x, str) and x for x in names)
        ):
            raise ValueError(f"contract needs ordered unique joint names: {key}")
        if (
            not isinstance(units, list)
            or len(units) != len(names)
            or not all(isinstance(x, str) and x for x in units)
            or not channel.get("frame")
        ):
            raise ValueError(f"contract needs units and frame: {key}")
        if key != "state" and channel.get("semantics") != "position_target":
            raise ValueError(f"unsupported contract action semantics: {key}")


def valid_vector(row, width: int) -> bool:
    return (
        row is not None
        and isinstance(row["data"], list)
        and len(row["data"]) == width
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in row["data"])
    )


def instruction(row: dict | None) -> dict | None:
    if row is None:
        return None
    try:
        payload = json.loads(row["text"])
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    text = payload.get("instruction")
    med = payload.get("target_med")
    if not text and isinstance(med, str) and med.strip():
        text = f"pick the {med} box from the shelf and place it in the tray"
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "text": text,
        "revision": f"{row['source']}:{row['row']}:{row['seq']}",
        "goal_id": payload.get("goal_id"),
        "evidence": row,
    }


def episode_windows(streams: dict, records: list) -> list:
    """Validate attribution before either exporting or admitting derived records.

    CON-5's initial reset is at simulation time zero. A later first row cannot
    stand in for it, even if the remaining reset count resembles an open tail.
    Goal reset/seed fields are attribution evidence only, never policy inputs.
    """
    out, ordinals = [], Counter()
    if not all(isinstance(r, dict) and type(r.get("episode")) is int for r in records):
        raise ValueError("episode records require integer episode identities")
    envs = {r["env_id"] for r in streams["reset_done"] if r["data"] == [1.0]}
    if len({(r.get("env_id", 0), r["episode"]) for r in records}) != len(records):
        raise ValueError("duplicate episode identity")
    for record in records:
        if "env_id" not in record and len(envs) > 1:
            raise ValueError("episode environment is ambiguous")
        env = record.get("env_id", next(iter(envs)) if len(envs) == 1 else 0)
        resets = [r for r in streams["reset_done"] if r["env_id"] == env and r["data"] == [1.0]]
        ordinal = ordinals[env]
        ordinals[env] += 1
        if record["episode"] != ordinal:
            raise ValueError("episode order must be contiguous from zero per environment")
        if not resets or resets[0]["sim_time_ns"] != 0:
            raise ValueError("missing initial reset; episode attribution unavailable")
        if ordinal >= len(resets):
            raise ValueError("missing episode reset; episode attribution unavailable")
        start = resets[ordinal]["sim_time_ns"]
        end = resets[ordinal + 1]["sim_time_ns"] if ordinal + 1 < len(resets) else None
        for goal_row in streams["episode_goal"]:
            if (
                goal_row["env_id"] != env
                or goal_row["sim_time_ns"] < start
                or (end is not None and goal_row["sim_time_ns"] >= end)
            ):
                continue
            try:
                goal = json.loads(goal_row["text"])
            except (ValueError, TypeError):
                continue
            if not isinstance(goal, dict):
                continue
            if "reset_sim_ns" in goal and goal["reset_sim_ns"] != start:
                raise ValueError("goal reset evidence conflicts with episode window")
            if "seed" in goal and "seed" in record and goal["seed"] != record["seed"]:
                raise ValueError("goal seed evidence conflicts with episode outcome")
        out.append((record, env, start, end))
    for env, count in ordinals.items():
        resets = [r for r in streams["reset_done"] if r["env_id"] == env and r["data"] == [1.0]]
        if len(resets) not in (count, count + 1):
            raise ValueError("reset count conflicts with episode outcomes")
    return out


def build_episodes(
    streams: dict, records: list, source_manifest: dict, contract, pixels: dict
) -> list:
    out = []
    for record, env, start, end in episode_windows(streams, records):
        family = record.get("task_family") or source_manifest.get("task_family") or "unknown"
        split = split_for(family)
        episode = {
            "episode": record["episode"],
            "env_id": env,
            "outcome": record,
            "window_ns": [start, end],
            "task_family": family,
            "split": split,
            "exclusions": [] if end is not None else ["missing_final_reset"],
            "samples": [],
        }
        windows = {
            topic: [
                r
                for r in rows
                if r["env_id"] == env
                and r["sim_time_ns"] >= start
                and (end is None or r["sim_time_ns"] < end)
            ]
            for topic, rows in streams.items()
        }
        times = {topic: [r["sim_time_ns"] for r in rows] for topic, rows in windows.items()}
        request_times = set(times["target_request"])
        grip_times = set(times["gripper_cmd_safe"])

        def prior(topic, stamp, times=times, windows=windows, start=start):
            i = bisect_left(times[topic], stamp) - 1
            # Boundary rows remain archived but their pre/post-reset order is
            # unknown. A later command does not resolve that ambiguity.
            return windows[topic][i] if i >= 0 and times[topic][i] > start else None

        for cmd in windows["joint_cmd_safe"]:
            stamp = cmd["sim_time_ns"]
            frame, state, grip = (
                prior(t, stamp) for t in ("rgb_overhead", "joint_state", "gripper_cmd_safe")
            )
            # Requests are policy-visible. Never fall back to oracle episode_goal
            # (T4 can contain the final correction before the human says it).
            request = prior("target_request", stamp)
            task = instruction(request)
            pixel = pixels.get(frame["row"]) if frame else None
            reasons = list(episode["exclusions"])
            if stamp in request_times:
                reasons.append("ambiguous_instruction_order")
                task = None
            if stamp in grip_times:
                reasons.append("ambiguous_gripper_order")
            if stamp == start:
                reasons.append("reset_boundary")
            for name, row in (("frame", frame), ("state", state), ("gripper", grip)):
                if row is None:
                    reasons.append(f"missing_{name}")
            if task is None:
                reasons.append("missing_instruction")
            if pixel is None or pixel["status"] != "raw_timestamp_identity":
                reasons.append("pixels_" + (pixel["status"] if pixel else "missing"))
            if not all(r["producer"] for r in (cmd, frame, state, grip, request) if r is not None):
                reasons.append("unknown_producer")
            if contract is None:
                reasons.append("unknown_action_contract")
            else:
                for name, row in (("forwarded", cmd), ("state", state), ("gripper", grip)):
                    if not valid_vector(row, len(contract[name]["names"])):
                        reasons.append(f"invalid_{name}_vector")
                if (
                    contract == SO101_CONTRACT
                    and grip is not None
                    and valid_vector(grip, 1)
                    and not 0 <= grip["data"][0] <= 1
                ):
                    reasons.append("invalid_gripper_range")
            episode["samples"].append(
                {
                    "forwarded": cmd,
                    "proposed": prior("joint_cmd", stamp),
                    "proposed_gripper": prior("gripper_cmd", stamp),
                    "proposal_relation": "latest_prior_not_verified_command_pair",
                    "state": state,
                    "gripper": grip,
                    "frame": frame,
                    "pixels": pixel,
                    "instruction": task,
                    "skill": {
                        "name": None,
                        "version": None,
                        "branch": None,
                        "invocation": None,
                        "parameters": None,
                        "stage": None,
                        "span_ns": None,
                        "status": "unknown_no_execution_spans",
                    },
                    "split": split,
                    "exclusions": sorted(set(reasons)),
                    "eligible": not reasons,
                }
            )
        out.append(episode)
    return out


def training_samples(episodes: list) -> list:
    return [
        sample
        for episode in episodes
        if episode["outcome"].get("status") == "success" and episode["split"] == "train"
        for sample in episode["samples"]
        if sample["eligible"]
    ]


def export_dataset(run: Path, out: Path, contract=None, endpoints=None) -> dict:
    """Snapshot inputs, build records, then atomically publish a new directory."""
    import numpy as np

    run, out = Path(run), Path(out)
    if endpoints is not None and not isinstance(endpoints, dict):
        raise ValueError("endpoints must be a JSON object")
    if endpoints and set(endpoints) - set(TOPICS):
        raise ValueError("unknown topic in endpoints")
    if out.exists():
        raise ValueError("output already exists; use a new dataset directory")
    if out.resolve().is_relative_to(run.resolve()):
        raise ValueError("dataset output must be outside the source run")
    chosen = {topic: resolve_stream(run, topic, endpoints or {}) for topic in TOPICS}
    paths = [run / "manifest.json", run / "episodes.jsonl"]
    paths += [p for p in chosen.values() if p is not None]
    # The recorder keys raw files by camera name, not producer. Preserve other
    # camera endpoints as evidence of ambiguity even after explicit selection.
    camera_paths = sorted((run / "traces").glob("*__rgb_overhead.arrow"))
    if (run / "traces/rgb_overhead.arrow").exists():
        camera_paths.append(run / "traces/rgb_overhead.arrow")
    paths += [p for p in camera_paths if p not in paths]
    paths += [
        p
        for p in (run / "graph.yaml", run / "authored-graph.yaml", run / "traces/overhead.mp4")
        if p.is_file()
    ]
    paths += sorted((run / "traces/frames/overhead").glob("*.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".demo-export-", dir=out.parent) as temp:
        root = Path(temp) / "dataset"
        root.mkdir()
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(run.resolve()):
                raise ValueError(f"source outside run: {path}")
            target = root / "source" / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha256(path) != sha256(target):
                raise ValueError(f"source changed during snapshot: {path}")
        streams = {
            topic: strict_stream(root / "source" / path.relative_to(run), root) if path else []
            for topic, path in chosen.items()
        }
        source_manifest = json.loads((root / "source/manifest.json").read_text())
        if not isinstance(source_manifest, dict):
            raise ValueError("source manifest must be a JSON object")
        records = [
            json.loads(line)
            for line in (root / "source/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        validate_contract(contract, source_manifest.get("embodiment"))
        pixels = pixel_index(root, streams["rgb_overhead"], len(camera_paths) == 1)
        episodes = build_episodes(streams, records, source_manifest, contract, pixels)
        write_json(root / "episodes.json", episodes)
        selected = training_samples(episodes)
        if selected and contract == SO101_CONTRACT:
            state = np.asarray([s["state"]["data"] for s in selected])
            action = np.asarray(
                [s["forwarded"]["data"][:5] + s["gripper"]["data"] for s in selected]
            )
            write_json(
                root / "stats.json",
                {
                    "state_mean": state.mean(axis=0).tolist(),
                    "state_std": np.maximum(state.std(axis=0), 1e-6).tolist(),
                    "action_mean": action.mean(axis=0).tolist(),
                    "action_std": np.maximum(action.std(axis=0), 1e-6).tolist(),
                    "samples": len(selected),
                    "split": "train",
                },
            )
        counts = Counter(
            reason
            for episode in episodes
            for sample in episode["samples"]
            for reason in sample["exclusions"]
        )
        manifest = {
            "format": "aisle_offline_demonstrations",
            "format_version": FORMAT_VERSION,
            "evidence_kind": "offline_data_not_benchmark_certification",
            "embodiment": source_manifest.get("embodiment"),
            "contract": contract,
            "contract_evidence": "operator_declared" if contract else "unknown",
            "source_manifest": source_manifest,
            "exporter_sha256": {
                name: sha256(Path(__file__).with_name(name))
                for name in ("demo_dataset.py", "demo_export.py", "finetune_smolvla.py")
            },
            "episode_mapping": "zero_anchored_reset_ordinal_with_goal_evidence_checks",
            "endpoints": {k: str(v.relative_to(run)) if v else None for k, v in chosen.items()},
            "join_policy": "strictly_earlier_same_episode_and_environment",
            "split_policy": "sha256_task_family_80_10_10_before_samples",
            "training_view": "success_and_eligible_and_train_split",
            "skill_attribution": "unknown_no_execution_spans",
            "completeness": "closed_Arrow_streams_and_decoded_assets_not_capture_attestation",
            "video": inspect_video(
                root / "source/traces/overhead.mp4", len(streams["rgb_overhead"])
            ),
            "coverage": {
                "episodes": len(episodes),
                "commands": sum(len(e["samples"]) for e in episodes),
                "source_commands": len(streams["joint_cmd_safe"]),
                "unassigned_commands": len(streams["joint_cmd_safe"])
                - sum(len(e["samples"]) for e in episodes),
                "eligible": sum(s["eligible"] for e in episodes for s in e["samples"]),
                "training_samples": len(selected),
                "exclusions": dict(sorted(counts.items())),
            },
            "artifacts_sha256": {
                str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
            },
        }
        write_json(root / "manifest.json", manifest)
        root.rename(out)
    return manifest


def read_dataset(root: Path) -> dict:
    """Validate the format and every bundled byte before returning records."""
    import numpy as np

    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("unsupported dataset version: missing manifest")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a JSON object with format/version")
    if (
        manifest.get("format") != "aisle_offline_demonstrations"
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported dataset format/version; re-export the source run")
    artifacts = manifest.get("artifacts_sha256", {})
    if not {"episodes.json", "source/manifest.json", "source/episodes.jsonl"} <= artifacts.keys():
        raise ValueError("missing dataset artifact hashes")
    for name, digest in artifacts.items():
        artifact = root / name
        if (
            not artifact.resolve().is_relative_to(root.resolve())
            or not artifact.is_file()
            or sha256(artifact) != digest
        ):
            raise ValueError(f"dataset artifact hash mismatch: {name}")
    episodes = json.loads((root / "episodes.json").read_text())
    sources = {}
    pixel_digests = {}

    def source_rows(topic):
        endpoint = manifest["endpoints"].get(topic)
        if endpoint is None:
            return []
        endpoint_path = Path(endpoint)
        if (
            endpoint_path.parent != Path("traces")
            or endpoint_path.suffix != ".arrow"
            or endpoint_path.stem.rsplit("__", 1)[-1] != topic
        ):
            raise ValueError(f"invalid channel endpoint: {topic}")
        name = f"source/{endpoint}"
        if name not in artifacts:
            raise ValueError("channel endpoint missing from artifact hashes")
        if name not in sources:
            sources[name] = strict_stream(root / name, root)
            if any(row["topic"] != topic for row in sources[name]):
                raise ValueError(f"source topic does not match channel endpoint: {topic}")
        return sources[name]

    records = [
        json.loads(line)
        for line in (root / "source/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    layout = episode_windows(
        {topic: source_rows(topic) for topic in ("reset_done", "episode_goal")}, records
    )
    if len(episodes) != len(layout):
        raise ValueError("episode count differs from source outcomes")

    def verify_row(row, topic):
        if row is None:
            return
        name = row["source"]
        endpoint = manifest["endpoints"].get(topic)
        if endpoint is None or name != f"source/{endpoint}" or row["topic"] != topic:
            raise ValueError(f"row does not match channel endpoint: {topic}")
        rows = source_rows(topic)
        index = row["row"]
        if not isinstance(index, int) or not 0 <= index < len(rows) or rows[index] != row:
            raise ValueError("source row identity/value mismatch")

    for episode, (record, env, start, end) in zip(episodes, layout, strict=True):
        if (
            episode["outcome"] != record
            or episode["episode"] != record["episode"]
            or episode["env_id"] != env
            or episode["window_ns"] != [start, end]
        ):
            raise ValueError("episode identity/window/outcome differs from source evidence")
        for sample in episode["samples"]:
            for channel, topic in CHANNEL_TOPICS.items():
                verify_row(sample[channel], topic)
            if sample["instruction"] is not None:
                verify_row(sample["instruction"]["evidence"], "target_request")
                if instruction(sample["instruction"]["evidence"]) != sample["instruction"]:
                    raise ValueError("instruction revision mismatch")
            pixel = sample.get("pixels")
            if pixel and pixel["path"] and pixel["path"] not in artifacts:
                raise ValueError("pixel path missing from artifact hashes")
            if pixel and pixel["status"] == "raw_timestamp_identity":
                expected = (
                    f"source/traces/frames/overhead/{sample['frame']['sim_time_ns']:020d}.npz"
                )
                if pixel["path"] != expected:
                    raise ValueError("pixel identity timestamp mismatch")
                if expected not in pixel_digests:
                    with np.load(root / expected, allow_pickle=False) as data:
                        rgb = data["rgb"]
                        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                            raise ValueError("pixel identity shape/dtype mismatch")
                        pixel_digests[expected] = (
                            list(rgb.shape),
                            hashlib.sha256(rgb.tobytes()).hexdigest(),
                        )
                if pixel_digests[expected] != (pixel["shape"], pixel["rgb_sha256"]):
                    raise ValueError("pixel identity content mismatch")
            if sample["eligible"]:
                start, end = episode["window_ns"]
                command = sample["forwarded"]
                if (
                    end is None
                    or not start < command["sim_time_ns"] < end
                    or command["env_id"] != episode["env_id"]
                    or sample["exclusions"]
                ):
                    raise ValueError("invalid eligible episode boundary")
                for row in (
                    sample["state"],
                    sample["gripper"],
                    sample["frame"],
                    sample["instruction"]["evidence"],
                ):
                    if (
                        not start < row["sim_time_ns"] < command["sim_time_ns"]
                        or row["env_id"] != episode["env_id"]
                    ):
                        raise ValueError("noncausal or cross-environment sample")
                if pixel["status"] != "raw_timestamp_identity":
                    raise ValueError("unverified eligible pixels")
            if sample["split"] != episode["split"] or episode["split"] != split_for(
                episode["task_family"]
            ):
                raise ValueError("inconsistent dataset split")
    return {"manifest": manifest, "episodes": episodes}
