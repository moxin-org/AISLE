"""P3 policy-only evidence export and deterministic history view (HAR-4/6)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from aisle.harness.execution_capture import encoded, verify_capture

# Endpoint identities are deliberately bounded to the first T4/L1 candidate.
# Scorer endpoints and rollout-client payloads are never projection inputs.
POLICY_ENDPOINTS = {
    "dora-genesis": {
        "bridge_info",
        "joint_state",
        "gripper_state",
        "rgb_overhead",
        "rgb_wrist",
        "depth_overhead",
        "seg_overhead",
    },
    "reset": {"reset_done"},
    "human-sim": {"human_msg"},
    "task-state-machine": {"robot_msg", "target_request", "return_request", "episode_feedback"},
    "segmented-pose": {"target_pose"},
    "grasp-planner-topdown": {"grasp_pose"},
    "return-planner": {"grasp_pose"},
    "ik-trajectory": {"joint_cmd", "gripper_cmd", "plan_done"},
    "budget-guard": {"joint_cmd_safe", "gripper_cmd_safe", "violation"},
}
ALLOWED = {f"{node}__{topic}" for node, topics in POLICY_ENDPOINTS.items() for topic in topics}
GUARD = {"budget-guard/joint_cmd_safe", "budget-guard/gripper_cmd_safe"}
FORMAT = "aisle_p3_policy_evidence"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(encoded(value))


def export_policy(run, out, *, launch=0):
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if type(launch) is not int or launch < 0:
        raise ValueError("invalid launch index")
    source_manifest = json.loads((run / "manifest.json").read_text())
    if source_manifest.get("perception") != "L1":
        raise ValueError("this projection is qualified only for the L1 candidate")
    graph_path = run / ("graph.yaml" if launch == 0 else f"graph-r{launch}.yaml")
    hashes = source_manifest.get("exec_graph_hashes", [])
    if launch >= len(hashes) or not graph_path.is_file() or digest(graph_path) != hashes[launch]:
        raise ValueError("instrumented graph hash mismatch")
    doc = yaml.safe_load(graph_path.read_text())
    nodes = [n for n in doc["nodes"] if "AISLE_EXECUTION_CAPTURE_CONFIG" in n.get("env", {})]
    if len(nodes) != 1 or nodes[0]["id"] != "ik-trajectory":
        raise ValueError("projection requires exactly one bound IK executor")
    config = json.loads(nodes[0]["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"])
    capture_path = Path(config["path"])
    # Archived captures can move; the original graph's absolute path must not
    # select an unrelated live run. Use only the recorded launch directory name.
    trace_relative = Path("traces") if launch == 0 else Path("traces") / f"relaunch-{launch}"
    expected_suffix = (*trace_relative.parts, "executions", "ik-trajectory.jsonl")
    if capture_path.parts[-len(expected_suffix) :] != expected_suffix:
        raise ValueError("unexpected capture trace directory")
    traces = run / trace_relative
    capture_path = traces / "executions/ik-trajectory.jsonl"
    verify_capture(capture_path, traces, config, guard_endpoints=GUARD)
    check_journal(capture_path)

    def snapshot(source, target):
        if source.is_symlink() or not source.resolve().is_relative_to(traces.resolve()):
            raise ValueError("policy artifact source escapes trace directory")
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise ValueError("policy artifact source changed during export")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        public_traces = temporary / "traces"
        (public_traces / "executions").mkdir(parents=True)
        for endpoint in sorted(ALLOWED):
            source = traces / f"{endpoint}.arrow"
            if source.is_file():
                snapshot(source, public_traces / source.name)
        identities = []
        for line in (traces / "identities.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["endpoint"] in ALLOWED:
                check_identity(row)
                identities.append(row)
        (public_traces / "identities.jsonl").write_bytes(b"".join(encoded(r) for r in identities))
        snapshot(capture_path, public_traces / "executions/ik-trajectory.jsonl")
        for camera in ("overhead", "wrist"):
            for source in sorted((traces / "frames" / camera).glob("*.npz")):
                if not re.fullmatch(r"[0-9]{20}\.npz", source.name):
                    raise ValueError("unexpected frame artifact")
                check_camera(source)
                target = public_traces / "frames" / camera / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                snapshot(source, target)
        if (traces / "overhead.mp4").is_file():
            snapshot(traces / "overhead.mp4", public_traces / "overhead.mp4")
        artifacts = {
            str(p.relative_to(temporary)): digest(p)
            for p in sorted(temporary.rglob("*"))
            if p.is_file()
        }
        manifest = dict(
            format=FORMAT,
            version=1,
            perception="L1",
            sim_engine=source_manifest.get("sim_engine"),
            source_git_sha=source_manifest.get("git_sha"),
            launch=launch,
            source_graph_sha256=hashes[launch],
            capture_binding={k: config[k] for k in ("capture_id", "node_id", "inputs", "skill")},
            artifacts=artifacts,
            delivery_completion="unknown",
        )
        write_json(temporary / "manifest.json", manifest)
        read_policy(temporary)
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return read_policy(out)


def read_policy(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest) != {
        "format",
        "version",
        "perception",
        "sim_engine",
        "source_git_sha",
        "launch",
        "source_graph_sha256",
        "capture_binding",
        "artifacts",
        "delivery_completion",
    }:
        raise ValueError("unexpected policy manifest fields")
    if manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise ValueError("unsupported policy evidence bundle")
    artifacts = manifest["artifacts"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"manifest.json"}
    if actual != set(artifacts):
        raise ValueError("policy artifact set mismatch")
    for relative, expected in artifacts.items():
        path = root / relative
        if path.is_symlink() or path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("unsafe policy artifact")
        if digest(path) != expected:
            raise ValueError("policy artifact hash mismatch")
        parts = Path(relative).parts
        allowed = (
            relative
            in {
                "traces/identities.jsonl",
                "traces/executions/ik-trajectory.jsonl",
                "traces/overhead.mp4",
            }
            or len(parts) == 2
            and parts[0] == "traces"
            and Path(relative).suffix == ".arrow"
            and Path(relative).stem in ALLOWED
            or len(parts) == 4
            and parts[:2] == ("traces", "frames")
            and parts[2] in {"overhead", "wrist"}
            and re.fullmatch(r"[0-9]{20}\.npz", parts[3])
        )
        if not allowed:
            raise ValueError("non-policy artifact in bundle")
        if path.suffix == ".npz":
            check_camera(path)
    traces = root / "traces"
    for line in (traces / "identities.jsonl").read_text().splitlines():
        check_identity(json.loads(line))
    check_journal(traces / "executions/ik-trajectory.jsonl")
    journal = verify_capture(
        traces / "executions/ik-trajectory.jsonl",
        traces,
        manifest["capture_binding"],
        guard_endpoints=GUARD,
    )
    return dict(manifest=manifest, journal=journal)


def check_identity(record):
    if (
        set(record) != {"format", "version", "endpoint", "row", "trace_row_sha256", "metadata"}
        or record["endpoint"] not in ALLOWED
        or not set(record["metadata"])
        <= {
            "goal_id",
            "request_id",
            "invocation_id",
            "skill_name",
            "skill_version",
            "command_origin_endpoint",
            "command_origin_seq",
        }
    ):
        raise ValueError("non-policy identity artifact")


def check_journal(path):
    """Refuse undeclared annotations rather than projecting arbitrary JSON into a policy view."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(records) < 2 or not all(isinstance(record, dict) for record in records):
        raise ValueError("invalid policy journal records")
    for record in records:
        if "skill" in record and not set(record["skill"]) <= {"name", "version", "sources"}:
            raise ValueError("unexpected policy skill annotations")
    if set(records[0]) != {"format", "version", "capture_id", "node_id", "inputs", "skill"}:
        raise ValueError("unexpected policy journal header")
    sources = set(records[0]["inputs"].values())
    if not sources <= {s.replace("__", "/") for s in ALLOWED} | {"turn-barrier/turn"}:
        raise ValueError("non-policy journal input binding")
    if set(records[-1]) != {"kind", "events", "open_invocation", "open_proposals", "sha256"}:
        raise ValueError("unexpected policy journal footer")
    common = {
        "kind",
        "event_seq",
        "sim_time_ns",
        "env_id",
        "episode",
        "invocation_id",
        "goal_revision",
    }
    extras = {
        "reset": {"source"},
        "proposed": {"source", "skill", "parameters", "delivery_completion"},
        "accepted": set(),
        "rejected": {"reason"},
        "started": {"source", "observation", "delivery_completion"},
        "command_emitted": {"source", "observation", "stage"},
        "returned": {"observation", "delivery_completion"},
        "interrupted": {"reason", "source", "delivery_completion"},
        "cancelled": {"reason", "source", "delivery_completion"},
    }
    for record in records[1:-1]:
        if set(record) != common | extras.get(record.get("kind"), set()):
            raise ValueError("unexpected policy journal fields")
        for field in ("source", "observation"):
            if record.get(field) is not None and set(record[field]) != {
                "endpoint",
                "sim_time_ns",
                "env_id",
                "seq",
                "trace_row_sha256",
            }:
                raise ValueError("unexpected policy journal reference")
        if record.get("stage") is not None and set(record["stage"]) != {"name", "read_attempt"}:
            raise ValueError("unexpected policy journal stage")
        if "parameters" in record and not set(record["parameters"]) <= {
            "operation",
            "request",
            "pose",
            "place_xy",
            "place_tcp_z",
            "approach_m",
        }:
            raise ValueError("unexpected policy journal parameters")


def check_camera(path):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 256 * 1024 * 1024:
            raise ValueError("oversized camera artifact")
    with np.load(path, allow_pickle=False) as arrays:
        if not arrays.files or not set(arrays.files) <= {"rgb", "depth", "seg"}:
            raise ValueError("unexpected camera arrays")
        for key in arrays.files:
            array = arrays[key]
            if (
                key == "rgb"
                and (array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3)
                or key == "depth"
                and (array.dtype != np.float32 or array.ndim != 2)
                or key == "seg"
                and (array.dtype != np.int32 or array.ndim != 2)
            ):
                raise ValueError("unexpected camera array shape/type")


def summarize_history(bundle):
    """An optional organized view derived exclusively from both arms' common raw bundle."""
    journal = bundle["journal"]
    invocations = {}
    for event in journal["events"]:
        identifier = event["invocation_id"]
        kind = event["kind"]
        if kind == "proposed":
            invocations[identifier] = dict(
                invocation_id=identifier,
                episode=event["episode"],
                goal_revision=event["goal_revision"],
                skill=event["skill"],
                parameters=event["parameters"],
                state="proposed",
                first_event_seq=event["event_seq"],
                last_event_seq=event["event_seq"],
                commands_emitted=0,
                commands_forwarded=0,
                stages=[],
                delivery_completion="unknown",
                source=event["source"],
            )
        elif identifier is not None:
            item = invocations[identifier]
            item["last_event_seq"] = event["event_seq"]
            if kind == "command_emitted":
                item["commands_emitted"] += 1
                stage = event.get("stage")
                if stage is not None and (not item["stages"] or item["stages"][-1] != stage):
                    item["stages"].append(stage)
            else:
                item["state"] = kind
    forwarded = {
        (e["invocation_id"], e["origin"]["endpoint"], e["origin"]["seq"])
        for e in journal["forwarded"]
    }
    for identifier, _, _ in forwarded:
        invocations[identifier]["commands_forwarded"] += 1
    return list(invocations.values())


def query_raw(root, endpoint, *, offset=0, limit=100):
    import pyarrow as pa

    name = endpoint.replace("/", "__")
    if name not in ALLOWED:
        raise ValueError("non-policy raw endpoint")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("invalid raw query bounds")
    read_policy(root)
    traces = Path(root) / "traces"
    path = traces / f"{name}.arrow"
    if not path.is_file():
        raise ValueError("endpoint was not captured")
    identities = {
        (r["endpoint"], r["row"]): r["metadata"]
        for r in map(json.loads, (traces / "identities.jsonl").read_text().splitlines())
    }
    with pa.memory_map(str(path), "r") as stream:
        table = pa.ipc.open_stream(stream).read_all().slice(offset, limit)
    return [
        dict(
            endpoint=endpoint,
            row=offset + i,
            **row,
            metadata=identities.get((name, offset + i), {}),
        )
        for i, row in enumerate(table.to_pylist())
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--launch", type=int, default=0)
    history = sub.add_parser("history")
    history.add_argument("--bundle", required=True, type=Path)
    raw = sub.add_parser("raw")
    raw.add_argument("--bundle", required=True, type=Path)
    raw.add_argument("--endpoint", required=True)
    raw.add_argument("--offset", type=int, default=0)
    raw.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_policy(args.run, args.out, launch=args.launch)
            payload = dict(
                ok=True, bundle=str(args.out), artifacts=len(result["manifest"]["artifacts"])
            )
        elif args.command == "raw":
            payload = dict(
                ok=True,
                rows=query_raw(args.bundle, args.endpoint, offset=args.offset, limit=args.limit),
            )
        else:
            payload = dict(ok=True, invocations=summarize_history(read_policy(args.bundle)))
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, zipfile.BadZipFile) as exc:
        print(json.dumps(dict(ok=False, error=str(exc))))
        return 1
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
