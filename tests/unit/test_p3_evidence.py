"""Policy-only evidence parity for the bounded P3 recovery pilot."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from p3_evidence import export_policy, read_policy, summarize_history

pytestmark = pytest.mark.unit


def run_fixture(tmp_path):
    from aisle.harness.execution_capture import ExecutionCapture, RecordedTransport, reference
    from aisle.harness.trace_recorder import TRACE_SCHEMA, trace_identity

    run = tmp_path / "run"
    traces = run / "traces"
    (traces / "executions").mkdir(parents=True)
    bindings = {
        "reset_done": "reset/reset_done",
        "grasp_pose": "grasp-planner-topdown/grasp_pose",
        "joint_state": "dora-genesis/joint_state",
    }
    config = dict(
        path=str(traces / "executions/ik-trajectory.jsonl"),
        capture_id="fixture",
        node_id="ik-trajectory",
        inputs=bindings,
        skill={
            "name": "ik-trajectory",
            "version": "fixture",
            "sources": {"src/aisle/nodes/ik_trajectory.py": "c" * 64},
        },
    )
    capture = ExecutionCapture(**config)
    streams = {}

    def record(endpoint, value, metadata):
        streams.setdefault(endpoint, []).append((value, metadata))

    def event(port, stamp, values, **extra):
        value = pa.array(values)
        meta = dict(sim_time_ns=stamp, seq=stamp, env_id=0, **extra)
        record(bindings[port], value, meta)
        return dict(id=port, value=value, metadata=meta)

    capture.reset(event("reset_done", 0, [1]))
    proposal = capture.propose(event("grasp_pose", 1, [0.0] * 7, goal_id="public-goal"), {})
    capture.accept(proposal)
    capture.observe(event("joint_state", 2, [0.0] * 9))

    class Raw:
        def send_output(self, topic, value, metadata):
            record(f"ik-trajectory/{topic}", value, metadata)
            record(f"budget-guard/{topic}_safe", value, {**metadata, "seq": 99})

    RecordedTransport(Raw(), capture).send_output(
        "joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, seq=1, env_id=0)
    )
    capture.finish(dict(sim_time_ns=2, seq=2, env_id=0))
    capture.close()
    record(
        "verifier-oracle/episode_result",
        pa.array(["SECRET_VERDICT"]),
        dict(sim_time_ns=3, seq=0, env_id=0),
    )
    record(
        "rollout-client/episode_goal",
        pa.array(["SECRET_GOAL"]),
        dict(sim_time_ns=0, seq=0, env_id=0),
    )
    identities = []
    for endpoint, values in streams.items():
        rows = []
        for index, (value, metadata) in enumerate(values):
            ref = reference(endpoint, value, metadata)
            row = {k: ref[k] for k in ("sim_time_ns", "env_id", "seq")}
            row.update(
                data=None
                if pa.types.is_string(value.type)
                else [float(x) for x in value.to_pylist()],
                text=value[0].as_py() if pa.types.is_string(value.type) else None,
            )
            rows.append(row)
            identities.append(trace_identity(endpoint.replace("/", "__"), index, row, metadata))
        with pa.OSFile(str(traces / (endpoint.replace("/", "__") + ".arrow")), "wb") as sink:
            with pa.ipc.new_stream(sink, TRACE_SCHEMA) as writer:
                writer.write_table(pa.Table.from_pylist(rows, schema=TRACE_SCHEMA))
    (traces / "identities.jsonl").write_text("".join(json.dumps(r) + "\n" for r in identities))
    graph = {
        "nodes": [
            {
                "id": "ik-trajectory",
                "inputs": bindings,
                "env": {"AISLE_EXECUTION_CAPTURE_CONFIG": json.dumps(config)},
            }
        ]
    }
    graph_bytes = yaml.safe_dump(graph).encode()
    (run / "graph.yaml").write_bytes(graph_bytes)
    (run / "manifest.json").write_text(
        json.dumps(
            dict(
                sim_engine="genesis",
                perception="L1",
                git_sha="a" * 40,
                exec_graph_hashes=[hashlib.sha256(graph_bytes).hexdigest()],
            )
        )
    )
    (run / "episodes.jsonl").write_text("SECRET_SCORING_RECORD\n")
    return run


def test_both_arms_share_identical_policy_raw_sources_and_summary_is_derived(tmp_path):
    """HAR-4/6; TC-8: structured history adds organization, never scorer evidence."""
    run = run_fixture(tmp_path)
    out = tmp_path / "public"
    export_policy(run, out)
    result = read_policy(out)
    assert result["manifest"]["perception"] == "L1"
    history = summarize_history(result)
    assert len(history) == 1
    assert history[0]["state"] == "returned"
    assert history[0]["delivery_completion"] == "unknown"
    assert history[0]["commands_emitted"] == history[0]["commands_forwarded"] == 1
    for path in out.rglob("*"):
        if path.is_file():
            assert b"SECRET_" not in path.read_bytes()
    assert not (out / "graph.yaml").exists()
    assert not (out / "episodes.jsonl").exists()


def test_export_rejects_graph_drift_and_overwrite(tmp_path):
    """CON-5/8; HAR-4: evidence must bind the actually launched graph and preserve prior export."""
    run = run_fixture(tmp_path)
    out = tmp_path / "public"
    export_policy(run, out)
    with pytest.raises(FileExistsError):
        export_policy(run, out)
    (run / "graph.yaml").write_text("nodes: []\n")
    with pytest.raises(ValueError, match="graph"):
        export_policy(run, tmp_path / "drift")


def test_public_bundle_refuses_tampered_source_and_unlisted_files(tmp_path):
    """HAR-4: public evidence revalidates hashes and rejects hidden extra artifacts."""
    run = run_fixture(tmp_path)
    out = tmp_path / "public"
    export_policy(run, out)
    (out / "secret.txt").write_text("extra")
    with pytest.raises(ValueError, match="artifact"):
        read_policy(out)
    (out / "secret.txt").unlink()
    path = out / "traces/ik-trajectory__joint_cmd.arrow"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_policy(out)


def test_relaunch_export_resolves_archived_trace_directory(tmp_path):
    """HAR-4: moved runs retain launch-specific evidence without reading old absolute paths."""
    run = run_fixture(tmp_path)
    moved = tmp_path / "moved-traces"
    (run / "traces").rename(moved)
    (run / "traces").mkdir()
    moved.rename(run / "traces/relaunch-1")
    graph = yaml.safe_load((run / "graph.yaml").read_text())
    config = json.loads(graph["nodes"][0]["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"])
    config["path"] = "/old/host/run/traces/relaunch-1/executions/ik-trajectory.jsonl"
    graph["nodes"][0]["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"] = json.dumps(config)
    raw = yaml.safe_dump(graph).encode()
    (run / "graph-r1.yaml").write_bytes(raw)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["exec_graph_hashes"].append(hashlib.sha256(raw).hexdigest())
    (run / "manifest.json").write_text(json.dumps(manifest))
    result = export_policy(run, tmp_path / "public", launch=1)
    assert result["manifest"]["launch"] == 1


def test_camera_archive_cannot_smuggle_unrecognized_arrays(tmp_path):
    """TC-9; HAR-4: policy camera artifacts contain only declared sensor arrays."""
    import numpy as np

    run = run_fixture(tmp_path)
    frames = run / "traces/frames/overhead"
    frames.mkdir(parents=True)
    np.savez(
        frames / "00000000000000000000.npz",
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        hidden_oracle=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="camera"):
        export_policy(run, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_unknown_journal_fields_are_refused_at_policy_boundary(tmp_path):
    """TC-8: extra privileged annotations cannot enter through a valid journal hash."""
    from aisle.harness.execution_capture import encoded

    run = run_fixture(tmp_path)
    path = run / "traces/executions/ik-trajectory.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["oracle_result"] = "SECRET_RESULT"
    raw = b"".join(encoded(r) for r in records[:-1])
    records[-1]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw + encoded(records[-1]))
    with pytest.raises(ValueError, match="journal"):
        export_policy(run, tmp_path / "public")


def test_raw_query_returns_original_rows_with_source_ids_and_bounds(tmp_path):
    """HAR-6; CON-8: raw access keeps original values and refuses scorer endpoints."""
    from p3_evidence import query_raw

    run = run_fixture(tmp_path)
    out = tmp_path / "public"
    export_policy(run, out)
    rows = query_raw(out, "ik-trajectory/joint_cmd", offset=0, limit=1)
    assert rows[0]["row"] == 0
    assert rows[0]["data"] == [0.0] * 9
    assert rows[0]["metadata"]["goal_id"] == "public-goal"
    with pytest.raises(ValueError, match="endpoint"):
        query_raw(out, "verifier-oracle/episode_result")
    with pytest.raises(ValueError, match="bounds"):
        query_raw(out, "ik-trajectory/joint_cmd", limit=1001)


def test_cli_reports_policy_refusal_as_one_json_object(tmp_path, capsys):
    """CON-8; HAR-6: failed raw queries are structured refusals, never scorer fallbacks."""
    from p3_evidence import main

    assert (
        main(["raw", "--bundle", str(tmp_path), "--endpoint", "verifier-oracle/episode_result"])
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and "endpoint" in payload["error"]


def test_public_manifest_cannot_add_privileged_annotations(tmp_path):
    """TC-8: manifest metadata is also part of the policy boundary."""
    run = run_fixture(tmp_path)
    out = tmp_path / "public"
    export_policy(run, out)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["oracle_score"] = "SECRET_SCORE"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        read_policy(out)


def test_nested_skill_annotations_are_not_policy_evidence(tmp_path):
    """TC-8: a bound capture still cannot add undeclared skill annotations to the public view."""
    from aisle.harness.execution_capture import encoded

    run = run_fixture(tmp_path)
    path = run / "traces/executions/ik-trajectory.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if "skill" in row:
            row["skill"]["hidden_outcome"] = "SECRET_SKILL_ANNOTATION"
    raw = b"".join(encoded(r) for r in records[:-1])
    records[-1]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw + encoded(records[-1]))
    graph = yaml.safe_load((run / "graph.yaml").read_text())
    config = json.loads(graph["nodes"][0]["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"])
    config["skill"] = records[0]["skill"]
    graph["nodes"][0]["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"] = json.dumps(config)
    raw_graph = yaml.safe_dump(graph).encode()
    (run / "graph.yaml").write_bytes(raw_graph)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["exec_graph_hashes"][0] = hashlib.sha256(raw_graph).hexdigest()
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="skill"):
        export_policy(run, tmp_path / "public")


def test_export_refuses_symlinked_policy_artifacts(tmp_path):
    """HAR-4; TC-8: a policy filename cannot redirect export to private storage."""
    run = run_fixture(tmp_path)
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"SECRET_PRIVATE_BYTES")
    (run / "traces/overhead.mp4").symlink_to(outside)
    with pytest.raises(ValueError, match="source"):
        export_policy(run, tmp_path / "public")
