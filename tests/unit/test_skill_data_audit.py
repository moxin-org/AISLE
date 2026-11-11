"""P0 synthetic evidence only; no recorded robot outcomes are manufactured."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import audit_skill_data as audit  # noqa: E402
import demo_export as exporter  # noqa: E402
import finetune_smolvla as trainer  # noqa: E402

pytestmark = pytest.mark.unit


def write_topic(root, topic, stamps, data):
    from aisle.harness.trace_recorder import TRACE_SCHEMA

    table = pa.Table.from_pydict(
        {
            "sim_time_ns": stamps,
            "env_id": [0] * len(stamps),
            "seq": list(range(len(stamps))),
            "data": data,
            "text": [None] * len(stamps),
        },
        schema=TRACE_SCHEMA,
    )
    with pa.OSFile(str(root / "traces" / f"{topic}.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


@pytest.fixture
def synthetic_run(tmp_path):
    """CON-12: tiny Arrow fixture, expressly not a physical execution."""
    (tmp_path / "traces").mkdir()
    rows = [
        {"episode": 0, "status": "success", "seed": 0},
        {"episode": 1, "status": "fail", "seed": 1, "failure": "synthetic"},
    ]
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "manifest.json").write_text(json.dumps({"env_baseline": "local"}))
    (tmp_path / "graph.yaml").write_text("nodes: []\n")
    write_topic(tmp_path, "reset__reset_done", [0, 100, 200], [[1]] * 3)
    write_topic(tmp_path, "camera__rgb_overhead", [20, 90, 120], [None] * 3)
    write_topic(tmp_path, "guard__joint_cmd_safe", [10, 30, 110, 130], [[n] * 7 for n in range(4)])
    write_topic(tmp_path, "guard__gripper_cmd_safe", [0, 100], [[0], [1]])
    write_topic(tmp_path, "bridge__joint_state", [25, 100], [[9] * 7, [8] * 7])
    return tmp_path


def test_audit_detects_alignment_and_provenance_gaps(synthetic_run):
    """HAR-4/6, CON-5: inspect both synthetic outcomes without certifying them."""
    result = audit.inspect_run(synthetic_run, [0, 1], decode_video=False)
    assert result == audit.inspect_run(synthetic_run, [0, 1], decode_video=False)
    first, second = result["episodes"]
    assert first["commands_without_episode_frame"] == 1
    assert first["commands_without_prior_state"] == 1
    assert second["commands_without_episode_frame"] == 1
    assert second["legacy_cross_episode_frame_pairs"] == 1
    assert first["legacy_command_index_mismatches"] == 1
    assert result["video"]["status"] == "not_requested"
    assert not result["requested_skill_referenced"]


def test_characterizes_export_loader_index_loss(synthetic_run, tmp_path):
    """P0/HAR-6: reproduce current index loss; this is not desired P1 behavior."""
    record = exporter.episode_tuples(synthetic_run, 0)
    assert record["action_ts"] == [30]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "ep_0000.json").write_text(json.dumps({"episode": 0, **record}))
    loaded = trainer.load_tuples(dataset, synthetic_run, stride=1, limit=None)
    # At t=30 the command is 1, but dropping t=10 loses its source row index.
    assert loaded[0]["action"][:5] == [0] * 5


def test_characterizes_previous_episode_frame(synthetic_run):
    """P0/HAR-6: the all-run frame query can retain an earlier episode's image."""
    assert exporter.episode_tuples(synthetic_run, 1)["frame_indices"] == [1, 2]


def test_characterizes_future_state_fallback(synthetic_run, tmp_path):
    """P0/HAR-6: a missing preceding state currently selects the first future state."""
    write_topic(synthetic_run, "camera__rgb_overhead", [0, 90, 120], [None] * 3)
    record = exporter.episode_tuples(synthetic_run, 0)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "ep_0000.json").write_text(json.dumps({"episode": 0, **record}))
    loaded = trainer.load_tuples(dataset, synthetic_run, stride=1, limit=None)
    # First command is at 10; the selected state was captured at 25.
    assert loaded[0]["state6"] == [9] * 6


def test_nonmonotonic_camera_refused(synthetic_run):
    """CON-8/HAR-6: do not silently apply binary search to unordered observations."""
    write_topic(synthetic_run, "camera__rgb_overhead", [90, 20], [None] * 2)
    with pytest.raises(ValueError, match="nonmonotonic camera"):
        audit.inspect_run(synthetic_run, [0], decode_video=False)


def test_cli_success_does_not_claim_certification(synthetic_run, capsys):
    """CON-8: successful audit may contain findings; its evidence kind says so."""
    assert audit.main(["--run", str(synthetic_run), "--skip-video"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["evidence_kind"] == "read_only_audit_not_certification"


def test_inventory_reports_missing_root_and_skill_reference(synthetic_run):
    """HAR-4: search records absence without inventing an execution."""
    (synthetic_run / "graph.yaml").write_text("nodes: [{id: ik-transfer-v2}]\n")
    result = audit.inventory([synthetic_run, synthetic_run / "absent"], "ik-transfer-v2")
    assert result[0]["episode_files"] == 1
    assert result[0]["skill_runs"][0]["run"] == str(synthetic_run)
    assert result[1]["exists"] is False


def test_cli_error_is_json(tmp_path, capsys):
    """CON-8: missing recordings produce a JSON error and nonzero exit."""
    assert audit.main(["--run", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
