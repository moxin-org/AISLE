"""P1 portable fixtures; these are synthetic data, never physical outcomes."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import demo_dataset as ds  # noqa: E402
import demo_export as exporter  # noqa: E402
import finetune_smolvla as trainer  # noqa: E402

pytestmark = pytest.mark.unit


def topic(run, name, times, values=None, texts=None, envs=None):
    from aisle.harness.trace_recorder import TRACE_SCHEMA

    n = len(times)
    table = pa.Table.from_pydict(
        dict(
            sim_time_ns=times,
            env_id=envs or [0] * n,
            seq=list(range(n)),
            data=values or [None] * n,
            text=texts or [None] * n,
        ),
        schema=TRACE_SCHEMA,
    )
    path = run / "traces" / f"{name}.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return path


@pytest.fixture
def recorded(tmp_path):
    """CON-12: two tiny synthetic episodes, including a failed outcome."""
    run = tmp_path / "run"
    (run / "traces/frames/overhead").mkdir(parents=True)
    records = [
        dict(episode=i, status=s, task_family="fixture-pick")
        for i, s in enumerate(["success", "fail"])
    ]
    (run / "episodes.jsonl").write_text("\n".join(map(json.dumps, records)))
    (run / "manifest.json").write_text(
        json.dumps(dict(embodiment="so101", tier="T1", sim_engine="genesis"))
    )
    (run / "authored-graph.yaml").write_text("nodes: []\n")
    topic(run, "reset__reset_done", [0, 100, 200], [[1]] * 3)
    topic(run, "camera__rgb_overhead", [20, 90, 120])
    topic(run, "guard__joint_cmd_safe", [10, 30, 100, 130], [[float(i)] * 6 for i in range(4)])
    topic(run, "ik__joint_cmd", [9, 29, 99, 129], [[float(i) + 0.5] * 6 for i in range(4)])
    topic(run, "guard__gripper_cmd_safe", [2, 102], [[0], [1]])
    topic(run, "bridge__joint_state", [25, 125], [[9.0] * 6, [8.0] * 6])
    topic(
        run,
        "task__target_request",
        [1, 101],
        texts=[json.dumps({"target_med": x}) for x in ["amoxicillin", "ibuprofen"]],
    )
    for stamp in [20, 90, 120]:
        np.savez(
            run / f"traces/frames/overhead/{stamp:020d}.npz",
            rgb=np.full((2, 3, 3), stamp, dtype=np.uint8),
        )
    return run


def export(run, tmp_path, contract=None, name="dataset"):
    out = tmp_path / name
    ds.export_dataset(run, out, contract=contract or ds.SO101_CONTRACT)
    return out, ds.read_dataset(out)


def test_round_trip_preserves_original_rows_and_distinct_values(recorded, tmp_path):
    """CON-5, HAR-4/6, TC-5: compacting never renumbers source commands."""
    out, data = export(recorded, tmp_path)
    first = data["episodes"][0]
    assert "missing_frame" in first["samples"][0]["exclusions"]
    sample = first["samples"][1]
    assert sample["forwarded"]["row"] == sample["forwarded"]["seq"] == 1
    assert sample["forwarded"]["producer"] == "guard"
    assert sample["forwarded"]["sim_time_ns"] == 30
    assert sample["forwarded"]["data"] == [1.0] * 6
    assert sample["proposed"]["data"] == [1.5] * 6
    assert sample["state"]["data"] == [9.0] * 6
    assert sample["skill"]["version"] is None
    assert (
        sample["instruction"]["text"]
        == "pick the amoxicillin box from the shelf and place it in the tray"
    )
    assert sample["eligible"]
    assert data["manifest"]["training_view"] == "success_and_eligible_and_train_split"
    # Loading is self-contained: run_dir is compatibility-only, never reread.
    loaded = trainer.load_tuples(out, tmp_path / "absent", 1, None)
    assert loaded[0]["action"] == [1.0] * 5 + [0.0]
    assert loaded[0]["state6"] == [9.0] * 6
    assert loaded[0]["task"] == sample["instruction"]["text"]
    assert trainer.build_frame_cache(out, {loaded[0]["frame_path"]})[
        loaded[0]["frame_path"]
    ].shape == (2, 3, 3)
    assert data["episodes"][1]["outcome"]["status"] == "fail"
    assert len(loaded) == 1
    _, again = export(recorded, tmp_path, name="again")
    assert data == again
    _, rebuilt = export(out / "source", tmp_path, name="rebuilt")
    assert data == rebuilt
    stats = json.loads((out / "stats.json").read_text())
    assert stats["samples"] == 1
    assert stats["action_mean"] == loaded[0]["action"]


def test_no_future_or_cross_episode_or_cross_environment_join(recorded, tmp_path):
    """HAR-6: missing causal state and reset-boundary samples stay excluded."""
    topic(recorded, "camera__rgb_overhead", [5, 90, 120])
    topic(recorded, "bridge__joint_state", [5, 125], [[7.0] * 6, [8.0] * 6], envs=[1, 0])
    _, data = export(recorded, tmp_path)
    assert "missing_state" in data["episodes"][0]["samples"][0]["exclusions"]
    boundary = data["episodes"][1]["samples"][0]
    assert boundary["frame"] is None
    assert "reset_boundary" in boundary["exclusions"]
    assert boundary["instruction"] is None


def test_correction_changes_revision_without_future_goal_leak(recorded, tmp_path):
    """HAR-6, TC-5: request revisions bind by evidence, never final oracle labels."""
    topic(
        recorded,
        "task__target_request",
        [1, 26],
        texts=[json.dumps({"target_med": x}) for x in ["amoxicillin", "ibuprofen"]],
    )
    topic(
        recorded, "rollout__episode_goal", [1], texts=[json.dumps({"target_med": "final-secret"})]
    )
    _, data = export(recorded, tmp_path)
    rows = data["episodes"][0]["samples"]
    assert "amoxicillin" in rows[0]["instruction"]["text"]
    assert "ibuprofen" in rows[1]["instruction"]["text"]
    assert rows[0]["instruction"]["revision"] != rows[1]["instruction"]["revision"]


@pytest.mark.parametrize("times", [[20, 20], [90, 20], [None, 20]])
def test_invalid_source_timestamps_refused(recorded, tmp_path, times):
    """CON-8, HAR-6: duplicate, nonmonotonic and missing stamps are not repaired."""
    topic(recorded, "camera__rgb_overhead", times)
    with pytest.raises(ValueError, match="timestamp"):
        export(recorded, tmp_path)
    assert not (tmp_path / "dataset").exists()


def test_missing_stream_and_final_reset_are_explicit(recorded, tmp_path):
    """HAR-4/6: incomplete episodes are retained but cannot train."""
    (recorded / "traces/guard__gripper_cmd_safe.arrow").unlink()
    topic(recorded, "reset__reset_done", [0, 100], [[1]] * 2)
    _, data = export(recorded, tmp_path)
    assert "missing_gripper" in data["episodes"][0]["samples"][1]["exclusions"]
    assert "missing_final_reset" in data["episodes"][1]["exclusions"]
    assert not any(s["eligible"] for e in data["episodes"] for s in e["samples"])


@pytest.mark.parametrize("cut", [1, 20])
def test_truncated_arrow_refused(recorded, tmp_path, cut):
    """HAR-4: missing Arrow EOS or a partial batch cannot certify completeness."""
    path = recorded / "traces/bridge__joint_state.arrow"
    path.write_bytes(path.read_bytes()[:-cut])
    with pytest.raises(ValueError, match="Arrow"):
        export(recorded, tmp_path)


@pytest.mark.parametrize("damage", ["decode", "resolution", "missing"])
def test_capture_failures_are_excluded(recorded, tmp_path, damage):
    """HAR-4/6: failed raw decode, changing shape and absent pixels are explicit."""
    path = recorded / "traces/frames/overhead/00000000000000000120.npz"
    if damage == "decode":
        path.write_bytes(b"broken")
    elif damage == "resolution":
        np.savez(path, rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    else:
        path.unlink()
    _, data = export(recorded, tmp_path)
    row = data["episodes"][1]["samples"][1]
    assert not row["eligible"]
    assert any("pixel" in reason for reason in row["exclusions"])


def test_video_counts_do_not_certify_identity_and_truncation_is_reported():
    """HAR-4: decoder completeness is distinct from row-to-pixel attribution."""

    class Capture:
        def __init__(self, frames):
            self.frames = iter(frames)

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

    image = np.zeros((2, 3, 3), dtype=np.uint8)
    assert ds.inspect_video_frames(Capture([image]), 2, 2)["status"] == "incomplete"
    result = ds.inspect_video_frames(Capture([image, image]), 2, 2)
    assert result["status"] == "decoded"
    assert result["mapping"] == "unverified_ordinal"
    result = ds.inspect_video_frames(Capture([image, np.zeros((3, 3, 3))]), 2, 2)
    assert result["status"] == "resolution_changed"


def test_franka_unknown_contract_and_unsupported_format_refused(recorded, tmp_path):
    """TC-5, CON-8: SO101 loader never truncates Franka or assumes a format."""
    out, _ = export(recorded, tmp_path)
    manifest = out / "manifest.json"
    value = json.loads(manifest.read_text())
    value["format_version"] = 999
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="version"):
        trainer.load_tuples(out, recorded, 1, None)
    (recorded / "manifest.json").write_text(json.dumps(dict(embodiment="franka")))
    out = tmp_path / "franka"
    ds.export_dataset(recorded, out)
    with pytest.raises(ValueError, match="SO-101"):
        trainer.load_tuples(out, recorded, 1, None)
    unknown = ds.read_dataset(out)
    assert unknown["manifest"]["contract"] is None


def test_contract_validation_and_content_tampering(recorded, tmp_path):
    """TC-1/5, HAR-4: order, dimensions, units and content hashes are checked."""
    bad = copy.deepcopy(ds.SO101_CONTRACT)
    bad["state"]["units"] = ["degrees"] * 6
    out, _ = export(recorded, tmp_path, contract=bad)
    with pytest.raises(ValueError, match="contract"):
        trainer.load_tuples(out, recorded, 1, None)
    _, data = export(recorded, tmp_path, name="good")
    path = tmp_path / "good" / data["episodes"][0]["samples"][1]["pixels"]["path"]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        ds.read_dataset(tmp_path / "good")


def test_split_groups_precede_samples_and_unknown_family_stays_together(recorded, tmp_path):
    """CON-5: an episode/task family cannot leak across train/test windows."""
    _, data = export(recorded, tmp_path)
    episodes = data["episodes"]
    assert episodes[0]["split"] == episodes[1]["split"] == "train"
    assert all(s["split"] == e["split"] for e in episodes for s in e["samples"])
    assert ds.split_for("unknown") == ds.split_for("unknown")


def test_json_cli_and_legacy_dataset_rejection(recorded, tmp_path, capsys):
    """CON-8: diagnostic exports succeed; unsupported legacy inputs fail loudly."""
    out = tmp_path / "cli"
    assert exporter.main(["--run", str(recorded), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert exporter.main(["--run", str(recorded), "--out", str(out)]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    with pytest.raises(ValueError, match="version"):
        trainer.load_tuples(recorded, recorded, 1, None)


@pytest.mark.parametrize("width", [5, 9])
def test_wrong_dimensions_excluded_even_when_embodiment_claims_so101(recorded, tmp_path, width):
    """TC-5: declared embodiment does not license silent dimension truncation."""
    topic(recorded, "guard__joint_cmd_safe", [30], [[1.0] * width])
    _, data = export(recorded, tmp_path)
    assert "invalid_forwarded_vector" in data["episodes"][0]["samples"][0]["exclusions"]


@pytest.mark.parametrize("change", ["order", "frame", "units"])
def test_loader_rejects_incompatible_semantics(recorded, tmp_path, change):
    """TC-1/5: matching width alone cannot authorize a training interpretation."""
    contract = copy.deepcopy(ds.SO101_CONTRACT)
    if change == "order":
        contract["state"]["names"].reverse()
    elif change == "frame":
        contract["forwarded"]["frame"] = "world"
    else:
        contract["gripper"]["units"] = ["rad"]
    out, _ = export(recorded, tmp_path, contract=contract)
    with pytest.raises(ValueError, match="contract"):
        trainer.load_tuples(out, recorded, 1, None)


def test_equal_time_and_malformed_correction_do_not_reuse_old_evidence(recorded, tmp_path):
    """HAR-6: equal stamps have no cross-producer order; malformed revisions exclude."""
    topic(recorded, "bridge__joint_state", [30], [[1.0] * 6])
    topic(
        recorded,
        "task__target_request",
        [1, 26],
        texts=['{"target_med":"amoxicillin"}', '{"unknown":"correction"}'],
    )
    _, data = export(recorded, tmp_path)
    sample = data["episodes"][0]["samples"][1]
    assert sample["state"] is None
    assert sample["instruction"] is None
    assert "missing_instruction" in sample["exclusions"]


def test_same_time_goal_and_gripper_changes_are_ambiguous(recorded, tmp_path):
    """HAR-6: no total order means an equal-time correction cannot label an action."""
    topic(
        recorded,
        "task__target_request",
        [1, 30],
        texts=['{"target_med":"amoxicillin"}', '{"target_med":"ibuprofen"}'],
    )
    topic(recorded, "guard__gripper_cmd_safe", [2, 30], [[0], [1]])
    _, data = export(recorded, tmp_path)
    sample = data["episodes"][0]["samples"][1]
    assert "ambiguous_instruction_order" in sample["exclusions"]
    assert "ambiguous_gripper_order" in sample["exclusions"]
    assert sample["instruction"] is None


def test_source_identity_checked_even_if_derived_file_is_rehashed(recorded, tmp_path):
    """HAR-4/6: derived values must agree with the referenced immutable Arrow row."""
    out, data = export(recorded, tmp_path)
    data["episodes"][0]["samples"][1]["forwarded"]["data"] = [42.0] * 6
    ds.write_json(out / "episodes.json", data["episodes"])
    data["manifest"]["artifacts_sha256"]["episodes.json"] = ds.sha256(out / "episodes.json")
    ds.write_json(out / "manifest.json", data["manifest"])
    with pytest.raises(ValueError, match="identity/value mismatch"):
        ds.read_dataset(out)


def test_pixel_identity_checked_against_frame_timestamp(recorded, tmp_path):
    """HAR-4/6: a valid asset hash cannot substitute a different frame's pixels."""
    out, data = export(recorded, tmp_path)
    data["episodes"][0]["samples"][1]["pixels"]["path"] = (
        "source/traces/frames/overhead/00000000000000000090.npz"
    )
    ds.write_json(out / "episodes.json", data["episodes"])
    data["manifest"]["artifacts_sha256"]["episodes.json"] = ds.sha256(out / "episodes.json")
    ds.write_json(out / "manifest.json", data["manifest"])
    with pytest.raises(ValueError, match="pixel identity"):
        ds.read_dataset(out)


def test_multiple_producers_require_selection_and_raw_identity_stays_ambiguous(recorded, tmp_path):
    """HAR-4/6: camera filename alone cannot identify one of multiple producers."""
    topic(recorded, "second__rgb_overhead", [20, 90, 120])
    with pytest.raises(ValueError, match="ambiguous producers"):
        export(recorded, tmp_path)
    out = tmp_path / "selected"
    ds.export_dataset(
        recorded, out, ds.SO101_CONTRACT, {"rgb_overhead": "camera__rgb_overhead.arrow"}
    )
    assert not ds.read_dataset(out)["episodes"][0]["samples"][1]["eligible"]


def test_shuffled_episode_identity_is_refused(recorded, tmp_path):
    """HAR-6: reset ordinal cannot be silently rebound to a different episode."""
    path = recorded / "episodes.jsonl"
    path.write_text("\n".join(reversed(path.read_text().splitlines())))
    with pytest.raises(ValueError, match="episode order"):
        export(recorded, tmp_path)


def test_trainer_rejects_before_model_import(recorded, tmp_path, monkeypatch, capsys):
    """CON-8: invalid input returns JSON before costly model initialization."""
    monkeypatch.setattr(
        sys, "argv", ["finetune", "--dataset", str(recorded), "--out", str(tmp_path / "model")]
    )
    assert trainer.main() == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert not (tmp_path / "model").exists()


def test_malformed_contract_cli_is_json(recorded, tmp_path, capsys):
    """CON-8: malformed configuration must not escape as a traceback."""
    contract = tmp_path / "contract.json"
    contract.write_text("[]")
    assert (
        exporter.main(
            ["--run", str(recorded), "--out", str(tmp_path / "out"), "--contract", str(contract)]
        )
        == 1
    )
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_statistics_are_required_and_validated_before_training(recorded, tmp_path):
    """CON-8, TC-5: model initialization cannot precede validating training statistics."""
    out, data = export(recorded, tmp_path)
    stats = json.loads((out / "stats.json").read_text())
    stats["state_std"] = [-1.0] * 6
    ds.write_json(out / "stats.json", stats)
    data["manifest"]["artifacts_sha256"]["stats.json"] = ds.sha256(out / "stats.json")
    ds.write_json(out / "manifest.json", data["manifest"])
    with pytest.raises(ValueError, match="statistics"):
        trainer.load_tuples(out, recorded, 1, None)


def rehash_artifact(out, name):
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["artifacts_sha256"][name] = ds.sha256(out / name)
    ds.write_json(out / "manifest.json", manifest)


def test_missing_initial_reset_cannot_shift_success_labels(recorded, tmp_path):
    """CON-5, HAR-6: losing reset zero must not label the next failure a success."""
    topic(recorded, "reset__reset_done", [100, 200], [[1], [1]])
    with pytest.raises(ValueError, match="initial reset"):
        export(recorded, tmp_path)
    assert not (tmp_path / "dataset").exists()


def test_reader_rechecks_initial_reset_from_bundled_source(recorded, tmp_path):
    """CON-5, HAR-4: a rehashed bundle cannot hide missing reset evidence."""
    out, _ = export(recorded, tmp_path)
    topic(out / "source", "reset__reset_done", [100, 200], [[1], [1]])
    rehash_artifact(out, "source/traces/reset__reset_done.arrow")
    with pytest.raises(ValueError, match="initial reset"):
        trainer.load_tuples(out, None, 1, None)


@pytest.mark.parametrize("mismatch", ["reset", "seed"])
def test_goal_evidence_cannot_contradict_episode_mapping(recorded, tmp_path, mismatch):
    """HAR-4/6: use retained reset/seed evidence to refuse shifted episode attribution."""
    path = recorded / "episodes.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    for i, record in enumerate(records):
        record["seed"] = i
    path.write_text("\n".join(map(json.dumps, records)))
    goals = [dict(reset_sim_ns=0, seed=0), dict(reset_sim_ns=100, seed=1)]
    goals[1]["reset_sim_ns" if mismatch == "reset" else "seed"] = 999
    topic(recorded, "rollout__episode_goal", [1, 101], texts=list(map(json.dumps, goals)))
    with pytest.raises(ValueError, match="goal.*(reset|seed)"):
        export(recorded, tmp_path)


@pytest.mark.parametrize("channel", ["frame", "state", "gripper", "instruction"])
def test_later_command_excludes_reset_stamped_observation(recorded, tmp_path, channel):
    """HAR-6: a command after reset cannot establish the order of boundary observations."""
    if channel == "frame":
        topic(recorded, "camera__rgb_overhead", [20, 90, 100])
        np.savez(
            recorded / "traces/frames/overhead/00000000000000000100.npz",
            rgb=np.full((2, 3, 3), 100, dtype=np.uint8),
        )
    elif channel == "state":
        topic(recorded, "bridge__joint_state", [25, 100], [[9.0] * 6, [8.0] * 6])
    elif channel == "gripper":
        topic(recorded, "guard__gripper_cmd_safe", [2, 100], [[0], [1]])
    else:
        topic(
            recorded,
            "task__target_request",
            [1, 100],
            texts=['{"target_med":"amoxicillin"}', '{"target_med":"ibuprofen"}'],
        )
    _, data = export(recorded, tmp_path)
    sample = data["episodes"][1]["samples"][1]
    assert sample["forwarded"]["sim_time_ns"] == 130
    assert sample[channel] is None
    assert f"missing_{channel}" in sample["exclusions"]
    assert not sample["eligible"]


def test_reader_rejects_reset_stamped_state_after_rehash(recorded, tmp_path):
    """HAR-6: readback must independently enforce the strict reset lower bound."""
    topic(recorded, "bridge__joint_state", [25, 100], [[9.0] * 6, [8.0] * 6])
    out, data = export(recorded, tmp_path)
    sample = data["episodes"][1]["samples"][1]
    sample["state"] = ds.strict_stream(out / "source/traces/bridge__joint_state.arrow", out)[1]
    sample.update(eligible=True, exclusions=[])
    ds.write_json(out / "episodes.json", data["episodes"])
    rehash_artifact(out, "episodes.json")
    with pytest.raises(ValueError, match="noncausal"):
        ds.read_dataset(out)


@pytest.mark.parametrize("channel", ["forwarded", "state"])
def test_reader_rejects_proposed_row_in_other_channel(recorded, tmp_path, channel):
    """HAR-4/6, TC-5: valid row hashes do not authorize changing channel roles."""
    out, data = export(recorded, tmp_path)
    sample = data["episodes"][0]["samples"][1]
    sample[channel], sample["proposed"] = sample["proposed"], sample[channel]
    ds.write_json(out / "episodes.json", data["episodes"])
    rehash_artifact(out, "episodes.json")
    with pytest.raises(ValueError, match="channel endpoint"):
        trainer.load_tuples(out, None, 1, None)


def test_reader_rejects_same_topic_from_unselected_producer(recorded, tmp_path):
    """HAR-4/6: endpoint binding includes the selected producer, not just the topic."""
    out, data = export(recorded, tmp_path)
    path = topic(out / "source", "other__joint_cmd_safe", [30], [[2.0] * 6])
    rehash_artifact(out, str(path.relative_to(out)))
    data["episodes"][0]["samples"][1]["forwarded"] = ds.strict_stream(path, out)[0]
    ds.write_json(out / "episodes.json", data["episodes"])
    rehash_artifact(out, "episodes.json")
    with pytest.raises(ValueError, match="channel endpoint"):
        trainer.load_tuples(out, None, 1, None)


def test_reader_binds_outcome_and_window_to_source_episode(recorded, tmp_path):
    """HAR-4/6: a rehashed derived outcome cannot promote a source failure to success."""
    out, data = export(recorded, tmp_path)
    data["episodes"][1]["outcome"]["status"] = "success"
    ds.write_json(out / "episodes.json", data["episodes"])
    rehash_artifact(out, "episodes.json")
    with pytest.raises(ValueError, match="episode.*source"):
        trainer.load_tuples(out, None, 1, None)


def test_reader_checks_reset_endpoint_topic(recorded, tmp_path):
    """HAR-4/6: the attribution validator cannot reinterpret gripper rows as resets."""
    out, _ = export(recorded, tmp_path)
    path = topic(out / "source", "other__gripper_cmd_safe", [0, 100, 200], [[1]] * 3)
    rehash_artifact(out, str(path.relative_to(out)))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["endpoints"]["reset_done"] = "traces/other__gripper_cmd_safe.arrow"
    ds.write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="channel endpoint"):
        ds.read_dataset(out)
