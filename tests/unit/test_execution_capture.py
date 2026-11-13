"""P3 executor-side provenance, independent of physical task scoring."""

import json

import pyarrow as pa
import pytest

from aisle.harness.execution_capture import ExecutionCapture, RecordedTransport, read_capture

pytestmark = pytest.mark.unit
SKILL = {"name": "fixture-skill", "version": "sha256:fixture"}
INPUTS = {
    "reset_done": "reset/reset_done",
    "grasp_pose": "planner/grasp_pose",
    "joint_state": "bridge/joint_state",
}


def event(port, stamp, values, *, seq=None, **extra):
    return dict(
        id=port,
        value=pa.array(values),
        metadata=dict(sim_time_ns=stamp, seq=stamp if seq is None else seq, env_id=0, **extra),
    )


def make(tmp_path):
    return ExecutionCapture(
        tmp_path / "capture.jsonl",
        capture_id="fixture",
        node_id="executor",
        inputs=INPUTS,
        skill=SKILL,
    )


def arm(capture):
    capture.reset(event("reset_done", 0, [1], request_id="r0"))
    proposal = capture.propose(event("grasp_pose", 1, [0.1] * 7, goal_id="g1"), {"kind": "grasp"})
    capture.accept(proposal)
    capture.observe(event("joint_state", 2, [0.0] * 9))
    return proposal


class Raw:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    def send_output(self, topic, value, metadata):
        if self.fail:
            raise RuntimeError("transport refused")
        self.sent.append((topic, value.to_pylist(), metadata))

    def __iter__(self):
        return iter([])


def test_start_is_actual_send_not_plan_creation_and_return_is_unconfirmed(tmp_path):
    """HAR-4; TC-8: actual send establishes emission, never physical delivery."""
    capture = make(tmp_path)
    invocation = arm(capture)
    raw = Raw()
    transport = RecordedTransport(raw, capture)
    before = [json.loads(line) for line in capture.path.read_text().splitlines()]
    assert not any(r.get("kind") == "started" for r in before)
    transport.send_output("joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, env_id=0, seq=1))
    capture.finish(dict(sim_time_ns=2, env_id=0, seq=2))
    capture.close()
    result = read_capture(capture.path)
    assert [e["kind"] for e in result["events"]] == [
        "reset",
        "proposed",
        "accepted",
        "started",
        "command_emitted",
        "returned",
    ]
    command = next(e for e in result["events"] if e["kind"] == "command_emitted")
    assert command["invocation_id"] == invocation
    assert command["source"]["endpoint"] == "executor/joint_cmd"
    assert command["observation"]["endpoint"] == "bridge/joint_state"
    assert command["source"]["seq"] == 1
    assert command["observation"]["seq"] == 2
    assert result["events"][-1]["delivery_completion"] == "unknown"
    assert raw.sent[0][2]["invocation_id"] == invocation
    assert raw.sent[0][2]["command_origin_seq"] == 1
    assert raw.sent[0][2]["goal_id"] == "g1"


def test_failed_transport_does_not_claim_started_or_emitted(tmp_path):
    """HAR-4: a refused transport send cannot establish execution."""
    capture = make(tmp_path)
    arm(capture)
    with pytest.raises(RuntimeError, match="transport"):
        RecordedTransport(Raw(fail=True), capture).send_output(
            "joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, env_id=0, seq=1)
        )
    capture.close()
    result = read_capture(capture.path)
    assert not any(e["kind"] in {"started", "command_emitted"} for e in result["events"])
    assert result["open_invocation"] is not None


def test_reset_records_interruption_and_drops_command_labels(tmp_path):
    """TC-6; HAR-4: reset invalidates active invocation and attribution."""
    capture = make(tmp_path)
    old = arm(capture)
    raw = Raw()
    transport = RecordedTransport(raw, capture)
    transport.send_output("joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, env_id=0, seq=1))
    capture.reset(event("reset_done", 3, [1], request_id="r1"))
    transport.send_output("joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=4, env_id=0, seq=2))
    capture.close()
    result = read_capture(capture.path)
    interrupted = next(e for e in result["events"] if e["kind"] == "interrupted")
    assert interrupted["invocation_id"] == old
    assert "invocation_id" not in raw.sent[-1][2]


def test_busy_proposal_rejection_preserves_existing_invocation(tmp_path):
    """HAR-3/4: a rejected second request does not relabel the executing plan."""
    capture = make(tmp_path)
    old = arm(capture)
    new = capture.propose(event("grasp_pose", 3, [0.2] * 7, goal_id="g2"), {})
    capture.reject(new, "controller_busy")
    raw = Raw()
    RecordedTransport(raw, capture).send_output(
        "joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=4, env_id=0, seq=1)
    )
    capture.close()
    assert raw.sent[0][2]["invocation_id"] == old
    assert raw.sent[0][2]["goal_id"] == "g1"
    assert any(
        e["kind"] == "rejected" and e["invocation_id"] == new
        for e in read_capture(capture.path)["events"]
    )


def test_missing_goal_identity_is_not_inferred(tmp_path):
    """TC-7; HAR-4: missing causal goal metadata stays unknown."""
    capture = make(tmp_path)
    capture.reset(event("reset_done", 0, [1]))
    request = capture.propose(event("grasp_pose", 1, [0.0] * 7), {})
    capture.accept(request)
    capture.close()
    assert read_capture(capture.path)["events"][1]["goal_revision"] is None


def test_equal_simulation_timestamps_have_explicit_local_order(tmp_path):
    """CON-5: lifecycle order is witnessed by the executor, not fabricated timestamps."""
    capture = make(tmp_path)
    arm(capture)
    RecordedTransport(Raw(), capture).send_output(
        "joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, env_id=0, seq=1)
    )
    capture.finish(dict(sim_time_ns=2, env_id=0, seq=2))
    capture.close()
    events = read_capture(capture.path)["events"]
    assert [e["event_seq"] for e in events] == list(range(len(events)))
    assert [e["sim_time_ns"] for e in events[-3:]] == [2, 2, 2]


def test_incomplete_capture_is_refused_and_never_overwritten(tmp_path):
    """HAR-4; CON-8: no sealed capture claim for a truncated writer or occupied path."""
    capture = make(tmp_path)
    arm(capture)
    with pytest.raises(ValueError, match="closed"):
        read_capture(capture.path)
    with pytest.raises(FileExistsError):
        make(tmp_path)
    capture.close()


def test_capture_content_change_breaks_footer_binding(tmp_path):
    """HAR-4: capture closure binds the preceding header and event bytes."""
    capture = make(tmp_path)
    arm(capture)
    capture.close()
    capture.path.write_text(capture.path.read_text().replace('"g1"', '"g2"'))
    with pytest.raises(ValueError, match="digest"):
        read_capture(capture.path)


def test_cross_environment_reset_is_refused(tmp_path):
    """TC-6; BRG-5: a reset cannot silently move the capture to another env."""
    capture = make(tmp_path)
    capture.reset(event("reset_done", 0, [1]))
    other = event("reset_done", 1, [1])
    other["metadata"]["env_id"] = 2
    with pytest.raises(ValueError, match="environments"):
        capture.reset(other)
    capture.close()


def test_unknown_request_goal_does_not_inherit_observation_goal(tmp_path):
    """TC-7: observations cannot supply a missing invocation goal identity."""
    capture = make(tmp_path)
    capture.reset(event("reset_done", 0, [1]))
    capture.accept(capture.propose(event("grasp_pose", 1, [0.0] * 7), {}))
    labels = capture.command_metadata(
        "joint_cmd", dict(sim_time_ns=2, env_id=0, seq=1, goal_id="stale")
    )
    assert "goal_id" not in labels
    capture.close()


def test_reset_cancels_proposals_and_prevents_reaccept(tmp_path):
    """TC-6: pending pre-reset proposals cannot become post-reset commands."""
    capture = make(tmp_path)
    capture.reset(event("reset_done", 0, [1]))
    pending = capture.propose(event("grasp_pose", 1, [0.0] * 7), {})
    capture.reset(event("reset_done", 2, [1]))
    with pytest.raises(ValueError, match="reused"):
        capture.accept(pending)
    capture.close()


def test_rehashed_invalid_lifecycle_is_not_a_valid_capture(tmp_path):
    """HAR-4: a self-consistent byte digest cannot legitimize fabricated lifecycle order."""
    import hashlib

    from aisle.harness.execution_capture import encoded

    capture = make(tmp_path)
    arm(capture)
    capture.close()
    records = [json.loads(line) for line in capture.path.read_bytes().splitlines()]
    records[-2]["kind"] = "returned"  # proposed -> returned without acceptance
    raw = b"".join(encoded(r) for r in records[:-1])
    records[-1]["sha256"] = hashlib.sha256(raw).hexdigest()
    capture.path.write_bytes(raw + encoded(records[-1]))
    with pytest.raises(ValueError, match="lifecycle"):
        read_capture(capture.path)


def test_lockstep_capture_hashes_final_transport_stamp(tmp_path):
    """TC-2; HAR-4: capture sees the authoritative turn stamp, not pre-wrapper metadata."""
    from aisle.turn_node import Node

    capture = make(tmp_path)
    arm(capture)
    raw = Raw()
    raw.events = [
        dict(
            type="INPUT",
            id="turn",
            value=pa.array([1]),
            metadata=dict(
                turn_epoch=1,
                turn_id=1,
                sim_time_ns=30,
                env_id=0,
                seq=1,
                target_node="executor",
                expected_inputs=[],
                expected_counts=[],
            ),
        )
    ]

    class IterableRaw(Raw):
        def __iter__(self):
            return iter(raw.events)

    actual = IterableRaw()
    node = Node(
        RecordedTransport(actual, capture),
        dict(
            AISLE_LOCKSTEP="1", AISLE_TURN_NODE="executor", AISLE_TURN_OUTPUTS="joint_cmd,turn_done"
        ),
    )
    for _ in node:
        node.send_output("joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, env_id=0, seq=1))
    result = read_capture(capture.path)
    command = next(e for e in result["events"] if e["kind"] == "command_emitted")
    assert command["sim_time_ns"] == command["source"]["sim_time_ns"] == 30
    assert command["observation"]["sim_time_ns"] == 2
    assert actual.sent[0][2]["sim_time_ns"] == 30


def trace_fixture(tmp_path):
    """Small source capture with a distinct guarded command and one observed invocation."""
    from aisle.harness.execution_capture import reference

    capture = make(tmp_path)
    invocation = arm(capture)
    raw = Raw()
    RecordedTransport(raw, capture).send_output(
        "joint_cmd", pa.array([0.0] * 9), dict(sim_time_ns=2, env_id=0, seq=1)
    )
    capture.close()
    traces = tmp_path / "traces"
    traces.mkdir()
    streams = [
        ("reset/reset_done", event("reset_done", 0, [1], request_id="r0")),
        ("planner/grasp_pose", event("grasp_pose", 1, [0.1] * 7, goal_id="g1")),
        ("bridge/joint_state", event("joint_state", 2, [0.0] * 9)),
        ("executor/joint_cmd", dict(value=pa.array([0.0] * 9), metadata=raw.sent[0][2])),
        (
            "guard/joint_cmd_safe",
            dict(value=pa.array([0.01] * 9), metadata={**raw.sent[0][2], "seq": 17}),
        ),
    ]
    identities = []
    schema = pa.schema(
        [
            ("sim_time_ns", pa.int64()),
            ("env_id", pa.int32()),
            ("seq", pa.int64()),
            ("data", pa.list_(pa.float64())),
            ("text", pa.string()),
        ]
    )
    for endpoint, e in streams:
        ref = reference(endpoint, e["value"], e["metadata"])
        row = {k: ref[k] for k in ("sim_time_ns", "env_id", "seq")}
        row.update(data=[float(x) for x in e["value"].to_pylist()], text=None)
        name = endpoint.replace("/", "__")
        with pa.OSFile(str(traces / f"{name}.arrow"), "wb") as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_table(pa.Table.from_pylist([row], schema=schema))
        identities.append(
            dict(
                format="aisle_trace_identity",
                version=1,
                endpoint=name,
                row=0,
                trace_row_sha256=ref["trace_row_sha256"],
                metadata=e["metadata"],
            )
        )
    (traces / "identities.jsonl").write_text("".join(json.dumps(r) + "\n" for r in identities))
    config = dict(capture_id="fixture", node_id="executor", inputs=INPUTS, skill=SKILL)
    return capture.path, traces, config, invocation


def test_verify_matches_exact_source_rows_and_guard_identity(tmp_path):
    """HAR-4; BG-1: source joins use captured row hashes and explicit command ancestry."""
    from aisle.harness.execution_capture import verify_capture

    path, traces, config, invocation = trace_fixture(tmp_path)
    result = verify_capture(path, traces, config, guard_endpoints={"guard/joint_cmd_safe"})
    assert len(result["forwarded"]) == 1
    assert result["forwarded"][0]["invocation_id"] == invocation
    assert result["forwarded"][0]["source"]["seq"] == 17
    assert result["forwarded"][0]["origin"]["seq"] == 1


def test_verify_refuses_missing_observation_and_forged_skill(tmp_path):
    """HAR-4; CON-5: missing raw evidence or mismatched instrument binding cannot verify."""
    from aisle.harness.execution_capture import verify_capture

    path, traces, config, _ = trace_fixture(tmp_path)
    with pytest.raises(ValueError, match="binding"):
        verify_capture(
            path,
            traces,
            {**config, "skill": {"name": "other", "version": "1"}},
            guard_endpoints=set(),
        )
    (traces / "bridge__joint_state.arrow").unlink()
    with pytest.raises(ValueError, match="source"):
        verify_capture(path, traces, config, guard_endpoints=set())


def test_guard_timing_without_origin_labels_is_not_forwarding_evidence(tmp_path):
    """BG-1; HAR-4: same-time guarded commands cannot be assigned by timing alone."""
    from aisle.harness.execution_capture import verify_capture

    path, traces, config, _ = trace_fixture(tmp_path)
    identities = [
        json.loads(line) for line in (traces / "identities.jsonl").read_text().splitlines()
    ]
    identities[-1]["metadata"] = {}
    (traces / "identities.jsonl").write_text("".join(json.dumps(r) + "\n" for r in identities))
    result = verify_capture(path, traces, config, guard_endpoints={"guard/joint_cmd_safe"})
    assert result["forwarded"] == []


def test_source_goal_metadata_cannot_be_relabelled(tmp_path):
    """TC-7; HAR-4: journal goal attribution must agree with captured request metadata."""
    from aisle.harness.execution_capture import verify_capture

    path, traces, config, _ = trace_fixture(tmp_path)
    sidecar = traces / "identities.jsonl"
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    records[1]["metadata"]["goal_id"] = "different-goal"
    sidecar.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="goal"):
        verify_capture(path, traces, config, guard_endpoints=set())


def test_observation_from_another_environment_is_refused(tmp_path):
    """BRG-5; HAR-4: an observation from another world cannot support a command."""
    capture = make(tmp_path)
    arm(capture)
    other = event("joint_state", 3, [0.0] * 9)
    other["metadata"]["env_id"] = 1
    with pytest.raises(ValueError, match="environments"):
        capture.observe(other)
    capture.close()


def test_delayed_pre_reset_request_cannot_be_attributed_to_new_episode(tmp_path):
    """TC-6/7; HAR-4: a delayed request cannot acquire a newer reset identity."""
    capture = make(tmp_path)
    capture.reset(event("reset_done", 0, [1]))
    capture.reset(event("reset_done", 10, [1]))
    with pytest.raises(ValueError, match="predates reset"):
        capture.propose(event("grasp_pose", 5, [0.0] * 7, goal_id="old"), {})
    capture.close()


def test_recorder_preserves_command_ancestry_and_request_identity():
    """HAR-4; TC-7; BG-1: sidecar binds invocation labels to exact raw command rows."""
    from aisle.harness.trace_recorder import IDENTITY_TOPICS, trace_identity

    metadata = dict(
        sim_time_ns=1,
        env_id=0,
        seq=2,
        invocation_id="inv",
        goal_id="goal",
        command_origin_endpoint="executor/joint_cmd",
        command_origin_seq=1,
        skill_name="skill",
        skill_version="v1",
    )
    row = {k: metadata[k] for k in ("sim_time_ns", "env_id", "seq")}
    row.update(data=[0.0], text=None)
    identity = trace_identity("guard__joint_cmd_safe", 0, row, metadata)
    assert identity["metadata"]["invocation_id"] == "inv"
    assert identity["metadata"]["command_origin_seq"] == 1
    assert {"grasp_pose", "target_pose", "return_request"} <= IDENTITY_TOPICS


def test_journal_command_clock_must_match_its_raw_source(tmp_path):
    """HAR-4: source presence cannot validate a contradictory lifecycle clock."""
    import hashlib

    from aisle.harness.execution_capture import encoded, verify_capture

    path, traces, config, _ = trace_fixture(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    next(e for e in records if e.get("kind") == "command_emitted")["sim_time_ns"] = 99
    raw = b"".join(encoded(e) for e in records[:-1])
    records[-1]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw + encoded(records[-1]))
    with pytest.raises(ValueError, match="clock"):
        verify_capture(path, traces, config, guard_endpoints=set())
