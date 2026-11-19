"""Public P3 task traffic over the trusted lockstep Node wrapper."""

import json

import pyarrow as pa
import pytest

from aisle.turn_node import Node

pytestmark = pytest.mark.unit


class Raw:
    def __init__(self, events):
        self.events, self.sent = events, []

    def __iter__(self):
        yield from self.events

    def send_output(self, topic, value, metadata):
        self.sent.append((topic, value.to_pylist(), metadata))


def stamp(turn=0, **extra):
    return dict(
        turn_epoch=7,
        turn_id=turn,
        sim_time_ns=turn * 1_000_000_000,
        env_id=0,
        seq=turn + 1,
        **extra,
    )


def event(topic, value, turn=0, **metadata):
    return dict(type="INPUT", id=topic, value=value, metadata={**stamp(turn), **metadata})


def turn_events(inputs, turn=0):
    return [
        *inputs,
        event(
            "turn",
            pa.array([turn], type=pa.uint64()),
            turn,
            target_node="task",
            expected_inputs=sorted(x["id"] for x in inputs),
            expected_counts=[1] * len(inputs),
        ),
    ]


def transport(events):
    from aisle.harness.p3_dora_transport import DoraTaskTransport

    raw = Raw(events)
    node = Node(
        raw,
        dict(
            AISLE_LOCKSTEP="1",
            AISLE_TURN_NODE="task",
            AISLE_TURN_OUTPUTS="robot_msg,target_request,return_request,episode_feedback,turn_done",
        ),
    )
    return DoraTaskTransport(node, turn_epoch=7), raw


def reset():
    return event("reset_done", pa.array([1], type=pa.uint32()), seed=300, request_id="private")


def test_arrow_projection_output_stamps_and_host_turn_acknowledgement():
    """TC-2/6/7; VAL-6: keep private metadata out and leave turn acknowledgement to the host."""
    human = dict(kind="request", med="a", text="please give a")
    service, raw = transport(
        [
            *turn_events(
                [
                    reset(),
                    event(
                        "human_msg",
                        pa.array([json.dumps(human)]),
                        goal_id="goal",
                        private_field="hidden",
                    ),
                ]
            ),
            dict(type="INPUT_CLOSED", id="turn"),
        ]
    )
    topic, payload, meta = service.next_event()
    assert (topic, payload) == ("reset_done", [1])
    assert set(meta) == {"env_id", "seq", "sim_time_ns", "turn_epoch", "turn_id"}
    assert service.next_event() == ("human_msg", human, {**stamp(), "goal_id": "goal"})
    service.send("robot_msg", dict(kind="confirm", med="a", text="a?"), {"goal_id": "goal"})
    service.send("target_request", {"target_med": "a"}, {"goal_id": "goal", "request_id": "req"})
    with pytest.raises(StopIteration):
        service.next_event()
    assert [r[0] for r in raw.sent] == ["robot_msg", "target_request", "turn_done"]
    assert json.loads(raw.sent[1][1][0]) == {"target_med": "a"}
    assert raw.sent[1][2] == {**stamp(), "goal_id": "goal", "request_id": "req"}
    assert dict(
        zip(raw.sent[-1][2]["closed_outputs"], raw.sent[-1][2]["emitted_counts"], strict=True)
    ) == {
        "robot_msg": 1,
        "target_request": 1,
        "return_request": 0,
        "episode_feedback": 0,
        "turn_done": 1,
    }
    with pytest.raises(ValueError, match="closed"):
        service.send(
            "target_request", {"target_med": "a"}, {"goal_id": "goal", "request_id": "req"}
        )


def test_sim_ticks_do_not_use_wall_timer_or_reset_private_fields():
    """CON-5; TC-2/6: public ticks follow simulation turns after reset, never wall time."""
    service, _ = transport(
        [
            dict(type="INPUT", id="tick", value=pa.array([], type=pa.uint8()), metadata={}),
            *turn_events([reset()]),
            *turn_events([], 1),
            *turn_events([], 2),
            dict(type="INPUT_CLOSED", id="turn"),
        ]
    )
    assert service.next_event()[0] == "reset_done"
    one = service.next_event()
    two = service.next_event()
    assert one == ("tick", None, {**stamp(1), "seq": 1})
    assert two == ("tick", None, {**stamp(2), "seq": 2})
    with pytest.raises(StopIteration):
        service.next_event()


@pytest.mark.parametrize(
    "topic", ["oracle_state", "episode_result", "episode_goal", "episode_meta"]
)
def test_private_input_is_refused(topic):
    """VAL-2/6; TRT-6: accidental private graph wiring is a fatal transport refusal."""
    service, _ = transport(turn_events([event(topic, pa.array(["hidden"]))]))
    with pytest.raises(ValueError, match="undeclared"):
        service.next_event()
    with pytest.raises(ValueError, match="closed"):
        service.next_event()


@pytest.mark.parametrize(
    "value", [pa.array([1]), pa.array([0], type=pa.uint32()), pa.array([None], type=pa.uint32())]
)
def test_reset_wire_contract_is_checked(value):
    """TC-6: only UInt32[1]=1 may cross the reset boundary."""
    service, _ = transport(turn_events([event("reset_done", value)]))
    with pytest.raises(ValueError, match="reset"):
        service.next_event()


@pytest.mark.parametrize(
    "value",
    [
        pa.array(['{"ok": true, "ok": false}']),
        pa.array(['{"ok": NaN}']),
        pa.array(["[]"]),
        pa.array(["{}", "{}"]),
        pa.array([1]),
        pa.array(["x" * 8193]),
    ],
)
def test_malformed_json_input_is_refused(value):
    """CON-4; HAR-4: malformed, ambiguous or oversized task traffic cannot be decoded silently."""
    service, _ = transport(turn_events([event("plan_done", value, goal_id="g", request_id="r")]))
    with pytest.raises(ValueError):
        service.next_event()


def test_producer_incarnation_change_is_refused():
    """TC-2; TRT-9: a fresh bridge incarnation requires a new worker binding."""
    events = turn_events([reset()])
    for item in events:
        item["metadata"]["turn_epoch"] = 8
    service, _ = transport(events)
    with pytest.raises(ValueError, match="epoch"):
        service.next_event()


@pytest.mark.parametrize("terminal", [None, {"type": "STOP"}, {"type": "ERROR", "error": "bad"}])
def test_unexplained_end_is_not_successful_exhaustion(terminal):
    """HAR-4; TRT-10: only explicit coordinator-input closure ends the transport normally."""
    service, _ = transport([] if terminal is None else [terminal])
    with pytest.raises(ValueError):
        service.next_event()


@pytest.mark.parametrize(
    "topic,payload,meta",
    [
        ("joint_cmd", {"target_med": "a"}, {"goal_id": "g", "request_id": "r"}),
        ("target_request", {"target_med": "a", "extra": True}, {"goal_id": "g", "request_id": "r"}),
        (
            "target_request",
            {"target_med": "a"},
            {"goal_id": "g", "request_id": "r", "turn_epoch": 8},
        ),
        ("return_request", {"return_med": "a"}, {"goal_id": "g"}),
    ],
)
def test_output_endpoint_payload_and_identity_authority_is_fixed(topic, payload, meta):
    """VAL-2; TRT-5: transport admits only fixed high-level requests and controller identities."""
    service, raw = transport(turn_events([reset()]))
    service.next_event()
    with pytest.raises(ValueError):
        service.send(topic, payload, meta)
    assert raw.sent == []


def test_requires_lockstep_and_exact_output_grants():
    """TC-2; TRT-5: arbitrary Node transports or actuator grants cannot masquerade as P3."""
    from aisle.harness.p3_dora_transport import DoraTaskTransport

    with pytest.raises(ValueError):
        DoraTaskTransport(Node(Raw([]), {}), turn_epoch=7)
    node = Node(
        Raw([]),
        dict(AISLE_LOCKSTEP="1", AISLE_TURN_NODE="task", AISLE_TURN_OUTPUTS="joint_cmd,turn_done"),
    )
    with pytest.raises(ValueError):
        DoraTaskTransport(node, turn_epoch=7)


def test_completion_ids_and_output_sequence_are_preserved():
    """TC-2/7: executor correlation survives projection and output sequences are per topic."""
    service, raw = transport(
        [
            *turn_events([reset()]),
            *turn_events(
                [
                    event(
                        "plan_done",
                        pa.array(['{"ok": true}']),
                        1,
                        goal_id="g",
                        request_id="r",
                        private_field="hidden",
                    )
                ],
                1,
            ),
        ]
    )
    service.next_event()
    for index in range(2):
        service.send(
            "target_request", {"target_med": "a"}, {"goal_id": "g", "request_id": str(index)}
        )
    assert [r[2]["seq"] for r in raw.sent] == [1, 2]
    assert service.next_event() == (
        "plan_done",
        {"ok": True},
        {**stamp(1), "goal_id": "g", "request_id": "r"},
    )


def test_same_turn_dialogue_waits_for_reset_in_actual_boundary():
    """TC-6/7: Node's lexical input ordering cannot deliver initial dialogue before reset."""
    from aisle.harness.p3_task_boundary import TaskBoundary

    service, _ = transport(
        turn_events(
            [
                event(
                    "human_msg", pa.array(['{"kind":"request","med":"a","text":"a?"}']), goal_id="g"
                ),
                reset(),
            ]
        )
    )
    boundary = TaskBoundary(binding_id="fixture", medicines=["a"], max_motion_requests=1)
    assert boundary.observe(*service.next_event())["kind"] == "reset"
    assert boundary.observe(*service.next_event())["kind"] == "request"


def test_turn_replay_is_refused():
    """TC-2; TRT-9: duplicate coordinator turns cannot open a new action interval."""
    service, _ = transport([*turn_events([reset()]), *turn_events([reset()])])
    service.next_event()
    with pytest.raises(ValueError, match="monotonic"):
        service.next_event()


def test_controller_feedback_uses_fixed_wire_fields_and_goal_correlation():
    """TC-2/7; HAR-3: controller feedback travels as stamped Arrow with separate counters."""
    service, raw = transport(turn_events([reset()]))
    service.next_event()
    payload = dict(t=1.0, phase="delivering", retries=2, dialogue_corrections=1)
    service.send("episode_feedback", payload, {"goal_id": "g"})
    assert raw.sent[0][0] == "episode_feedback"
    assert json.loads(raw.sent[0][1][0]) == payload
    assert raw.sent[0][2] == {**stamp(), "goal_id": "g"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("t", float("nan")),
        ("t", -1),
        ("phase", "success"),
        ("retries", 9),
        ("retries", True),
        ("dialogue_corrections", -1),
    ],
)
def test_malformed_feedback_is_refused(field, value):
    """TC-7/8; HAR-3: feedback cannot inject outcome labels or invalid retry accounting."""
    service, raw = transport(turn_events([reset()]))
    service.next_event()
    payload = dict(t=0, phase="confirming", retries=0, dialogue_corrections=0)
    payload[field] = value
    with pytest.raises(ValueError):
        service.send("episode_feedback", payload, {"goal_id": "g"})
    assert not raw.sent
