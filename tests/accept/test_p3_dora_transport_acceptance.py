"""Public task boundary over native Dora; synthetic inputs, no physics or provider."""

import json
import shlex
import shutil
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.accept
ROOT = Path(__file__).resolve().parents[2]


def test_public_task_requests_cross_native_dora_turns(tmp_path, dataflow):
    """CON-4; TC-2/6/7; VAL-6: Arrow task requests retain identities and host turn accounting."""
    if shutil.which("dora") is None:
        pytest.skip("Dora CLI required")
    pytest.importorskip("dora")
    base = tmp_path.resolve()
    host = base / "host.py"
    host.write_text("""
import json
from pathlib import Path
from aisle.turn_node import Node
from aisle.harness.p3_dora_transport import DoraTaskTransport
from aisle.harness.p3_task_boundary import TaskBoundary

transport = DoraTaskTransport(Node(), turn_epoch=7)
boundary = TaskBoundary(binding_id="native-fixture", medicines=["a"], max_motion_requests=1)
events = []
while True:
    try:
        raw = transport.next_event()
    except StopIteration:
        break
    event = boundary.observe(*raw)
    if event is None:
        continue
    events.append(event)
    if event["kind"] in ("request", "confirm_reply"):
        command = boundary.action(dict(revision=boundary.revision, med="a",
                          kind="confirm" if event["kind"] == "request" else "deliver"))
        transport.send(command["topic"], command["payload"], command["metadata"])
assert [r["kind"] for r in events] == ["reset", "request", "confirm_reply", "motion_returned"]
assert boundary.active is None and boundary.motion_requests == 1
assert "PRIVATE-SENTINEL" not in json.dumps(events)
Path("public.json").write_text(json.dumps(events))
""")
    driver = base / "driver.py"
    driver.write_text("""
import json
from pathlib import Path
import pyarrow as pa
from dora import Node

node = Node()
stage = -1
seen = {}
rows = []
def send_stage(stage):
    meta = dict(env_id=0, seq=stage+1, turn_epoch=7, turn_id=stage, sim_time_ns=stage*10_000_000)
    if stage == 0:
        node.send_output("reset_done", pa.array([1], type=pa.uint32()),
                         dict(meta, seed=300, request_id="PRIVATE-SENTINEL"))
        dialogue = dict(kind="request", med="a", text="a?")
        node.send_output("human_msg", pa.array([json.dumps(dialogue)]),
                         dict(meta, goal_id="private-goal", secret="PRIVATE-SENTINEL"))
        names = ["human_msg", "reset_done"]
    elif stage == 1:
        dialogue = dict(kind="confirm_reply", med="a", text="yes")
        node.send_output("human_msg", pa.array([json.dumps(dialogue)]),
                         dict(meta, goal_id="private-goal"))
        names = ["human_msg"]
    else:
        node.send_output("plan_done", pa.array(['{"ok": true}']), dict(meta, **identity))
        names = ["plan_done"]
    node.send_output("turn", pa.array([stage], type=pa.uint64()),
                     dict(meta, target_node="task", expected_inputs=names,
                          expected_counts=[1]*len(names)))

for event in node:
    if event["type"] != "INPUT": continue
    topic = event["id"]
    if topic == "tick" and stage == -1:
        stage = 0
        send_stage(stage)
    elif topic in ("robot_msg", "target_request", "turn_done"):
        meta = event["metadata"]
        assert meta["turn_epoch"] == 7 and meta["turn_id"] == stage
        seen[topic] = True
        rows.append(dict(topic=topic, metadata=dict(meta)))
        if topic == "target_request":
            assert json.loads(event["value"][0].as_py()) == {"target_med": "a"}
            identity = {k: meta[k] for k in ("goal_id", "request_id")}
        expected = {"turn_done"} | [{"robot_msg"}, {"target_request"}, set()][stage]
        if set(seen) == expected:
            if stage == 2: break
            seen = {}
            stage += 1
            send_stage(stage)
Path("wire.json").write_text(json.dumps(rows, default=str))
""")

    def q(source):
        return dict(source=source, queue_size=100)

    graph = base / "graph.yaml"
    graph.write_text(
        yaml.safe_dump(
            dict(
                nodes=[
                    dict(
                        id="driver",
                        path=sys.executable,
                        args=shlex.join([str(driver)]),
                        inputs={
                            "tick": "dora/timer/millis/100",
                            **{
                                k: q(f"task/{k}")
                                for k in ("robot_msg", "target_request", "turn_done")
                            },
                        },
                        outputs=["reset_done", "human_msg", "plan_done", "turn"],
                    ),
                    dict(
                        id="task",
                        path=sys.executable,
                        args=shlex.join([str(host)]),
                        env=dict(
                            PYTHONPATH=str(ROOT / "src"),
                            AISLE_LOCKSTEP="1",
                            AISLE_TURN_NODE="task",
                            AISLE_TURN_OUTPUTS="robot_msg,target_request,return_request,turn_done",
                        ),
                        inputs={
                            k: q(f"driver/{k}")
                            for k in ("reset_done", "human_msg", "plan_done", "turn")
                        },
                        outputs=["robot_msg", "target_request", "return_request", "turn_done"],
                    ),
                ]
            )
        )
    )
    result = dataflow.run(graph, timeout_s=30)
    assert not result.timed_out and result.returncode == 0, result.stderr
    rows = json.loads((base / "wire.json").read_text())
    done = [r["metadata"] for r in rows if r["topic"] == "turn_done"]
    assert [r["turn_id"] for r in done] == [0, 1, 2]
    assert [r["emitted_counts"] for r in done] == [[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
    events = json.loads((base / "public.json").read_text())
    assert events[-1]["physical_completion"] == "unknown"
