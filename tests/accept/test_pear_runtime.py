"""Real HTTP and Dora transport checks; stand-ins do NOT measure PEAR task quality."""

import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from aisle.nodes.pear_backend import ChatClient, parse_action

pytestmark = pytest.mark.accept
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def endpoint(request):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            time.sleep(getattr(request, "param", 0))
            action = {
                "arm": {"x": 0, "y": 0, "z": 0.2, "roll": 0, "pitch": 0, "yaw": 0},
                "gripper": {"width": 0},
            }
            result = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "```json\n" + json.dumps(action) + "\n```"},
                    }
                ],
                "usage": {"completion_tokens": 50},
            }
            encoded = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_http_protocol(endpoint):
    """CON-1/4, TC-1: actual HTTP request/response parses a finite cm/radian action."""
    url, requests = endpoint
    response = ChatClient(url, "stand-in").infer([{"role": "user", "content": "test"}], seed=4)
    assert parse_action(response.answer).position_cm == (0, 0, 0.2)
    assert requests[0]["seed"] == 4 and requests[0]["model"] == "stand-in"


WORLD = """
import json, os
import numpy as np
import pyarrow as pa
from dora import Node
from aisle.nodes.budget_guard import load_limits, gripper_to_fingers
from aisle.topics import make_sender
from aisle.turns import BridgeTurn, TurnStamp, watermark_metadata

node = Node()
send = make_sender(node)
limits = load_limits("franka")
q = np.array(limits.fallback_qpos, dtype=np.float32)
tick = 0
lockstep = os.environ.get("WORLD_LOCKSTEP") == "1"
active = None
ports = ["joint_state", "rgb_overhead", "rgb_wrist", "target_request", "reset_done"]

def publish():
    global tick, active
    tick += 1
    stamp = TurnStamp(1, tick-1, tick*10_000_000)
    meta = {"sim_time_ns":stamp.sim_time_ns, "env_id":0, "goal_id":"smoke-1"}
    if lockstep:
        meta.update(stamp.metadata())
        active = BridgeTurn(stamp)
    counts = dict.fromkeys(ports, 0)
    def emit(name, value, metadata):
        send(name, value, metadata)
        counts[name] += 1
    if tick == 1:
        emit("reset_done", pa.array([1],type=pa.uint32()), meta)
        request = {"instruction":"raise the empty gripper slightly"}
        emit("target_request",pa.array([json.dumps(request)]),meta)
    if tick % 10 == 0:
        pixels=pa.array(np.arange(18,dtype=np.uint8))
        image_meta={**meta,"h":2,"w":3,"enc":"rgb8"}
        emit("rgb_overhead",pixels,image_meta)
        emit("rgb_wrist",pixels,image_meta)
    emit("joint_state",pa.array(q,type=pa.float32()),meta)
    if lockstep:
        counts["sim_turn"] = 1
        send("sim_turn", pa.array([stamp.turn_id], type=pa.uint64()),
             watermark_metadata(stamp, counts))

with open(os.environ["RECORD"], "w", buffering=1) as output:
    for event in node:
        if event["type"] != "INPUT": continue
        name = event["id"]
        commands = None
        if name in {"joint_cmd", "gripper_cmd"}:
            if name == "joint_cmd":
                values = event["value"].to_numpy(zero_copy_only=False).copy()
                output.write(json.dumps({"event":"guarded_command", "q":values.tolist(),
                    "metadata":dict(event["metadata"])}, default=str) + "\\n")
            if lockstep:
                active.accept(name, event, event["metadata"])
                if active.ready_to_commit: commands = active.finish()
            else:
                commands = [(name, event)]
        elif name == "turn_commit":
            commands = active.commit(event["metadata"])
        elif name == "policy_result":
            output.write(event["value"][0].as_py() + "\\n")
        elif name == "violation":
            record = {"event":"guard_violation", "value":event["value"][0].as_py()}
            output.write(json.dumps(record) + "\\n")
        elif name == "tick":
            if not lockstep or tick == 0: publish()
        if commands is not None:
            for kind, command in commands:
                if kind == "joint_cmd":
                    q = command["value"].to_numpy(zero_copy_only=False).copy()
                elif kind == "gripper_cmd":
                    q[7:] = gripper_to_fingers(command["value"][0].as_py(), limits)
            if lockstep: publish()
"""


@pytest.mark.parametrize(
    "mode,endpoint,expected_acceptance",
    [("async-live", 0, True), ("lockstep-eval", 0.7, True), ("async-live", 0.7, False)],
    indirect=["endpoint"],
)
def test_dora_observe_act_observe_through_real_guard(tmp_path, endpoint, mode, expected_acceptance):
    """TC-1/2/3/6/9, BG-1/2/3: live Dora/HTTP round trips retain the actual guard."""
    dora, python = os.environ.get("AISLE_PEAR_TEST_DORA"), os.environ.get("AISLE_PEAR_TEST_PYTHON")
    if not dora or not python:
        pytest.skip("set AISLE_PEAR_TEST_DORA and AISLE_PEAR_TEST_PYTHON for Dora 1.0.1")
    url, requests = endpoint
    world = tmp_path / "world.py"
    world.write_text(WORLD)
    records, trace = tmp_path / "records.jsonl", tmp_path / "trace"
    node_env = {"AISLE_EMBODIMENT": "franka", "AISLE_LOCKSTEP": "0"}

    def node(name, script, inputs, outputs, extra=None):
        return {
            "id": name,
            "path": python,
            "args": str(script),
            "inputs": inputs,
            "outputs": outputs,
            "env": {**node_env, **(extra or {})},
        }

    world_outputs = ["joint_state", "rgb_overhead", "rgb_wrist", "target_request", "reset_done"]
    graph = {
        "nodes": [
            node(
                "world",
                world,
                {
                    "tick": "dora/timer/millis/10",
                    "joint_cmd": "budget-guard/joint_cmd_safe",
                    "gripper_cmd": "budget-guard/gripper_cmd_safe",
                    "policy_result": "pear-policy/policy_result",
                    "violation": "budget-guard/violation",
                },
                world_outputs,
                {"RECORD": str(records)},
            ),
            node(
                "pear-policy",
                ROOT / "src/aisle/nodes/pear_policy.py",
                {key: {"source": f"world/{key}", "queue_size": 100} for key in world_outputs},
                ["joint_cmd", "gripper_cmd", "policy_result"],
                {
                    "AISLE_PEAR_ENDPOINT": url,
                    "AISLE_PEAR_MODEL": "stand-in",
                    "AISLE_PEAR_TRACE_DIR": str(trace),
                },
            ),
            node(
                "budget-guard",
                ROOT / "src/aisle/nodes/budget_guard.py",
                {
                    "joint_cmd": "pear-policy/joint_cmd",
                    "gripper_cmd": "pear-policy/gripper_cmd",
                    "reset_done": "world/reset_done",
                    "tick": "dora/timer/millis/5000",
                },
                ["joint_cmd_safe", "gripper_cmd_safe", "violation", "guard_stats"],
            ),
        ]
    }
    if mode == "lockstep-eval":
        world_node, policy, guard = graph["nodes"]
        world_node["env"]["WORLD_LOCKSTEP"] = "1"
        world_node["outputs"] = [*world_outputs, "sim_turn"]
        world_node["inputs"]["turn_commit"] = "turn-barrier/turn_commit"
        for participant in (policy, guard):
            participant["inputs"]["turn"] = "turn-barrier/turn"
            participant["outputs"].append("turn_done")
            participant["env"].update(
                AISLE_LOCKSTEP="1",
                AISLE_TURN_NODE=participant["id"],
                AISLE_TURN_OUTPUTS=",".join(participant["outputs"]),
            )
        policy["env"]["AISLE_PEAR_LOCKSTEP"] = "1"
        guard["env"]["AISLE_TURN_WALL_OUTPUTS"] = "guard_stats"
        plan = {
            "bridge": "world",
            "bridge_outputs": world_node["outputs"],
            "bridge_inputs": {
                name: {"source": "budget-guard", "output": name + "_safe"}
                for name in ("joint_cmd", "gripper_cmd")
            },
            "participants": {},
        }
        for participant in (policy, guard):
            edges = {}
            for name, source in participant["inputs"].items():
                if name in {"turn", "tick"}:
                    continue
                source = source["source"] if isinstance(source, dict) else source
                producer, port = source.split("/")
                edges[name] = {"source": producer, "output": port, "edge": "forward"}
            plan["participants"][participant["id"]] = {
                "inputs": edges,
                "outputs": participant["outputs"],
            }
        graph["nodes"].append(
            node(
                "turn-barrier",
                ROOT / "src/aisle/nodes/turn_barrier.py",
                {
                    "tick": "dora/timer/millis/100",
                    "sim_turn": "world/sim_turn",
                    "done_policy": "pear-policy/turn_done",
                    "done_guard": "budget-guard/turn_done",
                },
                ["turn", "turn_commit"],
                {"AISLE_TURN_PLAN": json.dumps(plan)},
            )
        )
    graph_path = tmp_path / "flow.yaml"
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
    with (tmp_path / "runtime.log").open("w") as log:
        proc = subprocess.Popen(
            [dora, "run", str(graph_path)],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        rows = []
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.1)
                if records.exists():
                    rows = [json.loads(line) for line in records.read_text().splitlines()]
                    if sum(r.get("event") == "model_reply" for r in rows) >= 2:
                        break
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
    logs = (tmp_path / "runtime.log").read_text()
    replies = [r for r in rows if r.get("event") == "model_reply"]
    assert len(replies) >= 2, logs[-5000:]
    assert all(r["accepted"] == expected_acceptance for r in replies), replies
    commands = [r for r in rows if r.get("event") == "guarded_command"]
    assert len(requests) >= 2
    assert len(list(trace.glob("*/*.png"))) >= 4
    if not expected_acceptance:
        assert not commands
        assert all(r["feedback"] == "stale_observation" for r in replies)
        return
    assert commands and all(
        set(r["metadata"]) >= {"seq", "env_id", "sim_time_ns"} for r in commands
    )
    assert not any(r.get("event") == "guard_violation" for r in rows)
    if mode == "lockstep-eval":
        assert all("turn_id" in r["metadata"] for r in commands)
    assert all(r["goal_id"] == "smoke-1" for r in rows if r.get("event") == "model_reply")
    first = json.loads(requests[0]["messages"][-1]["content"][0]["text"].split("\n")[0])
    second = json.loads(requests[1]["messages"][-1]["content"][0]["text"].split("\n")[0])
    assert second["arm"]["z"] > first["arm"]["z"] + 0.1
    assert len(requests[1]["messages"]) > len(requests[0]["messages"])
