"""Actual Nano URDF + Nexus/Metal + Dora; model is an explicitly labeled HTTP stand-in."""

import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from aisle.harness.pear import prepare_graph
from aisle.nodes.pear_nano import NanoAdapter

pytestmark = pytest.mark.accept


def test_real_nano_nexus_observe_act_observe(tmp_path):
    """TC-1/2/3/5/9, BG-1/2: three real renders, dual-arm IK, guarded measured feedback."""
    values = [
        os.environ.get(name)
        for name in ("AISLE_PEAR_TEST_DORA", "AISLE_PEAR_TEST_PYTHON", "AISLE_PEAR_TEST_NANO_URDF")
    ]
    if not all(values):
        pytest.skip("set Dora 1.0.1, Nexus Python, and the Nano URDF with meshes for this test")
    dora, python, urdf = values
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            action = {}
            for side in ("left", "right"):
                action[side + "_arm"] = {"x": 0, "y": 0, "z": 0.2, "pitch": 0, "roll": 0}
                action[side + "_gripper"] = {"angle": 0}
            content = "```json\n" + json.dumps(action) + "\n```"
            encoded = json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            ).encode()
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
    proc = None
    try:
        result = prepare_graph(
            tmp_path / "run",
            f"http://127.0.0.1:{server.server_port}/v1",
            "nano-stand-in",
            embodiment="nano",
            nano_urdf=Path(urdf),
            python=Path(python),
            duration_s=3,
            instruction="raise both tools slightly",
        )
        run = Path(result["graph"]).parent
        with (run / "dora.log").open("w") as log:
            proc = subprocess.Popen(
                [dora, "run", result["graph"]], stdout=log, stderr=log, start_new_session=True
            )
            deadline = time.monotonic() + 120
            while proc.poll() is None:
                captured = (run / "dora.log").read_text()
                assert "Traceback (most recent call last)" not in captured, captured[-6000:]
                assert time.monotonic() < deadline, captured[-6000:]
                time.sleep(0.1)
        assert proc.returncode == 0, (run / "dora.log").read_text()[-6000:]
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    rows = [json.loads(line) for line in (run / "nano-runtime.jsonl").read_text().splitlines()]
    replies = [r for r in rows if r.get("event") == "model_reply"]
    assert len(replies) >= 2 and all(r["accepted"] for r in replies), replies
    assert len(requests) >= 2
    first, second = [
        json.loads(r["messages"][-1]["content"][0]["text"].split("\n")[0]) for r in requests[:2]
    ]
    for side in ("left", "right"):
        assert second[side + "_arm"]["xyz_cm"][2] > first[side + "_arm"]["xyz_cm"][2] + 0.1
    robot = NanoAdapter(Path(urdf))
    commands = np.array([r["q"] for r in rows if r.get("event") == "guarded_command"])
    assert len(commands) > 3
    assert np.all(commands >= robot.lower - 1e-6) and np.all(commands <= robot.upper + 1e-6)
    assert np.max(np.abs(np.diff(commands, axis=0))) <= 0.00301
    assert np.max(np.abs(commands[:, robot.held] - commands[0, robot.held])) < 1e-6
    measured = np.array([r["q"] for r in rows if r.get("event") == "joint_state"])
    assert np.max(np.abs(measured[:, robot.held] - measured[0, robot.held])) < 0.05
    assert np.all(measured >= robot.lower - 0.001) and np.all(measured <= robot.upper + 0.001)
    assert len(list((run / "traces").glob("*/*.png"))) >= 6
    scene = json.loads((run / "nano-scene.json").read_text())
    assert -0.04 < scene["floor_z"] < -0.03 and scene["fixed_base"] is True
