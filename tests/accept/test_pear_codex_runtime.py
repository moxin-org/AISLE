"""Opt-in authenticated Codex -> real Nano/Nexus -> fresh RGB feedback check."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aisle.harness.pear import prepare_graph
from aisle.nodes.pear_nano import NanoAdapter

pytestmark = pytest.mark.accept

INSTRUCTION = (
    "Inspect the RGB cameras and briefly name the visible cube colors. Select the arm "
    "best placed to approach the red cube. Make small non-contact probes toward it, "
    "at most 0.3 cm per action, using the images to choose the arm and direction. "
    "Keep the other arm and both grippers still. Reobserve after each action: compare "
    "the fresh images and measured tool state with the previous observation, and "
    "refine your next move. If a direction is uncertain, probe cautiously or hold. "
    "Do not attempt contact, grasping, or declare task success."
)


def test_codex_closes_real_nano_rgb_motion_feedback_loop(tmp_path):
    """TC-9, BG-1/2, CON-2/5: actual model decisions and fresh measured feedback."""
    if os.environ.get("AISLE_PEAR_TEST_CODEX") != "1":
        pytest.skip("opt in to authenticated model requests with AISLE_PEAR_TEST_CODEX=1")
    dora, python, urdf = [
        os.environ.get(name)
        for name in ("AISLE_PEAR_TEST_DORA", "AISLE_PEAR_TEST_PYTHON", "AISLE_PEAR_TEST_NANO_URDF")
    ]
    if not all((dora, python, urdf)):
        pytest.skip("requires the task-owned Dora/Nexus runtime and Nano model with meshes")
    model = os.environ.get("AISLE_PEAR_TEST_CODEX_MODEL", "gpt-6.1-terra")
    result = prepare_graph(
        tmp_path / "run",
        None,
        model,
        provider="codex",
        embodiment="nano",
        nano_urdf=Path(urdf),
        python=Path(python),
        duration_s=6,
        instruction=INSTRUCTION,
        mode="lockstep-eval",
    )
    run = Path(result["graph"]).parent
    process = None
    try:
        with (run / "dora.log").open("w") as log:
            process = subprocess.Popen(
                [dora, "run", result["graph"]], stdout=log, stderr=log, start_new_session=True
            )
            deadline = time.monotonic() + 600
            while process.poll() is None:
                captured = (run / "dora.log").read_text()
                assert "Traceback (most recent call last)" not in captured, captured[-6000:]
                assert time.monotonic() < deadline, captured[-6000:]
                time.sleep(0.1)
        assert process.returncode == 0, (run / "dora.log").read_text()[-6000:]
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    events = [json.loads(line) for line in (run / "traces/events.jsonl").read_text().splitlines()]
    replies = [e for e in events if e["event"] == "model_reply"]
    assert len(replies) >= 2, events
    assert sum(r["accepted"] for r in replies) >= 2, replies
    assert not any(e["event"] == "inference_error" for e in events), events
    for reply in replies:
        assert reply["provider"] == "codex" and reply["model"] == model
        assert reply["usage"]["tool_calls"] == 0
        assert reply["usage"]["environments"] == []
        assert reply["usage"]["inputTokens"] > 0
        assert reply["usage"]["thread_id"] and reply["usage"]["turn_id"]
    requests = [json.loads(p.read_text()) for p in sorted((run / "traces").glob("*/request.json"))]
    states = [json.loads(r["messages"][-1]["content"][0]["text"].split("\n")[0]) for r in requests]
    deltas = {
        side: (
            np.array(states[-1][side + "_arm"]["xyz_cm"])
            - np.array(states[0][side + "_arm"]["xyz_cm"])
        )
        for side in ("left", "right")
    }
    assert max(np.linalg.norm(d) for d in deltas.values()) > 0.1, deltas
    snapshots = sorted((run / "traces").glob("*/request.json"))
    changed_pixels = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        with Image.open(snapshots[0].parent / f"{camera}.png") as image:
            first = np.array(image)
        with Image.open(snapshots[-1].parent / f"{camera}.png") as image:
            last = np.array(image)
        changed_pixels[camera] = int(np.count_nonzero(np.any(first != last, axis=2)))
    assert max(changed_pixels.values()) > 10, changed_pixels
    assert len(requests[-1]["messages"]) > len(requests[0]["messages"])
    assert requests[-1]["observation_ns"] > requests[0]["observation_ns"]
    assert any("reached" in r["messages"][-1]["content"][0]["text"] for r in requests[1:])
    rows = [json.loads(line) for line in (run / "nano-runtime.jsonl").read_text().splitlines()]
    robot = NanoAdapter(Path(urdf))
    commands = np.array([r["q"] for r in rows if r.get("event") == "guarded_command"])
    measured = np.array([r["q"] for r in rows if r.get("event") == "joint_state"])
    assert len(commands) > 3
    assert np.all(commands >= robot.lower - 1e-6) and np.all(commands <= robot.upper + 1e-6)
    assert np.max(np.abs(np.diff(commands, axis=0))) <= 0.00301
    assert np.max(np.abs(commands[:, robot.held] - commands[0, robot.held])) < 1e-6
    assert np.all(measured >= robot.lower - 0.001) and np.all(measured <= robot.upper + 0.001)
    summary = {
        "ok": True,
        "provider": "codex",
        "model": model,
        "actual_vision_model": True,
        "actual_pear_checkpoint": False,
        "accepted_actions": sum(r["accepted"] for r in replies),
        "requests": len(requests),
        "guarded_commands": len(commands),
        "observed_tool_delta_cm": {side: d.tolist() for side, d in deltas.items()},
        "changed_pixels_first_to_last": changed_pixels,
        "inference_wall_s": [r["usage"]["wall_time_s"] for r in replies],
        "condition": "lockstep-eval",
        "task_success_evaluated": False,
        "attested": False,
        "evidence": str(run),
    }
    (run / "codex-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
