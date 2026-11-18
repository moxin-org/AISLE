"""Opt-in local PEAR checkpoint -> real Nano/Nexus -> fresh RGB feedback."""

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aisle.harness.pear import prepare_graph
from aisle.nodes.pear_nano import NanoAdapter

pytestmark = pytest.mark.accept
REVISION = "65176e435e5fa80a059ebfafaf8215e6ad00a83e"
INSTRUCTION = (
    "Inspect the RGB cameras and briefly name the visible cube colors. Select the arm "
    "best placed to approach the red cube. Make small non-contact probes toward it, "
    "at most 0.3 cm per action, using the images to choose the arm and direction. "
    "Keep the other arm and both grippers still. Reobserve after each action and "
    "compare fresh images and measured tool state with the previous observation. "
    "If a direction is uncertain, probe cautiously or hold. "
    "Do not attempt contact, grasping, or declare task success."
)


def test_local_pear_closes_nano_rgb_feedback_loop(tmp_path):
    """TC-9, BG-1/2, CON-2/5: actual local checkpoint, RGB decisions, measured motion."""
    endpoint = os.environ.get("AISLE_PEAR_TEST_OMINIX")
    if not endpoint:
        pytest.skip("set AISLE_PEAR_TEST_OMINIX to the local OminiX /v1 endpoint")
    values = [
        os.environ.get(name)
        for name in ("AISLE_PEAR_TEST_DORA", "AISLE_PEAR_TEST_PYTHON", "AISLE_PEAR_TEST_NANO_URDF")
    ]
    assert all(values), "Requires explicit Dora 1.0.1, Nexus Python and Nano URDF"
    dora, python, urdf = values
    robot = NanoAdapter(Path(urdf))
    with urllib.request.urlopen(endpoint.removesuffix("/v1") + "/health", timeout=5) as response:
        health = json.load(response)
    assert health["ok"] and health["backend"] == "ominix-mlx"
    provenance = health["provenance"]
    assert provenance["source_repo"] == "mit-han-lab/Qwen3.8-27B-PEAR"
    assert provenance["source_revision"] == REVISION
    assert len(provenance["source_sha256"]) == 2
    assert provenance["vision_dtype"] == "bfloat16"
    model = health["model"]
    result = prepare_graph(
        tmp_path / "run",
        endpoint,
        model,
        model_revision=REVISION,
        embodiment="nano",
        nano_urdf=Path(urdf),
        python=Path(python),
        duration_s=6,
        instruction=INSTRUCTION,
        mode="lockstep-eval",
        enable_thinking=False,
        max_tokens=512,
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
    assert sum(r["accepted"] for r in replies) >= 2, replies
    assert not any(e["event"] == "inference_error" for e in events), events
    for reply in replies:
        usage = reply["usage"]
        assert usage["backend"] == "ominix-mlx"
        assert usage["source_revision"] == REVISION
        assert usage["image_count"] >= 3 and usage["completion_tokens"] > 0
        if reply["accepted"]:
            action = robot.parse_action(reply["answer"])
            moving_arms = 0
            for side in ("left", "right"):
                arm = action[side + "_arm"]
                distance = np.linalg.norm([arm[key] for key in ("x", "y", "z")])
                assert distance <= 0.30001, action
                moving_arms += distance > 0
                assert action[side + "_gripper"]["angle"] == 0, action
            assert moving_arms <= 1, action
    snapshots = sorted((run / "traces").glob("*/request.json"))
    requests = [json.loads(path.read_text()) for path in snapshots]
    states = [json.loads(r["messages"][-1]["content"][0]["text"].split("\n")[0]) for r in requests]
    deltas = {
        side: np.array(states[-1][side + "_arm"]["xyz_cm"])
        - np.array(states[0][side + "_arm"]["xyz_cm"])
        for side in ("left", "right")
    }
    assert max(np.linalg.norm(d) for d in deltas.values()) > 0.1, deltas
    changed = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        with Image.open(snapshots[0].parent / f"{camera}.png") as image:
            first = np.array(image)
        with Image.open(snapshots[-1].parent / f"{camera}.png") as image:
            last = np.array(image)
        changed[camera] = int(np.count_nonzero(np.any(first != last, axis=2)))
    assert max(changed.values()) > 10, changed
    assert requests[-1]["observation_ns"] > requests[0]["observation_ns"]
    assert len(requests[-1]["messages"]) > len(requests[0]["messages"])
    assert any("reached" in r["messages"][-1]["content"][0]["text"] for r in requests[1:])
    rows = [json.loads(line) for line in (run / "nano-runtime.jsonl").read_text().splitlines()]
    commands = np.array([r["q"] for r in rows if r.get("event") == "guarded_command"])
    measured = np.array([r["q"] for r in rows if r.get("event") == "joint_state"])
    assert len(commands) > 3
    assert np.all(commands >= robot.lower - 1e-6) and np.all(commands <= robot.upper + 1e-6)
    assert np.max(np.abs(np.diff(commands, axis=0))) <= 0.00301
    assert np.max(np.abs(commands[:, robot.held] - commands[0, robot.held])) < 1e-6
    assert np.all(measured >= robot.lower - 0.001) and np.all(measured <= robot.upper + 0.001)
    summary = {
        "ok": True,
        "backend": "ominix-mlx",
        "model": model,
        "source_revision": REVISION,
        "actual_pear_checkpoint": True,
        "actual_vision_model": True,
        "nano_urdf": str(Path(urdf).resolve()),
        "nano_urdf_sha256": robot.sha256,
        "accepted_actions": sum(r["accepted"] for r in replies),
        "requests": len(requests),
        "guarded_commands": len(commands),
        "observed_tool_delta_cm": {side: d.tolist() for side, d in deltas.items()},
        "changed_pixels_first_to_last": changed,
        "inference_wall_s": [r["usage"]["wall_time_s"] for r in replies],
        "condition": "lockstep-eval",
        "enable_thinking": False,
        "max_tokens": 512,
        "task_success_evaluated": False,
        "attested": False,
        "evidence": str(run),
    }
    (run / "ominix-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
