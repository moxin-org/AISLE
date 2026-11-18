"""Experimental graph construction preserves the frozen observation/guard boundary."""

import json
from pathlib import Path

import pytest
import yaml

from aisle.harness.pear import main, prepare_graph

pytestmark = pytest.mark.unit


def test_prepared_graph_keeps_rgb_only_and_guarded_motion(tmp_path):
    """TC-9, VAL-5/6, CON-7: no privileged policy edge and no frozen-file edits."""
    report = prepare_graph(tmp_path / "run", "http://localhost:8000/v1", "qwen3_8_27b")
    assert report["ok"] and not report["attested"]
    graph = yaml.safe_load(Path(report["graph"]).read_text())
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["dora-genesis"]["env"]["AISLE_PERCEPTION"] == "L2"
    assert nodes["dora-genesis"]["env"]["AISLE_SIM_ENGINE"] == "nexus"
    assert nodes["dora-genesis"]["env"]["AISLE_SCENE"] == "pharmacy"
    assert nodes["dora-genesis"]["env"]["AISLE_N_ENVS"] == "1"
    assert nodes["task-state-machine"]["env"]["AISLE_TASK_TIER"] == "T1"
    policy = nodes["pear-policy"]
    assert policy["env"]["AISLE_ENV_PIN"] == "0"
    assert policy["env"]["AISLE_PEAR_TIMEOUT_S"] == "180"
    assert set(policy["inputs"]) == {
        "turn",
        "rgb_overhead",
        "rgb_wrist",
        "joint_state",
        "target_request",
        "reset_done",
        "episode_result",
    }
    for port in ("joint_cmd", "gripper_cmd"):
        assert nodes["dora-genesis"]["inputs"][port]["source"] == f"budget-guard/{port}_safe"
        assert nodes["budget-guard"]["inputs"][port]["source"] == f"pear-policy/{port}"
    plan = json.loads(Path(report["turn_plan"]).read_text())
    assert "vla-policy-smolvla" not in plan["participants"]
    assert "policy_result" in plan["participants"]["pear-policy"]["outputs"]
    assert (
        plan["participants"]["pear-policy"]["inputs"]["episode_result"]["source"]
        == "verifier-oracle"
    )
    assert policy["env"]["AISLE_PEAR_LOCKSTEP"] == "1"
    assert not report["registered_skill"]


def test_prepare_refuses_overwrite_and_invalid_mode(tmp_path):
    """CON-8: fail before overwriting another run or guessing a measurement condition."""
    with pytest.raises(ValueError):
        prepare_graph(tmp_path / "run", "http://localhost/v1", "pear", mode="bad")
    prepare_graph(tmp_path / "run", "http://localhost/v1", "pear", mode="async-live")
    with pytest.raises(FileExistsError):
        prepare_graph(tmp_path / "run", "http://localhost/v1", "pear")


def test_generated_turn_plan_closes_without_a_forward_cycle(tmp_path):
    """TC-2, VAL-5: the full generated topology must admit a terminal turn commit."""
    from aisle.turns import TurnBarrier, TurnStamp, watermark_metadata

    result = prepare_graph(tmp_path / "run", "http://localhost/v1", "pear")
    plan = json.loads(Path(result["turn_plan"]).read_text())
    barrier = TurnBarrier(plan)
    stamp = TurnStamp(1, 0, 0)
    ready = barrier.open_bridge(watermark_metadata(stamp, dict.fromkeys(plan["bridge_outputs"], 0)))
    closed = set()
    while ready:
        participant = next(iter(ready))
        ready.pop(participant)
        counts = dict.fromkeys(plan["participants"][participant]["outputs"], 0)
        ready.update(barrier.close(participant, watermark_metadata(stamp, counts)))
        closed.add(participant)
    assert closed == set(plan["participants"])
    assert barrier.complete
    assert barrier.bridge_expected_inputs() == {"gripper_cmd": 0, "joint_cmd": 0, "reset": 0}


def test_cli_reports_json_failure(capsys, tmp_path):
    """CON-8: invalid endpoint produces JSON on stdout and nonzero status."""
    assert (
        main(
            [
                "prepare",
                "--out",
                str(tmp_path / "run"),
                "--endpoint",
                "file:///bad",
                "--model",
                "pear",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_local_http_options_reach_nano_graph_and_manifest(tmp_path):
    """CON-5, VAL-5: the local inference condition is explicit and reproducible."""
    from aisle.nodes.pear_nano import MODEL

    result = prepare_graph(
        tmp_path / "local",
        "http://127.0.0.1:8087/v1",
        "qwen3.8-27b-pear",
        embodiment="nano",
        nano_urdf=MODEL,
        enable_thinking=False,
        max_tokens=512,
    )
    graph = yaml.safe_load(Path(result["graph"]).read_text())
    env = next(n["env"] for n in graph["nodes"] if n["id"] == "pear-policy")
    assert env["AISLE_PEAR_THINKING"] == "0"
    assert env["AISLE_PEAR_MAX_TOKENS"] == "512"
    assert result["inference_options"] == {"enable_thinking": False, "max_tokens": 512}


def test_explicit_python_preserves_venv_symlink(tmp_path):
    """CON-2: resolving a venv interpreter to its base binary loses the environment."""
    interpreter = tmp_path / "venv-python"
    import sys

    interpreter.symlink_to(sys.executable)
    result = prepare_graph(tmp_path / "run", "http://localhost/v1", "pear", python=interpreter)
    graph = yaml.safe_load(Path(result["graph"]).read_text())
    assert all(node["path"] == str(interpreter.absolute()) for node in graph["nodes"])


def test_nano_graph_binds_real_model_three_cameras_and_separate_guard(tmp_path):
    """TC-5/9, BG-1/2: Nano never runs through a Franka bridge or its seven-axis guard."""
    from aisle.nodes.pear_nano import MODEL, NanoAdapter

    result = prepare_graph(
        tmp_path / "nano",
        "http://localhost/v1",
        "pear",
        embodiment="nano",
        nano_urdf=MODEL,
        instruction="raise left tool",
    )
    graph = yaml.safe_load(Path(result["graph"]).read_text())
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert set(nodes) == {"nano-sim", "nano-guard", "pear-policy", "turn-barrier"}
    assert nodes["nano-sim"]["inputs"]["joint_cmd"]["source"] == "nano-guard/joint_cmd_safe"
    assert nodes["nano-guard"]["inputs"]["joint_cmd"]["source"] == "pear-policy/joint_cmd"
    assert {k for k in nodes["pear-policy"]["inputs"] if k.startswith("rgb_")} == {
        "rgb_head",
        "rgb_left_wrist",
        "rgb_right_wrist",
    }
    assert result["robot_model_sha256"] == NanoAdapter().sha256
    assert result["embodiment"] == "nano" and not result["attested"]
    assert result["task_success_verifier"] is False
    assert nodes["pear-policy"]["env"]["AISLE_PEAR_STOP_ON_RESULT"] == "1"


@pytest.mark.parametrize("embodiment", ["franka", "nano"])
def test_generated_watchdog_accepts_the_inference_budget(tmp_path, embodiment):
    """TC-2, CON-5: a generated turn barrier must start with valid watchdog budgets."""
    from aisle.nodes.pear_nano import MODEL
    from aisle.turns import TurnWatchdog

    report = prepare_graph(
        tmp_path / "run", "http://localhost/v1", "pear", embodiment=embodiment, nano_urdf=MODEL
    )
    graph = yaml.safe_load(Path(report["graph"]).read_text())
    env = next(n for n in graph["nodes"] if n["id"] == "turn-barrier")["env"]
    TurnWatchdog(
        ordinary_s=float(env["AISLE_TURN_WATCHDOG_S"]),
        verdict_s=float(env.get("AISLE_VERDICT_TURN_WATCHDOG_S", "15")),
        clock=lambda: 0,
    )


def test_cli_reports_json_failure_for_an_unparseable_nano_urdf(capsys, tmp_path):
    """CON-8: a Nano URDF that is not well-formed XML is refused as JSON, not a traceback."""
    urdf = tmp_path / "broken.urdf"
    urdf.write_text("<robot><joint")
    status = main(
        [
            "prepare",
            "--out",
            str(tmp_path / "run"),
            "--endpoint",
            "http://127.0.0.1:9/v1",
            "--model",
            "pear",
            "--embodiment",
            "nano",
            "--nano-urdf",
            str(urdf),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert status == 1
    assert report["ok"] is False
    assert "URDF" in report["error"]
    assert not (tmp_path / "run").exists()
