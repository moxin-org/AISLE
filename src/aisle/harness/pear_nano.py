"""Generate a private Nano/Nexus bring-up graph without touching frozen profiles."""

import json
import shlex
from pathlib import Path

import yaml

from aisle.nodes.pear_nano import NanoAdapter


def prepare_nano(out, model, urdf, mode, python, revision, instruction, duration_s, provider_env):
    robot = NanoAdapter(urdf)
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 8192:
        raise ValueError("Nano instruction must be nonempty bounded text")
    if type(duration_s) is not int or not 1 <= duration_s <= 300:
        raise ValueError("Nano bring-up duration must be 1..300 simulator seconds")
    out = Path(out).resolve()
    root = Path(__file__).resolve().parents[3]
    plan_path = out / "turn-plan.json"
    common = {
        "AISLE_EMBODIMENT": "nano",
        "AISLE_NANO_URDF": str(robot.path),
        "AISLE_NANO_MODEL_SHA256": robot.sha256,
        "AISLE_ENV_PIN": "0",
        "AISLE_LOCKSTEP": "1",
        "AISLE_PEAR_TRACE_DIR": str(out / "traces"),
    }

    def edge(source):
        return {"source": source, "queue_size": 100, "queue_policy": "backpressure"}

    def node(name, script, inputs, outputs, env=None):
        path = str(root / "src/aisle/nodes" / script)
        result = {
            "id": name,
            "path": path,
            "inputs": inputs,
            "outputs": outputs,
            "env": {
                **common,
                "AISLE_TURN_NODE": name,
                "AISLE_TURN_OUTPUTS": ",".join(outputs),
                **(env or {}),
            },
        }
        if python is not None:
            result.update(path=str(python.absolute()), args=shlex.quote(path))
        return result

    observations = [
        "joint_state",
        *["rgb_" + c for c in robot.cameras],
        "reset_done",
        "target_request",
        "episode_result",
    ]
    bridge = node(
        "nano-sim",
        "pear_nano_bridge.py",
        {
            "tick": "dora/timer/millis/10",
            "joint_cmd": edge("nano-guard/joint_cmd_safe"),
            "turn_commit": edge("turn-barrier/turn_commit"),
            "policy_result": edge("pear-policy/policy_result"),
            "guard_result": edge("nano-guard/violation"),
        },
        [*observations, "sim_turn"],
        {
            "AISLE_NANO_INSTRUCTION": instruction,
            "AISLE_NANO_DURATION_S": str(duration_s),
            "AISLE_NANO_RUN_DIR": str(out),
        },
    )
    policy = node(
        "pear-policy",
        "pear_policy.py",
        {
            **{name: edge("nano-sim/" + name) for name in observations},
            "turn": edge("turn-barrier/turn"),
        },
        ["joint_cmd", "gripper_cmd", "policy_result", "turn_done"],
        {
            **provider_env,
            "AISLE_PEAR_MODEL": model,
            "AISLE_PEAR_MODEL_REVISION": revision,
            "AISLE_PEAR_TIMEOUT_S": "180",
            "AISLE_PEAR_LOCKSTEP": "1" if mode == "lockstep-eval" else "0",
            "AISLE_PEAR_STOP_ON_RESULT": "1",
        },
    )
    guard = node(
        "nano-guard",
        "pear_nano_guard.py",
        {
            "joint_cmd": edge("pear-policy/joint_cmd"),
            "joint_state": edge("nano-sim/joint_state"),
            "reset_done": edge("nano-sim/reset_done"),
            "turn": edge("turn-barrier/turn"),
        },
        ["joint_cmd_safe", "violation", "turn_done"],
    )
    plan = {
        "bridge": "nano-sim",
        "bridge_outputs": bridge["outputs"],
        "bridge_inputs": {"joint_cmd": {"source": "nano-guard", "output": "joint_cmd_safe"}},
        "participants": {},
        "barrier": "turn-barrier",
    }
    for participant in (policy, guard):
        inputs = {}
        for name, link in participant["inputs"].items():
            if name == "turn":
                continue
            source, output = link["source"].split("/")
            inputs[name] = {"source": source, "output": output, "edge": "forward"}
        plan["participants"][participant["id"]] = {
            "inputs": inputs,
            "outputs": participant["outputs"],
            "verdict_bearing": False,
        }
    barrier = node(
        "turn-barrier",
        "turn_barrier.py",
        {
            "tick": "dora/timer/millis/100",
            "sim_turn": edge("nano-sim/sim_turn"),
            "done_policy": edge("pear-policy/turn_done"),
            "done_guard": edge("nano-guard/turn_done"),
        },
        ["turn", "turn_commit"],
        {
            "AISLE_TURN_PLAN": str(plan_path),
            "AISLE_TURN_WATCHDOG_S": "210",
            "AISLE_VERDICT_TURN_WATCHDOG_S": "210",
        },
    )
    graph = {"nodes": [bridge, policy, guard, barrier]}
    out.mkdir(parents=True, exist_ok=False)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    (out / "dataflow.yaml").write_text(yaml.safe_dump(graph, sort_keys=False))
    report = {
        "ok": True,
        "graph": str(out / "dataflow.yaml"),
        "turn_plan": str(plan_path),
        "condition": mode,
        "sim_engine": "nexus",
        "model": model,
        "provider": provider_env["AISLE_PEAR_PROVIDER"],
        "model_revision": revision,
        "inference_options": (
            {
                "enable_thinking": provider_env["AISLE_PEAR_THINKING"] == "1",
                "max_tokens": int(provider_env["AISLE_PEAR_MAX_TOKENS"]),
            }
            if provider_env["AISLE_PEAR_PROVIDER"] == "chat-completions"
            else None
        ),
        **robot.description,
        "attested": False,
        "registered_skill": False,
        "task_success_verifier": False,
        "base_control": "fixed for arm bring-up",
        "notice": "Experimental Nano arm runtime; no task-success or hardware claim.",
    }
    (out / "integration.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
