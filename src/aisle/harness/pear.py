"""Prepare an unregistered PEAR bring-up graph; no benchmark-acceptance claim (ADR-73)."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import yaml

from aisle.nodes.pear_backend import ChatClient

ROOT = Path(__file__).resolve().parents[3]
OLD = "vla-policy-smolvla"
NEW = "pear-policy"


def _rename(value):
    if isinstance(value, dict):
        return {_rename(k): _rename(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, str) and (value == OLD or value.startswith(OLD + "/")):
        return NEW + value[len(OLD) :]
    return value


def prepare_graph(
    output_dir: Path,
    endpoint: str | None,
    model: str,
    *,
    mode="lockstep-eval",
    sim_engine="nexus",
    python: Path | None = None,
    model_revision="unverified",
    embodiment="franka",
    nano_urdf: Path | None = None,
    instruction="Observe the objects and raise the left tool slightly.",
    duration_s=30,
    provider="chat-completions",
    codex_executable="codex",
    reasoning_effort="medium",
    enable_thinking=True,
    max_tokens=4096,
) -> dict:
    provider_env = {"AISLE_PEAR_PROVIDER": provider}
    if provider == "chat-completions":
        if not endpoint:
            raise ValueError("Chat Completions requires --endpoint")
        ChatClient(endpoint, model, enable_thinking=enable_thinking, max_tokens=max_tokens)
        provider_env["AISLE_PEAR_ENDPOINT"] = endpoint
        provider_env["AISLE_PEAR_THINKING"] = "1" if enable_thinking else "0"
        provider_env["AISLE_PEAR_MAX_TOKENS"] = str(max_tokens)
    elif provider == "codex":
        from aisle.nodes.pear_codex import CodexClient

        client = CodexClient(model, executable=codex_executable, effort=reasoning_effort)
        provider_env.update(
            AISLE_CODEX_EXECUTABLE=client.executable, AISLE_CODEX_EFFORT=reasoning_effort
        )
    else:
        raise ValueError("unsupported inference provider")
    if mode not in {"lockstep-eval", "async-live"} or sim_engine not in {
        "nexus",
        "genesis",
        "rapier",
    }:
        raise ValueError("unsupported inference condition or simulator")
    if python is not None and not python.is_file():
        raise ValueError("--python must name an existing interpreter")
    if embodiment == "nano":
        from aisle.harness.pear_nano import prepare_nano

        if sim_engine != "nexus" or nano_urdf is None:
            raise ValueError("Nano requires --sim-engine nexus and --nano-urdf with its meshes")
        return prepare_nano(
            output_dir,
            model,
            nano_urdf,
            mode,
            python,
            model_revision,
            instruction,
            duration_s,
            provider_env,
        )
    if embodiment != "franka":
        raise ValueError("unsupported PEAR embodiment")
    graph = _rename(yaml.safe_load((ROOT / "graphs/eval_vla_smolvla_t1.yaml").read_text()))
    plan = _rename(json.loads((ROOT / "graphs/turn_plans/eval_vla_smolvla_t1.json").read_text()))
    nodes = {node["id"]: node for node in graph["nodes"]}
    output_dir = output_dir.resolve()
    plan_path = output_dir / "turn-plan.json"
    graph_path = output_dir / "dataflow.yaml"
    for node in graph["nodes"]:
        node["path"] = str((ROOT / "graphs" / node["path"]).resolve())
        node.setdefault("env", {}).update(
            AISLE_EMBODIMENT="franka",
            AISLE_TIER="T1",
            AISLE_TASK_TIER="T1",
            AISLE_ENV_PIN="0",
            AISLE_N_ENVS="1",
            AISLE_SCENE="pharmacy",
        )
    bridge = nodes["dora-genesis"]
    bridge["env"].update(AISLE_PERCEPTION="L2", AISLE_SIM_ENGINE=sim_engine)
    policy = nodes[NEW]
    policy["path"] = str(ROOT / "src/aisle/nodes/pear_policy.py")
    policy["inputs"]["episode_result"] = {
        "source": "verifier-oracle/episode_result",
        "queue_size": 100,
    }
    policy["outputs"] = ["joint_cmd", "gripper_cmd", "policy_result", "turn_done"]
    policy["env"].update(
        **provider_env,
        AISLE_PEAR_MODEL=model,
        AISLE_PEAR_MODEL_REVISION=model_revision,
        AISLE_PEAR_LOCKSTEP="1" if mode == "lockstep-eval" else "0",
        AISLE_PEAR_TIMEOUT_S="180",
        AISLE_PEAR_TRACE_DIR=str(output_dir / "traces"),
        AISLE_TURN_OUTPUTS=",".join(policy["outputs"]),
    )
    barrier = nodes["turn-barrier"]
    barrier["env"].update(
        AISLE_TURN_PLAN=str(plan_path),
        AISLE_TURN_WATCHDOG_S="210",
        AISLE_VERDICT_TURN_WATCHDOG_S="210",
    )
    nodes["rollout-client"]["env"].update(
        AISLE_RESULTS=str(output_dir / "episodes.jsonl"), AISLE_SEEDS="0", AISLE_TIMEOUT_S="30"
    )
    participant = plan["participants"][NEW]
    participant["outputs"] = sorted(policy["outputs"])
    participant["inputs"]["episode_result"] = {
        "edge": "forward",
        "output": "episode_result",
        "source": "verifier-oracle",
    }
    if python is not None:
        for node in graph["nodes"]:
            node["args"], node["path"] = shlex.quote(node["path"]), str(python.absolute())
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
    report = {
        "ok": True,
        "graph": str(graph_path),
        "turn_plan": str(plan_path),
        "condition": mode,
        "sim_engine": sim_engine,
        "model": model,
        "provider": provider,
        "model_revision": model_revision,
        "inference_options": (
            {"enable_thinking": enable_thinking, "max_tokens": max_tokens}
            if provider == "chat-completions"
            else None
        ),
        "attested": False,
        "registered_skill": False,
        "notice": (
            "Experimental bring-up only. Not harness-validated, registered or benchmark-accepted."
        ),
    }
    (output_dir / "integration.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None) -> int:
    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--endpoint", help="Vision-capable Chat Completions base URL (HTTP provider only)"
    )
    parser.add_argument("--model", required=True, help="Inference model name")
    parser.add_argument("--model-revision", default="unverified")
    parser.add_argument(
        "--provider", choices=["chat-completions", "codex"], default="chat-completions"
    )
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument(
        "--no-thinking", action="store_true", help="Disable HTTP model thinking explicitly"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="HTTP model output token budget"
    )
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default="medium"
    )
    parser.add_argument("--mode", choices=["lockstep-eval", "async-live"], default="lockstep-eval")
    parser.add_argument("--sim-engine", choices=["nexus", "genesis", "rapier"], default="nexus")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--embodiment", choices=["franka", "nano"], default="franka")
    parser.add_argument("--nano-urdf", type=Path)
    parser.add_argument(
        "--instruction", default="Observe the objects and raise the left tool slightly."
    )
    parser.add_argument("--duration-s", type=int, default=30)
    try:
        args = parser.parse_args(argv)
        report = prepare_graph(
            args.out,
            args.endpoint,
            args.model,
            mode=args.mode,
            sim_engine=args.sim_engine,
            python=args.python,
            model_revision=args.model_revision,
            embodiment=args.embodiment,
            nano_urdf=args.nano_urdf,
            instruction=args.instruction,
            duration_s=args.duration_s,
            provider=args.provider,
            codex_executable=args.codex_executable,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=not args.no_thinking,
            max_tokens=args.max_tokens,
        )
    except (OSError, ValueError) as error:
        report = {"ok": False, "error": str(error)}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
