"""T4 rollout-client completion over real dora transport, without physics."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.accept
ROOT = Path(__file__).resolve().parents[2]


def test_real_client_retains_both_goals_before_chain_completion(tmp_path, dataflow):
    """HAR-1/4; TC-7: the real client writes two goal identities for one corrected T4 seed."""
    if shutil.which("dora") is None:
        pytest.skip("dora CLI is required")
    from aisle.harness.rollout import t4_chain_completion

    results = tmp_path / "episodes.jsonl"
    driver = tmp_path / "driver.py"
    driver.write_text("""
import json
import pyarrow as pa
from dora import Node

node = Node()
pending = None
ticks = 0
for event in node:
    if event["type"] != "INPUT":
        continue
    metadata = event.get("metadata") or {}
    if event["id"] == "reset":
        if metadata.get("request_id") == "reset-cleanup":
            break
        node.send_output("reset_done", pa.array([1]),
                         dict(request_id=metadata["request_id"], env_id=0, sim_time_ns=0, seq=0))
    elif event["id"] == "episode_goal":
        goal_id = metadata["goal_id"]
        pending = (goal_id, 2 if goal_id.endswith("r") else 0)
        ticks = 0
    elif event["id"] == "tick" and pending is not None:
        ticks += 1
        if ticks <= pending[1]:
            continue
        goal_id = pending[0]
        recovery = goal_id.endswith("r")
        result = dict(goal_id=goal_id, seed=200, verifier="oracle",
                      status="fail" if recovery else "success",
                      failure="not_returned" if recovery else None,
                      t_end=60.0 if recovery else 15.0)
        node.send_output("episode_result", pa.array([json.dumps(result)]),
                         dict(goal_id=goal_id, env_id=0,
                              sim_time_ns=60_000_000_000 if recovery else 15_000_000_000,
                              seq=1 if recovery else 0))
        pending = None
""")
    graph = tmp_path / "graph.yaml"
    graph.write_text(
        yaml.safe_dump(
            dict(
                nodes=[
                    dict(
                        id="driver",
                        path=sys.executable,
                        args=str(driver),
                        inputs={
                            "reset": "client/reset",
                            "episode_goal": "client/episode_goal",
                            "tick": "dora/timer/millis/100",
                        },
                        outputs=["reset_done", "episode_result"],
                    ),
                    dict(
                        id="client",
                        path=sys.executable,
                        args=str(ROOT / "src/aisle/harness/rollout_client.py"),
                        env={
                            "AISLE_TIER": "T4",
                            "AISLE_T4_INCREMENT_TWO": "1",
                            "AISLE_SEEDS": "200",
                            "AISLE_RESULTS": str(results),
                            "AISLE_TIMEOUT_S": "60",
                            "AISLE_LOCKSTEP": "0",
                        },
                        inputs={
                            "tick": "dora/timer/millis/100",
                            "reset_done": "driver/reset_done",
                            "episode_result": "driver/episode_result",
                        },
                        outputs=["reset", "episode_goal", "episode_meta"],
                    ),
                ]
            )
        )
    )
    result = dataflow.run(graph, timeout_s=20)
    assert not result.timed_out, result.stderr
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert [r["goal_id"] for r in records] == ["ep-0000", "ep-0000r"]
    assert t4_chain_completion(records[:1], [200])["ok"] is False
    completion = t4_chain_completion(records, [200])
    assert completion["ok"] is True and completion["complete_episodes"] == 1
    assert records[1]["status"] == "fail"
