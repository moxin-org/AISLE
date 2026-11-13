"""Real dora transport/recorder acceptance with synthetic control events, no physics."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.accept
ROOT = Path(__file__).resolve().parents[2]


def test_goal_identity_survives_dora_transport_and_chain_export(tmp_path, dataflow):
    """HAR-4/6; TC-6/7: goal metadata survives actual transport and reset-scoped export."""
    if shutil.which("dora") is None:
        pytest.skip("dora CLI is required for this acceptance test")
    pytest.importorskip("imageio")
    run = tmp_path / "run"
    run.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text("""
import json
import os
from pathlib import Path
import pyarrow as pa
from dora import Node

node = Node()
for event in node:
    if event["type"] != "INPUT":
        continue
    def send(topic, stamp, seq, value, **meta):
        node.send_output(topic, pa.array([json.dumps(value)] if isinstance(value, dict) else value),
                         dict(sim_time_ns=stamp, env_id=0, seq=seq, **meta))
    send("reset_done", 0, 0, [1], request_id="r0")
    send("episode_goal", 1, 0, dict(seed=4, reset_sim_ns=0, target_med="a"), goal_id="g1")
    first = dict(goal_id="g1", seed=4, status="success", failure=None, verifier="oracle", t_end=1.0)
    send("episode_result", 10, 0, first, goal_id="g1")
    send("episode_goal", 11, 1,
         dict(seed=4, reset_sim_ns=10, target_med="b", expects_return=True), goal_id="g2")
    second = dict(goal_id="g2", seed=4, status="fail", failure="not_returned",
                  verifier="oracle", t_end=2.0)
    send("episode_result", 20, 1, second, goal_id="g2")
    send("reset_done", 30, 1, [1], request_id="r1")
    Path(os.environ["OUTCOMES"]).write_text(json.dumps(dict(episode=0, **first)) + "\\n" +
        json.dumps(dict(episode=0, recovery=True, **second)) + "\\n")
    break
""")
    topics = ("reset_done", "episode_goal", "episode_result")
    graph = tmp_path / "dataflow.yaml"
    graph.write_text(
        yaml.safe_dump(
            dict(
                nodes=[
                    dict(
                        id="driver",
                        path=sys.executable,
                        args=str(driver),
                        env={"OUTCOMES": str(run / "episodes.jsonl")},
                        inputs={"tick": "dora/timer/millis/100"},
                        outputs=list(topics),
                    ),
                    dict(
                        id="recorder",
                        path=sys.executable,
                        args=str(ROOT / "src/aisle/harness/trace_recorder.py"),
                        env={
                            "AISLE_TRACE_DIR": str(run / "traces"),
                            "AISLE_FRAME_CAPTURE_PERIOD_S": "0",
                        },
                        inputs={
                            f"driver__{t}": {"source": f"driver/{t}", "queue_size": 100}
                            for t in topics
                        },
                    ),
                ]
            )
        )
    )
    completed = dataflow.run(graph, timeout_s=20)
    assert not completed.timed_out, completed.stderr
    assert completed.returncode == 0, completed.stderr
    (run / "manifest.json").write_text(json.dumps(dict(evidence_kind="synthetic_dora_transport")))
    sys.path.insert(0, str(ROOT / "tools"))
    import goal_chains

    goal_chains.export_chains(run, tmp_path / "export")
    result = goal_chains.read_chains(tmp_path / "export")
    assert result["manifest"]["coverage"]["verdict_records"] == 2
    assert result["episodes"][0]["window_ns"] == [0, 30]
    assert [g["goal_id"] for g in result["episodes"][0]["goals"]] == ["g1", "g2"]
    assert result["episodes"][0]["goals"][1]["outcome"]["status"] == "fail"
