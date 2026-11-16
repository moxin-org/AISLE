"""HAR-4 capture is bound by the instrumented graph, not guessed by a node."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from aisle.harness.rollout import instrumented_graph

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_opt_in_capture_binds_exact_executor_sources_and_inputs(tmp_path, monkeypatch):
    """HAR-4; CON-5: opt-in capture pins actual executor source and input edges."""
    monkeypatch.setenv("AISLE_EXECUTION_CAPTURE", "1")
    out = instrumented_graph(ROOT / "graphs/expert_t4_inc2.yaml", ROOT, tmp_path)
    doc = yaml.safe_load(out.read_text())
    node = next(n for n in doc["nodes"] if n["id"] == "ik-trajectory")
    config = json.loads(node["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"])
    assert config["node_id"] == node["id"]
    assert config["inputs"]["grasp_pose_2"] == "return-planner/grasp_pose"
    assert config["skill"]["name"] == "ik-trajectory"
    assert (
        config["skill"]["sources"]["src/aisle/nodes/ik_trajectory.py"]
        == hashlib.sha256((ROOT / "src/aisle/nodes/ik_trajectory.py").read_bytes()).hexdigest()
    )
    assert Path(config["path"]).parent == tmp_path / "traces/executions"


def test_disabled_capture_has_no_binding_or_files(tmp_path, monkeypatch):
    """HAR-4: existing runs retain default behavior unless capture is requested."""
    monkeypatch.delenv("AISLE_EXECUTION_CAPTURE", raising=False)
    out = instrumented_graph(ROOT / "graphs/expert_t0.yaml", ROOT, tmp_path)
    assert not any(
        "AISLE_EXECUTION_CAPTURE_CONFIG" in n.get("env", {})
        for n in yaml.safe_load(out.read_text())["nodes"]
    )
    assert not (tmp_path / "traces/executions").exists()


def test_authored_capture_path_is_refused(tmp_path):
    """HAR-4: authored graphs cannot choose recorder paths or impersonate another skill."""
    graph = ROOT / "graphs/expert_t0.yaml"
    doc = yaml.safe_load(graph.read_text())
    doc["nodes"][0].setdefault("env", {})["AISLE_EXECUTION_CAPTURE_CONFIG"] = "{}"
    with pytest.raises(RuntimeError, match="capture configuration"):
        instrumented_graph(graph, ROOT, tmp_path, graph_snapshot=yaml.safe_dump(doc).encode())


def test_relaunch_capture_uses_distinct_path_and_identity(tmp_path, monkeypatch):
    """HAR-4: wall-clamp relaunches cannot overwrite or alias prior invocations."""
    monkeypatch.setenv("AISLE_EXECUTION_CAPTURE", "1")
    configs = []
    for trace in (tmp_path / "traces", tmp_path / "traces-relaunch-1"):
        out = instrumented_graph(ROOT / "graphs/expert_t0.yaml", ROOT, tmp_path, trace_dir=trace)
        node = next(
            n for n in yaml.safe_load(out.read_text())["nodes"] if n["id"] == "ik-trajectory"
        )
        configs.append(json.loads(node["env"]["AISLE_EXECUTION_CAPTURE_CONFIG"]))
    assert configs[0]["path"] != configs[1]["path"]
    assert configs[0]["capture_id"] != configs[1]["capture_id"]
