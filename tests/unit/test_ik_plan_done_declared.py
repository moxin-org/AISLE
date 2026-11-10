"""`plan_done` goes out only on graphs that declared it (HAR-3, ADR-30).

Fix carried from #601 (G1-F4): `ik-trajectory` sent `plan_done` whenever a
grasp plan completed. `expert_t0` does not declare that output, so under
lockstep the send raised ProtocolError and killed the node.
"""

from pathlib import Path

import pytest
import yaml

from aisle.nodes.ik_trajectory import plan_done_declared

GRAPHS = sorted((Path(__file__).resolve().parents[2] / "graphs").glob("*.yaml"))


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("joint_cmd,gripper_cmd,plan_done,turn_done", True),
        (" joint_cmd , plan_done ", True),
        ("joint_cmd,gripper_cmd,turn_done", False),
        ("joint_cmd,replan_done,turn_done", False),
        ("", False),
        (None, False),
    ],
)
def test_plan_done_is_read_from_the_declared_outputs(declared, expected):
    """HAR-3: the retry-arming signal is sent only when the node's declared
    output list names `plan_done` exactly."""
    assert plan_done_declared(declared) is expected


def _ik_nodes():
    for path in GRAPHS:
        for node in yaml.safe_load(path.read_text()).get("nodes", []):
            if str(node.get("path", "")).endswith("ik_trajectory.py"):
                yield path.name, node


def test_every_graph_that_wires_plan_done_declares_it_to_the_node():
    """HAR-3, ADR-30: the gate reads AISLE_TURN_OUTPUTS, so a graph that
    lists `plan_done` among the node's outputs must also name it there, or
    the in-context retry would be silently disarmed."""
    wired = [
        (name, node) for name, node in _ik_nodes() if "plan_done" in (node.get("outputs") or [])
    ]
    assert wired, "no graph wires plan_done: the fixture glob is wrong"
    for name, node in wired:
        declared = (node.get("env") or {}).get("AISLE_TURN_OUTPUTS", "")
        assert plan_done_declared(declared), name


def test_the_frozen_t0_graph_does_not_declare_plan_done():
    """ADR-30: `expert_t0` is the case the gate exists for."""
    node = next(node for name, node in _ik_nodes() if name == "expert_t0.yaml")
    assert "plan_done" not in (node.get("outputs") or [])
    assert not plan_done_declared(node["env"]["AISLE_TURN_OUTPUTS"])
