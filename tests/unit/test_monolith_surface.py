"""SPEC 440 — the monolithic control surface: MON-1 table, MON-3 single
module and typed-facility denial, MON-4 interface exactness, MON-5 same
primitives and guard route, MON-7 bypass attempts, MON-9 provenance
honesty, MON-10 parity decisions at and around the boundary, MON-11
campaign purpose, MON-14 coverage pins.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from cli_helpers import REPO_ROOT

from aisle.harness import monolith as mono
from aisle.monolith import confinement, primitives
from aisle.nodes import grasp_topdown, ik_trajectory, segmented_pose
from aisle.nodes.monolith_broker import (
    ACTIONS,
    OBSERVATIONS,
    Broker,
    validate_action,
)

pytestmark = pytest.mark.unit

ROOT = Path(REPO_ROOT)
EXPERT = ROOT / "experts" / "monolithic" / "expert_t1.py"

MINIMAL = """
API_VERSION = "1.0"

class Controller:
    def __init__(self, primitives, log):
        self.p = primitives
        self.log = log
    def on_event(self, event):
        return []
"""


def _module(tmp_path: Path, body: str, name: str = "m.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _broker(tmp_path: Path, body: str = MINIMAL) -> Broker:
    return Broker(_module(tmp_path, body), "franka", log=lambda _m: None)


# -- MON-3: one module, ordinary errors, no typed facilities ---------------


def test_allowlist_is_one_python_file():
    """MON-3, MON-14: the monolithic deliverable is exactly one .py file;
    the typed allowlist is the agent region of the graph."""
    allow = json.loads((ROOT / "docs/monolithic/allowlist.json").read_text())
    editable = allow["monolithic"]["editable"]
    assert len(editable) == 1 and editable[0].endswith(".py")
    assert (ROOT / editable[0]).is_file()
    assert all((ROOT / p).exists() for p in allow["typed"]["editable"])


def test_expert_module_loads_and_declares_api_version(tmp_path):
    """MON-3: the expert is an ordinary module against API 1.0."""
    broker = Broker(EXPERT, "franka", log=lambda _m: None)
    assert broker.record["api_version"] == primitives.API_VERSION
    assert broker.record["module_sha256"]


def test_module_errors_surface_unchanged(tmp_path):
    """MON-3: a syntax error, a missing Controller, and a wrong API version
    are reported as the ordinary Python failures they are."""
    assert "SyntaxError" in mono.check_module(_module(tmp_path, "def (:\n"))["error"]
    assert "Controller" in mono.check_module(_module(tmp_path, 'API_VERSION = "1.0"\n'))["error"]
    bad = MINIMAL.replace('"1.0"', '"0.9"')
    assert "API_VERSION" in mono.check_module(_module(tmp_path, bad))["error"]


@pytest.mark.parametrize(
    "body",
    [
        "import aisle.harness.validate\n",
        "from aisle.harness import registry\n",
        "import yaml\n",
        "import dora\n",
    ],
)
def test_typed_facilities_are_denied_in_the_monolithic_view(tmp_path, body):
    """MON-3, MON-14: validator, registry, YAML and the dora runtime are
    not importable from the module."""
    report = mono.check_module(_module(tmp_path, body + MINIMAL))
    assert report["ok"] is False and report["infrastructure_invalid"] is True


def test_primitives_public_surface_is_pinned():
    """MON-3, MON-14: the primitive object's public members are exactly
    the documented set."""
    p = primitives.Primitives._load("franka")
    public = {n for n in dir(p) if not n.startswith("_")}
    assert public == set(primitives.PUBLIC_API)
    doc = p.describe()
    assert set(doc["members"]) == set(primitives.PUBLIC_API)
    assert all(doc["members"].values())


# -- MON-4: interface exactness -------------------------------------------


def test_interface_map_is_exact_against_both_graphs():
    """MON-4: observation fields equal the typed region's crossing edges
    and the broker's wiring; actions likewise."""
    assert mono.interface_report(ROOT) == {
        "ok": True,
        "interface_map": "mon-interface-map-v1",
        "fields": 12,
        "errors": [],
    }


def test_interface_map_fails_on_missing_extra_or_privileged_field():
    """MON-4: a missing, extra, or differently privileged semantic field
    is an error."""
    imap = mono.load_json(ROOT, "interface-map.json")
    missing = copy.deepcopy(imap)
    missing["fields"] = [f for f in missing["fields"] if f["name"] != "joint_state"]
    assert any("joint_state" in e for e in mono.interface_errors(ROOT, missing))
    extra = copy.deepcopy(imap)
    sem = {"role": "observe", "privileged": False, "semantics": "ground-truth poses"}
    extra["fields"].append(
        {"name": "oracle_state", "authority": "task", "typed": sem, "monolithic": dict(sem)}
    )
    assert any("oracle_state" in e for e in mono.interface_errors(ROOT, extra))
    priv = copy.deepcopy(imap)
    priv["fields"][0]["monolithic"]["privileged"] = True
    assert any("mismatch" in e or "privileged" in e for e in mono.interface_errors(ROOT, priv))


# -- MON-5: same primitives, same guard route -------------------------------


def test_primitives_delegate_to_the_typed_nodes_own_implementations():
    """MON-5, MON-14: the pinned implementations are the typed nodes'
    function objects, not copies."""
    p = primitives.Primitives._load("franka")
    assert type(p.pose_session()) is segmented_pose.L1Session
    assert primitives.PINNED_IMPLEMENTATIONS["grasp"] == "aisle.nodes.grasp_topdown.plan_grasp"
    assert p.plan_grasp.__func__.__globals__["grasp_topdown"] is grasp_topdown
    assert p.staged_plan.__func__.__globals__["ik_trajectory"] is ik_trajectory
    np.testing.assert_allclose(p.home, p.physics["embodiment"]["franka"]["home_qpos"], rtol=1e-6)


def test_monolithic_graph_routes_commands_through_the_same_guard():
    """MON-5: budget-guard consumes the broker's joint_cmd/gripper_cmd and
    the trusted nodes are wired as in expert_t1.yaml."""
    typed = {
        n["id"]: n for n in yaml.safe_load((ROOT / "graphs/expert_t1.yaml").read_text())["nodes"]
    }
    monol = {
        n["id"]: n
        for n in yaml.safe_load((ROOT / "graphs/monolithic_t1.yaml").read_text())["nodes"]
    }
    guard = monol["budget-guard"]["inputs"]
    assert guard["joint_cmd"]["source"] == "monolith-broker/joint_cmd"
    assert guard["gripper_cmd"]["source"] == "monolith-broker/gripper_cmd"
    assert monol["rollout-client"]["inputs"]["episode_feedback"]["source"] == (
        "monolith-broker/episode_feedback"
    )
    for node_id in ("dora-genesis", "reset", "verifier-oracle"):
        assert monol[node_id] == typed[node_id], node_id
    assert monol["budget-guard"]["env"] == typed["budget-guard"]["env"]
    assert monol["budget-guard"]["path"] == typed["budget-guard"]["path"]
    assert set(monol) == (
        set(typed) - set(mono.load_json(ROOT, "interface-map.json")["typed"]["agent_nodes"])
    ) | {"monolith-broker"}


# -- MON-7: bypass attempts are denied or recorded ---------------------------


@pytest.mark.parametrize(
    "body",
    [
        "import aisle.verifier.oracle\n",  # scorer/verifier import
        "from aisle.verifier import stages\n",
        "import aisle.reset.service\n",  # reset outside the broker
        "import aisle.nodes.dora_genesis\n",  # direct bridge
        "import aisle.nodes.budget_guard\n",  # the guard itself
        "import subprocess\n",
        "import socket\n",
        "import os\n",
        "import sys\n",
        "import importlib\n",
        "import ctypes\n",
        "from . import sibling\n",
        "__import__('subprocess')\n",
        "open('/etc/passwd')\n",  # path traversal / filesystem
    ],
)
def test_bypass_attempts_are_denied_before_any_command(tmp_path, body):
    """MON-7, MON-14: every listed route raises ConfinementViolation at
    module load, so no command or score can follow."""
    with pytest.raises(confinement.ConfinementViolation):
        Broker(_module(tmp_path, body + MINIMAL), "franka", log=lambda _m: None)


def test_lazy_import_inside_a_callback_is_denied(tmp_path):
    """MON-7: an import performed inside on_event is checked the same way
    and the broker refuses the callback."""
    body = MINIMAL.replace("        return []", "        import subprocess\n        return []")
    broker = _broker(tmp_path, body)
    with pytest.raises(confinement.ConfinementViolation):
        broker.deliver("tick", 1, 0)


def test_monkeypatching_a_trusted_callable_is_caught(tmp_path):
    """MON-7, MON-14: replacing a pinned primitive from inside the module
    is detected by the integrity check after the callback."""
    body = MINIMAL.replace(
        "        return []",
        "        g = self.p.plan_grasp.__func__.__globals__['grasp_topdown']\n"
        "        g.plan_grasp = lambda *a, **k: None\n"
        "        return []",
    )
    broker = _broker(tmp_path, body)
    original = grasp_topdown.plan_grasp
    try:
        with pytest.raises(confinement.ConfinementViolation, match="monkeypatch"):
            broker.deliver("tick", 1, 0)
    finally:
        grasp_topdown.plan_grasp = original


def test_invalid_record_is_written_beside_the_results(tmp_path, monkeypatch):
    """MON-7: a refusal produces an infrastructure-invalid record in the
    run directory."""
    from aisle.nodes import monolith_broker

    monkeypatch.setenv("AISLE_RESULTS", str(tmp_path / "episodes.jsonl"))
    monolith_broker._invalid_record("import: denied module 'socket'")
    record = json.loads((tmp_path / "monolith_invalid.json").read_text())
    assert record["infrastructure_invalid"] is True and "socket" in record["reason"]


# -- the broker's action gate -------------------------------------------------


def test_action_gate_accepts_only_the_declared_shapes():
    """MON-4: joint_cmd is n_dof finite floats, gripper_cmd a finite float,
    feedback a JSON object; anything else is a module error."""
    n = 9
    assert validate_action({"joint_cmd": [0.0] * n}, n)["joint_cmd"].shape == (n,)
    assert validate_action({"gripper_cmd": 0.5}, n) == {"gripper_cmd": 0.5}
    assert validate_action({"feedback": {"t": 1}}, n) == {"feedback": {"t": 1}}
    for bad in (
        {"joint_cmd": [0.0] * (n - 1)},
        {"joint_cmd": [float("nan")] * n},
        {"gripper_cmd": float("inf")},
        {"feedback": [1]},
        {"reset": True},
        {"joint_cmd": [0.0] * n, "gripper_cmd": 0.0},
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_action(bad, n)
    assert set(ACTIONS) == {"joint_cmd", "gripper_cmd", "feedback"}


def test_broker_ticks_and_default_feedback_follow_the_sim_clock(tmp_path):
    """MON-4 cadence: 1 Hz ticks derive from the turn clock as in the typed
    state machine, and a silent module still reports every tick."""
    broker = _broker(tmp_path)
    assert broker.ticks_due(5_000_000_000) == []  # no goal
    broker.deliver("episode_goal", {"target_med": "ibuprofen"}, 1_000_000_000, "ep-0")
    assert broker.ticks_due(1_500_000_000) == []
    assert broker.ticks_due(3_000_000_000) == [1, 2]
    actions = broker.deliver("tick", 2, 3_000_000_000)
    assert actions == [{"feedback": {"t": 2, "phase": "unknown"}}]
    broker.deliver("episode_result", {"status": "success"}, 4_000_000_000, "ep-0")
    assert broker.ticks_due(9_000_000_000) == []
    assert set(OBSERVATIONS) >= {"bridge_info", "seg_overhead", "depth_overhead", "joint_state"}


# -- MON-1: treatment table -------------------------------------------------


def test_t1_l2_candidate_treatment_table_rendering_is_current():
    """MON-1: the T1-L2 candidate table and record carry current surface hashes."""
    documents = mono.candidate_documents(ROOT, "t1-l2-realistic")
    report = mono.table_report(ROOT, write=False, documents=documents)
    assert report["ok"] is True and report["errors"] == [], report


def test_treatment_table_is_complete_and_rendered():
    """MON-1, MON-14: every row classified with both sides, paths present,
    differences justified, and the committed rendering current."""
    report = mono.table_report(ROOT, write=False)
    assert report["ok"] is True and report["rows"] == 14 and report["errors"] == [], report
    assert report["immutable_id"].startswith("sha256:")
    table = mono.load_json(ROOT, "treatment-table.json")
    surfaces = {r["surface"] for r in table["rows"]}
    for needed in (
        "agent deliverable",
        "static validation and diagnostics",
        "guard route and limits",
    ):
        assert any(needed in s for s in surfaces), needed


def test_treatment_table_blocks_on_an_undeclared_difference():
    """MON-1: an unjustified difference or unresolved row is an error."""
    table = mono.load_json(ROOT, "treatment-table.json")
    row = copy.deepcopy(table["rows"][0])
    row["justification"] = ""
    assert mono.table_errors(ROOT, {"rows": [row]})
    row = copy.deepcopy(table["rows"][0])
    row["class"] = "intentionally_different"
    assert mono.table_errors(ROOT, {"rows": [row]})
    row = copy.deepcopy(table["rows"][4])
    row["typed"]["paths"].append("src/aisle/nodes/does_not_exist.py")
    assert any("missing" in e for e in mono.table_errors(ROOT, {"rows": [row]}))


# -- MON-9 / MON-10 / MON-11: parity -------------------------------------


def _episodes(statuses, failure=None, invalid=()):
    return {
        seed: {
            "seed": seed,
            "status": s,
            "failure": None if s == "success" else failure,
            **({"infrastructure_invalid": True} if seed in invalid else {}),
        }
        for seed, s in enumerate(statuses)
    }


@pytest.fixture
def protocol():
    proto = mono.load_json(ROOT, "parity-protocol.json")
    proto["acceptance"]["preconditions"] = {k: True for k in proto["acceptance"]["preconditions"]}
    return proto


def test_parity_passes_at_the_equivalence_boundary(protocol):
    """MON-10, MON-14: 8/8 vs 7/8 sits exactly on the 0.125 margin and
    passes; both arms clear the functional floor."""
    typed = _episodes(["success"] * 8)
    monol = _episodes(["success"] * 7 + ["fail"], failure="never_grasped")
    decision = mono.parity_decision(protocol, typed, monol)
    assert decision["gate"] == "pass", decision["reasons"]
    assert decision["success_rate"]["difference"] == pytest.approx(0.125)


def test_parity_blocks_just_past_the_margin_and_below_the_floor(protocol):
    """MON-10, MON-14: 8/8 vs 6/8 exceeds the margin and misses the floor;
    the gate is blocked with both reasons."""
    decision = mono.parity_decision(
        protocol, _episodes(["success"] * 8), _episodes(["success"] * 6 + ["fail"] * 2, "timeout")
    )
    assert decision["gate"] == "blocked"
    assert any("margin" in r for r in decision["reasons"])
    assert any("floor" in r for r in decision["reasons"])


def test_parity_blocks_on_safety_failure_missing_pair_and_unresolved_exclusion(protocol):
    """MON-10: a forbidden failure, a missing paired run, or an
    infrastructure-invalid run keeps the gate blocked and in the record."""
    typed = _episodes(["success"] * 8)
    unsafe = _episodes(["success"] * 7 + ["fail"], failure="collision")
    assert any("safety" in r for r in mono.parity_decision(protocol, typed, unsafe)["reasons"])
    short = _episodes(["success"] * 7)
    assert any("missing" in r for r in mono.parity_decision(protocol, typed, short)["reasons"])
    invalid = _episodes(["success"] * 8, invalid=(3,))
    decision = mono.parity_decision(protocol, typed, invalid)
    assert decision["valid_pairs"] == 7
    assert any("infrastructure-invalid" in r for r in decision["reasons"])
    assert [p["seed"] for p in decision["pairs"]] == list(range(8))  # every run stays


def test_parity_gate_is_blocked_by_the_honest_preconditions():
    """MON-9, MON-10: with the committed protocol — same author for both
    experts, seeds not revealed by a separate operator — a perfect 8/8 vs
    8/8 still cannot pass."""
    proto = mono.load_json(ROOT, "parity-protocol.json")
    experts = mono.load_json(ROOT, "experts.json")
    assert all(a["blind"] is False for a in experts["artifacts"]) and experts["frozen"] is False
    pre = proto["acceptance"]["preconditions"]
    assert pre["MON-9 independent blind authorship of both experts"] is False
    decision = mono.parity_decision(proto, _episodes(["success"] * 8), _episodes(["success"] * 8))
    assert decision["gate"] == "blocked"
    assert all("precondition" in r for r in decision["reasons"])


def test_parity_records_carry_the_expert_parity_purpose(tmp_path):
    """MON-11: parity output is tagged expert_parity and the protocol's
    purpose matches, so pooling with agent sessions is detectable."""
    proto = mono.load_json(ROOT, "parity-protocol.json")
    assert proto["purpose"] == mono.CAMPAIGN_PURPOSE == "expert_parity"
    typed = tmp_path / "typed.jsonl"
    monol = tmp_path / "mono.jsonl"
    typed.write_text("\n".join(json.dumps(r) for r in _episodes(["success"] * 8).values()))
    monol.write_text("\n".join(json.dumps(r) for r in _episodes(["success"] * 8).values()))
    report = mono.parity_report(ROOT, typed, monol)
    assert report["campaign_purpose"] == "expert_parity" and report["ok"] is True


# -- launcher -----------------------------------------------------------------


def test_launcher_stamps_the_frozen_graph_with_the_module(tmp_path):
    """MON-3: the launcher's graph copy differs from the frozen template
    only in absolutized paths and the module under test."""
    module = _module(tmp_path, MINIMAL)
    out = mono.stamp_graph(ROOT, module, tmp_path / "out")
    doc = yaml.safe_load(out.read_text())
    broker = next(n for n in doc["nodes"] if n["id"] == "monolith-broker")
    assert broker["env"]["AISLE_MONOLITH_MODULE"] == str(module.resolve())
    assert all(Path(n["path"]).is_absolute() and Path(n["path"]).is_file() for n in doc["nodes"])
    template = yaml.safe_load((ROOT / mono.TEMPLATE_GRAPH).read_text())
    assert [n["id"] for n in doc["nodes"]] == [n["id"] for n in template["nodes"]]


@pytest.mark.parametrize("tier", ["T0", "T2", "T3", "T4", "unknown"])
def test_launcher_refuses_unsupported_tier_before_participant_execution(
    tmp_path, monkeypatch, tier
):
    """MON-4, BMK-4: the T1 surface cannot stand in for another task tier."""

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported tier reached participant construction or rollout")

    monkeypatch.setattr(mono, "check_module", unexpected)
    monkeypatch.setattr(mono, "stamp_graph", unexpected)
    report = mono.run(tmp_path, tmp_path / "absent.py", [0], 1, tier=tier)
    assert report["ok"] is False
    assert report["tier"] == tier
    assert report["supported_tiers"] == ["T1"]
    assert report["error"] == "unsupported_monolithic_tier"
    assert not (tmp_path / "graphs").exists()


def test_launcher_preserves_t1_rollout_arguments(tmp_path, monkeypatch):
    """MON-3, MON-4: the supported T1 surface still reaches the trusted rollout."""
    from aisle.harness import rollout

    observed = {}
    module = tmp_path / "participant.py"
    graph = tmp_path / "stamped.yaml"
    monkeypatch.setattr(mono, "check_module", lambda *args: {"ok": True})
    monkeypatch.setattr(mono, "stamp_graph", lambda *args: graph)

    def launch(**kwargs):
        observed.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(rollout, "rollout", launch)
    report = mono.run(tmp_path, module, [0, 1], 2, run_id="tier-test", record_simulator_work=True)
    assert report["ok"] is True
    assert observed["record_simulator_work"] is True
    assert observed["tier"] == "T1"
    assert observed["graph"] == graph
    assert observed["seeds"] == [0, 1]
    assert observed["episodes"] == 2
    assert report["campaign_purpose"] == "expert_parity"


def test_cli_reports_unsupported_tier_as_json(tmp_path):
    """CON-8, MON-4: unsupported task tiers produce structured CLI refusal."""
    from cli_helpers import run_json

    code, report = run_json(
        "aisle.harness.cli",
        "monolith",
        "run",
        "--module",
        str(tmp_path / "absent.py"),
        "--tier",
        "T2",
        "--episodes",
        "1",
        "--seeds",
        "0",
        "--root",
        str(tmp_path),
    )
    assert code == 1
    assert report["error"] == "unsupported_monolithic_tier"
    assert report["ok"] is False


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"verifier": "both"}, "unsupported_monolithic_verifier"),
        ({"reset_mode": "sideways"}, "unsupported_monolithic_reset"),
        ({"template": "graphs/no_such_template.yaml"}, "unsupported_monolithic_template"),
    ],
)
def test_run_refuses_unsupported_verifier_reset_or_template(kwargs, error):
    """MON-3/MON-4: the launcher accepts only the admitted verifier, reset and template
    and refuses before touching the module or the simulator."""
    from aisle.harness import monolith

    result = monolith.run(ROOT, EXPERT, seeds=[0], episodes=1, **kwargs)
    assert result["ok"] is False and result["error"] == error


def test_run_refuses_a_template_that_binds_no_monolithic_module(tmp_path, monkeypatch):
    """MON-3: a template without AISLE_MONOLITH_MODULE cannot masquerade as the monolithic arm."""
    from aisle.harness import monolith

    monkeypatch.setattr(monolith, "check_module", lambda *args, **kwargs: {"ok": True})
    result = monolith.run(ROOT, EXPERT, seeds=[0], episodes=1, template="graphs/expert_t1.yaml")
    assert result["ok"] is False and result["error"] == "unsupported_monolithic_template"
    result = monolith.run(ROOT, EXPERT, seeds=[0], episodes=1, template="pyproject.toml")
    assert result["ok"] is False and result["error"] == "unsupported_monolithic_template"


def test_l2_pose_session_is_the_pinned_detected_pose_object():
    """MON-3/MON-5/BND-2: the L2 primitive is the same L2Session the typed
    detected-pose node runs, driven by rgb + depth, never a segmentation mask;
    it is part of the public API and pinned by import path."""
    from aisle.monolith import primitive_api
    from aisle.nodes import l2_pose

    assert "l2_pose_session" in primitives.PUBLIC_API
    assert primitives.PINNED_IMPLEMENTATIONS["pose_l2"] == "aisle.nodes.l2_pose.L2Session"
    assert "l2_pose_session" in primitive_api.CALLS["primitives"]
    assert primitive_api.CREATES["l2_pose_session"] == "pose"
    assert {"on_rgb", "on_depth", "on_bridge_info"} <= primitive_api.CALLS["pose"]
    assert callable(primitives.Primitives.l2_pose_session)
    assert callable(l2_pose.lazy_pinned_detector())  # no model load until the first frame
    assert l2_pose.L2Session.on_rgb is l2_pose.FramePairSession.on_obs
    assert callable(l2_pose.pinned_detector) and callable(l2_pose.pinned_backprojector)


def test_t1_l2_module_checks_against_the_broker():
    """MON-3: the L2 expert compiles and constructs with no simulator."""
    from aisle.harness import monolith

    result = monolith.check_module(ROOT / "experts/monolithic/expert_t1_l2.py")
    assert result["ok"] is True, result
