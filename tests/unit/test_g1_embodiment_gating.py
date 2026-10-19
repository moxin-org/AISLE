"""G1-E embodiment gating: the harness and generic nodes accept `g1_right`,
and unknown embodiments are refused explicitly instead of silently falling
onto the Franka DH chain (VAL-2 EMBODIMENT_MISMATCH, HAR-1, MOB-4).

The research-bound embodiment tables (matched-session, exposure corpus,
monolith) stay untouched — only the rollout CLI, the graph validator's arm
mapping, and the embodiment-agnostic node manifests move."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from cli_helpers import REPO_ROOT, make_registry_root, run_json, write_manifest

pytestmark = pytest.mark.unit


def run_validate(graph: Path | str, *args: str) -> tuple[int, dict]:
    """The same CLI wrapper test_validator.py uses."""
    return run_json("aisle.harness.cli", "validate", str(graph), *args)


MANIFESTS = REPO_ROOT / "registry" / "manifests"
#: sources with literally zero embodiment branches (G1-E audit)
OPEN_NODES = ("oracle-pose", "rollout-client", "turn-barrier")
#: sources that still branch on franka/so101 or read physics[embodiment]
#: (budget-guard left this set when G1-F1 added g1_right limits and FK)
STILL_PINNED = ("ik-trajectory", "grasp-planner-topdown")


def _manifest(node_id: str) -> dict:
    return yaml.safe_load((MANIFESTS / f"{node_id}.yaml").read_text())


def _single_node_graph(root: Path, node_id: str) -> Path:
    manifest = _manifest(node_id)
    graph = root / "g.yaml"
    graph.write_text(
        yaml.safe_dump(
            {"nodes": [{"id": node_id, "inputs": {}, "outputs": list(manifest["outputs"])}]},
            sort_keys=False,
        )
    )
    return graph


def test_rollout_parser_accepts_g1_right_and_rejects_unknown():
    """HAR-1: `harness rollout --embodiment` accepts g1_right (parse-level);
    an unknown embodiment like humanoid is an argparse refusal, not a silent
    Franka default."""
    from aisle.harness.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "rollout",
            "--graph",
            "g.yaml",
            "--episodes",
            "1",
            "--seeds",
            "1",
            "--embodiment",
            "g1_right",
        ]
    )
    assert args.embodiment == "g1_right"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "rollout",
                "--graph",
                "g.yaml",
                "--episodes",
                "1",
                "--seeds",
                "1",
                "--embodiment",
                "humanoid",
            ]
        )


@pytest.mark.parametrize("node_id", OPEN_NODES)
def test_open_nodes_validate_under_g1_right(tmp_path, node_id):
    """VAL-2: a node whose source has no franka/so101 branch carries
    `g1_right` in its manifest arm list, so a g1_right graph gets NO
    EMBODIMENT_MISMATCH from it."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _manifest(node_id))
    code, report = run_validate(
        _single_node_graph(root, node_id), "--root", str(root), "--embodiment", "g1_right"
    )
    assert code == 0, report
    assert "EMBODIMENT_MISMATCH" not in {e["code"] for e in report["errors"]}


@pytest.mark.parametrize("node_id", STILL_PINNED)
def test_pinned_nodes_still_refuse_g1_right(tmp_path, node_id):
    """VAL-2: ik-trajectory/grasp-planner-topdown still have
    franka/so101-only kinematics — G1-F's scope — so a g1_right graph
    MUST keep reporting EMBODIMENT_MISMATCH (pinning 'not yet supported')."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _manifest(node_id))
    code, report = run_validate(
        _single_node_graph(root, node_id), "--root", str(root), "--embodiment", "g1_right"
    )
    assert code != 0, report
    assert "EMBODIMENT_MISMATCH" in {e["code"] for e in report["errors"]}


def test_open_nodes_still_validate_under_so101(tmp_path):
    """VAL-2 regression: opening the three agnostic manifests to g1_right
    changes nothing for the existing embodiments."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _manifest("oracle-pose"))
    for embodiment in ("franka", "so101"):
        code, report = run_validate(
            _single_node_graph(root, "oracle-pose"), "--root", str(root), "--embodiment", embodiment
        )
        assert code == 0, (embodiment, report)


def test_validator_arm_mapping_knows_g1_right():
    """VAL-2: EMBODIMENT_ARM maps g1_right to its own arm kind — the
    g1-driver manifest (arm: [g1_right]) validates under g1_right."""
    from aisle.harness.validate import EMBODIMENT_ARM

    assert EMBODIMENT_ARM["g1_right"] == "g1_right"
    assert EMBODIMENT_ARM["mobile"] == "franka"  # MOB-4 unchanged


def test_g1_driver_manifest_validates_under_g1_right(tmp_path):
    """VAL-2: the G1-D driver manifest round-trips the new arm kind."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _manifest("g1-driver"))
    code, report = run_validate(
        _single_node_graph(root, "g1-driver"), "--root", str(root), "--embodiment", "g1_right"
    )
    assert code == 0, report


def test_ik_refuses_unknown_embodiments_instead_of_franka_fallthrough():
    """HAR-1/G1-E: fk_tcp/_pose_error/_ik_once/ik_solve raise a named
    ValueError for g1_right (and any unknown embodiment) instead of
    silently solving the Franka DH chain — G1 kinematics is G1-F's scope."""
    from aisle.nodes.ik_trajectory import _ik_once, _pose_error, fk_tcp, ik_solve

    for bad in ("g1_right", "humanoid"):
        with pytest.raises(ValueError, match=f"unsupported embodiment {bad!r}"):
            fk_tcp(np.zeros(7, dtype=np.float32), bad)
        with pytest.raises(ValueError, match=f"unsupported embodiment {bad!r}"):
            _pose_error(np.zeros(7), np.zeros(3), np.eye(3), bad)
        with pytest.raises(ValueError, match=f"unsupported embodiment {bad!r}"):
            _ik_once(np.zeros(3), np.eye(3), np.zeros(7), bad)
        with pytest.raises(ValueError, match=f"unsupported embodiment {bad!r}"):
            ik_solve(np.zeros(3), np.eye(3), np.zeros(7), bad)


def test_ik_supported_paths_unchanged():
    """HAR-1 regression: franka/mobile stay the Franka chain and so101 its
    pinned URDF chain — the rejection adds no behavior for supported kinds."""
    from aisle.nodes.ik_trajectory import fk_tcp

    franka = fk_tcp(np.zeros(7, dtype=np.float32))
    assert franka.shape == (3,)
    assert fk_tcp(np.zeros(7, dtype=np.float32), "mobile").shape == (3,)
    so101 = fk_tcp(np.zeros(5, dtype=np.float32), "so101")
    assert np.asarray(so101).shape == (3,)


def test_grasp_planner_refuses_unknown_embodiment():
    """HAR-1/G1-E: the grasp planner's embodiment read refuses g1_right and
    unknown kinds by name before touching layout/FK (it would otherwise
    KeyError deep in physics profiles or mis-plan against Franka geometry)."""
    from aisle.nodes.grasp_topdown import require_supported_embodiment

    for bad in ("g1_right", "humanoid"):
        with pytest.raises(ValueError, match=f"unsupported embodiment {bad!r}"):
            require_supported_embodiment(bad)
    for good in ("franka", "so101", "mobile"):
        require_supported_embodiment(good)  # no raise


def test_fixture_mirrors_carry_the_open_arms():
    """VAL-2 (test_validator.py:727 pins byte equality; this pins WHICH
    nodes were opened): the reserved_dists mirrors of the three open nodes
    match the real manifests, so corpus runs see the same g1_right arms."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "roots" / "reserved_dists"
    for node_id in OPEN_NODES:
        real = _manifest(node_id)
        copy = yaml.safe_load((fixture / "registry" / "manifests" / f"{node_id}.yaml").read_text())
        assert "g1_right" in copy["embodiment"]["arm"], node_id
        assert copy["embodiment"] == real["embodiment"], node_id


#: G1-C2: the seven bridge/layout-dependent nodes whose g1_right support
#: lands with the bridge's wire-9 command path (dora-genesis runs it;
#: the other six depend on its layout or its topics).
BRIDGE_DEPENDENT_NODES = (
    "dora-genesis",
    "arm-driver-sim",
    "gripper-driver-sim",
    "camera-source",
    "verifier-oracle",
    "task-state-machine",
    "reset",
)


@pytest.mark.parametrize("node_id", BRIDGE_DEPENDENT_NODES)
def test_bridge_dependent_manifests_carry_g1_right(node_id):
    """TC-5, VAL-2 (G1-C2): the seven bridge-dependent manifests open
    `g1_right` beside franka/so101, and the reserved_dists mirrors match,
    so a g1_right graph no longer reports EMBODIMENT_MISMATCH for them."""
    manifest = _manifest(node_id)
    assert "g1_right" in manifest["embodiment"]["arm"], node_id
    fixture = REPO_ROOT / "tests" / "fixtures" / "roots" / "reserved_dists"
    copy = yaml.safe_load((fixture / "registry" / "manifests" / f"{node_id}.yaml").read_text())
    assert "g1_right" in copy["embodiment"]["arm"], node_id
    assert copy["embodiment"] == manifest["embodiment"], node_id


def test_expert_t0_g1_right_mismatches_only_the_pinned_three():
    """TC-5, VAL-2 (G1-C2): with the bridge supporting g1_right, the
    expert graph under --embodiment g1_right reports EMBODIMENT_MISMATCH
    for EXACTLY the G1-F-pinned trio — every other node validates."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t0.yaml", "--embodiment", "g1_right")
    mismatched = {e["node"] for e in report["errors"] if e["code"] == "EMBODIMENT_MISMATCH"}
    assert mismatched == set(STILL_PINNED), sorted(mismatched)
    assert code != 0  # the pinned trio still fails validation, as pinned
