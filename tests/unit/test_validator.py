"""Unit tests for the dataflow validator (SPEC 060 VAL-1..7, CON-8).

Acceptance tests named by the spec:
test_bad_corpus_all_rejected_with_expected_codes (VAL-1..6),
test_good_corpus_passes, test_hints_nonempty (VAL-3).
The validator imports neither genesis nor dora.
"""

import tomllib
from pathlib import Path

import pytest
import yaml
from cli_helpers import REPO_ROOT, make_registry_root, run_json, write_manifest

pytestmark = pytest.mark.unit

BAD_DIR = REPO_ROOT / "tests" / "fixtures" / "graphs" / "bad"
GOOD_DIR = REPO_ROOT / "tests" / "fixtures" / "graphs" / "good"

with open(BAD_DIR / "expected.toml", "rb") as _f:
    EXPECTED = tomllib.load(_f)


def run_validate(graph: Path | str, *args: str) -> tuple[int, dict]:
    return run_json("aisle.harness.cli", "validate", str(graph), *args)


def corpus_args(expectation: dict) -> list[str]:
    args = []
    if "embodiment" in expectation:
        args += ["--embodiment", expectation["embodiment"]]
    if "root" in expectation:
        args += ["--root", str(REPO_ROOT / expectation["root"])]
    return args


_CORPUS_CACHE: dict[str, tuple[int, dict]] = {}


def corpus_report(stem: str) -> tuple[int, dict]:
    """One CLI run per bad-corpus file per session, shared across tests."""
    if stem not in _CORPUS_CACHE:
        _CORPUS_CACHE[stem] = run_validate(BAD_DIR / f"{stem}.yaml", *corpus_args(EXPECTED[stem]))
    return _CORPUS_CACHE[stem]


def codes(report: dict, level: str) -> set[str]:
    return {entry["code"] for entry in report[level]}


def test_corpus_minimums():
    """VAL-7: the golden corpus holds >=20 deliberately broken graphs and
    >=3 valid graphs, and includes the design-doc §8.1.4 typo case BY
    CONTENT: an edge referencing controller/joint_cmd while no node has the
    id controller. graphs/expert_t0.yaml joined the good corpus at T08
    (issue 2 resolved by the poses topic, see test_expert_t0_is_good)."""
    bad = list(BAD_DIR.glob("*.yaml"))
    assert len(bad) >= 20
    assert len(list(GOOD_DIR.glob("*.yaml"))) >= 3
    assert {f.stem for f in bad} == set(EXPECTED)
    typo = yaml.safe_load((BAD_DIR / "input_no_producer_controller_typo.yaml").read_text())
    sources = [source for node in typo["nodes"] for source in (node.get("inputs") or {}).values()]
    assert any(str(s).startswith("controller/joint_cmd") for s in sources)
    assert "controller" not in {n["id"] for n in typo["nodes"]}


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_bad_corpus_all_rejected_with_expected_codes(stem):
    """VAL-1, VAL-2, VAL-4, VAL-5, VAL-6, VAL-7: every bad-corpus graph is rejected
    with EXACTLY its expected stable error codes (so a regression can neither
    drop the named code nor sneak in a second one), or flagged with its
    expected warning code for warning-class checks; exit 0 iff ok (CON-8)."""
    expectation = EXPECTED[stem]
    code, report = corpus_report(stem)
    assert codes(report, "errors") == set(expectation["codes"]), report
    assert codes(report, "warnings") == set(expectation.get("warnings", [])), report
    if expectation["codes"]:
        assert report["ok"] is False
        assert code != 0
    else:
        assert report["ok"] is True
        assert code == 0


@pytest.mark.parametrize("path", sorted(GOOD_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_good_corpus_passes(path):
    """VAL-7: every good-corpus graph validates with ok=true, no errors,
    exit 0 (CON-8)."""
    code, report = run_validate(path)
    assert code == 0, report
    assert report["ok"] is True
    assert report["errors"] == []


def test_expert_t0_is_good():
    """VAL-7's good-corpus requirement for graphs/expert_t0.yaml is
    satisfied HERE by validating the real file in place (it stays outside
    tests/fixtures/graphs/good/ so no copy can drift): NORMAL validation
    — no --allow-unproven, which HAR-2's rollout gate never sets — passes
    with zero errors AND zero warnings (the M0 evalcards exist, ADR-3
    retired at T08)."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t0.yaml")
    assert code == 0, report
    assert report["ok"] is True and report["errors"] == []
    assert report["warnings"] == []


def test_expert_t1_is_good():
    """VAL-7 for graphs/expert_t1.yaml, validated in place like expert_t0:
    the T1 expert baseline at perception rung L1 (TC-9) passes NORMAL
    validation with zero errors and zero warnings — the rung declaration on
    the sim bridge, the absent `poses` topic, and the segmented-pose edges
    (seg_overhead + depth_overhead in, target_pose out) all pass VAL-8 and
    the schema checks together."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t1.yaml")
    assert code == 0, report
    assert report["ok"] is True and report["errors"] == []
    assert report["warnings"] == []


def _graph_params() -> list:
    # a list, not a generator: pytest deprecated non-Collection argvalues
    # (PytestRemovedIn10Warning), and a warning this suite prints on every
    # run is one nobody reads by the time it matters
    return [
        pytest.param(path, id=path.name) for path in sorted((REPO_ROOT / "graphs").glob("*.yaml"))
    ]


@pytest.mark.parametrize("graph", _graph_params())
def test_every_shipped_graph_validates(graph):
    """VAL-7 over the WHOLE corpus, not a hand-kept three.

    `expert_t0`/`expert_t1`/`expert_t1_l2` were validated in place and the
    other eight were not, so an edge added to those eight had no producer
    check at all — measured in the round-2 review of #208, where pointing a
    new edge at a non-existent producer left every test green while
    `harness validate` rejected the file. A per-graph parametrize means a
    new graph is covered the day it lands.

    Mobile graphs are validated under their own embodiment; asserting only
    `ok` (not zero warnings) because the sibling tests above own the
    stricter no-warnings claim for the graphs they name."""
    mobile = {"expert_s1", "eval_s1_driver_v2", "eval_s3_driver_v1", "agent_campaign"}
    extra = ["--embodiment", "mobile"] if graph.stem in mobile else []
    code, report = run_validate(graph, *extra)
    assert code == 0, report
    assert report["ok"] is True and report["errors"] == [], report["errors"]


def test_a_graph_that_cannot_deliver_a_refusal_is_rejected(tmp_path):
    """TC-6/ADR-34: dora DROPS `send_output` to an undeclared output — one
    stderr warning, `Ok(())` to the caller — so a graph whose reset node
    omits `reset_refused` does not fail, it HANGS: the refusal evaporates
    and the requester waits in `awaiting_reset` until the wall clamp.

    That is strictly worse than the silent degradation the topic split
    removed, and no other rule sees it: VAL-4 checks that declared outputs
    exist in the manifest, never that a manifest output is declared. Both
    halves are pinned, because either alone still hangs the run."""
    doc = yaml.safe_load((REPO_ROOT / "graphs" / "expert_t0.yaml").read_text())

    def written(mutate, name):
        copy = yaml.safe_load(yaml.safe_dump(doc))
        mutate(copy["nodes"])
        path = tmp_path / name
        path.write_text(yaml.safe_dump(copy, sort_keys=False))
        return path

    def drop_output(nodes):
        node = next(n for n in nodes if n["id"] == "reset")
        node["outputs"] = [o for o in node["outputs"] if o != "reset_refused"]

    def drop_consumer(nodes):
        for node in nodes:
            (node.get("inputs") or {}).pop("reset_refused", None)

    for mutate, name in ((drop_output, "no_output.yaml"), (drop_consumer, "no_consumer.yaml")):
        code, report = run_validate(written(mutate, name))
        assert code != 0, f"{name} validated despite being unable to deliver a refusal"
        assert "REFUSAL_UNROUTED" in {e["code"] for e in report["errors"]}, report["errors"]


def test_expert_t1_with_poses_routed_is_rejected(tmp_path):
    """VAL-8/TC-9 mutation-proofing on the REAL graph: take expert_t1.yaml
    verbatim, re-add the bridge's ground-truth `poses` output and route it
    into segmented-pose — the exact quiet downgrade the rung exists to
    prevent — and the validator must reject it with
    PERCEPTION_RUNG_VIOLATION naming the offending edge."""
    doc = yaml.safe_load((REPO_ROOT / "graphs" / "expert_t1.yaml").read_text())
    for node in doc["nodes"]:
        # keep node paths resolvable from the copy's location
        node["path"] = str((REPO_ROOT / "graphs" / node["path"]).resolve())
        if node["id"] == "dora-genesis":
            node["outputs"].append("poses")
        if node["id"] == "segmented-pose":
            node["inputs"]["poses"] = {"source": "dora-genesis/poses", "queue_size": 100}
    mutated = tmp_path / "expert_t1_poses_leak.yaml"
    mutated.write_text(yaml.safe_dump(doc, sort_keys=False))

    code, report = run_validate(mutated, "--root", str(REPO_ROOT))
    assert code != 0 and report["ok"] is False
    violations = [e for e in report["errors"] if e["code"] == "PERCEPTION_RUNG_VIOLATION"]
    assert violations, report["errors"]
    assert any("dora-genesis/poses" in e.get("edge", "") for e in violations), violations


def test_expert_t1_l2_is_good():
    """VAL-7 for graphs/expert_t1_l2.yaml, validated in place: the T1
    baseline at rung L2 (TC-9's top rung — the bridge publishes NEITHER
    simulator ground-truth pose nor segmentation; identity comes from RGB and
    same-stamp sensor depth supplies metric geometry, idea I7) passes NORMAL
    validation with zero errors and zero warnings."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t1_l2.yaml")
    assert code == 0, report
    assert report["ok"] is True and report["errors"] == []
    assert report["warnings"] == []


def test_expert_t1_l2_with_seg_routed_is_rejected(tmp_path):
    """VAL-8/TC-9 mutation-proofing at L2: re-add `seg_overhead` to the
    bridge and route it into detected-pose — the quiet downgrade from
    detection to ground-truth segmentation — and the validator must reject
    it with PERCEPTION_RUNG_VIOLATION naming the edge."""
    doc = yaml.safe_load((REPO_ROOT / "graphs" / "expert_t1_l2.yaml").read_text())
    for node in doc["nodes"]:
        node["path"] = str((REPO_ROOT / "graphs" / node["path"]).resolve())
        if node["id"] == "dora-genesis":
            node["outputs"].append("seg_overhead")
        if node["id"] == "detected-pose":
            node["inputs"]["seg_overhead"] = {
                "source": "dora-genesis/seg_overhead",
                "queue_size": 100,
            }
    mutated = tmp_path / "expert_t1_l2_seg_leak.yaml"
    mutated.write_text(yaml.safe_dump(doc, sort_keys=False))

    code, report = run_validate(mutated, "--root", str(REPO_ROOT))
    assert code != 0 and report["ok"] is False
    violations = [e for e in report["errors"] if e["code"] == "PERCEPTION_RUNG_VIOLATION"]
    assert violations, report["errors"]
    assert any("dora-genesis/seg_overhead" in e.get("edge", "") for e in violations), violations


def test_hints_nonempty():
    """VAL-3: every error and warning across the whole bad corpus carries a
    non-empty hint naming a registry capability or concrete fix, and the
    report is a single JSON object of the specified shape."""
    for stem in sorted(EXPECTED):
        _, report = corpus_report(stem)
        # VAL-3 as amended by ADR-24 D5: dist_state is the labeled
        # non-attesting diagnostic
        assert set(report) == {"ok", "graph", "errors", "warnings", "dist_state"}
        for entry in report["errors"] + report["warnings"]:
            assert entry["code"], (stem, entry)
            assert entry["hint"].strip(), (stem, entry)
            assert entry["detail"].strip(), (stem, entry)
            assert "edge" in entry or "node" in entry, (stem, entry)
            assert all(isinstance(v, str) for v in entry.values()), (stem, entry)


def test_manifest_missing_hint_names_closest():
    """VAL-3: MANIFEST_MISSING hints name the closest registry id, turning
    the design-doc typo class into a one-edit fix for the agent."""
    _, report = corpus_report("manifest_missing_typo_oracle_pos")
    hints = " ".join(e["hint"] for e in report["errors"])
    assert "oracle-pose" in hints


def test_schema_mismatch_hint_names_schemas():
    """VAL-3, VAL-4: SCHEMA_MISMATCH hints name both the produced and the
    expected schema from the CAP-2 vocabulary."""
    _, report = corpus_report("schema_mismatch_depth_to_rgb")
    mismatch = [e for e in report["errors"] if e["code"] == "SCHEMA_MISMATCH"]
    assert any("depth_f32" in e["hint"] and "rgb8_image" in e["hint"] for e in mismatch)


def fixture_root(tmp_path: Path, mutations: dict[str, dict]) -> Path:
    """Registry-shaped root with selected real manifests copied in, applying
    {manifest_id: {direction.port: new_schema}} mutations."""
    root = make_registry_root(tmp_path)
    for manifest_id, changes in mutations.items():
        manifest = yaml.safe_load(
            (REPO_ROOT / "registry" / "manifests" / f"{manifest_id}.yaml").read_text()
        )
        for path, schema in changes.items():
            direction, port = path.split(".")
            manifest[direction][port]["schema"] = schema
        write_manifest(root, manifest)
    return root


def write_graph(root: Path, nodes: list[dict]) -> Path:
    graph = root / "g.yaml"
    graph.write_text(yaml.safe_dump({"nodes": nodes}, sort_keys=False))
    return graph


def test_schema_unknown_from_bad_manifest(tmp_path):
    """VAL-4: a manifest schema name outside the CAP-2 vocabulary is its own
    error (SCHEMA_UNKNOWN), never silently passed."""
    root = fixture_root(
        tmp_path,
        {"camera-source": {"outputs.rgb_overhead": "not_a_schema"}, "detector-openvocab": {}},
    )
    graph = write_graph(
        root,
        [
            {"id": "camera-source", "outputs": ["rgb_overhead"]},
            {
                "id": "detector-openvocab",
                "inputs": {"rgb": "camera-source/rgb_overhead"},
                "outputs": ["boxes", "labels"],
            },
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0
    assert "SCHEMA_UNKNOWN" in codes(report, "errors")


def test_schema_unknown_reports_every_name_deterministically(tmp_path):
    """VAL-4, CON-5: when both ends of an edge carry unknown schema names,
    both are reported, in producer-then-consumer order, identically across
    runs (no hash-seed dependence)."""
    root = fixture_root(
        tmp_path,
        {
            "camera-source": {"outputs.rgb_overhead": "zzz_unknown"},
            "detector-openvocab": {"inputs.rgb": "aaa_unknown"},
        },
    )
    graph = write_graph(
        root,
        [
            {"id": "camera-source", "outputs": ["rgb_overhead"]},
            {
                "id": "detector-openvocab",
                "inputs": {"rgb": "camera-source/rgb_overhead"},
                "outputs": ["boxes", "labels"],
            },
        ],
    )
    _, first = run_validate(graph, "--root", str(root))
    _, second = run_validate(graph, "--root", str(root))
    unknown = [e["detail"] for e in first["errors"] if e["code"] == "SCHEMA_UNKNOWN"]
    assert len(unknown) == 2
    assert "zzz_unknown" in unknown[0] and "aaa_unknown" in unknown[1]
    assert first == second


def test_oracle_leak_not_hidden_by_schema_unknown(tmp_path):
    """VAL-6: an oracle leak is reported even when the same edge also has an
    unknown schema name — safety findings are never masked by schema
    errors."""
    root = fixture_root(
        tmp_path,
        {"camera-source": {"outputs.oracle_state": "mystery"}, "oracle-pose": {}},
    )
    graph = write_graph(
        root,
        [
            {"id": "camera-source", "outputs": ["oracle_state"]},
            {
                "id": "oracle-pose",
                "inputs": {"poses": "camera-source/oracle_state"},
                "outputs": ["target_pose"],
            },
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0
    assert {"ORACLE_LEAK", "SCHEMA_UNKNOWN"} <= codes(report, "errors")


def test_malformed_manifests_reported_as_json(tmp_path):
    """CON-8: manifests violating the CAP-1 JSON Schema in ANY field —
    missing id, scalar embodiment, non-mapping ports, non-numeric rate_hz,
    non-string schema value, bad latency_class enum — become GRAPH_INVALID
    registry errors that name the file, never a traceback (the screen is
    the full capability schema, not a bespoke shape check)."""
    for index, mutate in enumerate(
        (
            lambda m: m.pop("id"),
            lambda m: m.update(embodiment="franka"),
            lambda m: m.update(inputs="rgb"),
            lambda m: m["outputs"]["rgb_overhead"].update(schema=42),
            lambda m: m["inputs"].update(tick={"schema": "scalar_f32", "rate_hz": "fast"}),
            lambda m: m["outputs"]["rgb_overhead"].update(latency_class="warp"),
        )
    ):
        root = fixture_root(tmp_path / str(index), {"detector-openvocab": {}})
        manifest = yaml.safe_load(
            (REPO_ROOT / "registry" / "manifests" / "camera-source.yaml").read_text()
        )
        mutate(manifest)
        (root / "registry" / "manifests" / "camera-source.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        graph = write_graph(root, [{"id": "detector-openvocab", "outputs": ["boxes", "labels"]}])
        code, report = run_validate(graph, "--root", str(root))
        assert code != 0
        assert "GRAPH_INVALID" in codes(report, "errors")
        assert any("camera-source.yaml" in e["detail"] for e in report["errors"])


GUARD_ROOT = REPO_ROOT / "tests" / "fixtures" / "roots" / "with_guard"


@pytest.mark.parametrize(
    "port, driver", [("joint_cmd", "arm-driver-sim"), ("gripper_cmd", "gripper-driver-sim")]
)
@pytest.mark.parametrize("suffix", ["", "_0", "_12"])
@pytest.mark.parametrize("route", ["raw", "guarded", "timer"])
def test_motion_gate_covers_indexed_ports(tmp_path, port, driver, suffix, route):
    """VAL-5, VAL-3: indexed command inputs require the guard and retain their edge identity."""
    root = fixture_root(tmp_path, {"ik-trajectory": {}, "budget-guard": {}, driver: {}})
    source = {
        "raw": f"ik-trajectory/{port}",
        "guarded": f"budget-guard/{port}_safe",
        "timer": "dora/timer/millis/10",
    }[route]
    graph = write_graph(
        root,
        [
            {"id": "ik-trajectory", "outputs": [port]},
            {
                "id": "budget-guard",
                "inputs": {f"{port}_0": f"ik-trajectory/{port}"},
                "outputs": [f"{port}_safe"],
            },
            {"id": driver, "inputs": {port + suffix: {"source": source, "queue_size": 10}}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    if route == "guarded":
        assert code == 0, report
        assert report["errors"] == []
    else:
        assert code != 0, report
        assert codes(report, "errors") == {"MOTION_UNGATED"}, report
        assert report["errors"][0]["edge"] == f"{source} -> {driver}/{port}{suffix}"


def test_motion_gate_is_topological(tmp_path):
    """VAL-5: every backward path into a driver command input must traverse
    the resolved budget-guard — a multi-hop path THROUGH the guard passes,
    a direct guard edge passes, and an indirect path that bypasses the
    guard is MOTION_UNGATED. Uses a fixture registry that includes the
    guard manifest (real registry gains it at T07). See ADR 4, as amended
    by the ADR-5 ratification (2026-08-03): the fixture guard is
    motion-class, so the DIRECT case — the guard's own joint_cmd fed from
    a raw timer — exercises the guard-sink exemption: without
    `node_id != GUARD_ID` in _validate_edge it would be MOTION_UNGATED."""
    direct = write_graph(
        tmp_path,
        [
            {
                "id": "budget-guard",
                "inputs": {"joint_cmd": "dora/timer/millis/10"},
                "outputs": ["joint_cmd_safe"],
            },
            {
                "id": "arm-driver-sim",
                "inputs": {"joint_cmd": "budget-guard/joint_cmd_safe"},
                "outputs": ["joint_state"],
            },
        ],
    )
    code, report = run_validate(direct, "--root", str(GUARD_ROOT))
    assert code == 0, report
    assert report["ok"] is True

    multihop = tmp_path / "multihop.yaml"
    multihop.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {
                        "id": "budget-guard",
                        "inputs": {"joint_cmd": "dora/timer/millis/10"},
                        "outputs": ["joint_cmd_safe"],
                    },
                    {
                        "id": "command-smoother",
                        "inputs": {"cmd": "budget-guard/joint_cmd_safe"},
                        "outputs": ["joint_cmd"],
                    },
                    {
                        "id": "arm-driver-sim",
                        "inputs": {"joint_cmd": "command-smoother/joint_cmd"},
                        "outputs": ["joint_state"],
                    },
                ]
            }
        )
    )
    code, report = run_validate(multihop, "--root", str(GUARD_ROOT))
    assert code == 0, report

    bypass = tmp_path / "bypass.yaml"
    bypass.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {
                        "id": "command-smoother",
                        "inputs": {"cmd": "dora/timer/millis/10"},
                        "outputs": ["joint_cmd"],
                    },
                    {
                        "id": "arm-driver-sim",
                        "inputs": {"joint_cmd": "command-smoother/joint_cmd"},
                        "outputs": ["joint_state"],
                    },
                ]
            }
        )
    )
    code, report = run_validate(bypass, "--root", str(GUARD_ROOT))
    assert code != 0
    assert "MOTION_UNGATED" in codes(report, "errors")


def test_fixture_root_schemas_match_real_registry():
    """The CAP-1 schema copies inside every fixture root must be identical
    to the real registry's — fixture validation must never drift from the
    live contract."""
    real = (REPO_ROOT / "registry" / "schema" / "capability.schema.json").read_bytes()
    roots = sorted((REPO_ROOT / "tests" / "fixtures" / "roots").iterdir())
    assert roots
    for root in roots:
        copy = root / "registry" / "schema" / "capability.schema.json"
        assert copy.read_bytes() == real, copy


def test_motion_gate_mixed_fanin_is_ungated(tmp_path):
    """VAL-5 (ADR 4): one unguarded input taints the node — a smoother fed
    by BOTH the guard and a bare timer leaves an unguarded path into the
    driver, so the sink is MOTION_UNGATED."""
    graph = write_graph(
        tmp_path,
        [
            {
                "id": "budget-guard",
                "inputs": {"joint_cmd": "dora/timer/millis/10"},
                "outputs": ["joint_cmd_safe"],
            },
            {
                "id": "command-smoother",
                "inputs": {"cmd": "budget-guard/joint_cmd_safe", "aux": "dora/timer/millis/10"},
                "outputs": ["joint_cmd"],
            },
            {
                "id": "arm-driver-sim",
                "inputs": {"joint_cmd": "command-smoother/joint_cmd"},
                "outputs": ["joint_state"],
            },
        ],
    )
    code, report = run_validate(graph, "--root", str(GUARD_ROOT))
    assert code != 0
    assert "MOTION_UNGATED" in codes(report, "errors")


def test_motion_gate_cycle_without_guard_is_ungated(tmp_path):
    """VAL-5 (ADR 4): a cycle with no guard on it never reaches a gated
    root, so a driver fed from the cycle is MOTION_UNGATED (and the
    validator terminates rather than recursing forever)."""
    graph = write_graph(
        tmp_path,
        [
            {
                "id": "command-smoother",
                "inputs": {"cmd": "command-mixer/joint_cmd"},
                "outputs": ["joint_cmd"],
            },
            {
                "id": "command-mixer",
                "inputs": {"cmd": "command-smoother/joint_cmd"},
                "outputs": ["joint_cmd"],
            },
            {
                "id": "arm-driver-sim",
                "inputs": {"joint_cmd": "command-smoother/joint_cmd"},
                "outputs": ["joint_state"],
            },
        ],
    )
    code, report = run_validate(graph, "--root", str(GUARD_ROOT))
    assert code != 0
    assert "MOTION_UNGATED" in codes(report, "errors")


def test_verifier_feedback_loop_is_legal():
    """VAL-6 (ADR 5): verifier verdicts feeding lifecycle nodes is the
    sanctioned pattern — episode_result consumption downstream of the
    verifier is not an oracle leak."""
    code, report = run_validate(GOOD_DIR / "verifier_feedback_loop.yaml")
    assert code == 0, report
    assert "ORACLE_LEAK" not in codes(report, "errors")


def test_unwired_manifest_inputs_are_legal(tmp_path):
    """ADR 5: wiring none of a manifest's declared inputs is legal (dora
    permits subsets; source nodes have zero inputs by design)."""
    graph = write_graph(tmp_path, [{"id": "oracle-pose", "inputs": {}, "outputs": ["target_pose"]}])
    code, report = run_validate(graph)
    assert code == 0, report


def test_missing_node_hint_lists_graph_nodes():
    """VAL-3: when no similar node id exists, the INPUT_NO_PRODUCER hint
    lists the graph's actual nodes instead of an empty did-you-mean."""
    _, report = corpus_report("input_no_producer_missing_node")
    hints = " ".join(e["hint"] for e in report["errors"])
    assert "did you mean ''" not in hints
    assert "detector-openvocab" in hints


def test_weak_similarity_gives_search_hint():
    """VAL-3: a node id with no close manifest match (warp-drive) gets the
    registry-search command, not a misleading did-you-mean."""
    _, report = corpus_report("manifest_missing_unknown_node")
    hints = " ".join(e["hint"] for e in report["errors"])
    assert "arm-driver-sim" not in hints
    assert "search --provides" in hints


def test_undeclared_port_hint_is_actionable():
    """VAL-3: the undeclared-input hint names the corrective action and the
    real ports, not just a bare list."""
    _, report = corpus_report("schema_mismatch_undeclared_port")
    mismatch = [e for e in report["errors"] if e["code"] == "SCHEMA_MISMATCH"]
    assert any("rename the input" in e["hint"] and "'rgb'" in e["hint"] for e in mismatch)


def test_non_utf8_graph_reported_as_json(tmp_path):
    """CON-8: a non-UTF-8 graph file is a GRAPH_INVALID JSON report, not an
    UnicodeDecodeError traceback."""
    graph = tmp_path / "binary.yaml"
    graph.write_bytes(b"\x80\x81\x82\xff")
    code, report = run_validate(graph)
    assert code != 0
    assert "GRAPH_INVALID" in codes(report, "errors")


def test_allow_unproven_downgrades_eval_error():
    """VAL-2: --allow-unproven downgrades EVAL_MISSING_FOR_MOTION to a
    warning (design doc §8.2.1); the harness never sets it for agents."""
    graph = BAD_DIR / "eval_missing_for_motion.yaml"
    code, report = run_validate(
        graph, "--allow-unproven", "--root", str(REPO_ROOT / "tests/fixtures/roots/eval_null")
    )
    assert "EVAL_MISSING_FOR_MOTION" not in codes(report, "errors")
    assert "EVAL_MISSING_FOR_MOTION" in codes(report, "warnings")


def test_missing_graph_file_reported_as_json(tmp_path):
    """CON-8: validating a nonexistent graph path yields a JSON error report
    and nonzero exit, not a traceback."""
    code, report = run_validate(tmp_path / "nope.yaml")
    assert code != 0
    assert report["ok"] is False
    assert "GRAPH_INVALID" in codes(report, "errors")


def test_expert_graph_accepts_so101_after_motion_stack_lands():
    """VAL-2, M0-5: once both official SO-101 motion manifests advertise
    support, the exact T0 graph validates under a profile swap."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t0.yaml", "--embodiment", "so101")
    assert code == 0, report
    assert "EMBODIMENT_MISMATCH" not in codes(report, "errors")


def test_install_missing_hint_names_installed_alternative():
    """VAL-2 INSTALL_MISSING (H1-discovered class, PR #33 findings): a
    pip:-sourced manifest whose distribution is absent errors rather than
    validating a graph that cannot launch; per VAL-3 the hint MUST name an
    installed registry alternative with the same capability when one
    exists (oracle-pose provides object_pose alongside pose-estimator).
    Runs against the reserved_dists root like every INSTALL_MISSING
    expectation (PR #65 review: this dedicated test was the one remaining
    ambient-env coupling)."""
    code, report = corpus_report("install_missing_detector")
    assert code != 0
    entries = [e for e in report["errors"] if e["code"] == "INSTALL_MISSING"]
    assert {e["node"] for e in entries} == {"detector-openvocab", "pose-estimator"}
    pose_hint = next(e["hint"] for e in entries if e["node"] == "pose-estimator")
    assert "oracle-pose" in pose_hint
    # the NO-alternative fallback branch (detector-openvocab is the sole
    # object_detection provider) must stay actionable, not just non-empty —
    # naming the RESERVED distribution, not the ambient-coupled real one
    det_hint = next(e["hint"] for e in entries if e["node"] == "detector-openvocab")
    assert "aisle-corpus-reserved-yolo" in det_hint and "install" in det_hint


def test_corpus_and_hint_survive_dora_dists_installed(monkeypatch):
    """PR #65 review P1 (the reviewer's reproduction, pinned): with fake
    installed metadata for ALL THREE real dora perception dists — the
    exact env-change the INSTALL_MISSING hint invites — the corpus
    fixture AND the dedicated hint path must be unmoved: reserved names
    keep firing INSTALL_MISSING."""
    import aisle.harness.validate as val
    from aisle.harness.validate import validate

    real = val._pip_installed.__wrapped__  # unwrap the functools.cache

    def fake_installed(dist):
        return True if dist in ("dora-yolo", "dora-pose", "dora-ocr") else real(dist)

    monkeypatch.setattr(val, "_pip_installed", fake_installed)
    report = validate(
        BAD_DIR / "install_missing_detector.yaml",
        REPO_ROOT / "tests" / "fixtures" / "roots" / "reserved_dists",
        "franka",
        False,
    )
    codes_fired = {e["code"] for e in report["errors"]}
    assert codes_fired == {"INSTALL_MISSING"}, codes_fired
    det_hint = next(e["hint"] for e in report["errors"] if e["node"] == "detector-openvocab")
    assert "aisle-corpus-reserved-yolo" in det_hint


def _pip_manifest(base_id: str, **overrides) -> dict:
    manifest = yaml.safe_load(
        (REPO_ROOT / "registry" / "manifests" / f"{base_id}.yaml").read_text()
    )
    manifest.update(overrides)
    return manifest


def _single_node_graph(root, manifest):
    return write_graph(
        root, [{"id": manifest["id"], "inputs": {}, "outputs": list(manifest["outputs"])}]
    )


def test_install_missing_corpus_is_env_independent():
    """Issue #37 (full fix of the PR #34 finding): the INSTALL_MISSING
    corpus entries run against the reserved_dists fixture root, whose
    perception manifests name RESERVED never-published distributions —
    installing the real dora-yolo/dora-pose/dora-ocr (exactly what the
    INSTALL_MISSING hint invites) must shift NOTHING. The reserved names
    stay uninstallable by convention; assert it so a collision fails in
    one labeled place."""
    from aisle.harness.validate import _pip_installed

    for dist in (
        "aisle-corpus-reserved-yolo",
        "aisle-corpus-reserved-pose",
        "aisle-corpus-reserved-ocr",
    ):
        assert not _pip_installed(dist), (
            f"reserved corpus name {dist!r} is installed in this environment — "
            "these names exist to be permanently absent; pick a new reserved "
            "name in tests/fixtures/roots/reserved_dists and expected.toml"
        )


def test_reserved_root_mirrors_real_registry():
    """Issue #37: the reserved_dists fixture root is the REAL registry
    with exactly three perception sources swapped to reserved names —
    pinned here so registry changes cannot silently drift the corpus
    environment."""
    import yaml as _yaml

    swaps = {
        "detector-openvocab": "pip:aisle-corpus-reserved-yolo",
        "pose-estimator": "pip:aisle-corpus-reserved-pose",
        "ocr-label": "pip:aisle-corpus-reserved-ocr",
    }
    real_dir = REPO_ROOT / "registry" / "manifests"
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "roots" / "reserved_dists"
    fixture_manifests = fixture_dir / "registry" / "manifests"
    assert {p.name for p in real_dir.glob("*.yaml")} == {
        p.name for p in fixture_manifests.glob("*.yaml")
    }
    for real_path in sorted(real_dir.glob("*.yaml")):
        real = _yaml.safe_load(real_path.read_text())
        copy = _yaml.safe_load((fixture_manifests / real_path.name).read_text())
        if real["id"] in swaps:
            assert copy["source"] == swaps[real["id"]], real["id"]
            real, copy = dict(real), dict(copy)
            real.pop("source"), copy.pop("source")
        assert copy == real, f"{real_path.name} drifted from the real registry"
        source = _yaml.safe_load((fixture_manifests / real_path.name).read_text()).get("source")
        if isinstance(source, str) and ":" not in source:
            assert (fixture_dir / source).is_file(), f"missing stub for {source}"


def test_install_missing_installed_distribution_passes(tmp_path):
    """VAL-2: the POSITIVE path — a pip: source whose distribution IS
    installed (pytest, guaranteed in the test env) must not error; case
    and PEP 503 name normalization are importlib.metadata's job (PyTest ==
    pytest). Kills the `_pip_installed = lambda d: False` mutation the
    corpus alone cannot catch (PR #34 review)."""
    root = make_registry_root(tmp_path)
    for source in ("pip:pytest", "pip:PyTest"):
        write_manifest(root, _pip_manifest("oracle-pose", source=source))
        graph = _single_node_graph(root, _pip_manifest("oracle-pose"))
        code, report = run_validate(graph, "--root", str(root))
        assert code == 0, (source, report)
        assert "INSTALL_MISSING" not in codes(report, "errors")


def test_install_missing_empty_and_decorated_dist_names(tmp_path):
    """PR #34 adversarial review (as evolved by issue #35): `source:
    "pip:"` once crashed the CLI with an uncaught ValueError — the CAP-1
    source pattern now rejects an empty distribution at the schema layer
    (structured GRAPH_INVALID naming the manifest, still no traceback,
    CON-8). Decorations (`pip:pytest[x]`, `pip:pytest==1.0`,
    `pip: pytest`) stay valid: the dist name is normalized (strip + cut
    at [=<>!~;@) before probing, so INSTALLED dists are never falsely
    flagged."""
    root = make_registry_root(tmp_path)
    graph = _single_node_graph(root, _pip_manifest("oracle-pose"))
    # PR #63 review P2: whitespace-only dists normalize to empty and must
    # die at the SAME schema layer as the bare `pip:`
    for source in ("pip:", "pip:  ", "pip:\t"):
        write_manifest(root, _pip_manifest("oracle-pose", source=source))
        code, report = run_validate(graph, "--root", str(root))
        assert code != 0, source
        assert "GRAPH_INVALID" in codes(report, "errors"), source  # JSON, no crash
        assert any("does not match" in e["detail"] for e in report["errors"]), source
    for source in ("pip:pytest[extra]", "pip:pytest==1.0", "pip: pytest"):
        write_manifest(root, _pip_manifest("oracle-pose", source=source))
        code, report = run_validate(graph, "--root", str(root))
        assert code == 0, (source, report)


def test_pip_scheme_casing_is_a_schema_violation(tmp_path):
    """PR #34 adversarial review (as evolved by issue #35): `PIP:x` once
    dodged INSTALL_MISSING via exact-lowercase matching. The CAP-1 source
    pattern now rejects any colon-bearing source that is not lowercase
    `pip:` at the SCHEMA layer — casing (and every unknown scheme) is
    closed off before validation, not merely normalized."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _pip_manifest("oracle-pose", source="PIP:aisle-review-absent-dist"))
    graph = _single_node_graph(root, _pip_manifest("oracle-pose"))
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0
    assert "GRAPH_INVALID" in codes(report, "errors")
    assert any("oracle-pose" in e["detail"] for e in report["errors"])  # names the manifest


def test_install_missing_alternatives_are_usable(tmp_path):
    """PR #34 adversarial review: the hint must not send the agent to an
    alternative that fails the NEXT compile — an uninstalled pip peer, an
    embodiment-incompatible node, or a partial provider (covers only some
    of the missing node's capabilities)."""
    root = make_registry_root(tmp_path)
    missing = _pip_manifest("pose-estimator", source="pip:aisle-review-absent-dist")
    write_manifest(root, missing)
    # uninstalled pip peer: same capability, also absent — must NOT appear
    write_manifest(
        root,
        _pip_manifest("oracle-pose", id="pose-peer-pip", source="pip:aisle-review-absent-peer"),
    )
    # embodiment-incompatible peer: so101-only arm on a franka graph
    so101_only = _pip_manifest("oracle-pose", id="pose-peer-so101")
    so101_only["embodiment"] = {"arm": ["so101"], "gripper": "any"}
    write_manifest(root, so101_only)
    # the genuinely usable peer
    write_manifest(root, _pip_manifest("oracle-pose"))
    graph = _single_node_graph(root, missing)
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0
    hint = next(e["hint"] for e in report["errors"] if e["code"] == "INSTALL_MISSING")
    assert "oracle-pose" in hint
    assert "pose-peer-pip" not in hint
    assert "pose-peer-so101" not in hint


def test_path_manifest_mismatch_rule_edges(tmp_path):
    """VAL-2 PATH_MANIFEST_MISMATCH (issue #36): pip sources accept the
    manifest source verbatim or the bare distribution name — anything
    else is a mismatch; path-less nodes are dora's launch problem, not
    this check's; absolute paths resolving to the manifest source (the
    instrumented/staged-copy form) match."""
    from aisle.harness.validate import validate_nodes

    manifests = {
        "detector-openvocab": {
            "id": "detector-openvocab",
            "provides": [],
            "source": "pip:dora-yolo",
            "safety_class": "perception",
            "embodiment": {"arm": ["franka"]},
        },
        "oracle-pose": {
            "id": "oracle-pose",
            "provides": [],
            "source": "src/aisle/nodes/oracle_pose.py",
            "safety_class": "perception",
            "embodiment": {"arm": ["franka"]},
        },
    }

    def codes_for(node):
        errors, _ = validate_nodes(
            [node],
            manifests,
            set(),
            "franka",
            True,
            graph_dir=REPO_ROOT / "graphs",
            root=REPO_ROOT,
        )
        return {e["code"] for e in errors}

    ok_verbatim = {"id": "detector-openvocab", "path": "pip:dora-yolo"}
    ok_bare = {"id": "detector-openvocab", "path": "dora-yolo"}
    spoofed_pip = {"id": "detector-openvocab", "path": "../skills/evil.py"}
    pathless = {"id": "detector-openvocab"}
    # PR #62 review P2: decorated/case-varied pip sources must accept the
    # NORMALIZED bare dist (reuse _pip_dist), matching the INSTALL_MISSING
    # normalization contract — not the unparsed suffix
    manifests["detector-decorated"] = {
        "id": "detector-decorated",
        "provides": [],
        "source": "pip:dora-yolo[gpu]",
        "safety_class": "perception",
        "embodiment": {"arm": ["franka"]},
    }
    manifests["detector-pinned"] = {
        "id": "detector-pinned",
        "provides": [],
        "source": "PIP:dora-yolo==1.0",
        "safety_class": "perception",
        "embodiment": {"arm": ["franka"]},
    }
    ok_decorated_bare = {"id": "detector-decorated", "path": "dora-yolo"}
    ok_decorated_verbatim = {"id": "detector-decorated", "path": "pip:dora-yolo[gpu]"}
    ok_cased_bare = {"id": "detector-pinned", "path": "dora-yolo"}
    spoofed_decorated = {"id": "detector-decorated", "path": "dora-yolo-evil"}
    ok_absolute = {
        "id": "oracle-pose",
        "path": str((REPO_ROOT / "src" / "aisle" / "nodes" / "oracle_pose.py").resolve()),
    }
    spoofed_path = {"id": "oracle-pose", "path": "../src/aisle/nodes/grasp_topdown.py"}

    assert "PATH_MANIFEST_MISMATCH" not in codes_for(ok_verbatim)
    assert "PATH_MANIFEST_MISMATCH" not in codes_for(ok_bare)
    assert "PATH_MANIFEST_MISMATCH" in codes_for(spoofed_pip)
    assert "PATH_MANIFEST_MISMATCH" not in codes_for(pathless)
    assert "PATH_MANIFEST_MISMATCH" not in codes_for(ok_absolute)
    assert "PATH_MANIFEST_MISMATCH" in codes_for(spoofed_path)
    assert "PATH_MANIFEST_MISMATCH" not in codes_for(ok_decorated_bare)
    assert "PATH_MANIFEST_MISMATCH" not in codes_for(ok_decorated_verbatim)
    assert "PATH_MANIFEST_MISMATCH" not in codes_for(ok_cased_bare)
    assert "PATH_MANIFEST_MISMATCH" in codes_for(spoofed_decorated)


def test_path_check_has_no_second_base_bypass(tmp_path):
    """PR #62 review P1: the graphs-dir fallback approved any path that
    WOULD match if the graph lived in graphs/ — but a staged graph (the
    live-swap tmpdir) lives elsewhere, and dora resolves against ITS
    base, so validator and runtime could resolve different code under an
    approved id. There is exactly ONE base: the graph's own directory. A
    relative path in a tmpdir-staged graph that only matches via the
    graphs/ composition is a MISMATCH."""
    from aisle.harness.validate import validate_nodes

    manifests = {
        "oracle-pose": {
            "id": "oracle-pose",
            "provides": [],
            "source": "src/aisle/nodes/oracle_pose.py",
            "safety_class": "perception",
            "embodiment": {"arm": ["franka"]},
        }
    }
    staged_dir = tmp_path / "aisle-swap-adversary"
    staged_dir.mkdir()
    node = {"id": "oracle-pose", "path": "../src/aisle/nodes/oracle_pose.py"}
    errors, _ = validate_nodes(
        [node], manifests, set(), "franka", True, graph_dir=staged_dir, root=REPO_ROOT
    )
    assert "PATH_MANIFEST_MISMATCH" in {e["code"] for e in errors}
    absolute = {
        "id": "oracle-pose",
        "path": str((REPO_ROOT / "src/aisle/nodes/oracle_pose.py").resolve()),
    }
    errors, _ = validate_nodes(
        [absolute], manifests, set(), "franka", True, graph_dir=staged_dir, root=REPO_ROOT
    )
    assert "PATH_MANIFEST_MISMATCH" not in {e["code"] for e in errors}


def _path_manifest(mid, source):
    return {
        "id": mid,
        "kind": "node",
        "provides": ["object_pose"],
        "requires": [],
        "inputs": {},
        "outputs": {},
        "embodiment": {"arm": ["franka"], "gripper": "any"},
        "safety_class": "perception",
        "eval": None,
        "origin": "hub",
        "source": source,
    }


def test_snapshot_validation_keeps_authored_path_provenance(tmp_path):
    """HAR-2/VAL-2: captured bytes retain the authored graph's directory.

    Snapshot validation must close rollout's read race without treating the
    run artifact as the graph identity or adding a second resolution base.
    A source-file edit after capture is ignored, while the relative node path
    in the capture still resolves from the original graph directory.
    """
    from aisle.harness.validate import validate

    root = make_registry_root(tmp_path / "root")
    source = root / "src" / "node.py"
    source.parent.mkdir(parents=True)
    source.write_text("# vetted node\n")
    write_manifest(root, _path_manifest("snapshot-node", "src/node.py"))
    graph_dir = root / "graphs"
    graph_dir.mkdir()
    graph = graph_dir / "snapshot.yaml"
    graph.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {
                        "id": "snapshot-node",
                        "path": "../src/node.py",
                        "inputs": {},
                        "outputs": [],
                    }
                ]
            },
            sort_keys=False,
        )
    )
    snapshot = graph.read_bytes()
    graph.write_text(graph.read_text().replace("../src/node.py", "../src/other.py"))

    captured = validate(
        root=root,
        graph_path=graph,
        embodiment="franka",
        allow_unproven=False,
        graph_snapshot=snapshot,
    )
    current = validate(graph, root, "franka", False)
    assert captured["ok"] is True, captured
    assert "PATH_MANIFEST_MISMATCH" in {entry["code"] for entry in current["errors"]}


def test_source_invalid_requires_contained_regular_file(tmp_path):
    """PR #63 review P1: `root / source` is NOT containment — an absolute
    source (`/etc/hosts`) survives the join, `../` escapes the root, a
    symlink can resolve outside it, and exists() accepts directories.
    The source must be a REGULAR FILE resolved UNDER the root."""
    from aisle.harness.validate import validate_nodes

    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "src" / "real.py").write_text("ok")
    outside = tmp_path / "outside.py"
    outside.write_text("external mutable code")
    (root / "src" / "link.py").symlink_to(outside)

    def codes_for(source):
        manifests = {"n": _path_manifest("n", source)}
        errors, _ = validate_nodes(
            [{"id": "n"}], manifests, set(), "franka", True, graph_dir=root, root=root
        )
        return {e["code"] for e in errors}

    assert "SOURCE_INVALID" not in codes_for("src/real.py")  # the honest case
    assert "SOURCE_INVALID" in codes_for(str(outside.resolve()))  # absolute
    assert "SOURCE_INVALID" in codes_for("../outside.py")  # traversal
    assert "SOURCE_INVALID" in codes_for("src")  # directory, not a file
    assert "SOURCE_INVALID" in codes_for("src/link.py")  # symlink escaping root


def test_install_missing_alternatives_exclude_invalid_sources(tmp_path):
    """PR #63 review P2: the INSTALL_MISSING hint MUST never name an
    alternative that fails the NEXT compile (VAL-2) — a same-capability
    manifest whose path source names no file is exactly such an
    alternative and must not be recommended."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _pip_manifest("pose-estimator", source="pip:aisle-review-absent-dist"))
    ghost = _path_manifest("ghost-pose", "src/aisle/nodes/does_not_exist.py")
    ghost["provides"] = ["object_pose"]
    path = root / "registry" / "manifests" / "ghost-pose.yaml"
    path.write_text(yaml.safe_dump(ghost, sort_keys=False))  # NO auto-stub: stays ghost
    real = _path_manifest("real-pose", "src/aisle/nodes/real_pose.py")
    real["provides"] = ["object_pose"]
    write_manifest(root, real)  # helper stubs the source file: launchable
    # PR #64 review P2: EVERY manifest-level next-compile check must be
    # mirrored — a base-requiring peer on a fixed-base graph fails
    # EMBODIMENT_MISMATCH, an eval-null MOTION peer fails
    # EVAL_MISSING_FOR_MOTION; neither may be recommended
    mobile_peer = _path_manifest("mobile-pose", "src/aisle/nodes/mobile_pose.py")
    mobile_peer["provides"] = ["object_pose"]
    mobile_peer["embodiment"] = {"arm": ["franka"], "gripper": "any", "base": ["mobile"]}
    write_manifest(root, mobile_peer)
    motion_peer = _path_manifest("motion-pose", "src/aisle/nodes/motion_pose.py")
    motion_peer["provides"] = ["object_pose"]
    motion_peer["safety_class"] = "motion"
    write_manifest(root, motion_peer)  # eval stays null: unproven motion
    graph = _single_node_graph(root, _pip_manifest("pose-estimator"))
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0
    hint = next(e["hint"] for e in report["errors"] if e["code"] == "INSTALL_MISSING")
    assert "real-pose" in hint
    assert "ghost-pose" not in hint  # would fail the next compile (SOURCE_INVALID)
    assert "mobile-pose" not in hint  # would fail it (EMBODIMENT_MISMATCH, base)
    assert "motion-pose" not in hint  # would fail it (EVAL_MISSING_FOR_MOTION)


def test_dist_state_present_on_early_graph_errors(tmp_path):
    """PR #69 review F4: a malformed GRAPH against a valid registry must
    still carry the registry's dist_state mappings — the diagnostic is
    computed from the registry, before graph parsing."""
    root = make_registry_root(tmp_path)
    write_manifest(root, _pip_manifest("oracle-pose", source="pip:aisle-review-absent-dist"))
    bad_graph = tmp_path / "broken.yaml"
    bad_graph.write_text("{not: [valid yaml")
    code, report = run_validate(bad_graph, "--root", str(root))
    assert code != 0
    assert report["dist_state"] == {"aisle-review-absent-dist": None}


def test_perception_rung_default_l0_allows_ground_truth_poses(tmp_path):
    """VAL-8: a graph that declares NO rung is L0, where `poses` is the
    sanctioned pose source. Every graph that existed before TC-9 is in this
    class, so the check must be silent on them (see also
    test_expert_t0_is_good, which validates the real T0 graph)."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "outputs": ["poses"]},
            {"id": "oracle-pose", "inputs": {"poses": "dora-genesis/poses"}},
        ],
    )
    _, report = run_validate(graph, "--root", str(root))
    assert "PERCEPTION_RUNG_VIOLATION" not in codes(report, "errors"), report


def test_perception_rung_l1_forbids_ground_truth_poses():
    """VAL-8, TC-9: at L1 the pose MUST be estimated from segmentation +
    depth, so consuming `poses` is an error naming the edge and the rung —
    otherwise a run labelled L1 would report grounding it never did."""
    code, report = corpus_report("perception_rung_l1_uses_poses")
    assert code != 0
    leak = [e for e in report["errors"] if e["code"] == "PERCEPTION_RUNG_VIOLATION"]
    assert len(leak) == 1, report
    assert leak[0]["edge"] == "dora-genesis/poses -> oracle-pose/poses"
    assert "L1" in leak[0]["detail"]
    # VAL-3: the hint must name the fix, not just the violation — and it must
    # NOT offer the downgrade. These messages are the research agent's
    # learning signal (SPEC 060) and the rung is self-declared in the graph
    # the agent authors, so "declare L0" would be a one-token way to silence
    # the error without changing what the run measures.
    assert "seg_overhead" in leak[0]["hint"]
    assert "L0" not in leak[0]["hint"]


def test_perception_rung_l1_allows_estimated_pose_path(tmp_path):
    """VAL-8: L1 forbids ONLY the ground-truth pose shortcut. The rung's own
    inputs — depth plus (with the bridge) segmentation — and the estimator's
    `target_pose` output stay legal, or the rung would be unusable."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "pose-estimator": {}})
    graph = write_graph(
        root,
        [
            {
                "id": "dora-genesis",
                "env": {"AISLE_PERCEPTION": "L1"},
                "outputs": ["depth_overhead"],
            },
            {"id": "pose-estimator", "inputs": {"depth": "dora-genesis/depth_overhead"}},
        ],
    )
    _, report = run_validate(graph, "--root", str(root))
    assert "PERCEPTION_RUNG_VIOLATION" not in codes(report, "errors"), report


@pytest.mark.parametrize("rung", ["L1", "L2"])
def test_store_rejects_unsupported_estimated_perception_rungs_at_validation(tmp_path, rung):
    """VAL-7/VAL-8, issue #130: the store's item-id namespace cannot be
    queried by the desk L1 or L2 estimators. Reject that unsupported pairing
    during validation, before a full Genesis build reaches the bridge's
    belt-and-braces refusal and produces zero episodes."""
    root = fixture_root(tmp_path, {"dora-genesis": {}})
    graph = write_graph(
        root,
        [
            {
                "id": "dora-genesis",
                "env": {"AISLE_SCENE": "store", "AISLE_PERCEPTION": rung},
                "outputs": ["rgb_overhead", "depth_overhead"],
            }
        ],
    )

    code, report = run_validate(graph, "--root", str(root))
    violations = [e for e in report["errors"] if e["code"] == "PERCEPTION_RUNG_VIOLATION"]
    assert code != 0 and report["ok"] is False
    assert len(violations) == 1, report
    assert violations[0]["node"] == "dora-genesis"
    assert rung in violations[0]["detail"] and "store" in violations[0]["detail"]
    assert "store namespace" in violations[0]["hint"] and "L0" in violations[0]["hint"]


def test_store_allows_supported_l0_perception_at_validation(tmp_path):
    """VAL-8, issue #130: store L0 remains supported. The compatibility
    check must not reject every store graph merely because AISLE_SCENE is
    declared on its bridge."""
    root = fixture_root(tmp_path, {"dora-genesis": {}})
    graph = write_graph(
        root,
        [
            {
                "id": "dora-genesis",
                "env": {"AISLE_SCENE": "store", "AISLE_PERCEPTION": "L0"},
                "outputs": ["poses"],
            }
        ],
    )

    code, report = run_validate(graph, "--root", str(root))
    # The minimal graph intentionally has no ADR-30 barrier, so VAL-2 now
    # rejects it on clock topology. The perception-specific assertion is that
    # supported store L0 contributes no rung error.
    assert code != 0 and report["ok"] is False, report
    assert "PERCEPTION_RUNG_VIOLATION" not in codes(report, "errors")
    assert codes(report, "errors") == {"CLOCK_COMMIT_COUNT", "CLOCK_PATH_INCOMPLETE"}


def test_perception_rung_violation_not_hidden_by_schema_error(tmp_path):
    """VAL-8 inherits VAL-6's rule: the rung violation is reported even when
    the same edge also carries a schema error. A leak that only surfaces once
    the graph is otherwise clean is a leak that ships."""
    root = fixture_root(tmp_path, {"dora-genesis": {"outputs.poses": "mystery"}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": "L1"}, "outputs": ["poses"]},
            {"id": "oracle-pose", "inputs": {"poses": "dora-genesis/poses"}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0
    assert {"PERCEPTION_RUNG_VIOLATION", "SCHEMA_UNKNOWN"} <= codes(report, "errors"), report


def test_perception_rung_table_matches_tc9():
    """VAL-8, TC-9's ladder as a table: L0 forbids nothing, L1 forbids the
    ground-truth pose, and L2 additionally forbids semantic segmentation
    while retaining ordinary sensor depth for metric geometry. The L2
    end-to-end edge case joins the golden corpus with the PR that makes
    `seg_overhead` a real bridge port."""
    from aisle.harness.validate import (
        FORBIDDEN_BY_RUNG,
        RUNG_REMEDY,
        graph_perception_rung,
    )

    assert FORBIDDEN_BY_RUNG["L0"] == ()
    assert FORBIDDEN_BY_RUNG["L1"] == ("poses",)
    assert set(FORBIDDEN_BY_RUNG["L2"]) == {"poses", "seg_overhead"}
    assert "depth_overhead" in RUNG_REMEDY["L2"]
    # declared on whichever node carries it, case-insensitively, defaulting to
    # L0, and a clean read carries no errors
    # read from the sim_bridge node, case-insensitively, defaulting to L0 when
    # nothing is declared; a clean read carries no errors
    bridge = {"b": {"provides": ["sim_bridge"]}}
    assert graph_perception_rung(
        [{"id": "a"}, {"id": "b", "env": {"AISLE_PERCEPTION": "l1"}}], bridge
    ) == ("L1", ["b"], [])
    assert graph_perception_rung([{"id": "a", "env": {}}], bridge) == ("L0", [], [])
    # VAL-2/VAL-3: no rung's remedy may name a topic that SAME rung forbids —
    # an author who follows such a hint fails the next compile. The shared
    # hint did exactly that at L2, sending the author to seg_overhead.
    for rung, remedy in RUNG_REMEDY.items():
        named = [topic for topic in FORBIDDEN_BY_RUNG[rung] if topic in remedy]
        assert not named, (rung, named, remedy)


def test_perception_rung_unknown_value_is_refused_not_ignored(tmp_path):
    """VAL-8: an AISLE_PERCEPTION value outside the table MUST be an error.

    This is the failure mode the check is least able to survive: an
    unrecognized rung forbids NOTHING, so before this was fixed a graph
    declaring `L3` while consuming ground-truth `poses` validated ok=true,
    exit 0 — the check silently absent rather than failing. The bridge
    refuses an unknown rung at launch (TC-9), so the validator, which is the
    earlier and cheaper gate, must refuse it too."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": "L3"}, "outputs": ["poses"]},
            {"id": "oracle-pose", "inputs": {"poses": "dora-genesis/poses"}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    rung_errors = [e for e in report["errors"] if e["code"] == "PERCEPTION_RUNG_VIOLATION"]
    assert any("unknown perception rung" in e["detail"] for e in rung_errors), report
    # and the forbidden edge is still named in the SAME report: a bad
    # declaration falls back to the strictest rung, not the most permissive
    assert any("consumed by" in e["detail"] for e in rung_errors), report


def test_perception_rung_whitespace_is_normalized(tmp_path):
    """VAL-8: `"L1 "` is L1. The bridge normalizes with .strip().upper(), so a
    validator that did not would disagree with the runtime about the same
    graph text — and disagree in the permissive direction, which is how a
    trailing space alone used to make an L1 graph consuming `poses` pass."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": " l1 "}, "outputs": ["poses"]},
            {"id": "oracle-pose", "inputs": {"poses": "dora-genesis/poses"}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    assert "PERCEPTION_RUNG_VIOLATION" in codes(report, "errors"), report
    assert not any("unknown perception rung" in e["detail"] for e in report["errors"]), report


def test_perception_rung_conflicting_declarations_are_refused(tmp_path):
    """VAL-8: a rung declared on a node that does NOT provide sim_bridge is an
    error. dora passes env to that node's process alone, so such a rung reaches
    nobody: the validator would forbid `poses` graph-wide while the bridge
    stayed at its default, published ground truth and rendered no
    segmentation — an L1 run starved of every pose source rather than an L1 run.

    Restricting the declaration to the bridge also makes a two-rung CONFLICT
    unreachable through the registry (one manifest provides sim_bridge, and
    node ids are unique), so the conflict branch is exercised at the function
    level below rather than through a graph."""
    from aisle.harness.validate import graph_perception_rung

    root = fixture_root(tmp_path, {"dora-genesis": {}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {
                "id": "oracle-pose",
                "env": {"AISLE_PERCEPTION": "L0"},
                "inputs": {"poses": "dora-genesis/poses"},
            },
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": "L1"}, "outputs": ["poses"]},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    assert any("does not provide sim_bridge" in e["detail"] for e in report["errors"]), report
    # kept as defense in depth for a future graph with two bridges: two rungs
    # attest neither, and resolving the ambiguity by YAML position is not a fix
    two_bridges = {"a": {"provides": ["sim_bridge"]}, "b": {"provides": ["sim_bridge"]}}
    _, _, errors = graph_perception_rung(
        [
            {"id": "a", "env": {"AISLE_PERCEPTION": "L0"}},
            {"id": "b", "env": {"AISLE_PERCEPTION": "L1"}},
        ],
        two_bridges,
    )
    assert any("conflicting perception rungs" in e["detail"] for e in errors), errors

    # PR #154 review: an omitted declaration is not neutral on a second
    # bridge — that process defaults to L0 at runtime. Treat its EFFECTIVE L0
    # as part of conflict detection instead of applying the desk bridge's L1
    # graph-wide and falsely reporting the already-L0 store as store+L1.
    rung, _, errors = graph_perception_rung(
        [
            {"id": "store", "env": {"AISLE_SCENE": "store"}},
            {
                "id": "desk",
                "env": {"AISLE_SCENE": "pharmacy", "AISLE_PERCEPTION": "L1"},
            },
        ],
        {
            "store": {"provides": ["sim_bridge"]},
            "desk": {"provides": ["sim_bridge"]},
        },
    )
    assert rung == "L2"  # strictest fallback while the declaration is invalid
    assert any("conflicting perception rungs ['L0', 'L1']" in e["detail"] for e in errors)
    assert not any("not supported for AISLE_SCENE='store'" in e["detail"] for e in errors)


def test_perception_rung_binds_nodes_without_manifests(tmp_path):
    """VAL-8 inherits VAL-6's manifest-less fallback: the rung binds the
    GRAPH, not the registry, so an unregistered node — hand-added, or injected
    by a harness — is not exempt. Measured before this was added: at L1 an
    unregistered consumer of `poses` reported MANIFEST_MISSING alone, where
    the byte-identical `oracle_state` graph reported MANIFEST_MISSING +
    ORACLE_LEAK."""
    root = fixture_root(tmp_path, {"dora-genesis": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": "L1"}, "outputs": ["poses"]},
            {"id": "my-unregistered-planner", "inputs": {"poses": "dora-genesis/poses"}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    assert {"MANIFEST_MISSING", "PERCEPTION_RUNG_VIOLATION"} <= codes(report, "errors"), report


def test_perception_rung_grants_verifiers_no_exemption(tmp_path):
    """VAL-8 has NO verifier carve-out, unlike VAL-6 sitting directly below it
    in the same function. TC-9 forbids `poses` to EVERY node at L1, and a
    verifier reading estimated-vs-ground-truth pose would be judging with the
    answer key. Pinned because the adjacent code's `not
    node_id.startswith("verifier-")` is exactly what a future edit would copy
    onto this check by analogy."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "verifier-oracle": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": "L1"}, "outputs": ["poses"]},
            {"id": "verifier-oracle", "inputs": {"oracle_state": "dora-genesis/poses"}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    assert "PERCEPTION_RUNG_VIOLATION" in codes(report, "errors"), report


def test_perception_rung_blank_declaration_is_not_absent(tmp_path):
    """VAL-8: a PRESENT but blank AISLE_PERCEPTION is an unknown rung, not an
    unset one. Skipping blanks fell through to the L0 default and permitted
    ground truth, so `AISLE_PERCEPTION: "   "` validated ok=true (Codex
    review) — the same silent-pass shape as the typo and whitespace cases."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {"id": "dora-genesis", "env": {"AISLE_PERCEPTION": "   "}, "outputs": ["poses"]},
            {"id": "oracle-pose", "inputs": {"poses": "dora-genesis/poses"}},
        ],
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    assert any("unknown perception rung" in e["detail"] for e in report["errors"]), report


def test_perception_rung_forbids_only_the_bridges_ground_truth(tmp_path):
    """VAL-8: the rung forbids the BRIDGE's `poses`, not every port that shares
    the name. An estimated pose published on a port called `poses` by a
    non-bridge node is exactly what L1 asks for, and name-only matching
    rejected it (Codex review). TC-9 says "the bridge's `poses` topic"."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "pose-estimator": {}, "oracle-pose": {}})
    graph = write_graph(
        root,
        [
            {
                "id": "dora-genesis",
                "env": {"AISLE_PERCEPTION": "L1"},
                "outputs": ["depth_overhead"],
            },
            {"id": "pose-estimator", "outputs": ["poses"]},
            {"id": "oracle-pose", "inputs": {"poses": "pose-estimator/poses"}},
        ],
    )
    _, report = run_validate(graph, "--root", str(root))
    assert "PERCEPTION_RUNG_VIOLATION" not in codes(report, "errors"), report


def test_perception_rung_yaml_null_declaration_is_refused(tmp_path):
    """VAL-8: `AISLE_PERCEPTION:` with no value parses from YAML as None. That
    is a graph DECLARING a rung, so it must refuse rather than inherit L0 and
    permit ground truth. Membership, not `is None`: each narrower version of
    this check let one more shape through (Codex review)."""
    root = fixture_root(tmp_path, {"dora-genesis": {}, "oracle-pose": {}})
    graph = root / "g.yaml"
    graph.write_text(
        "nodes:\n"
        "- id: dora-genesis\n"
        "  env:\n"
        "    AISLE_PERCEPTION:\n"
        "  outputs: [poses]\n"
        "- id: oracle-pose\n"
        "  inputs: {poses: dora-genesis/poses}\n"
    )
    code, report = run_validate(graph, "--root", str(root))
    assert code != 0, report
    assert any("unknown perception rung" in e["detail"] for e in report["errors"]), report


def test_expert_t4_is_good():
    """VAL-7 for graphs/expert_t4.yaml, validated in place: the T4
    dialogue baseline (ADR-32 increment one) passes NORMAL validation —
    episode_goal reaches only verifier-oracle, the state machine takes
    its task from human-sim/human_msg, and the episode_meta/robot_msg
    edges pass the schema checks."""
    code, report = run_validate(REPO_ROOT / "graphs" / "expert_t4.yaml")
    assert code == 0, report
    assert report["ok"] is True and report["errors"] == []
    assert report["warnings"] == []


def test_expert_t4_with_goal_routed_to_policy_is_rejected(tmp_path):
    """DIALOGUE_GOAL_LEAK mutation-proofing on the REAL graph (ADR-32 §1):
    take expert_t4.yaml verbatim and re-add the episode_goal edge every
    other tier wires into the task-state-machine — the exact un-blinding
    the rule exists to prevent (the goal's target_med is the final
    CORRECTED answer) — and the validator must reject it."""
    doc = yaml.safe_load((REPO_ROOT / "graphs" / "expert_t4.yaml").read_text())
    for node in doc["nodes"]:
        # keep node paths resolvable from the copy's location
        node["path"] = str((REPO_ROOT / "graphs" / node["path"]).resolve())
        if node["id"] == "task-state-machine":
            node["inputs"]["episode_goal"] = {
                "source": "rollout-client/episode_goal",
                "queue_size": 100,
            }
    mutated = tmp_path / "expert_t4_goal_leak.yaml"
    mutated.write_text(yaml.safe_dump(doc, sort_keys=False))

    code, report = run_validate(mutated, "--root", str(REPO_ROOT))
    assert code != 0 and report["ok"] is False
    leaks = [e for e in report["errors"] if e["code"] == "DIALOGUE_GOAL_LEAK"]
    assert leaks, report["errors"]
    assert any(e.get("node") == "task-state-machine" for e in leaks), leaks
