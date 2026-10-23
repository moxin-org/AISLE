"""Unit tests for the capability registry (SPEC 050 CAP-1..6, CON-8).

Acceptance tests named by the spec: test_all_lint (CAP-1..3),
test_search_cli_json (CAP-4), test_registry_completeness (CAP-5).
"""

import json
import subprocess

import pytest
import yaml
from cli_helpers import REPO_ROOT, make_registry_root, run_json, run_module, write_manifest

pytestmark = pytest.mark.unit

MANIFESTS_DIR = REPO_ROOT / "registry" / "manifests"

EXPECTED_IDS = {
    "camera-source",
    "oracle-pose",
    "detector-openvocab",
    "ocr-label",
    "pose-estimator",
    "grasp-planner-topdown",
    "human-sim",
    "ik-trajectory",
    "arm-driver-sim",
    "gripper-driver-sim",
    "task-state-machine",
    "turn-barrier",  # ADR-30 terminal run-to-quiescence coordinator
    "verifier-oracle",
    "vla-policy-smolvla",
    "verifier-realistic",  # A7's ADR-30 verdict-bearing participant
    "reset",
    "return-planner",
    "budget-guard",  # SPEC 080 BG-1 (T07)
    "dora-genesis",  # executable bridge identity (T08)
    "so101-driver",  # Phase 6 prep: the hardware bridge (ADR-phase6-prep)
    "g1-driver",  # G1-D: the G1 hardware bridge (ADR-g1-driver)
    "rollout-client",  # episode driver for runnable graphs (T08)
    "nav-action",  # SPEC 210 MOB-2 navigation action (T11)
    # T14 retail capability extension (design doc §11.4, ADR-17)
    "base-driver-sim",
    "waypoint-nav",
    "patrol-planner",
    "order-reader",
    "stock-detector",
    "misplacement-detector",
    "placement-controller",
    "task-planner",
    # T15 S1 expert graph (ADR-18)
    "verifier-retail",
    "s1-expert",
    # perception rung L1 (TC-9): the expert_t1 pose source
    "segmented-pose",
    # perception rung L2 (TC-9, idea I7): the expert_t1_l2 pose source
    "detected-pose",
}


def run_registry_raw(*args: str) -> subprocess.CompletedProcess:
    return run_module("aisle.harness.registry", *args)


def run_registry(*args: str) -> tuple[int, dict]:
    """Run the registry CLI; return (exit code, parsed JSON stdout)."""
    return run_json("aisle.harness.registry", *args)


@pytest.fixture(scope="module")
def repo_lint() -> subprocess.CompletedProcess:
    """One lint run over the committed registry, shared by the tests that
    assert different properties of the same invocation."""
    return run_registry_raw("lint")


make_root = make_registry_root


def valid_manifest(**overrides) -> dict:
    manifest = {
        "id": "fixture-node",
        "kind": "node",
        "provides": ["fixture_ability"],
        "requires": [],
        "inputs": {"rgb": {"schema": "rgb8_image", "rate_hz": 30}},
        "outputs": {"result": {"schema": "json_utf8", "latency_class": "best_effort"}},
        "embodiment": {"arm": ["franka", "so101"], "gripper": "parallel"},
        "safety_class": "perception",
        "eval": None,
        "origin": "hub",
        "source": "src/aisle/nodes/fixture.py",
    }
    manifest.update(overrides)
    return manifest


def test_all_lint(repo_lint):
    """CAP-1, CAP-2, CAP-3: every committed manifest validates against
    capability.schema.json and the closed schema vocabulary; lint is a CLI
    with a single JSON object on stdout and exit 0 iff ok (CON-8)."""
    report = json.loads(repo_lint.stdout)
    assert repo_lint.returncode == 0, report
    assert report["ok"] is True
    # every file on disk is linted: the curated core (EXPECTED_IDS,
    # pinned exactly by test_curated_core_and_agent_extras) plus any
    # registered CAP-7 agent skills — a hardcoded 26 went stale the
    # moment s1-driver-v2 landed (PR #54 review)
    assert report["checked"] == len(list(MANIFESTS_DIR.glob("*.yaml")))
    assert report["checked"] >= len(EXPECTED_IDS)
    assert report["errors"] == []


def test_search_cli_json():
    """CAP-4: search --provides grasp_planning returns matching manifests as
    JSON (CON-8); --embodiment filters by supported arm."""
    code, report = run_registry("search", "--provides", "grasp_planning")
    assert code == 0
    assert report["ok"] is True
    assert [m["id"] for m in report["matches"]] == ["grasp-planner-topdown"]

    code, report = run_registry("search", "--provides", "grasp_planning", "--embodiment", "franka")
    assert code == 0
    assert [m["id"] for m in report["matches"]] == ["grasp-planner-topdown"]

    code, report = run_registry("search", "--provides", "object_pose")
    ids = {m["id"] for m in report["matches"]}
    # the perception ladder (TC-9), now complete: L0 oracle passthrough,
    # L1 estimated from ground-truth segmentation, L2 detected on rendered
    # rgb — plus the abstract hub estimator exemplar
    assert ids == {
        "oracle-pose",
        "segmented-pose",
        "detected-pose",
        "pose-estimator",
        # registered 2026-08-24: the T2 breakthrough stack (0.5 suite)
        "t2-scan-pose",
    }


def test_search_no_match_is_ok_empty():
    """CAP-4, CON-8: an unmatched --provides returns ok with an empty matches
    list, not an error."""
    code, report = run_registry("search", "--provides", "does_not_exist")
    assert code == 0
    assert report["ok"] is True
    assert report["matches"] == []


def test_registry_completeness():
    """CAP-5 as evolved by T18: the CURATED core is pinned exactly, and any
    manifest beyond it must be a REGISTERED SKILL — origin agent-authored
    WITH a written evalcard (the §8.4 governance path is the only way in).
    The deliberate gap holds: no CURATED capability provides rearrangement
    (an agent-authored rearrangement skill is the intended fill, §3)."""
    import tomllib

    curated_file = REPO_ROOT / "registry" / "schema" / "curated_core.toml"
    curated = set(tomllib.loads(curated_file.read_text())["core"])
    assert curated == EXPECTED_IDS, "curated_core.toml drifted from the CAP-5 list"
    files = sorted(MANIFESTS_DIR.glob("*.yaml"))
    manifests = [yaml.safe_load(f.read_text()) for f in files]
    ids = {m["id"] for m in manifests}
    assert EXPECTED_IDS <= ids, f"curated core missing: {EXPECTED_IDS - ids}"
    for m in manifests:
        if m["id"] not in EXPECTED_IDS:
            assert m["origin"] == "agent-authored", m["id"]
            assert m["eval"] is not None, f"{m['id']} installed without an evalcard"
    core_provides = {p for m in manifests if m["id"] in EXPECTED_IDS for p in m["provides"]}
    assert not any("rearrang" in p for p in core_provides)


CAP1_REQUIRED = [
    "id",
    "kind",
    "provides",
    "requires",
    "inputs",
    "outputs",
    "embodiment",
    "safety_class",
    "eval",
    "origin",
    "source",
]


def test_schema_required_set_matches_cap1():
    """CAP-1: the JSON Schema requires exactly the CAP-1 field set. Optional
    top-level fields are `params` and — since ADR-40 (#265) — `trust_tier`.

    `trust_tier` is deliberately OPTIONAL rather than required with a
    default: every curated-core and pre-ADR-40 manifest predates it, and
    making it required would force a mechanical edit to the frozen registry
    to express something those entries do not claim. Absent means "no tier
    recorded", which is the truth for them."""
    schema = json.loads((REPO_ROOT / "registry" / "schema" / "capability.schema.json").read_text())
    assert set(schema["required"]) == set(CAP1_REQUIRED)
    assert set(schema["properties"]) - set(schema["required"]) == {
        "params",
        "trust_tier",
        "specializes",
        "applies_when",
    }


@pytest.mark.parametrize("field", CAP1_REQUIRED)
def test_lint_rejects_missing_required_field(tmp_path, field):
    """CAP-1: a manifest missing any required field fails lint with an error
    naming the manifest."""
    root = make_root(tmp_path)
    bad = valid_manifest()
    del bad[field]
    if field == "id":
        (root / "registry" / "manifests" / "anon.yaml").write_text(
            yaml.safe_dump(bad, sort_keys=False)
        )
    else:
        write_manifest(root, bad)
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert report["ok"] is False


def test_oracle_state_has_a_producer():
    """CAP-5: the bridge's observation facet (camera-source) declares the
    oracle_state output from the SPEC 010 topic table, so the verifier's
    permitted input has a producer when graphs are validated."""
    manifest = yaml.safe_load((MANIFESTS_DIR / "camera-source.yaml").read_text())
    assert manifest["outputs"]["oracle_state"]["schema"] == "posearray7d_f32"


def test_hot_topics_are_not_json():
    """CON-4: no manifest port at >=10 Hz uses the json_utf8 schema — JSON
    is allowed only on low-rate goal/result/report topics."""
    for f in sorted(MANIFESTS_DIR.glob("*.yaml")):
        manifest = yaml.safe_load(f.read_text())
        for port, spec in manifest.get("inputs", {}).items():
            if spec["schema"] == "json_utf8":
                assert spec["rate_hz"] < 10, f"{f.stem}/{port} is a hot JSON topic"


def test_lint_rejects_unknown_schema_name(tmp_path):
    """CAP-2: a schema name outside registry/schema/schemas.toml is a lint
    error, never silently passed."""
    root = make_root(tmp_path)
    bad = valid_manifest()
    bad["inputs"]["rgb"]["schema"] = "made_up_schema"
    write_manifest(root, bad)
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert any("made_up_schema" in e["message"] for e in report["errors"])


def test_lint_accepts_clock_and_episodic_input_declarations(tmp_path):
    """CAP-1/CAP-2: lockstep clocks and episodic back-edges are closed,
    typed manifest fields rather than graph comments or conventions."""
    root = make_root(tmp_path)
    manifest = valid_manifest()
    manifest["inputs"] = {
        "turn": {"schema": "sim_turn_u64", "rate_hz": 100, "is_clock": True},
        "result": {"schema": "json_utf8", "rate_hz": 1, "turn_edge": "episodic"},
    }
    write_manifest(root, manifest)
    code, report = run_registry("lint", "--root", str(root))
    assert code == 0, report


@pytest.mark.parametrize("bad_value", [0, 1, "true", None, [], {}])
def test_lint_rejects_non_boolean_is_clock(tmp_path, bad_value):
    """CAP-1: is_clock is exactly boolean; truthy coercions are forbidden."""
    root = make_root(tmp_path)
    manifest = valid_manifest()
    manifest["inputs"]["rgb"]["is_clock"] = bad_value
    write_manifest(root, manifest)
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert report["ok"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "daemon"},
        {"safety_class": "dangerous"},
        {"origin": "vendored"},
        {"embodiment": {"arm": ["ur5"], "gripper": "parallel"}},
        {"embodiment": {"arm": ["franka"], "gripper": "paralell"}},
        {"outputs": {"result": {"schema": "json_utf8", "latency_class": "asap"}}},
    ],
)
def test_lint_rejects_bad_enum_value(tmp_path, overrides):
    """CAP-1: every closed enum (kind, safety_class, origin, arm, gripper,
    latency_class) rejects values outside its set."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest(**overrides))
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert report["ok"] is False


def test_lint_rejects_malformed_vocabulary(tmp_path):
    """CAP-2: a schemas.toml entry that is not exactly an {arrow, shape}
    string mapping is a lint error, so the vocabulary itself stays closed."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest())
    (root / "registry" / "schema" / "schemas.toml").write_text(
        '[rgb8_image]\narrow = "UInt8"\nshape = "h*w*3"\n\n'
        '[json_utf8]\narrow = "Utf8"\nshape = "1"\n\n[broken]\nshape = "1"\n'
    )
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert any("broken" in e["message"] for e in report["errors"])


def test_lint_enforces_eval_rule(tmp_path):
    """CAP-6: eval may be null only while origin=hub and safety_class is not
    motion; a motion manifest with null eval fails lint."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest(safety_class="motion"))
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert any("eval" in e["message"] for e in report["errors"])


def test_lint_eval_rule_agent_authored(tmp_path):
    """CAP-6: agent-authored capabilities may not have null eval regardless
    of safety class."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest(origin="agent-authored"))
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert any("eval" in e["message"] for e in report["errors"])


def test_lint_accepts_populated_eval_for_motion(tmp_path):
    """CAP-6: a motion capability with a populated evalcard passes lint."""
    root = make_root(tmp_path)
    write_manifest(
        root,
        valid_manifest(
            safety_class="motion",
            eval={"suite": "tc_a1_a3", "pass_rate": 0.98, "last_run": "2026-07-18"},
        ),
    )
    code, report = run_registry("lint", "--root", str(root))
    assert code == 0, report


def test_sim_driver_evalcards_shipped_and_carveout_retired(repo_lint):
    """CAP-6 end state (ADR-3 retired at T08): the sim-driver manifests
    carry evalcards generated from the passing acceptance suite runs
    (tests/accept/test_contract.py),
    lint reports ZERO warnings, and the pending carve-out is an empty
    tombstone — the T10 gate's condition, met early."""
    from aisle.harness.registry import PENDING_M0_EVALCARDS

    report = json.loads(repo_lint.stdout)
    assert repo_lint.returncode == 0
    assert report["warnings"] == []
    assert PENDING_M0_EVALCARDS == set()
    for mid in ("arm-driver-sim", "gripper-driver-sim", "dora-genesis"):
        manifest = yaml.safe_load((MANIFESTS_DIR / f"{mid}.yaml").read_text())
        assert manifest["eval"] is not None, mid
        assert manifest["eval"]["pass_rate"] == 1.0
        assert "TC-A1..A3" in manifest["eval"]["suite"]


def test_bad_root_reported_as_json(tmp_path):
    """CON-8: lint and search with a --root missing schema files or the
    manifests dir emit a JSON error report and exit nonzero — no tracebacks,
    no silent ok-with-zero-results."""
    for command in (["lint"], ["search", "--provides", "grasp_planning"]):
        code, report = run_registry(*command, "--root", str(tmp_path / "nowhere"))
        assert code != 0
        assert report["ok"] is False


def test_lint_empty_manifests_dir_fails(tmp_path):
    """CAP-3: an empty registry is a lint error, never a green gate over
    nothing."""
    root = make_root(tmp_path)
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert report["ok"] is False


def test_search_scalar_provides_is_not_substring_matched(tmp_path):
    """CAP-4: a manifest whose provides is a YAML scalar (invalid per CAP-1)
    is never substring-matched by search."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest(provides="grasp_planning"))
    code, report = run_registry("search", "--root", str(root), "--provides", "grasp")
    assert code == 0
    assert report["matches"] == []


def test_search_survives_malformed_manifest_fields(tmp_path):
    """CON-8: search over manifests with malformed or missing fields
    (scalar embodiment, arm: null, no id) still returns a JSON report
    instead of crashing."""
    root = make_root(tmp_path)
    bad = valid_manifest(embodiment="franka")  # not a mapping
    del bad["id"]
    (root / "registry" / "manifests" / "anon.yaml").write_text(yaml.safe_dump(bad, sort_keys=False))
    write_manifest(root, valid_manifest(id="null-arm", embodiment={"arm": None, "gripper": "any"}))
    code, report = run_registry(
        "search", "--root", str(root), "--provides", "fixture_ability", "--embodiment", "franka"
    )
    assert code == 0
    assert report["matches"] == []


def test_search_serializes_yaml_dates(tmp_path):
    """CON-8: an unquoted YAML date (parsed as datetime.date) in a matching
    manifest is serialized in the JSON report, not a TypeError traceback."""
    root = make_root(tmp_path)
    path = root / "registry" / "manifests" / "fixture-node.yaml"
    manifest = valid_manifest(eval={"suite": "s", "pass_rate": 0.9, "last_run": "placeholder"})
    text = yaml.safe_dump(manifest, sort_keys=False).replace("placeholder", "2026-07-18")
    path.write_text(text)
    code, report = run_registry("search", "--root", str(root), "--provides", "fixture_ability")
    assert code == 0
    assert report["matches"][0]["eval"]["last_run"] == "2026-07-18"


def test_lint_findings_logged_to_stderr():
    """CON-8: lint findings are visible on stderr, not only inside the
    JSON report — exercised against the eval_null fixture root, whose
    eval-null motion drivers are ERRORS now that the ADR-3 carve-out is
    retired (no warning class remains in lint)."""
    proc = run_registry_raw(
        "lint", "--root", str(REPO_ROOT / "tests" / "fixtures" / "roots" / "eval_null")
    )
    assert proc.returncode == 1
    assert "arm-driver-sim" in proc.stderr
    assert "CAP-6" in proc.stderr


def test_lint_rejects_duplicate_ids(tmp_path):
    """CAP-5: two manifest files with the same id fail lint (search results
    must be unambiguous)."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest())
    dup = valid_manifest()
    (root / "registry" / "manifests" / "other-file.yaml").write_text(
        yaml.safe_dump(dup, sort_keys=False)
    )
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert report["ok"] is False


def test_search_annotates_launchability_and_installed_filter(tmp_path):
    """CAP-4 (issue #39): every search match carries `launchable` —
    search advertised uninstalled pip: nodes with no flag, the exact
    discovery-surface gap analysis/h1 documents — and --installed
    narrows to what can actually launch. Fixture root: no ambient-env
    coupling (issue #37 discipline)."""
    root = make_registry_root(tmp_path)
    absent = valid_manifest()
    absent.update(
        id="detector-absent",
        provides=["fixture_ability"],
        source="pip:aisle-search-reserved-absent",
    )
    write_manifest(root, absent)
    ghost = valid_manifest()
    ghost.update(
        id="detector-ghost", provides=["fixture_ability"], source="src/aisle/nodes/ghost.py"
    )
    path = root / "registry" / "manifests" / "detector-ghost.yaml"
    path.write_text(yaml.safe_dump(ghost, sort_keys=False))  # no auto-stub: stays ghost
    real = valid_manifest()
    real.update(id="detector-real", provides=["fixture_ability"])
    write_manifest(root, real)  # helper stubs the path source: launchable
    # PR #66 review: the installed-pip POSITIVE branch — an
    # implementation returning False for every pip source must fail
    # here, not survive on path-source cases alone (pytest is installed
    # by construction: it is running this test)
    installed_pip = valid_manifest()
    installed_pip.update(
        id="detector-pip-installed", provides=["fixture_ability"], source="pip:pytest"
    )
    write_manifest(root, installed_pip)

    code, report = run_registry("search", "--provides", "fixture_ability", "--root", str(root))
    assert code == 0
    launchable = {m["id"]: m["launchable"] for m in report["matches"]}
    assert launchable == {
        "detector-absent": False,  # uninstalled pip dist
        "detector-ghost": False,  # path source names no file
        "detector-pip-installed": True,  # pip source installed => launchable
        "detector-real": True,
    }

    code, report = run_registry(
        "search", "--provides", "fixture_ability", "--root", str(root), "--installed"
    )
    assert code == 0
    assert [m["id"] for m in report["matches"]] == ["detector-pip-installed", "detector-real"]


def test_pip_installed_probe_is_cached():
    """PR #66 review: a silent duplicate definition once overwrote the
    @functools.cache probe — repeated metadata scans per validation and
    the loss of one-snapshot consistency. Pin the cache: two calls for
    the same dist perform ONE metadata scan, and cache_clear exists."""
    import importlib.metadata as md

    from aisle.harness import registry as reg

    assert hasattr(reg._pip_installed, "cache_clear"), "probe lost its functools.cache"
    calls = {"n": 0}
    real = md.version

    def counting(dist):
        calls["n"] += 1
        return real(dist)

    reg._pip_installed.cache_clear()
    original = md.version
    md.version = counting
    try:
        assert reg._pip_installed("pytest") is True
        assert reg._pip_installed("pytest") is True
        assert calls["n"] == 1  # second call served from the cache
    finally:
        md.version = original
        reg._pip_installed.cache_clear()


def test_actuation_command_emitters_are_motion_with_evalcards():
    """ADR-5 safety-class boundary, RATIFIED 2026-08-03 (owner decision,
    PR #81 review): motion means EMITS ACTUATION COMMANDS. Every registry
    manifest whose outputs include a joint/gripper/base command (raw or
    guard-gated *_safe) must be safety_class motion and carry an evalcard
    (CAP-6 then requires eval non-null for the class)."""
    from aisle.harness.registry import ACTUATION_PORTS as actuation

    emitters = []
    for path in sorted(MANIFESTS_DIR.glob("*.yaml")):
        manifest = yaml.safe_load(path.read_text())
        if set(manifest.get("outputs") or {}) & actuation:
            emitters.append(manifest["id"])
            assert manifest["safety_class"] == "motion", (
                f"{manifest['id']} emits actuation commands but is "
                f"safety_class={manifest['safety_class']!r} (ADR-5 ratified)"
            )
            card = manifest.get("eval")
            # a vacuous card must not satisfy CAP-6: require the full shape
            assert isinstance(card, dict) and {"suite", "pass_rate", "last_run"} <= set(card), (
                f"{manifest['id']} is motion without a complete evalcard (CAP-6): {card!r}"
            )
    # the boundary decision covers the full emitter set, not one skill
    assert {
        "budget-guard",
        "ik-trajectory",
        "nav-action",
        "s1-driver-v2",
        "s1-expert",
        "s3-driver-v1",
        "waypoint-nav",
    } <= set(emitters)


def test_lint_rejects_nonmotion_actuation_emitter(tmp_path):
    """ADR-5 (ratified 2026-08-03): registry lint enforces the boundary at
    runtime, not only in CI — a decision-class manifest with actuation
    outputs must fail lint, or a mid-campaign `skill register` could drift
    the registry from the ratified policy unnoticed."""
    root = make_registry_root(tmp_path)
    write_manifest(
        root,
        valid_manifest(
            outputs={"joint_cmd": {"schema": "jointvec_f32", "latency_class": "hard_rt"}},
            safety_class="decision",
        ),
    )
    code, report = run_registry("lint", "--root", str(root))
    assert code != 0
    assert any("motion" in e["message"] and "ADR-5" in e["message"] for e in report["errors"])


def test_a_registered_skill_and_its_manifest_have_not_drifted():
    """CAP-1/CAP-7 (issue #225): `registry/manifests/<id>.yaml` is DERIVED
    from `skills/<id>/skill.yaml` — `harness skill register` writes it with
    `installed.write_text(yaml.safe_dump(_provisional_manifest(...)))`, and
    `_provisional_manifest` is the skill's own manifest with only `eval`
    replaced.

    Both files are checked in and edited by hand, and nothing compared them.
    That matters because `harness validate` reads the MANIFEST while
    re-registering rewrites it FROM the skill: if the two drift, an agent
    running the documented loop (`harness skill register skills/<id>`,
    CLAUDE.research.md) silently reverts the manifest to the stale copy.

    #219 made that concrete by adding ADR-30 lockstep declarations
    (`turn`, `turn_done`, `turn_edge: episodic`) to both files for the two
    retail drivers. Losing them from the manifest turns a validated graph
    into one that dies at the first watermark.

    `eval` is excluded and only `eval`: the registry carries the evaluated
    card, the skill ships `eval: null` until it earns one."""
    checked = []
    for skill_dir in sorted((REPO_ROOT / "skills").iterdir()):
        skill_yaml = skill_dir / "skill.yaml"
        if not skill_yaml.is_file():
            continue
        skill = yaml.safe_load(skill_yaml.read_text())
        manifest_path = MANIFESTS_DIR / f"{skill.get('id', skill_dir.name)}.yaml"
        if not manifest_path.is_file():
            continue  # registered skills only; an unregistered skill has no copy yet
        manifest = yaml.safe_load(manifest_path.read_text())
        drifted = {
            key
            for key in set(manifest) | set(skill)
            if key != "eval" and manifest.get(key) != skill.get(key)
        }
        assert not drifted, (
            f"{manifest_path.name} and {skill_yaml.relative_to(REPO_ROOT)} disagree on "
            f"{sorted(drifted)} — `harness skill register` would overwrite the manifest "
            "from the skill and drop whatever the manifest has that the skill lacks"
        )
        checked.append(skill_dir.name)
    assert checked, "no registered skill has a skill.yaml — this test went blind"


def test_a_sandbox_entry_is_never_counted_as_a_library_skill():
    """CAP-5/CAP-7 as amended by ADR-40 (#265): a sandbox entry exists so a
    graph naming its id VALIDATES. It must never satisfy anything that counts
    capabilities — the DoD skill count, reuse measurement, or the registry's
    "beyond the curated core, only registered skills" rule.

    The filter is the EVALCARD, not a second rule that could drift: a sandbox
    entry has `eval: null` by construction, and everything that counts skills
    already requires a non-null evalcard."""
    import tomllib

    curated = set(
        tomllib.loads((REPO_ROOT / "registry" / "schema" / "curated_core.toml").read_text())["core"]
    )
    manifests = [yaml.safe_load(f.read_text()) for f in sorted(MANIFESTS_DIR.glob("*.yaml"))]
    for m in manifests:
        if m.get("trust_tier") == "sandbox":
            assert m["eval"] is None, f"{m['id']}: a sandbox entry may not carry an evalcard"
            assert m["safety_class"] != "motion", f"{m['id']}: §9.4 per-tier ceiling"
            assert m["id"] not in curated, f"{m['id']}: the curated core is not sandboxable"
        # the converse: anything WITH an evalcard is at least reviewed
        if m["id"] not in curated and m.get("eval") is not None:
            assert m.get("trust_tier", "reviewed") != "sandbox"


def test_specializes_must_name_a_real_capability_sibling(tmp_path):
    """CAP-1 as amended by ADR-41 (#264): an evalcard says a skill WORKED on
    a suite; it never says when reaching for it is appropriate. With two
    entries under one capability at the same headline pass_rate, an agent has
    no basis to choose, and the accumulation benefit inverts as the library
    grows.

    `specializes` is the part of that gap a machine can check: it must name
    an EXISTING manifest that shares a capability. A dangling or unrelated
    reference is an error, not documentation — a field the validator cannot
    check is the failure mode #264 warns about."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest(id="base-cap", provides=["nav"]))
    write_manifest(root, valid_manifest(id="refined", provides=["nav"], specializes="base-cap"))
    report = json.loads(run_module("aisle.harness.registry", "lint", "--root", str(root)).stdout)
    assert report["ok"] is True, report["errors"]

    write_manifest(root, valid_manifest(id="dangling", provides=["nav"], specializes="nope"))
    report = json.loads(run_module("aisle.harness.registry", "lint", "--root", str(root)).stdout)
    # the DANGLING branch specifically: "nope" alone would also match the
    # shares-no-capability message, so a mutation deleting this branch
    # survived until the assertion named the reason (mutation-caught)
    assert any("not a registered manifest id" in e["message"] for e in report["errors"]), report


def test_specializes_must_share_a_capability_and_not_be_self_referential(tmp_path):
    """ADR-41 (#264): 'specializes' means "a narrower way to do the SAME
    thing". Pointing at an unrelated capability makes search's hierarchy
    meaningless, and pointing at yourself makes it a cycle of one."""
    root = make_root(tmp_path)
    write_manifest(root, valid_manifest(id="unrelated", provides=["grasping"]))
    write_manifest(root, valid_manifest(id="crosswired", provides=["nav"], specializes="unrelated"))
    report = json.loads(run_module("aisle.harness.registry", "lint", "--root", str(root)).stdout)
    assert any("capability" in e["message"] for e in report["errors"]), report

    root2 = make_root(tmp_path / "b")
    write_manifest(root2, valid_manifest(id="selfref", provides=["nav"], specializes="selfref"))
    report = json.loads(run_module("aisle.harness.registry", "lint", "--root", str(root2)).stdout)
    assert any("itself" in e["message"] for e in report["errors"]), report


def test_search_surfaces_what_an_agent_needs_to_choose():
    """CAP-4 (#264): the fields exist to be READ at the moment of choosing.
    Search returns whole manifests, so `specializes` and `applies_when` ride
    along — but this asserts it, because the gap was never that the data
    could not be stored, it was that nothing put it in front of the
    decision."""
    code, report = run_registry("search", "--provides", "trajectory_generation")
    assert code == 0
    ids = {m["id"] for m in report["matches"]}
    assert {"ik-trajectory", "ik-transfer-v2"} <= ids, ids
    # every match carries the choice-relevant keys, present or explicitly null
    for m in report["matches"]:
        assert "applies_when" in m and "specializes" in m, m["id"]
