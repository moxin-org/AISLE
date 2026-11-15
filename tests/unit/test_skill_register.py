"""T18: skill registration (design doc §8.4 item 1, §3 rule 3, §9.2;
CAP-5/CAP-7 curated-core amendment) — `harness skill register` validates the
skill (CAP-1..3), STAGES the candidate into the registry so its own eval
can discover it (the PR #30 blocker), runs the shipped eval suite bound
to the candidate, writes the evalcard (CAP-6), and rolls back exactly on
any failure. Governance: curated ids refused from the single-sourced
Class-C list regardless of file state. Rollouts are injected so these
tests never touch sim (CON-12); CON-8 for the CLI contract; CON-5 via
the injected clock.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

from aisle.harness.skill import (  # noqa: E402
    REGISTRY_MIN_PASS_RATE,
    RegistrationError,
    curated_ids,
    load_skill,
    register_skill,
)


def _write_skill(
    root: Path,
    name: str = "pour-arc",
    *,
    manifest_extra: dict | None = None,
    eval_extra: dict | None = None,
    omit_eval: bool = False,
    graph_nodes: list | None = None,
) -> Path:
    """A minimal, schema-valid node skill with a shipped eval suite whose
    graph USES the candidate (a node with the skill's id)."""
    d = root / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": name,
        "kind": "node",
        "provides": ["pour_control"],
        "requires": ["object_pose"],
        "inputs": {"object_pose": {"schema": "pose7d_f32", "rate_hz": 30}},
        "outputs": {"joint_cmd": {"schema": "jointvec_f32", "latency_class": "soft_rt"}},
        "embodiment": {"arm": ["franka"], "gripper": "parallel"},
        "safety_class": "motion",
        "eval": None,  # written by registration
        "origin": "agent-authored",
        "source": f"skills/{name}/node.py",
        **(manifest_extra or {}),
    }
    (d / "skill.yaml").write_text(yaml.safe_dump(manifest))
    (d / "node.py").write_text("def main():\n    pass\n")
    if not omit_eval:
        eval_cfg = {
            "suite": f"{name}_eval_v0",
            "graph": f"skills/{name}/eval_graph.yaml",
            "tier": "T1",
            "episodes": 4,
            "seeds": "0..3",
            "embodiment": "franka",
            "min_pass_rate": 0.5,
            **(eval_extra or {}),
        }
        (d / "eval.yaml").write_text(yaml.safe_dump(eval_cfg))
        nodes = graph_nodes if graph_nodes is not None else [{"id": name, "path": "node.py"}]
        (d / "eval_graph.yaml").write_text(yaml.safe_dump({"nodes": nodes}))
    return d


# ADR-5: a node whose outputs are actuation commands MUST be safety_class
# motion. Composed with §9.4's rule that sandbox may not be motion, that
# means a sandbox node can never emit actuation at all — so a sandbox
# fixture has to be a non-actuating shape.
_PERCEPTION_SHAPE = {
    "safety_class": "perception",
    "provides": ["object_pose_estimation"],
    "outputs": {"object_pose": {"schema": "pose7d_f32", "latency_class": "soft_rt"}},
}


def _registry(root: Path) -> Path:
    """A registry root with the REAL schema files (incl. the curated-core
    list) and one curated manifest."""
    (root / "registry" / "manifests").mkdir(parents=True)
    (root / "registry" / "schema").mkdir(parents=True)
    for f in (REPO_ROOT / "registry" / "schema").glob("*"):
        (root / "registry" / "schema" / f.name).write_text(f.read_text())
    core = (REPO_ROOT / "registry" / "manifests" / "oracle-pose.yaml").read_text()
    (root / "registry" / "manifests" / "oracle-pose.yaml").write_text(core)
    return root


def _ok_rollout(report=None):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return report or {"ok": True, "pass1": 0.75, "pass8": 0.75, "episodes": [{}] * 4}

    fake.calls = calls
    return fake


def test_register_writes_evalcard_and_installs_manifest(tmp_path):
    """§8.4: validate → stage → eval → evalcard → install."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root)
    fake = _ok_rollout()
    result = register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert result["ok"] is True and result["pass_rate"] == 0.75
    installed = yaml.safe_load((root / "registry" / "manifests" / "pour-arc.yaml").read_text())
    assert installed["eval"] == {
        "suite": "pour-arc_eval_v0",
        "pass_rate": 0.75,
        "last_run": "2026-07-25",
    }
    assert fake.calls and fake.calls[0]["tier"] == "T1" and fake.calls[0]["episodes"] == 4


def test_candidate_is_staged_and_discoverable_during_its_own_eval(tmp_path):
    """THE PR #30 blocker: during the eval rollout the candidate manifest
    must be IN the registry (with a provisional evalcard so CAP-6's
    motion gate passes) — proven with the REAL validator: unstaged, the
    eval graph gets MANIFEST_MISSING; at eval time, it validates."""
    from aisle.harness.validate import validate

    root = _registry(tmp_path)
    skill_dir = _write_skill(root)
    graph = skill_dir / "eval_graph.yaml"

    # unstaged: the real validator cannot discover the candidate
    before = validate(graph, root, "franka", allow_unproven=False)
    assert any(e["code"] == "MANIFEST_MISSING" for e in before["errors"]), before

    seen = {}

    def probing_rollout(**kwargs):
        staged = yaml.safe_load((root / "registry" / "manifests" / "pour-arc.yaml").read_text())
        seen["staged"] = staged
        report = validate(graph, root, "franka", allow_unproven=False)
        seen["missing_at_eval"] = [e for e in report["errors"] if e["code"] == "MANIFEST_MISSING"]
        return {"ok": True, "pass1": 1.0, "episodes": [{}]}

    register_skill(skill_dir, root, run_rollout=probing_rollout, now="2026-07-25")
    assert seen["missing_at_eval"] == []  # discoverable during its own eval
    assert seen["staged"]["eval"]["last_run"] == "provisional-2026-07-25"
    assert "(provisional)" in seen["staged"]["eval"]["suite"]  # CAP-6 pass, labelled


def test_update_evaluates_the_new_candidate_not_the_old(tmp_path):
    """PR #30: re-registering evaluates the STAGED new manifest — the old
    installed version must not shadow it during the eval."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root)
    register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-24")
    # v2 changes provides; the rollout must see v2 staged
    _write_skill(root, manifest_extra={"provides": ["pour_control_v2"]})
    seen = {}

    def probing_rollout(**kwargs):
        staged = yaml.safe_load((root / "registry" / "manifests" / "pour-arc.yaml").read_text())
        seen["provides"] = staged["provides"]
        return {"ok": True, "pass1": 1.0, "episodes": [{}]}

    register_skill(skill_dir, root, run_rollout=probing_rollout, now="2026-07-25")
    assert seen["provides"] == ["pour_control_v2"]


def test_failed_eval_rolls_back_exactly(tmp_path):
    """Any failure restores the registry byte-for-byte: a fresh candidate
    vanishes; an update keeps its PRIOR manifest."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root)
    bad = _ok_rollout(report={"ok": False, "refused": {"gate": "validate"}})
    with pytest.raises(RegistrationError, match="eval run failed"):
        register_skill(skill_dir, root, run_rollout=bad, now="2026-07-25")
    assert not (root / "registry" / "manifests" / "pour-arc.yaml").exists()

    # update case: v1 installs, v2's eval fails below threshold -> v1 stays
    register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-24")
    v1 = (root / "registry" / "manifests" / "pour-arc.yaml").read_text()
    _write_skill(root, eval_extra={"min_pass_rate": 0.99})
    with pytest.raises(RegistrationError, match="pass_rate"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")
    assert (root / "registry" / "manifests" / "pour-arc.yaml").read_text() == v1


def test_eval_graph_must_use_the_candidate(tmp_path):
    """PR #30: an unrelated green graph cannot mint the evalcard — the
    eval graph must contain a node with the candidate's id."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, graph_nodes=[{"id": "oracle-pose", "path": "x.py"}])
    fake = _ok_rollout()
    with pytest.raises(RegistrationError, match="does not use the candidate"):
        register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert not fake.calls


def test_curated_id_refused_even_when_core_file_deleted(tmp_path):
    """CAP-7: core protection comes from the Class-C curated list, not
    mutable file presence — deleting the core manifest opens nothing."""
    root = _registry(tmp_path)
    (root / "registry" / "manifests" / "oracle-pose.yaml").unlink()
    skill_dir = _write_skill(root, name="oracle-pose")
    with pytest.raises(RegistrationError, match="CURATED core"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")
    assert "oracle-pose" in curated_ids(root)


def test_register_refuses_non_agent_authored(tmp_path):
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, manifest_extra={"origin": "hub"})
    with pytest.raises(RegistrationError, match="origin"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")


def test_register_refuses_schema_invalid_before_any_eval(tmp_path):
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, manifest_extra={"safety_class": "chaotic"})
    fake = _ok_rollout()
    with pytest.raises(RegistrationError, match="schema"):
        register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert not fake.calls


def test_register_refuses_missing_eval_suite(tmp_path):
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, omit_eval=True)
    with pytest.raises(RegistrationError, match="eval"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")


def test_malformed_eval_fields_are_registration_errors(tmp_path):
    """PR #30 (CON-8): bad seed specs and non-numeric fields refuse
    cleanly — never a traceback."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra={"seeds": "abc"})
    with pytest.raises(RegistrationError, match="invalid"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")
    skill_dir = _write_skill(root, name="pour-arc2", eval_extra={"episodes": "many"})
    with pytest.raises(RegistrationError, match="invalid"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")
    (root / "skills" / "pour-arc3").mkdir(parents=True)
    (root / "skills" / "pour-arc3" / "skill.yaml").write_text("{bad: yaml: [")
    with pytest.raises(RegistrationError, match="YAML"):
        register_skill(root / "skills" / "pour-arc3", root, run_rollout=_ok_rollout(), now="x")


def test_a_self_declared_floor_below_the_registry_floor_is_refused(tmp_path):
    """CAP-6 (#243): `min_pass_rate` is the CANDIDATE's number, so without an
    absolute floor the exam is self-graded — a skill shipping 0.0 registers
    at pass_rate 0.0 and the gate reports ok. That is how t2-scan-tsm entered
    a campaign library at 0.0. The refusal lands at load, BEFORE any rollout
    is spent."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra={"min_pass_rate": 0.0})
    fake = _ok_rollout()
    with pytest.raises(RegistrationError, match="below the registry floor"):
        register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert not fake.calls, "the floor must refuse before spending an eval rollout"
    assert not (root / "registry" / "manifests" / "pour-arc.yaml").exists()


def test_a_stricter_self_declared_floor_still_governs(tmp_path):
    """CAP-6 (#243): the floor is a MINIMUM, not a replacement — a skill that
    holds itself to 0.9 is still refused at 0.75. The fix must not quietly
    loosen any skill's own bar."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra={"min_pass_rate": 0.9})
    with pytest.raises(RegistrationError, match="0.900"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-25")


@pytest.mark.parametrize(
    "minimum",
    [float("nan"), float("inf"), float("-inf"), 1.1, None, "invalid", True, 10**400],
    ids=["nan", "inf", "negative-inf", "above-one", "null", "text", "bool", "overflow"],
)
def test_invalid_pass_rate_threshold_is_refused_before_eval(tmp_path, minimum):
    """CAP-6, CON-8: invalid probability thresholds cannot bypass certification or spend eval."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra={"min_pass_rate": minimum})
    fake = _ok_rollout({"ok": True, "pass1": 0.0})
    with pytest.raises(RegistrationError, match="min_pass_rate"):
        register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert not fake.calls
    assert not (root / "registry" / "manifests" / "pour-arc.yaml").exists()


@pytest.mark.parametrize(
    "measured",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1, None, "invalid", True, 10**400],
    ids=["nan", "inf", "negative-inf", "negative", "above-one", "null", "text", "bool", "overflow"],
)
@pytest.mark.parametrize("updating", [False, True], ids=["new", "update"])
def test_invalid_measured_pass_rate_rolls_back_registration(tmp_path, measured, updating):
    """CAP-6, CAP-7, CON-8: invalid eval rates never certify a skill and restore prior state."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root)
    installed = root / "registry" / "manifests" / "pour-arc.yaml"
    if updating:
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-07-24")
    prior = installed.read_bytes() if installed.exists() else None
    fake = _ok_rollout({"ok": True, "pass1": measured})
    with pytest.raises(RegistrationError, match="measured pass_rate"):
        register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert len(fake.calls) == 1
    assert (installed.read_bytes() if installed.exists() else None) == prior


@pytest.mark.parametrize("rate", [0.5, 1.0, "0.5"])
def test_valid_rate_at_declared_threshold_is_certified(tmp_path, rate):
    """CAP-6: equality at valid probability boundaries and numeric strings remains accepted."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra={"min_pass_rate": rate})
    result = register_skill(
        skill_dir, root, run_rollout=_ok_rollout({"ok": True, "pass1": rate}), now="2026-07-25"
    )
    assert result["ok"] is True
    assert result["trust_tier"] == "reviewed"
    assert result["pass_rate"] == float(rate)


def test_every_shipped_skill_meets_the_registry_floor(tmp_path):
    """CAP-6 (#243): the floor is calibrated against the corpus, not picked in
    the abstract — every skill already in the library must satisfy it, or the
    change is a silent de-registration of mainline work."""
    for eval_path in (REPO_ROOT / "skills").glob("*/eval.yaml"):
        declared = float(yaml.safe_load(eval_path.read_text())["min_pass_rate"])
        assert declared >= REGISTRY_MIN_PASS_RATE, (
            f"{eval_path.parent.name} declares {declared}, below the floor "
            f"{REGISTRY_MIN_PASS_RATE} — the floor would evict a registered skill"
        )


def test_same_day_retries_get_distinct_run_ids(tmp_path):
    """PR #30: the evaluate-fix loop must allow a second attempt the same
    day — default run ids are uniquified."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root)
    fake = _ok_rollout()
    r1 = register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    r2 = register_skill(skill_dir, root, run_rollout=fake, now="2026-07-25")
    assert r1["eval_run_id"] != r2["eval_run_id"]
    assert fake.calls[0]["run_id"] != fake.calls[1]["run_id"]


def test_load_skill_reads_manifest_and_eval(tmp_path):
    root = _registry(tmp_path)
    skill = load_skill(_write_skill(root))
    assert skill.manifest["id"] == "pour-arc"
    assert skill.eval_cfg["min_pass_rate"] == 0.5


def test_cli_surface_parses():
    """CON-8: the CLI subcommand exists with the documented flags."""
    from aisle.harness.cli import build_parser

    args = build_parser().parse_args(["skill", "register", "skills/pour-arc"])
    assert args.command == "skill" and args.skill_command == "register"


@pytest.mark.parametrize("eval_extra", [{"seeds": "abc"}, {"min_pass_rate": float("nan")}])
def test_cli_register_json_contract(tmp_path, eval_extra):
    """CON-8 end to end: JSON on stdout, exit 1 on a refused registration
    (malformed seeds or threshold — no rollout machinery touched)."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra=eval_extra)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisle.harness.cli",
            "skill",
            "register",
            str(skill_dir),
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 1
    out = json.loads(proc.stdout)
    assert out["ok"] is False and "invalid" in out["error"]


def test_sandbox_registration_admits_an_id_without_claiming_quality(tmp_path):
    """CAP-6/CAP-7 (#265, ADR-40): ADR-37's floor closed a real hole and
    removed the only mechanism an agent had for a different, legitimate need
    — a new node has no manifest, so `validate` refuses the graph with
    MANIFEST_MISSING, and the only path to a manifest now requires a measured
    0.5. An agent that wants to RUN a new node must already have one that
    works.

    Campaign agents hit this and invented a workaround: both recovered T2
    skills shipped `min_pass_rate: 0.0` with a written rationale, using
    registration as ATTESTATION rather than certification. §9.4 names the
    missing tier (`sandbox -> reviewed -> certified`); this is it."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, manifest_extra=_PERCEPTION_SHAPE)
    fake = _ok_rollout()

    result = register_skill(skill_dir, root, run_rollout=fake, now="2026-08-17", sandbox=True)

    assert result["ok"] is True and result["trust_tier"] == "sandbox"
    assert not fake.calls, "a sandbox registration makes no quality claim, so it runs no eval"
    installed = yaml.safe_load((root / "registry" / "manifests" / "pour-arc.yaml").read_text())
    assert installed["trust_tier"] == "sandbox"
    assert installed["eval"] is None, "a sandbox entry must not carry an evalcard"


def test_a_sandbox_skill_may_never_hold_the_motion_class(tmp_path):
    """§9.4 (#265): the trust-tier roadmap's whole point is per-tier
    safety-class ceilings — an uncertified skill can never command the arm.
    The default `_write_skill` is motion, so this is the common case, not an
    edge one."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root)  # safety_class: motion
    with pytest.raises(RegistrationError, match="sandbox"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-08-17", sandbox=True)
    assert not (root / "registry" / "manifests" / "pour-arc.yaml").exists()


def test_sandbox_does_not_bypass_the_floor_for_a_real_registration(tmp_path):
    """#265 must not become a hole in #243. The default path is unchanged:
    a sub-floor declaration still refuses, and a sandbox entry cannot be
    promoted by re-registering without an eval."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, eval_extra={"min_pass_rate": 0.0})
    with pytest.raises(RegistrationError, match="below the registry floor"):
        register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-08-17")


def test_promoting_a_sandbox_entry_requires_passing_the_real_gate(tmp_path):
    """#265: the tier is a ladder, not a parking space. Re-registering the
    same id WITHOUT --sandbox must run the eval and apply the floor — and on
    success the entry stops being sandbox."""
    root = _registry(tmp_path)
    skill_dir = _write_skill(root, manifest_extra=_PERCEPTION_SHAPE)
    register_skill(skill_dir, root, run_rollout=_ok_rollout(), now="2026-08-17", sandbox=True)

    fake = _ok_rollout()
    result = register_skill(skill_dir, root, run_rollout=fake, now="2026-08-18")
    assert fake.calls, "promotion must actually run the eval"
    assert result["trust_tier"] == "reviewed"
    installed = yaml.safe_load((root / "registry" / "manifests" / "pour-arc.yaml").read_text())
    assert installed["eval"]["pass_rate"] == 0.75 and installed["trust_tier"] == "reviewed"
