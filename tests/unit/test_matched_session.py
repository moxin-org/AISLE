"""MON-8/MON-13: controller-side admission binds actual treatment artifacts."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from test_treatment_integrity import _candidate

from aisle.harness import monolith

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def prepared_pair(tmp_path, candidate="t1-oracle"):
    """Use the launcher's real table format in an isolated engineering fixture;
    the arm views and editable grants follow the named candidate's allowlist."""
    from aisle.harness.candidates import read_candidates
    from aisle.harness.matched_session import CONTROLLER_FILES

    control = tmp_path / "controller"
    table = json.loads((ROOT / "docs/monolithic/treatment-table.json").read_text())
    paths = {
        path
        for row in table["rows"]
        for arm in ("typed", "monolithic")
        for path in row[arm]["paths"]
    }
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "docs/monolithic").glob("*.json"))
    paths.update({"graphs/expert_t1.yaml", "graphs/monolithic_t1.yaml"})
    paths.update(CONTROLLER_FILES)
    rows = read_candidates(ROOT)[0]["candidates"]
    allowlist = json.loads((ROOT / rows[candidate]["documents"]["allowlist"]).read_text())
    for arm in ("typed", "monolithic"):
        paths.update(allowlist[arm]["editable"])
    # every declared candidate's artifacts and editable files (MON-8 document sets)
    from aisle.harness.candidates import candidate_artifacts

    for row in rows.values():
        paths.update(candidate_artifacts(row))
        candidate_allowlist = json.loads((ROOT / row["documents"]["allowlist"]).read_text())
        candidate_table = json.loads((ROOT / row["documents"]["treatment_table"]).read_text())
        for arm in ("typed", "monolithic"):
            paths.update(candidate_allowlist[arm]["editable"])
            paths.update(p for r in candidate_table["rows"] for p in r[arm]["paths"])
    for name in paths:
        target = control / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    assert monolith.table_report(control, write=True)["ok"]
    for row in rows.values():
        assert monolith.table_report(control, write=True, documents=row["documents"])["ok"]
    roots, candidates = {}, {}
    for arm in ("typed", "monolithic"):
        view = tmp_path / arm
        view.mkdir()
        candidate = _candidate(view)
        (view / "src/worker.py").unlink()
        (view / "src").rmdir()
        editable = sorted(allowlist[arm]["editable"])
        for name in editable:
            target = view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(control / name, target)
        candidate["assignment"]["arm"] = arm
        candidate["repository"]["visible_allowlist"] = sorted(["AGENTS.md", *editable])
        candidate["repository"]["editable_allowlist"] = editable
        candidates[arm], roots[arm] = candidate, view
    return control, candidates, roots


def test_pair_admission_binds_launcher_artifacts_without_claiming_study_readiness(tmp_path):
    """MON-8/MON-13/CSE-10: engineering admission is not independent gate evidence."""
    from aisle.harness.matched_session import admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    record = admit_pair(control, candidates, roots)
    assert record["schema_version"] == "aisle.matched-session-plan.v1"
    assert record["confirmatory_ready"] is False
    assert set(record["arms"]) == {"typed", "monolithic"}
    assert record["surface"]["treatment_table_id"]
    assert record["surface"]["interface_map_id"]
    assert record["surface"]["artifact_hashes"]
    assert all(arm["immutable_id"].startswith("sha256:") for arm in record["arms"].values())


@pytest.mark.parametrize("field", ["model", "budget", "policy", "confinement"])
def test_pair_admission_rejects_shared_identity_differences(tmp_path, field):
    """MON-8: model, budgets and authority cannot silently differ between arms."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    candidates = copy.deepcopy(candidates)
    candidates["monolithic"][field]["undeclared_difference"] = "different"
    with pytest.raises(AdmissionError, match="shared"):
        admit_pair(control, candidates, roots)


def test_pair_admission_rejects_extra_edit_grants(tmp_path):
    """MON-3/MON-8/MON-13: a sealed manifest cannot grant an undeclared helper."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    candidates["monolithic"]["repository"]["editable_allowlist"].append("AGENTS.md")
    with pytest.raises(AdmissionError, match="editable"):
        admit_pair(control, candidates, roots)


def test_pair_admission_rejects_stale_controller_surface(tmp_path):
    """MON-13: stale launcher table hashes block participant admission."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    broker = control / "src/aisle/nodes/monolith_broker.py"
    broker.write_text(broker.read_text() + "\n# drift\n")
    with pytest.raises(AdmissionError, match="surface"):
        admit_pair(control, candidates, roots)


def test_admission_rejects_overlapping_controller_and_participant_roots(tmp_path):
    """MON-6/MON-13: the participant view must not contain its trusted controller."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    roots["typed"] = tmp_path
    with pytest.raises(AdmissionError, match="^controller and participant roots overlap"):
        admit_pair(control, candidates, roots)


def test_admission_rejects_readonly_typed_facilities_in_monolithic_view(tmp_path):
    """MON-3: denying edit permission alone does not hide the typed registry."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    path = roots["monolithic"] / "registry/manifests/helper.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("name: forbidden\n")
    candidate = candidates["monolithic"]
    candidate["repository"]["visible_allowlist"].append("registry/manifests/helper.yaml")
    candidate["repository"]["visible_allowlist"].sort()
    with pytest.raises(AdmissionError, match="typed facilities"):
        admit_pair(control, candidates, roots)


def test_retained_plan_rechecks_actual_view_before_launch(tmp_path):
    """MON-13: admission at an earlier time cannot authorize changed executable bytes."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    record = admit_pair(control, candidates, roots)
    assert verify_plan(record, control, roots) == record
    module = roots["monolithic"] / "experts/monolithic/expert_t1.py"
    module.write_text(module.read_text() + "\n# changed after admission\n")
    with pytest.raises(AdmissionError, match="drift"):
        verify_plan(record, control, roots)


@pytest.mark.parametrize("arm", ["typed", "monolithic"])
@pytest.mark.parametrize("returncode", [0, 3])
def test_session_retains_real_process_and_common_evidence(tmp_path, arm, returncode):
    """MON-12: both arms retain real subprocess output, including failed attempts."""
    import subprocess
    import sys

    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    access = tmp_path / "controller-access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )

    def launch(output):
        # Controller-owned deterministic process, not a coding-agent baseline.
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"print('{json.dumps({'type': 'fixture', 'text': 'fixture event'})}'); "
                f"raise SystemExit({returncode})",
            ],
            cwd=roots[arm],
            capture_output=True,
            text=True,
            check=False,
        )
        (output / "session.jsonl").write_text(proc.stdout)
        (output / "session.stderr").write_text(proc.stderr)
        return {
            "rc": proc.returncode,
            "tokens": 0,
            "wall_s": 0.01,
            "classification": "agent_outcome"
            if proc.returncode == 0
            else "infrastructure_exclusion",
        }

    output = tmp_path / "retained"
    record = execute_session(
        plan,
        control,
        roots,
        arm,
        output,
        session_id="fixture-1",
        launch=launch,
        hidden_access_log=access,
    )
    assert record["ok"] is (returncode == 0)
    assert record["eligible_for_estimate"] is False
    assert record["arm"] == arm
    assert record["process"]["rc"] == returncode
    assert record["snapshots"]["authored"] and record["snapshots"]["final"]
    assert record["artifacts"]["session.jsonl"]
    assert record["postflight"]["confirmatory_ready"] is False
    assert json.loads((output / "matched-session.json").read_text()) == record
    assert json.loads((output / "session.jsonl").read_text())["text"] == "fixture event"


def test_drift_before_launch_is_retained_without_starting_process(tmp_path):
    """MON-13: a changed view is rejected and the rejected attempt remains recorded."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    (roots["typed"] / "AGENTS.md").write_text("changed contract")
    calls = []
    output = tmp_path / "rejected"
    record = execute_session(
        plan,
        control,
        roots,
        "typed",
        output,
        session_id="rejected-1",
        launch=lambda out: calls.append(out),
        hidden_access_log=tmp_path / "absent",
    )
    assert record["ok"] is False and not calls
    assert record["classification"] == "infrastructure_exclusion"
    assert (output / "matched-session.json").is_file()


def test_existing_attempt_cannot_be_overwritten_or_resumed(tmp_path):
    """MON-11/MON-13: retry must use a new retained attempt, not overwrite evidence."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("retain")
    with pytest.raises(AdmissionError, match="existing"):
        execute_session(
            plan,
            control,
            roots,
            "typed",
            output,
            session_id="retry",
            launch=lambda _: pytest.fail("must not launch"),
            hidden_access_log=tmp_path / "absent",
        )
    assert (output / "sentinel").read_text() == "retain"


def _access_log(tmp_path):
    path = tmp_path / "access.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )
    return path


def test_launcher_failure_detail_survives_postflight(tmp_path):
    """MON-12: the originating runner failure is retained alongside its process record."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)

    class FailedRunner(RuntimeError):
        record = {"rc": 7, "classification": "infrastructure_exclusion", "tokens": 12}

    def launch(output):
        (output / "session.jsonl").write_text("partial transcript\n")
        raise FailedRunner("telemetry stream truncated")

    record = execute_session(
        plan,
        control,
        roots,
        "typed",
        tmp_path / "attempt",
        session_id="failed-runner",
        launch=launch,
        hidden_access_log=_access_log(tmp_path),
    )
    assert record["ok"] is False
    assert "telemetry stream truncated" in record["error"]
    assert record["process"]["tokens"] == 12
    assert record["postflight"] is not None


def test_missing_final_deliverable_does_not_suppress_postflight(tmp_path):
    """MON-12/MON-13: destructive participant edits retain the independent audit outcome."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)

    def launch(output):
        (output / "session.jsonl").write_text("removed deliverable\n")
        (roots["monolithic"] / "experts/monolithic/expert_t1.py").unlink()
        return {"rc": 0}

    record = execute_session(
        plan,
        control,
        roots,
        "monolithic",
        tmp_path / "attempt",
        session_id="missing-final",
        launch=launch,
        hidden_access_log=_access_log(tmp_path),
    )
    assert record["ok"] is False
    assert record["postflight"]["classification"] == "infrastructure_exclusion"
    assert record["artifacts"]["session.jsonl"]


def test_access_evidence_is_retained_not_just_referenced(tmp_path):
    """MON-12: postflight can be reconstructed after the adapter's temporary log expires."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    access = _access_log(tmp_path)
    expected = access.read_bytes()
    output = tmp_path / "attempt"
    record = execute_session(
        plan,
        control,
        roots,
        "typed",
        output,
        session_id="access-retention",
        launch=lambda _: {"rc": 0},
        hidden_access_log=access,
    )
    access.unlink()
    assert (output / "hidden-access-log.json").read_bytes() == expected
    assert (
        record["artifacts"]["hidden-access-log.json"]
        == record["postflight"]["hidden_access_log"]["sha256"]
    )


def _confinement_pair(tmp_path, control, candidates, roots):
    import sys

    from aisle.harness.treatment_confinement import MacOSPolicy, compile_macos_profile

    binary = Path(sys.executable).resolve()
    scratch = {arm: tmp_path / f"{arm}-scratch" for arm in roots}
    for path in scratch.values():
        path.mkdir()
    bindings = {}
    for arm in roots:
        other = next(name for name in roots if name != arm)
        policy = MacOSPolicy(
            visible_roots=(roots[arm].resolve(),),
            output_roots=tuple(
                (roots[arm] / p).resolve()
                for p in candidates[arm]["repository"]["editable_allowlist"]
            )
            + (scratch[arm].resolve(),),
            runtime_read_roots=(binary.parent.parent,),
            allowed_executables=(binary,),
            hidden_roots=(control.resolve(), roots[other].resolve(), scratch[other].resolve()),
            network_policy="deny-external",
        )
        compiled = compile_macos_profile(policy)
        candidates[arm]["confinement"]["profile_sha256"] = compiled.sha256
        candidates[arm]["confinement"]["policy_sha256"] = compiled.policy_id
        bindings[arm] = {"policy": policy.canonical_dict(), "scratch": str(scratch[arm].resolve())}
    return bindings


def test_equivalent_permissions_bind_distinct_compiled_arm_profiles(tmp_path):
    """MON-8/MON-13: arm-specific roots differ while concrete hashes remain checked."""
    from aisle.harness.matched_session import admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    assert (
        candidates["typed"]["confinement"]["profile_sha256"]
        != candidates["monolithic"]["confinement"]["profile_sha256"]
    )
    plan = admit_pair(control, candidates, roots, confinement=bindings)
    assert plan["confinement_bindings"] == bindings
    assert verify_plan(plan, control, roots) == plan


def test_confinement_binding_refuses_unbound_profile_hash(tmp_path):
    """MON-13: semantic equivalence cannot excuse a changed concrete profile hash."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    candidates["typed"]["confinement"]["profile_sha256"] = "0" * 64
    with pytest.raises(AdmissionError, match="profile identity"):
        admit_pair(control, candidates, roots, confinement=bindings)


def test_confinement_binding_refuses_extra_write_authority(tmp_path):
    """MON-6/MON-8: mapped roots cannot conceal an extra writable directory."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    extra = tmp_path / "extra-write"
    extra.mkdir()
    bindings["monolithic"]["policy"]["output_roots"].append(str(extra.resolve()))
    with pytest.raises(AdmissionError, match="write authority"):
        admit_pair(control, candidates, roots, confinement=bindings)


@pytest.mark.parametrize("permission", ["runtime_read_roots", "allowed_executables"])
def test_extra_permissions_fail_even_with_valid_recompiled_hashes(tmp_path, permission):
    """MON-8: hash-valid profiles with unequal read or execute authority are not matched."""
    from aisle.harness.matched_session import AdmissionError, admit_pair
    from aisle.harness.treatment_confinement import MacOSPolicy, compile_macos_profile

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    if permission == "runtime_read_roots":
        extra = tmp_path / "additional-read"
        extra.mkdir()
    else:
        extra = Path("/bin/echo").resolve()
    declared = bindings["monolithic"]["policy"]
    declared[permission].append(str(extra))
    policy = MacOSPolicy(
        **{
            key: value if key == "network_policy" else tuple(Path(p) for p in value)
            for key, value in declared.items()
        }
    )
    compiled = compile_macos_profile(policy)
    candidates["monolithic"]["confinement"].update(
        profile_sha256=compiled.sha256, policy_sha256=compiled.policy_id
    )
    with pytest.raises(AdmissionError, match="shared"):
        admit_pair(control, candidates, roots, confinement=bindings)


def test_invalid_compiled_policy_is_an_admission_error(tmp_path):
    """MON-6/MON-13: rejected OS policy input uses the retained admission failure path."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    bindings["typed"]["policy"]["network_policy"] = "allow-all"
    with pytest.raises(AdmissionError, match="invalid confinement policy"):
        admit_pair(control, candidates, roots, confinement=bindings)


def _ambient_pair(candidates, bindings):
    from aisle.harness.matched_session import _private_tree_hash
    from aisle.harness.treatment_ambient import build_declared_environment

    ambient = {}
    for arm in candidates:
        home = Path(bindings[arm]["scratch"]) / "home"
        environment, record = build_declared_environment(home, source_env={"PATH": "/usr/bin:/bin"})
        candidates[arm]["state"]["environment_baseline_sha256"] = record["environment_sha256"]
        for variable, field in (
            ("HOME", "home_baseline_sha256"),
            ("XDG_CONFIG_HOME", "config_baseline_sha256"),
            ("XDG_CACHE_HOME", "cache_baseline_sha256"),
        ):
            candidates[arm]["state"][field] = _private_tree_hash(Path(environment[variable]))
        ambient[arm] = {"environment": environment, "record": record}
    return ambient


def test_distinct_private_homes_have_verified_equivalent_environments(tmp_path):
    """MON-8/TRT-1: private HOME paths can differ without hiding environment drift."""
    from aisle.harness.matched_session import admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    assert ambient["typed"]["environment"]["HOME"] != ambient["monolithic"]["environment"]["HOME"]
    plan = admit_pair(control, candidates, roots, confinement=bindings, ambient=ambient)
    assert verify_plan(plan, control, roots) == plan


def test_environment_difference_is_not_normalized_as_a_home_path(tmp_path):
    """MON-8: executable search authority remains matched after legitimate HOME binding."""
    from aisle.harness.matched_session import AdmissionError, admit_pair
    from aisle.harness.treatment_ambient import build_declared_environment

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    home = Path(bindings["monolithic"]["scratch"]) / "alternate-home"
    environment, record = build_declared_environment(home, source_env={"PATH": "/bin"})
    candidates["monolithic"]["state"]["environment_baseline_sha256"] = record["environment_sha256"]
    ambient["monolithic"] = {"environment": environment, "record": record}
    with pytest.raises(AdmissionError, match="shared"):
        admit_pair(control, candidates, roots, confinement=bindings, ambient=ambient)


def test_generated_environment_directory_cannot_be_replaced_by_symlink(tmp_path):
    """MON-6/MON-13: unchanged environment strings cannot hide a redirected state directory."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    plan = admit_pair(control, candidates, roots, confinement=bindings, ambient=ambient)
    cache = Path(ambient["typed"]["environment"]["XDG_CACHE_HOME"])
    cache.rmdir()
    cache.symlink_to(ambient["monolithic"]["environment"]["HOME"], target_is_directory=True)
    with pytest.raises(AdmissionError, match="ambient generated path"):
        verify_plan(plan, control, roots)


def _representation_documents(control, candidates, roots):
    table = monolith.load_json(control, "treatment-table.json")
    row = next(row for row in table["rows"] if row["surface"] == "documentation given to the agent")
    for arm in candidates:
        paths = row[arm]["paths"]
        for name in paths:
            target = roots[arm] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(control / name, target)
        candidates[arm]["repository"]["visible_allowlist"] = sorted(
            set(candidates[arm]["repository"]["visible_allowlist"]) | set(paths)
        )
        candidates[arm]["prompts"]["research_contract_sha256"] = monolith._set_digest(
            control, paths
        )
    return row


def test_declared_representation_documents_allow_only_the_bound_prompt_difference(tmp_path):
    """MON-8: representation instructions may differ only at the declared artifact identities."""
    from aisle.harness.matched_session import admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    row = _representation_documents(control, candidates, roots)
    plan = admit_pair(control, candidates, roots, prompt_row=row["id"])
    assert plan["prompt_row"] == row["id"]
    assert verify_plan(plan, control, roots) == plan


def test_visible_representation_document_must_match_controller_bytes(tmp_path):
    """MON-13: a correct declared hash does not excuse altered instructions in the view."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    row = _representation_documents(control, candidates, roots)
    (roots["monolithic"] / row["monolithic"]["paths"][0]).write_text("extra private hint")
    with pytest.raises(AdmissionError, match="representation document"):
        admit_pair(control, candidates, roots, prompt_row=row["id"])


def test_representation_binding_does_not_ignore_shared_system_prompt_changes(tmp_path):
    """MON-8: declaring representation-specific documentation never waives shared instructions."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    row = _representation_documents(control, candidates, roots)
    candidates["monolithic"]["prompts"]["system_sha256"] = "0" * 64
    with pytest.raises(AdmissionError, match="shared"):
        admit_pair(control, candidates, roots, prompt_row=row["id"])


def _launch_pair(candidates):
    import hashlib
    import sys

    executable = Path(sys.executable).resolve()
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    for candidate in candidates.values():
        candidate["agent"]["cli_binary_sha256"] = digest
        candidate["budget"]["wall_ceiling_s"] = 30
    return {
        arm: {
            "argv": [
                str(executable),
                "-c",
                "print('fixture')",
                "system prompt",
                "research contract",
            ],
            "system_prompt_arg": 3,
            "research_contract_arg": 4,
        }
        for arm in candidates
    }


def test_launch_binding_rechecks_actual_executable(tmp_path):
    """MON-8/MON-13: process executable bytes must match the admitted agent binary."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _launch_pair(candidates)
    plan = admit_pair(control, candidates, roots, launches=launches)
    assert plan["launch_bindings"] == launches
    assert verify_plan(plan, control, roots) == plan
    candidates["typed"]["agent"]["cli_binary_sha256"] = "0" * 64
    with pytest.raises(AdmissionError, match="executable"):
        admit_pair(control, candidates, roots, launches=launches)


def test_launch_binding_rejects_undeclared_argument_difference(tmp_path):
    """MON-8: a shared model declaration cannot conceal different process arguments."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _launch_pair(candidates)
    launches["monolithic"]["argv"].append("--extra-authority")
    with pytest.raises(AdmissionError, match="launch arguments"):
        admit_pair(control, candidates, roots, launches=launches)


@pytest.mark.parametrize("ceiling", [None, 0, -1, True, float("inf")])
def test_launch_binding_requires_finite_positive_wall_budget(tmp_path, ceiling):
    """MON-8: the process runner must receive a concrete bounded wall budget."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _launch_pair(candidates)
    for candidate in candidates.values():
        candidate["budget"]["wall_ceiling_s"] = ceiling
    with pytest.raises(AdmissionError, match="wall budget"):
        admit_pair(control, candidates, roots, launches=launches)


@pytest.mark.parametrize("arm", ["typed", "monolithic"])
@pytest.mark.parametrize("returncode", [0, 3])
@pytest.mark.parametrize("via_cli", [False, True])
@pytest.mark.parametrize("frontend_ceiling", [None, 1])
def test_bound_runner_retains_real_process_streams(
    tmp_path, monkeypatch, arm, returncode, via_cli, frontend_ceiling
):
    """MON-12/MON-13: both adapters retain actual process telemetry, including failed attempts.

    The pass-through adapter is an explicit engineering fixture, not OS
    confinement evidence or a study admission attestation.
    """
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import campaign
    from matched_campaign import run_engineering_session
    from test_treatment_confinement import _attestation

    from aisle.harness.matched_session import admit_pair
    from aisle.harness.treatment_confinement import MacOSPolicy, compile_macos_profile

    monkeypatch.setattr(campaign, "POLL_S", 0.01)
    control, candidates, roots = prepared_pair(tmp_path)
    launches = _launch_pair(candidates)
    for launch in launches.values():
        launch["argv"][2] = (
            "import sys,json; "
            "print(json.dumps({'type':'item.completed','item':{'id':'prompt','type':"
            "'agent_message','text':'fixture event\\n'+sys.argv[1]+'\\n'+sys.argv[2]}})); "
            f"raise SystemExit({returncode})"
        )
        if frontend_ceiling is not None:
            launch["argv"][2] = (
                "import sys,json; "
                "print(json.dumps({'type':'item.completed','item':{'id':'prompt','type':"
                "'agent_message','text':sys.argv[1]+'\\n'+sys.argv[2]}})); "
                "print(json.dumps({'type':'item.completed',"
                "'item':{'id':'tool','type':'file_change'}})); "
                f"raise SystemExit({returncode})"
            )
    if frontend_ceiling is not None:
        for candidate in candidates.values():
            candidate["budget"]["frontend_tool_ceiling"] = frontend_ceiling
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    adapter = tmp_path / "fixture-adapter"
    adapter.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    adapter.chmod(0o755)
    for candidate in candidates.values():
        candidate["confinement"]["adapter_binary_sha256"] = hashlib.sha256(
            adapter.read_bytes()
        ).hexdigest()
    declared = bindings[arm]["policy"]
    policy = MacOSPolicy(
        **{
            key: value if key == "network_policy" else tuple(Path(p) for p in value)
            for key, value in declared.items()
        }
    )
    compiled = compile_macos_profile(policy)
    profile = tmp_path / "fixture.sb"
    profile.write_text(compiled.text)
    attestation = _attestation(compiled, profile, adapter)
    plan = admit_pair(
        control, candidates, roots, confinement=bindings, ambient=ambient, launches=launches
    )
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )
    output = tmp_path / "attempt"
    if via_cli:
        request = tmp_path / "run-request.json"
        request.write_text(
            json.dumps(
                {
                    "plan": plan,
                    "root": str(control),
                    "visible_roots": {name: str(path) for name, path in roots.items()},
                    "profile_path": str(profile),
                    "attestation": attestation,
                    "hidden_access_log": str(access),
                }
            )
        )
        completed = _matched_cli(
            "run",
            "--request",
            str(request),
            "--output",
            str(output),
            "--arm",
            arm,
            "--session-id",
            "fixture-bound",
        )
        result = json.loads(completed.stdout)
        assert (completed.returncode == 0) is (returncode == 0)
    else:
        result = run_engineering_session(
            plan,
            control,
            roots,
            arm,
            output,
            session_id="fixture-bound",
            profile_path=profile,
            attestation=attestation,
            hidden_access_log=access,
        )
    assert result["ok"] is (returncode == 0)
    assert result["eligible_for_estimate"] is False
    assert result["process"]["rc"] == returncode
    transcript = (output / "session.jsonl").read_text()
    if frontend_ceiling is None:
        assert (
            json.loads(transcript)["item"]["text"]
            == "fixture event\nsystem prompt\nresearch contract"
        )
    else:
        events = [json.loads(line) for line in transcript.splitlines()]
        assert events[0]["item"]["text"] == "system prompt\nresearch contract"
        live = json.loads((output / "frontend-live.json").read_text())
        assert live["source"] == "live_stdout_pipe"
        assert live["observed_calls"] == 1
        assert result["artifacts"]["frontend-live.json"]
    assert result["artifacts"]["session-record.json"]
    assert result["artifacts"]["token_samples.jsonl"]
    for name in ("launch.json", "launch-profile.sb", "capability.json"):
        assert result["artifacts"][name]
    launch_record = json.loads((output / "launch.json").read_text())
    assert launch_record["argv"] == launches[arm]["argv"]
    assert launch_record["system_prompt_delivery"] == {
        "transport": "argv",
        "argument_index": 3,
        "sha256": plan["arms"][arm]["prompts"]["system_sha256"],
        "provider_role_verified": False,
    }
    assert launch_record["research_contract_delivery"] == {
        "transport": "argv",
        "argument_index": 4,
        "identity_sha256": plan["arms"][arm]["prompts"]["research_contract_sha256"],
        "encoding": "text",
        "frontend_interpretation_verified": False,
    }
    assert launch_record["cwd"] == str(roots[arm])
    assert launch_record["budget"] == plan["arms"][arm]["budget"]


def _matched_cli(*argv):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, str(ROOT / "tools/matched_campaign.py"), *argv],
        capture_output=True,
        text=True,
        check=False,
    )


def test_matched_cli_emits_json_for_malformed_request(tmp_path):
    """CON-8: unresolved controller requests produce one JSON error and nonzero exit."""
    request = tmp_path / "request.json"
    request.write_text("[]")
    result = _matched_cli(
        "admit", "--request", str(request), "--output", str(tmp_path / "plan.json")
    )
    assert result.returncode != 0
    assert json.loads(result.stdout)["ok"] is False
    assert "Traceback" not in result.stderr


def test_matched_cli_admits_and_rejects_study_collection_with_retained_attempt(tmp_path):
    """MON-8/CSE-10/CON-8: the CLI binds real inputs without granting study admission."""
    control, candidates, roots = prepared_pair(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "root": str(control),
                "visible_roots": {arm: str(path) for arm, path in roots.items()},
                "candidates": candidates,
            }
        )
    )
    plan_path = tmp_path / "plan.json"
    admitted = _matched_cli("admit", "--request", str(request), "--output", str(plan_path))
    assert admitted.returncode == 0, admitted.stdout + admitted.stderr
    assert json.loads(admitted.stdout)["ok"] is True
    assert json.loads(plan_path.read_text())["confirmatory_ready"] is False
    request.write_text(
        json.dumps(
            {
                "root": str(control),
                "visible_roots": {arm: str(path) for arm, path in roots.items()},
                "plan": json.loads(plan_path.read_text()),
                "profile_path": str(tmp_path / "absent.sb"),
                "attestation": {},
                "hidden_access_log": str(tmp_path / "absent-access.json"),
            }
        )
    )
    output = tmp_path / "refused"
    refused = _matched_cli(
        "run",
        "--request",
        str(request),
        "--output",
        str(output),
        "--arm",
        "typed",
        "--session-id",
        "not-a-pilot",
        "--purpose",
        "pilot",
    )
    assert refused.returncode != 0
    record = json.loads(refused.stdout)
    assert record["ok"] is False
    assert "CSE-10" in record["error"]
    assert record == json.loads((output / "matched-session.json").read_text())
    assert record["process"] is None


def test_matched_cli_argument_errors_are_json():
    """CON-8: argument parser failures must obey the controller's JSON error contract."""
    result = _matched_cli("run")
    assert result.returncode != 0
    assert json.loads(result.stdout)["ok"] is False


def test_evidence_sink_cannot_be_inside_participant_scratch(tmp_path):
    """MON-12/MON-13: retained audits and snapshots cannot be writable by the participant."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    plan = admit_pair(control, candidates, roots, confinement=bindings)
    output = Path(bindings["typed"]["scratch"]) / "tamperable-evidence"
    with pytest.raises(AdmissionError, match="scratch"):
        execute_session(
            plan,
            control,
            roots,
            "typed",
            output,
            session_id="bad-sink",
            launch=lambda _: pytest.fail("must not launch"),
            hidden_access_log=tmp_path / "unused",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    ["tools/matched_campaign.py", "tools/campaign.py", "src/aisle/harness/matched_session.py"],
)
def test_session_controller_and_runner_sources_are_bound(tmp_path, name):
    """MON-13: new controller code must be covered even when absent from the MON-1 table."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    target = control / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
    plan = admit_pair(control, candidates, roots)
    assert name in plan["surface"]["artifact_hashes"]
    target.write_text(target.read_text() + "\n# changed controller\n")
    with pytest.raises(AdmissionError, match="controller"):
        verify_plan(plan, control, roots)


def test_declared_controller_must_match_the_executing_checkout(tmp_path):
    """MON-13: hashing unrelated controller bytes does not bind the actual implementation."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    target = control / "tools/matched_campaign.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# a different launcher\n")
    with pytest.raises(AdmissionError, match="executing controller"):
        admit_pair(control, candidates, roots)


def test_common_envelope_distinguishes_missing_streams_from_empty_evidence(tmp_path):
    """MON-12/CSE-11: process success alone cannot establish complete common evidence."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    result = execute_session(
        plan,
        control,
        roots,
        "monolithic",
        tmp_path / "incomplete",
        session_id="no-streams",
        launch=lambda _: {"rc": 0},
        hidden_access_log=tmp_path / "absent",
    )
    common = result["common_evidence"]
    assert common["schema_version"] == "aisle.matched-common-evidence.v1"
    assert common["complete"] is False
    assert common["transcript"]["status"] == "not_collected"
    assert common["tool_events"]["status"] == "not_collected"
    assert common["runs"]["status"] == "not_collected"
    assert common["budgets"]["observed"]["tokens"] is None
    assert common["validation"]["typed_validator"] is False
    assert common["validation"]["declaration"] == "mon1-02"
    assert common["assignment"]["treatment_id"] == plan["arms"]["monolithic"]["immutable_id"]


@pytest.mark.parametrize("arm", ["typed", "monolithic"])
def test_common_envelope_retains_observed_budget_and_transcript_identity(tmp_path, arm):
    """MON-12: accounting uses observed values and references the retained raw stream."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)

    def launch(output):
        (output / "session.jsonl").write_text('{"type":"fixture"}\n')
        return {"rc": 0, "tokens": 17, "wall_s": 0.25}

    result = execute_session(
        plan,
        control,
        roots,
        arm,
        tmp_path / "accounted",
        session_id="accounting",
        launch=launch,
        hidden_access_log=tmp_path / "absent",
    )
    common = result["common_evidence"]
    assert common["budgets"]["declared"] == plan["arms"][arm]["budget"]
    assert common["budgets"]["observed"]["tokens"] == 17
    assert common["budgets"]["observed"]["wall_s"] == 0.25
    assert common["transcript"]["sha256"] == result["artifacts"]["session.jsonl"]
    assert common["validation"]["typed_validator"] is (arm == "typed")
    assert common["complete"] is False


@pytest.mark.parametrize(
    "process",
    [
        {"rc": False},
        {"rc": 0, "wall_s": float("nan")},
        {"rc": 0, "tokens": -1},
        {"rc": 0, "tokens": True},
    ],
)
def test_invalid_process_telemetry_retains_an_excluded_attempt(tmp_path, process):
    """MON-12/MON-13: invalid telemetry must not pass or break final evidence serialization."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )
    output = tmp_path / "invalid-telemetry"
    result = execute_session(
        plan,
        control,
        roots,
        "typed",
        output,
        session_id="bad-meter",
        launch=lambda _: process,
        hidden_access_log=access,
    )
    assert result["ok"] is False
    assert result["classification"] == "infrastructure_exclusion"
    assert "telemetry" in result["error"]
    assert json.loads((output / "matched-session.json").read_text()) == result
    assert result["postflight"] is not None


def test_transcript_symlink_is_an_integrity_failure(tmp_path):
    """MON-12/MON-13: evidence cannot substitute a link to bytes outside the retained attempt."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )
    foreign = tmp_path / "foreign-stream.jsonl"
    foreign.write_text("foreign evidence\n")

    def launch(output):
        (output / "session.jsonl").symlink_to(foreign)
        return {"rc": 0}

    result = execute_session(
        plan,
        control,
        roots,
        "typed",
        tmp_path / "linked-stream",
        session_id="bad-stream",
        launch=launch,
        hidden_access_log=access,
    )
    assert result["ok"] is False
    assert "symlink" in result["error"]
    assert "session.jsonl" not in result["artifacts"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "undeclared_difference"])
def test_validation_property_requires_one_explicit_treatment_declaration(tmp_path, change):
    """MON-8/MON-12: validator absence must have one unambiguous declared treatment row."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    table_path = control / "docs/monolithic/treatment-table.json"
    table = json.loads(table_path.read_text())
    row = next(
        row for row in table["rows"] if row["surface"] == "static validation and diagnostics"
    )
    if change == "missing":
        table["rows"].remove(row)
    elif change == "duplicate":
        duplicate = copy.deepcopy(row)
        duplicate["id"] = "duplicate-validation-declaration"
        table["rows"].append(duplicate)
    else:
        row["class"] = "identical"
    table_path.write_text(json.dumps(table))
    monolith.table_report(control, write=True)
    with pytest.raises(AdmissionError, match="validation declaration"):
        admit_pair(control, candidates, roots)


def test_active_session_accepts_only_its_declared_editable_changes(tmp_path):
    """MON-8/MON-13: allowed edits retain the original sealed session identity."""
    from aisle.harness.matched_session import (
        AdmissionError,
        admit_pair,
        verify_active_plan,
        verify_plan,
    )

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    editable = candidates["typed"]["repository"]["editable_allowlist"][0]
    path = roots["typed"] / editable
    path.write_text(path.read_text() + "\n# allowed agent edit\n")
    refreshed = verify_active_plan(plan, control, roots, "typed")
    assert refreshed["arms"]["typed"]["immutable_id"] != plan["arms"]["typed"]["immutable_id"]
    with pytest.raises(AdmissionError):
        verify_plan(plan, control, roots)
    (roots["typed"] / "AGENTS.md").write_text("changed shared instructions")
    with pytest.raises(AdmissionError):
        verify_active_plan(plan, control, roots, "typed")


def test_active_session_cannot_treat_other_arm_edits_as_its_own(tmp_path):
    """MON-13: the active arm's edit grant never extends to the paired view."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_active_plan

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    name = candidates["monolithic"]["repository"]["editable_allowlist"][0]
    (roots["monolithic"] / name).write_text("other arm changed")
    with pytest.raises(AdmissionError):
        verify_active_plan(plan, control, roots, "typed")


def _development_protocol():
    return {
        "schema_version": "aisle.matched-development.v1",
        "purpose": "expert_parity",
        "tier": "T1",
        "embodiment": "franka",
        "verifier": "oracle",
        "reset": "teleport",
        "seeds": [7],
        "run_ceiling": 1,
        "episode_ceiling": 1,
        "timeout_s": 30,
    }


def test_development_protocol_is_bound_and_reverified(tmp_path):
    """MON-8/MON-11: development run parameters have an immutable, unscored identity."""
    from aisle.harness.matched_session import admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    protocol = _development_protocol()
    plan = admit_pair(control, candidates, roots, development=protocol)
    assert plan["development"] == protocol
    assert verify_plan(plan, control, roots) == plan
    assert plan["confirmatory_ready"] is False


def test_v2_development_protocol_binds_the_candidate_table(tmp_path):
    """MON-3/MON-8: a v2 protocol admits by candidate id, retains the derived launch
    fields, binds the table hash into the plan, and re-verifies to itself."""
    from aisle.harness.candidates import candidates_sha256
    from aisle.harness.matched_session import admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    protocol = {
        "schema_version": "aisle.matched-development.v2",
        "purpose": "expert_parity",
        "candidate": "t1-oracle",
        "candidates_sha256": candidates_sha256(control),
        "seeds": [7],
        "run_ceiling": 1,
        "episode_ceiling": 1,
        "timeout_s": 30,
    }
    plan = admit_pair(control, candidates, roots, development=protocol)
    development = plan["development"]
    assert development["candidate"] == "t1-oracle"
    assert development["typed_graph"] == "graphs/expert_t1.yaml"
    assert development["verifier"] == "oracle" and development["embodiment"] == "franka"
    assert "docs/monolithic/candidates.json" in plan["surface"]["artifact_hashes"]
    assert verify_plan(plan, control, roots) == plan


def test_v2_development_protocol_admits_the_t1_l2_pair_with_its_own_documents(tmp_path):
    """MON-3/MON-8/BND-2: the T1-L2 candidate admits, its own treatment table and
    interface map identify the plan surface, and removing one artifact refuses."""
    from aisle.harness.candidates import candidates_sha256
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path, candidate="t1-l2-realistic")
    protocol = {
        "schema_version": "aisle.matched-development.v2",
        "purpose": "expert_parity",
        "candidate": "t1-l2-realistic",
        "candidates_sha256": candidates_sha256(control),
        "seeds": [7],
        "run_ceiling": 1,
        "episode_ceiling": 1,
        "timeout_s": 30,
    }
    plan = admit_pair(control, candidates, roots, development=protocol)
    assert plan["development"]["typed_graph"] == "graphs/pilot_t1_l2_typed.yaml"
    assert plan["development"]["verifier"] == "realistic"
    documents = plan["development"]["documents"]
    expected = monolith.table_report(control, write=False, documents=documents)["immutable_id"]
    assert plan["surface"]["treatment_table_id"] == expected
    assert expected != monolith.table_report(control, write=False)["immutable_id"]
    assert plan["surface"]["interface_map_id"] == "mon-interface-map-t1-l2-v1"
    assert "graphs/pilot_t1_l2_monolithic.yaml" in plan["surface"]["artifact_hashes"]
    assert "src/aisle/harness/pilot_policy_surface.py" in plan["surface"]["artifact_hashes"]
    assert verify_plan(plan, control, roots) == plan
    (control / "experts/monolithic/expert_t1_l2.py").unlink()
    with pytest.raises(AdmissionError, match="artifact missing"):
        admit_pair(control, candidates, roots, development=protocol)


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", "confirmatory"),
        ("tier", "T2"),
        ("seeds", [True]),
        ("seeds", []),
        ("run_ceiling", 0),
        ("episode_ceiling", 0),
        ("timeout_s", float("inf")),
        ("verifier", "realistic"),
    ],
)
def test_unsupported_development_protocol_refuses_admission(tmp_path, field, value):
    """MON-8/CSE-10: unsupported or unresolved run parameters cannot silently default."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    protocol = _development_protocol()
    protocol[field] = value
    with pytest.raises(AdmissionError, match="development"):
        admit_pair(control, candidates, roots, development=protocol)


@pytest.mark.parametrize("fault", ["missing", "bool", "out_of_range", "executable", "wrong_bytes"])
def test_launch_system_prompt_must_bind_delivered_argument(tmp_path, fault):
    """MON-8/MON-13: declared system-prompt bytes must match an explicit launch argument."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _launch_pair(candidates)
    for launch in launches.values():
        if fault == "missing":
            del launch["system_prompt_arg"]
        elif fault == "bool":
            launch["system_prompt_arg"] = True
        elif fault == "out_of_range":
            launch["system_prompt_arg"] = 99
        elif fault == "executable":
            launch["system_prompt_arg"] = 0
        else:
            launch["argv"][3] = "unadmitted prompt"
    with pytest.raises(AdmissionError, match="system.prompt"):
        admit_pair(control, candidates, roots, launches=launches)


@pytest.mark.parametrize("fault", [None, "missing", "wrong_bytes", "same_slot"])
def test_research_contract_argument_binding(tmp_path, fault):
    """MON-8/MON-13: executable admission binds the delivered research contract."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _launch_pair(candidates)
    for launch in launches.values():
        if fault == "missing":
            del launch["research_contract_arg"]
        elif fault == "wrong_bytes":
            launch["argv"][4] = "different contract"
        elif fault == "same_slot":
            launch["research_contract_arg"] = 3
    if fault is None:
        assert (
            admit_pair(control, candidates, roots, launches=launches)["launch_bindings"] == launches
        )
    else:
        with pytest.raises(AdmissionError, match="research contract"):
            admit_pair(control, candidates, roots, launches=launches)


def test_declared_document_bundle_is_the_only_permitted_launch_prompt_difference(tmp_path):
    """MON-8: representation launch differences must deliver the exact declared document bundle."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    row = _representation_documents(control, candidates, roots)
    launches = _launch_pair(candidates)
    for arm, launch in launches.items():
        launch["argv"][4] = json.dumps(
            [{"path": name, "text": (roots[arm] / name).read_text()} for name in row[arm]["paths"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    plan = admit_pair(control, candidates, roots, launches=launches, prompt_row=row["id"])
    assert plan["launch_bindings"] == launches
    launches["monolithic"]["argv"][4] += " "
    with pytest.raises(AdmissionError, match="research contract"):
        admit_pair(control, candidates, roots, launches=launches, prompt_row=row["id"])


@pytest.mark.parametrize("meter", ["tokens", "wall_s"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_final_observed_budget_cannot_pass_with_zero_exit(tmp_path, meter, offset):
    """MON-8/MON-12: final telemetry at a ceiling cannot bypass the supervisor's polling gate."""
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    for candidate in candidates.values():
        candidate["budget"]["wall_ceiling_s"] = 30
    plan = admit_pair(control, candidates, roots)
    process = {"rc": 0, "tokens": 0, "wall_s": 1.0}
    process[meter] = (
        plan["arms"]["typed"]["budget"]["ceiling" if meter == "tokens" else "wall_ceiling_s"]
        + offset
    )
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )
    result = execute_session(
        plan,
        control,
        roots,
        "typed",
        tmp_path / "budget-final",
        session_id="final-budget",
        launch=lambda _: process,
        hidden_access_log=access,
    )
    assert result["ok"] is (offset < 0)
    if offset >= 0:
        assert result["classification"] == "infrastructure_exclusion"
        assert "budget" in result["error"]
    assert result["process"] == process


@pytest.mark.parametrize("live_fault", [None, "missing", "wrong_count", "boolean_count"])
def test_session_envelope_retains_frontend_observations(tmp_path, live_fault):
    """MON-12: retained frontend invocations stay distinct from unverified global tool totals."""
    from aisle.harness.matched_frontend import FrontendToolBudget
    from aisle.harness.matched_session import admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    for candidate in candidates.values():
        candidate["budget"]["frontend_tool_ceiling"] = 1
    plan = admit_pair(control, candidates, roots)
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )

    def launch(output):
        (output / "session.jsonl").write_text(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "call-1", "type": "file_change"},
                }
            )
            + "\n"
        )
        guard = FrontendToolBudget("codex", 1)
        guard((output / "session.jsonl").read_text())
        report = guard.report()
        if live_fault == "wrong_count":
            report["observed_calls"] = 0
        elif live_fault == "boolean_count":
            report["observed_calls"] = True
        if live_fault != "missing":
            (output / "frontend-live.json").write_text(json.dumps(report))
        return {"rc": 0, "stream_complete": True, "stopped": "agent_done"}

    result = execute_session(
        plan,
        control,
        roots,
        "typed",
        tmp_path / "frontend-session",
        session_id="frontend-fixture",
        launch=launch,
        hidden_access_log=access,
    )
    if live_fault is not None:
        assert result["ok"] is False
        assert "frontend" in result["error"]
        assert result["common_evidence"]["budgets"]["observed"]["frontend_tool_calls"] is None
        return
    assert result["ok"] is True, result
    assert result["frontend_tools"]["observed_calls"] == 1
    assert result["artifacts"]["frontend-tools.json"]
    observed = result["common_evidence"]["budgets"]["observed"]
    assert observed["frontend_tool_calls"] == 1
    assert observed["tool_calls"] is None


def test_initial_home_content_must_match_declared_baseline(tmp_path):
    """MON-8/MON-13: equal declared HOME hashes cannot conceal different on-disk context."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    (Path(ambient["typed"]["environment"]["HOME"]) / "prior-context.txt").write_text("prior result")
    with pytest.raises(AdmissionError, match="baseline"):
        admit_pair(control, candidates, roots, confinement=bindings, ambient=ambient)


def test_active_home_writes_do_not_authorize_a_new_launch(tmp_path):
    """MON-8/MON-13: runtime private writes are allowed only within the already active arm."""
    from aisle.harness.matched_session import (
        AdmissionError,
        admit_pair,
        verify_active_plan,
        verify_plan,
    )

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    plan = admit_pair(control, candidates, roots, confinement=bindings, ambient=ambient)
    (Path(ambient["typed"]["environment"]["XDG_CONFIG_HOME"]) / "runtime.json").write_text("{}")
    assert verify_active_plan(plan, control, roots, "typed")["immutable_id"] == plan["immutable_id"]
    with pytest.raises(AdmissionError, match="baseline"):
        verify_plan(plan, control, roots)
    (Path(ambient["monolithic"]["environment"]["XDG_CACHE_HOME"]) / "foreign.txt").write_text(
        "leak"
    )
    with pytest.raises(AdmissionError, match="baseline"):
        verify_active_plan(plan, control, roots, "typed")


@pytest.mark.parametrize("fault", [None, "missing", "redirected"])
def test_session_retains_final_private_state_fingerprints(tmp_path, fault):
    """MON-12/MON-13: normal private runtime writes retain final hashes without private bytes."""
    from aisle.harness.matched_session import _private_tree_hash, admit_pair, execute_session

    control, candidates, roots = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, control, candidates, roots)
    ambient = _ambient_pair(candidates, bindings)
    plan = admit_pair(control, candidates, roots, confinement=bindings, ambient=ambient)
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )
    config = Path(ambient["typed"]["environment"]["XDG_CONFIG_HOME"])

    def launch(output):
        if fault is None:
            (config / "runtime.json").write_text('{"private":"session-state"}')
        else:
            config.rmdir()
            if fault == "redirected":
                outside = tmp_path / "foreign-private-state"
                outside.mkdir()
                (outside / "private.txt").write_text("must-not-be-retained")
                config.symlink_to(outside, target_is_directory=True)
        return {"rc": 0}

    result = execute_session(
        plan,
        control,
        roots,
        "typed",
        tmp_path / "private-postflight",
        session_id="private-final",
        launch=launch,
        hidden_access_log=access,
    )
    if fault is not None:
        assert result["ok"] is False
        assert result["private_state"]["final"] is None
        assert "private state postflight" in result["error"]
        assert "must-not-be-retained" not in json.dumps(result)
        return
    assert result["ok"] is True, result
    state = result["private_state"]
    assert (
        state["initial"]["config_baseline_sha256"]
        == plan["arms"]["typed"]["state"]["config_baseline_sha256"]
    )
    assert state["final"]["config_baseline_sha256"] == _private_tree_hash(config)
    assert state["initial"] != state["final"]
    assert "session-state" not in json.dumps(result)
    assert result["common_evidence"]["audits"]["private_state"] == state


@pytest.mark.parametrize(
    "name",
    [
        "src/aisle/monolith/worker.py",
        "src/aisle/harness/matched_run.py",
        "src/aisle/harness/matched_dynamic_run.py",
        "src/aisle/harness/typed_stage_provider.py",
        "src/aisle/harness/typed_worker_provisioning.py",
        "src/aisle/harness/worker_declaration.py",
        "src/aisle/harness/worker_capability.py",
        "src/aisle/harness/worker_authority_probe.py",
        "src/aisle/harness/worker_network_probe.py",
        "src/aisle/harness/matched_run_launch.py",
        "src/aisle/monolith/wire.py",
        "src/aisle/monolith/primitive_api.py",
        "src/aisle/monolith/requests.py",
        "src/aisle/monolith/proxy.py",
        "src/aisle/monolith/supervisor.py",
        "src/aisle/monolith/worker_launch.py",
        "src/aisle/monolith/worker_config.py",
        "src/aisle/harness/typed_validation.py",
        "src/aisle/harness/typed_node_requests.py",
        "src/aisle/harness/typed_node_worker.py",
        "src/aisle/harness/typed_node_supervisor.py",
        "src/aisle/harness/typed_execution_bundle.py",
        "src/aisle/harness/typed_worker_launch.py",
        "src/aisle/harness/typed_node_host.py",
        "src/aisle/harness/typed_graph_hosts.py",
        "src/aisle/harness/typed_graph_stage.py",
        "src/aisle/harness/typed_run_prepare.py",
        "src/aisle/harness/monolithic_run_evidence.py",
        "src/aisle/harness/monolithic_run_prepare.py",
        "src/aisle/harness/typed_graph_audit.py",
        "src/aisle/turn_node.py",
        "src/aisle/turns.py",
        "src/aisle/harness/validate.py",
        "src/aisle/harness/registry.py",
    ],
)
def test_matched_admission_binds_worker_implementation_drift(tmp_path, name):
    """MON-8/MON-13: every worker boundary implementation is part of session identity."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    root, candidates, views = prepared_pair(tmp_path)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, path)
    plan = admit_pair(root, candidates, views)
    assert name in plan["surface"]["artifact_hashes"]
    path.write_text(path.read_text() + "\n# changed controller boundary\n")
    # Registry/validator are also checked by the treatment table, before the
    # controller implementation comparison. Either binding must refuse drift.
    expected = (
        "controller surface checks failed"
        if name in ("src/aisle/harness/validate.py", "src/aisle/harness/registry.py")
        else "executing controller"
    )
    with pytest.raises(AdmissionError, match=expected):
        verify_plan(plan, root, views)


@pytest.mark.parametrize(
    "name",
    [
        "src/aisle/nodes/segmented_pose.py",
        "src/aisle/nodes/grasp_topdown.py",
        "src/aisle/nodes/ik_trajectory.py",
        "src/aisle/nodes/task_state_machine.py",
        "registry/manifests/segmented-pose.yaml",
        "registry/manifests/grasp-planner-topdown.yaml",
        "registry/manifests/ik-trajectory.yaml",
        "registry/manifests/task-state-machine.yaml",
    ],
)
def test_typed_authored_implementations_remain_editable_without_changing_controller(tmp_path, name):
    """MON-2/MON-6/MON-13: typed node and manifest edits remain deliverables, not trusted edits."""
    from aisle.harness.matched_session import (
        AdmissionError,
        admit_pair,
        verify_active_plan,
        verify_plan,
    )

    control, candidates, roots = prepared_pair(tmp_path)
    plan = admit_pair(control, candidates, roots)
    original = (control / name).read_bytes()
    candidate_path = roots["typed"] / name
    assert name in plan["arms"]["typed"]["repository"]["editable_allowlist"]
    candidate_path.write_bytes(candidate_path.read_bytes() + b"\n# authored candidate change\n")
    refreshed = verify_active_plan(plan, control, roots, "typed")
    assert refreshed["surface"] == plan["surface"]
    assert refreshed["arms"]["monolithic"] == plan["arms"]["monolithic"]
    assert refreshed["arms"]["typed"] != plan["arms"]["typed"]
    assert (control / name).read_bytes() == original
    with pytest.raises(AdmissionError):
        verify_plan(plan, control, roots)


def test_fresh_session_plans_match_without_reusing_prior_arm_state(tmp_path):
    """CSE-1/CSE-12/MON-8: independent session views preserve matching and exclude prior state."""
    from aisle.harness.matched_session import (
        AdmissionError,
        _ambient_ids,
        _permission_ids,
        _shared,
        admit_pair,
        verify_plan,
    )

    sessions = []
    for name in ("first", "second"):
        base = tmp_path / name
        base.mkdir()
        root, candidates, views = prepared_pair(base)
        bindings = _confinement_pair(base, root, candidates, views)
        ambient = _ambient_pair(candidates, bindings)
        plan = admit_pair(root, candidates, views, confinement=bindings, ambient=ambient)
        permissions = _permission_ids(root, plan["arms"], views, bindings)
        environments = _ambient_ids(plan["arms"], bindings, ambient)
        shared = {
            arm: _shared(plan["arms"][arm], permissions[arm], environments[arm])
            for arm in candidates
        }
        sessions.append((root, views, ambient, plan, shared))
    first, second = sessions
    assert first[4]["typed"] == second[4]["monolithic"]
    assert first[3]["surface"] == second[3]["surface"]
    assert first[3]["immutable_id"] != second[3]["immutable_id"]
    sentinel = "prior-session-private-sentinel"
    home = Path(first[2]["typed"]["environment"]["HOME"])
    (home / "session-state").write_text(sentinel)
    with pytest.raises(AdmissionError, match="HOME"):
        verify_plan(first[3], first[0], first[1])
    assert verify_plan(second[3], second[0], second[1]) == second[3]
    for arm in second[1]:
        for directory in (second[1][arm], Path(second[2][arm]["environment"]["HOME"])):
            assert not directory.is_relative_to(home)
            assert all(
                sentinel.encode() not in p.read_bytes() for p in directory.rglob("*") if p.is_file()
            )


def _pair_with_private_roles(tmp_path):
    from aisle.harness.treatment_confinement import MacOSPolicy, compile_macos_profile

    root, candidates, views = prepared_pair(tmp_path)
    bindings = _confinement_pair(tmp_path, root, candidates, views)
    roles = {name: str(tmp_path / name) for name in ("session_evidence", "capability_evidence")}
    for path in roles.values():
        Path(path).mkdir()
    for arm in candidates:
        policy = bindings[arm]["policy"]
        policy["hidden_roots"].extend(roles.values())
        compiled = compile_macos_profile(
            MacOSPolicy(
                **{
                    key: value if key == "network_policy" else tuple(Path(p) for p in value)
                    for key, value in policy.items()
                }
            )
        )
        candidates[arm]["confinement"].update(
            profile_sha256=compiled.sha256, policy_sha256=compiled.policy_id
        )
    ambient = _ambient_pair(candidates, bindings)
    return root, candidates, views, bindings, ambient, roles


def test_fresh_plans_bind_private_evidence_path_roles(tmp_path):
    """CSE-1/MON-8/MON-13: fresh private sinks compare by explicit verified roles."""
    from aisle.harness.matched_session import (
        _ambient_ids,
        _permission_ids,
        _shared,
        admit_pair,
        verify_plan,
    )

    observed = []
    for name in ("first", "second"):
        base = tmp_path / name
        base.mkdir()
        root, candidates, views, bindings, ambient, roles = _pair_with_private_roles(base)
        plan = admit_pair(
            root, candidates, views, confinement=bindings, ambient=ambient, private_roots=roles
        )
        assert plan["private_roots"] == roles
        assert verify_plan(plan, root, views) == plan
        permissions = _permission_ids(root, plan["arms"], views, bindings, roles)
        environments = _ambient_ids(plan["arms"], bindings, ambient)
        observed.append(
            {
                arm: _shared(plan["arms"][arm], permissions[arm], environments[arm])
                for arm in candidates
            }
        )
    assert observed[0]["typed"] == observed[1]["monolithic"]


@pytest.mark.parametrize(
    "mutation",
    [
        "visible",
        "controller",
        "runtime",
        "alias",
        "relative",
        "missing",
        "unhidden",
        "name",
        "symlink",
        "file",
    ],
)
def test_private_path_roles_refuse_unverified_roots(tmp_path, mutation):
    """MON-6/MON-13: role normalization cannot conceal grants, aliases or redirection."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    root, candidates, views, bindings, ambient, roles = _pair_with_private_roles(tmp_path)
    if mutation == "visible":
        roles["session_evidence"] = str(views["typed"])
    elif mutation == "controller":
        roles["session_evidence"] = str(root)
    elif mutation == "runtime":
        roles["session_evidence"] = bindings["typed"]["policy"]["runtime_read_roots"][0]
    elif mutation == "alias":
        roles["session_evidence"] = roles["capability_evidence"]
    elif mutation == "relative":
        roles["session_evidence"] = "relative"
    elif mutation == "missing":
        roles["session_evidence"] = str(tmp_path / "missing")
    elif mutation == "unhidden":
        other = tmp_path / "unhidden"
        other.mkdir()
        roles["session_evidence"] = str(other)
    elif mutation == "name":
        roles["../session_evidence"] = roles.pop("session_evidence")
    elif mutation == "symlink":
        link = tmp_path / "redirected"
        link.symlink_to(roles["session_evidence"], target_is_directory=True)
        roles["session_evidence"] = str(link)
    else:
        file = tmp_path / "file"
        file.write_text("not a directory")
        roles["session_evidence"] = str(file)
    with pytest.raises(AdmissionError):
        admit_pair(
            root, candidates, views, confinement=bindings, ambient=ambient, private_roots=roles
        )


def test_private_path_role_redirection_invalidates_retained_plan(tmp_path):
    """MON-13: private root role bindings are rechecked on retained-plan verification."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    root, candidates, views, bindings, ambient, roles = _pair_with_private_roles(tmp_path)
    plan = admit_pair(
        root, candidates, views, confinement=bindings, ambient=ambient, private_roots=roles
    )
    redirected = Path(roles["session_evidence"])
    redirected.rmdir()
    redirected.symlink_to(roles["capability_evidence"], target_is_directory=True)
    with pytest.raises(AdmissionError):
        verify_plan(plan, root, views)


def test_private_path_roles_require_confinement(tmp_path):
    """MON-6: a private label alone cannot establish hidden authority."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    root, candidates, views, _, _, roles = _pair_with_private_roles(tmp_path)
    with pytest.raises(AdmissionError):
        admit_pair(root, candidates, views, private_roots=roles)


def test_admission_cli_retains_private_root_roles(tmp_path):
    """CON-8/MON-13: CLI admission must not silently discard private evidence role bindings."""
    root, candidates, views, bindings, ambient, roles = _pair_with_private_roles(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "root": str(root),
                "candidates": candidates,
                "visible_roots": {arm: str(path) for arm, path in views.items()},
                "confinement": bindings,
                "ambient": ambient,
                "private_roots": roles,
            }
        )
    )
    output = tmp_path / "plan.json"
    result = _matched_cli("admit", "--request", str(request), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"]
    assert json.loads(output.read_text())["private_roots"] == roles


def test_undeclared_private_path_differences_remain_visible(tmp_path):
    """MON-13: extra hidden paths are not normalized without an explicit role binding."""
    from aisle.harness.matched_session import _permission_ids, admit_pair

    identities = []
    for name in ("first", "second"):
        base = tmp_path / name
        base.mkdir()
        root, candidates, views, bindings, ambient, _ = _pair_with_private_roles(base)
        plan = admit_pair(root, candidates, views, confinement=bindings, ambient=ambient)
        identities.append(_permission_ids(root, plan["arms"], views, bindings))
    assert identities[0]["typed"] != identities[1]["monolithic"]


@pytest.mark.parametrize("replace_private_directory", [False, True])
def test_postflight_revalidates_private_role_directory(tmp_path, replace_private_directory):
    """MON-8/MON-13: postflight cannot discard a private-root role checked at admission."""
    from aisle.harness.matched_session import admit_pair, execute_session

    root, candidates, views, bindings, ambient, roles = _pair_with_private_roles(tmp_path)
    plan = admit_pair(
        root, candidates, views, confinement=bindings, ambient=ambient, private_roots=roles
    )
    access = tmp_path / "access.json"
    access.write_text(
        json.dumps(
            {
                "schema_version": "aisle.hidden-access-log.v1",
                "adapter_active": True,
                "complete": True,
                "events": [],
            }
        )
    )

    def launch(output):
        (output / "session.jsonl").write_text(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "message", "type": "agent_message", "text": "fixture"},
                }
            )
            + "\n"
        )
        (output / "session.stderr").write_text("")
        if replace_private_directory:
            private = Path(roles["capability_evidence"])
            private.rmdir()
            private.write_text("no longer a directory")
        return {"rc": 0, "tokens": 0, "wall_s": 0.01, "classification": "agent_outcome"}

    record = execute_session(
        plan,
        root,
        views,
        "typed",
        tmp_path / "result",
        session_id="private-role-postflight",
        launch=launch,
        hidden_access_log=access,
    )
    assert record["ok"] is (not replace_private_directory), record
    if replace_private_directory:
        assert "private" in record["error"], record
        assert record["classification"] == "infrastructure_exclusion"


def _app_server_launch_pair(candidates):
    launches = _launch_pair(candidates)
    return {
        arm: {
            "argv": [launch["argv"][0], "app-server", "--listen", "stdio://"],
            "app_server": {
                "baseInstructions": "system prompt",
                "developerInstructions": "research contract",
            },
        }
        for arm, launch in launches.items()
    }


def test_app_server_prompt_binding_uses_thread_parameters(tmp_path):
    """MON-8/MON-13: admission binds actual thread prompts instead of unused argv slots."""
    from aisle.harness.matched_session import admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _app_server_launch_pair(candidates)
    plan = admit_pair(control, candidates, roots, launches=launches)
    assert verify_plan(plan, control, roots)["launch_bindings"] == launches


@pytest.mark.parametrize("field", ["baseInstructions", "developerInstructions"])
def test_app_server_prompt_drift_cannot_be_hidden_in_equal_arm_configuration(tmp_path, field):
    """MON-8/MON-13: equal wrong prompts still fail their admitted content hashes."""
    from aisle.harness.matched_session import AdmissionError, admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _app_server_launch_pair(candidates)
    for launch in launches.values():
        launch["app_server"][field] = "changed prompt"
    with pytest.raises(AdmissionError, match="differs"):
        admit_pair(control, candidates, roots, launches=launches)


def test_app_server_declared_document_bundle_is_bound_for_each_arm(tmp_path):
    """MON-1/MON-8: representation-specific documents retain their existing declared exception."""
    from aisle.harness.matched_session import admit_pair

    control, candidates, roots = prepared_pair(tmp_path)
    row = _representation_documents(control, candidates, roots)
    launches = _app_server_launch_pair(candidates)
    for arm, launch in launches.items():
        bundle = [
            {"path": name, "text": (control / name).read_bytes().decode()}
            for name in row[arm]["paths"]
        ]
        launch["app_server"]["developerInstructions"] = json.dumps(
            bundle, ensure_ascii=False, separators=(",", ":")
        )
    assert admit_pair(control, candidates, roots, launches=launches, prompt_row=row["id"])


def test_nested_host_binding_is_admitted_equally_and_rechecked(tmp_path):
    """MON-8/MON-13: both arms bind one host and drift invalidates the admitted plan."""
    import hashlib

    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    host = tmp_path / "host"
    host.write_bytes(b"#!/bin/sh\nexit 0\n")
    host.chmod(0o700)
    launches = _app_server_launch_pair(candidates)
    for arm, launch in launches.items():
        candidates[arm]["budget"]["frontend_tool_ceiling"] = 2
        launch["code_mode_host"] = {
            "path": str(host),
            "sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        }
    plan = admit_pair(control, candidates, roots, launches=launches)
    assert verify_plan(plan, control, roots)["launch_bindings"] == launches
    host.write_bytes(b"#!/bin/sh\nexit 1\n")
    with pytest.raises(AdmissionError, match="host"):
        verify_plan(plan, control, roots)


@pytest.mark.parametrize(
    "drift", [None, "peer", "ceiling", "credentials", "remote_http", "auth_mode"]
)
def test_provider_binding_requires_equal_admitted_transport_and_budget(tmp_path, drift):
    """MON-8/MON-13: provider selection is bound to both arms before launch."""
    from aisle.harness.matched_session import AdmissionError, admit_pair, verify_plan

    control, candidates, roots = prepared_pair(tmp_path)
    launches = _app_server_launch_pair(candidates)
    for arm, launch in launches.items():
        candidates[arm]["budget"]["frontend_tool_ceiling"] = 2
        launch["provider"] = {"base_url": "http://127.0.0.1:1234/v1", "requires_openai_auth": False}
    if drift == "peer":
        launches["monolithic"]["provider"]["base_url"] = "http://127.0.0.1:4321/v1"
    elif drift == "ceiling":
        for candidate in candidates.values():
            del candidate["budget"]["frontend_tool_ceiling"]
    elif drift == "credentials":
        for launch in launches.values():
            launch["provider"]["base_url"] = "https://user:password@example.test/v1"
    elif drift == "remote_http":
        for launch in launches.values():
            launch["provider"]["base_url"] = "http://example.test/v1"
    elif drift == "auth_mode":
        for launch in launches.values():
            launch["provider"]["requires_openai_auth"] = "false"
    if drift is None:
        plan = admit_pair(control, candidates, roots, launches=launches)
        assert verify_plan(plan, control, roots)["launch_bindings"] == launches
    else:
        with pytest.raises(AdmissionError):
            admit_pair(control, candidates, roots, launches=launches)


@pytest.mark.parametrize(
    ("launch", "message"),
    [
        ({"argv": ["codex"], "app_server": {}}, "requires an owned provider"),
        ({"argv": ["codex"]}, "requires an owned provider"),
        (
            {
                "argv": ["codex"],
                "app_server": {},
                "provider": {"relay_port": 4321},
                "mcp_harness": True,
            },
            "single relay port",
        ),
        (
            {
                "argv": ["codex"],
                "app_server": {},
                "provider": {"relay_port": 4321},
                "code_mode_host": {},
            },
            "single relay port",
        ),
    ],
)
def test_loopback_policy_admits_only_a_provider_only_launch(launch, message):
    """TRT-5 (ADR-69 §6): a one-port loopback grant serves exactly the owned
    provider relay; a launch without it, or with listeners on other ports,
    refuses at admission."""
    from aisle.harness.matched_session import AdmissionError, _verify_loopback_launch

    policy = {"network_policy": "loopback", "loopback_port": 4321}
    with pytest.raises(AdmissionError, match=message):
        _verify_loopback_launch(policy, launch)


def test_loopback_policy_admits_a_provider_only_launch_and_ignores_deny_external():
    """TRT-5: the provider-only loopback launch and every deny-external launch pass."""
    from aisle.harness.matched_session import _verify_loopback_launch

    _verify_loopback_launch(
        {"network_policy": "loopback", "loopback_port": 4321},
        {"argv": ["codex"], "app_server": {}, "provider": {"relay_port": 4321}},
    )
    _verify_loopback_launch(
        {"network_policy": "deny-external"}, {"argv": ["x"], "mcp_harness": True}
    )
    _verify_loopback_launch(None, {"argv": ["x"]})
