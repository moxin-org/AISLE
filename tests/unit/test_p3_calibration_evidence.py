"""P3 calibration decisions distinguish physical outcomes from executor events."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from p3_calibration_evidence import audit_chain, screen

pytestmark = pytest.mark.unit


def fixture(recovery_status="success"):
    goals = [
        dict(
            goal_id="g1",
            goal={"seed": 200},
            completeness="complete",
            verdict={"status": "success", "failure": None, "verifier": "oracle"},
        ),
        dict(
            goal_id="g2",
            goal={"seed": 200, "expects_return": True},
            completeness="complete",
            verdict={
                "status": recovery_status,
                "failure": None if recovery_status == "success" else "not_returned",
                "verifier": "oracle",
            },
        ),
    ]
    events = []
    forwarded = []
    for ident, goal, source in [
        ("i1", "g1", "grasp-planner-topdown/grasp_pose"),
        ("i2", "g2", "return-planner/grasp_pose"),
        ("i3", "g2", "grasp-planner-topdown/grasp_pose"),
    ]:
        events.extend(
            [
                dict(
                    kind="proposed",
                    invocation_id=ident,
                    goal_revision=goal,
                    event_seq=len(events),
                    source={"endpoint": source},
                ),
                dict(
                    kind="started",
                    invocation_id=ident,
                    goal_revision=goal,
                    event_seq=len(events) + 1,
                ),
            ]
        )
        forwarded.append({"invocation_id": ident})
    return [dict(goals=goals)], dict(events=events, forwarded=forwarded)


def test_executor_return_cannot_rescue_failed_physical_chain():
    """TC-8; HAR-4: verified motion reuse does not override independent failed delivery."""
    episodes, journal = fixture("fail")
    result = audit_chain(episodes, journal, 200)
    assert result["success"] is False
    assert result["recovery_sequence"] is True
    assert result["failures"] == {"not_returned": 1}


def test_missing_goal_outcome_is_not_zero_success():
    """TC-8: missing scoring evidence is an instrument failure, not a negative task trial."""
    episodes, journal = fixture()
    episodes[0]["goals"][1]["completeness"] = "unscored"
    episodes[0]["goals"][1]["verdict"] = None
    with pytest.raises(ValueError, match="complete"):
        audit_chain(episodes, journal, 200)


def test_return_proposal_without_start_is_not_reuse():
    """HAR-4: a declared return planner or rejected proposal cannot prove a recovery sequence."""
    episodes, journal = fixture()
    journal["events"] = [
        e for e in journal["events"] if not (e["kind"] == "started" and e["invocation_id"] == "i2")
    ]
    result = audit_chain(episodes, journal, 200)
    assert result["success"] is True
    assert result["recovery_sequence"] is False
    assert result["started_invocations"] == 2


def test_screen_requires_full_declared_sample_and_variation():
    """CON-5; TC-8: calibration uses all planned chains, and a floor/ceiling is not admission."""
    seeds = list(range(200, 248, 4))
    rows = [
        dict(seed=seed, status="complete", success=index < 2, recovery_sequence=True)
        for index, seed in enumerate(seeds)
    ]
    assert screen(rows, seeds)["status"] == "eligible"
    assert screen(rows[:-1], seeds)["status"] == "incomplete"
    assert screen([{**r, "success": False} for r in rows], seeds)["status"] == "ineligible"
    assert screen([{**r, "success": True} for r in rows], seeds)["status"] == "ineligible"
    rows[0] = dict(seed=seeds[0], status="instrument_failure", success=None)
    result = screen(rows, seeds)
    assert result["status"] == "instrument_failure"
    assert result["successes"] is None


def test_duplicate_or_unplanned_seed_cannot_replace_a_slot():
    """CON-5: replacement/repeated assignments cannot silently improve calibration admission."""
    with pytest.raises(ValueError, match="seed"):
        screen([dict(seed=200, status="complete"), dict(seed=200, status="complete")], [200, 204])


def test_failed_initial_attempt_is_not_an_observed_initial_delivery():
    """TC-8: ordered commands alone do not establish that the scripted misdelivery happened."""
    episodes, journal = fixture()
    episodes[0]["goals"][0]["verdict"].update(status="fail", failure="never_grasped")
    result = audit_chain(episodes, journal, 200)
    assert result["success"] is False
    assert result["recovery_sequence"] is False


def test_assessment_joins_source_verdicts_and_policy_capture(tmp_path):
    """HAR-4; TC-8: the assessment checks real file bindings before deriving chain facts."""
    import hashlib
    import json

    import pyarrow as pa
    from p3_calibration_evidence import assess_run
    from test_p3_evidence import run_fixture

    from aisle.harness.trace_recorder import TRACE_SCHEMA, trace_identity

    run = run_fixture(tmp_path)
    traces = run / "traces"
    identities = [
        json.loads(line) for line in (traces / "identities.jsonl").read_text().splitlines()
    ]
    replaced = {"rollout-client__episode_goal", "verifier-oracle__episode_result"}
    identities = [r for r in identities if r["endpoint"] not in replaced]
    first = dict(
        goal_id="public-goal",
        seed=200,
        status="success",
        failure=None,
        verifier="oracle",
        t_end=2.0,
    )
    second = dict(
        goal_id="recovery",
        seed=200,
        status="fail",
        failure="not_returned",
        verifier="oracle",
        t_end=4.0,
    )
    streams = {
        "rollout-client__episode_goal": [
            (1, "public-goal", dict(seed=200, reset_sim_ns=0, target_med="a")),
            (3, "recovery", dict(seed=200, reset_sim_ns=2, target_med="b", expects_return=True)),
        ],
        "verifier-oracle__episode_result": [(2, "public-goal", first), (4, "recovery", second)],
    }
    for endpoint, values in streams.items():
        rows = []
        for index, (stamp, goal, payload) in enumerate(values):
            row = dict(sim_time_ns=stamp, env_id=0, seq=index, data=None, text=json.dumps(payload))
            rows.append(row)
            identities.append(trace_identity(endpoint, index, row, {"goal_id": goal}))
        with pa.OSFile(str(traces / f"{endpoint}.arrow"), "wb") as sink:
            with pa.ipc.new_stream(sink, TRACE_SCHEMA) as writer:
                writer.write_table(pa.Table.from_pylist(rows, schema=TRACE_SCHEMA))
    (traces / "identities.jsonl").write_text("".join(json.dumps(r) + "\n" for r in identities))
    (run / "episodes.jsonl").write_text(
        json.dumps(dict(episode=0, **first))
        + "\n"
        + json.dumps(dict(episode=0, recovery=True, **second))
        + "\n"
    )
    graph = (run / "graph.yaml").read_bytes()
    graph_hash = hashlib.sha256(graph).hexdigest()
    (run / "authored-graph.yaml").write_bytes(graph)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(
        graph_hash=graph_hash,
        env_fingerprint="b" * 64,
        env_attested=True,
        relaunches=0,
        wall_clamped=[],
        seeds=[200],
    )
    (run / "manifest.json").write_text(json.dumps(manifest))
    args = dict(
        seed=200,
        instrument_commit="a" * 40,
        graph_sha256=graph_hash,
        env_fingerprint="b" * 64,
        skill_sources={"src/aisle/nodes/ik_trajectory.py": "c" * 64},
    )
    result = assess_run(run, tmp_path / "assessment", **args)
    assert result["status"] == "complete" and result["success"] is False
    assert result["recovery_sequence"] is False
    assert result["failures"] == {"not_returned": 1}
    with pytest.raises(ValueError, match="attested"):
        assess_run(run, tmp_path / "wrong-env", **{**args, "env_fingerprint": "d" * 64})
    with pytest.raises(ValueError, match="skill source"):
        assess_run(
            run, tmp_path / "wrong-skill", **{**args, "skill_sources": {"different": "e" * 64}}
        )
