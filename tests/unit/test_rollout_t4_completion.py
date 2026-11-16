"""HAR-1/4: an initial T4 result does not complete an increment-two chain."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.unit


def run_fake_chain(
    tmp_path, monkeypatch, *, recovery_at, exit_at, deadline=200, wall=300, inc2=True
):
    from aisle.harness import rollout as ro

    graph = tmp_path / "graphs/g.yaml"
    graph.parent.mkdir()
    graph.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {
                        "id": "rollout-client",
                        "path": "../src/aisle/harness/rollout_client.py",
                        "env": {"AISLE_T4_INCREMENT_TWO": "1" if inc2 else "0"},
                    }
                ]
            }
        )
    )
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness/budget.toml").write_text("[campaign]\ntokens=1\nepisodes=10\nwall_h=1\n")
    records = [
        dict(
            episode=0, seed=200, status="success", t_end=15.0, goal_id="ep-0000", verifier="oracle"
        )
    ]
    results = None

    class Clock:
        now = 0

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.now += duration
            if recovery_at is not None and self.now >= recovery_at and len(records) == 1:
                records.append(
                    dict(
                        episode=0,
                        seed=200,
                        status="fail",
                        failure="not_returned",
                        t_end=60.0,
                        goal_id="ep-0000r",
                        verifier="oracle",
                        recovery=True,
                    )
                )
                results.write_text("".join(json.dumps(r) + "\n" for r in records))

    clock = Clock()
    process = SimpleNamespace(poll=lambda: 0 if clock.now >= exit_at else None)
    stopped = []

    def spawn(graph, run_dir, environment, **kwargs):
        nonlocal results
        results = Path(environment["AISLE_RESULTS"])
        results.write_text(json.dumps(records[0]) + "\n")
        return process

    monkeypatch.setattr(ro, "instrumented_graph", lambda *a, **k: graph)
    monkeypatch.setattr(ro, "perception_check", lambda *a, **k: {"ok": True, "rung": "L1"})
    monkeypatch.setattr(ro, "engine_check", lambda *a, **k: {"ok": True, "engine": "genesis"})
    monkeypatch.setattr(
        ro,
        "run_gates",
        lambda *a, **k: {
            "ok": True,
            "env_hash": "fixture",
            "sim_extra": "sim",
            "sim_engine": "genesis",
            "sim_backend": "metal",
            "sim_device": "mps",
        },
    )
    original_run = ro.subprocess.run
    monkeypatch.setattr(
        ro.subprocess,
        "run",
        lambda command, *a, **k: (
            SimpleNamespace(stdout="fixture", returncode=0)
            if command[0] == "git"
            else original_run(command, *a, **k)
        ),
    )
    monkeypatch.setattr(ro, "time", clock)
    monkeypatch.setattr(ro, "_spawn_dora", spawn)
    monkeypatch.setattr(ro, "_terminate", lambda p: stopped.append(clock.now))
    monkeypatch.setattr(ro, "reap_orphans", lambda *a, **k: None)
    monkeypatch.setattr(ro, "build_budget_s", lambda *a: 0)
    monkeypatch.setattr(ro, "pre_data_stall_s", lambda *a: 1000)
    report = ro.rollout(
        root=tmp_path,
        graph=graph,
        tier="T4",
        episodes=1,
        seeds=[200],
        reset_mode="teleport",
        verifier="oracle",
        run_id="chain",
        branch="test",
        no_idea_gate=True,
        env_baseline="local",
        timeout_s=deadline,
        per_episode_wall_s=wall,
    )
    return report, stopped, tmp_path / "runs/chain/manifest.json"


def test_delayed_recovery_survives_ninety_seconds_of_result_silence(tmp_path, monkeypatch):
    """HAR-1/4; ADR-32: the live second goal survives the old post-count quiet cutoff."""
    report, stopped, manifest = run_fake_chain(tmp_path, monkeypatch, recovery_at=140, exit_at=142)
    assert stopped == [142]
    assert report["ok"] is True
    assert len(report["episodes"]) == 2
    assert report["chain_completion"]["complete_episodes"] == 1
    assert json.loads(manifest.read_text())["chain_completion"] == report["chain_completion"]


@pytest.mark.parametrize("exit_at, expected_stop", [(140, 140), (999, 200)])
def test_missing_recovery_never_reports_success(tmp_path, monkeypatch, exit_at, expected_stop):
    """HAR-4; CON-8: process exit or a hard deadline cannot turn a partial chain into success."""
    report, stopped, _ = run_fake_chain(tmp_path, monkeypatch, recovery_at=None, exit_at=exit_at)
    assert stopped == [expected_stop]
    assert report["ok"] is False
    assert report["chain_completion"]["missing_goal_ids"] == ["ep-0000r"]
    assert len(report["episodes"]) == 1
    assert report["episodes"][0]["status"] == "success"


def test_recovery_wall_clamp_retains_missing_outcome_without_synthetic_score(tmp_path, monkeypatch):
    """HAR-1/4: a bounded incomplete chain is stopped without inventing a recovery verdict."""
    report, stopped, manifest = run_fake_chain(
        tmp_path, monkeypatch, recovery_at=None, exit_at=999, wall=10
    )
    assert stopped[0] <= 12
    assert report["ok"] is False
    assert len(report["episodes"]) == 1
    assert not report["episodes"][0].get("synthetic")
    assert json.loads(manifest.read_text())["wall_clamped"] == [200]


def test_ordinary_t4_keeps_its_existing_completion_behavior(tmp_path, monkeypatch):
    """HAR-1; CON-5: the increment-two fix does not require recovery from an ordinary graph."""
    report, stopped, _ = run_fake_chain(
        tmp_path, monkeypatch, recovery_at=None, exit_at=999, inc2=False
    )
    assert stopped[0] <= 94
    assert report["ok"] is True
    assert "chain_completion" not in report


def test_chain_completion_counts_physical_episodes_and_checks_record_identities():
    """HAR-4; TC-7: extra goal rows cannot substitute for another physical episode."""
    from aisle.harness.rollout import t4_chain_completion

    records = [
        dict(episode=0, seed=200, goal_id="ep-0000", status="success"),
        dict(episode=0, seed=200, goal_id="ep-0000r", status="fail", recovery=True),
    ]
    partial = t4_chain_completion(records, [200, 201])
    assert partial["complete_episodes"] == 1
    assert partial["missing_goal_ids"] == ["ep-0001"]
    assert partial["ok"] is False
    records.append(dict(episode=1, seed=201, goal_id="ep-0001", status="success"))
    assert t4_chain_completion(records, [200, 201])["ok"] is True
    assert t4_chain_completion(records + [records[0]], [200, 201])["ok"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 204},
        {"episode": True},
        {"recovery": True},
        {"goal_id": "foreign"},
        {"synthetic": True},
        {"status": []},
    ],
)
def test_chain_completion_refuses_misattributed_or_synthetic_records(change):
    """HAR-4; TC-7: only the scheduled goal's actual result can satisfy completion."""
    from aisle.harness.rollout import t4_chain_completion

    record = dict(episode=0, seed=201, goal_id="ep-0000", status="success")
    record.update(change)
    result = t4_chain_completion([record], [201])
    assert result["ok"] is False
    assert result["invalid_record_indices"] == [0]
    assert result["missing_goal_ids"] == ["ep-0000"]


def test_increment_two_is_selected_by_graph_not_ambient_environment():
    """HAR-2; CON-5: a shell setting cannot change the graph-attested goal protocol."""
    from aisle.harness.rollout import scrub_bringup_env

    assert "AISLE_T4_INCREMENT_TWO" not in scrub_bringup_env({"AISLE_T4_INCREMENT_TWO": "1"})
