"""Harness-side engine selection (ADR-67).

The bridge learned `AISLE_SIM_ENGINE` in ADR-67; these tests cover the
harness half: the runner owns the choice (never the operator's shell), the
graph declares it where the graph hash attests it, every sim-launching entry
point can select it, and the attested identity and budgets follow the engine
actually resolved rather than a hardcoded Genesis default.

Unit-marked: nothing here imports a simulator.
"""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_MANIFESTS = {"dora-genesis": {"provides": ["sim_bridge"]}}


def _graph_with_bridge_env(tmp_path: Path, env: dict, name: str = "engine.yaml") -> Path:
    """The real T0 graph with extra bridge env, node paths absolutized."""
    doc = yaml.safe_load((REPO_ROOT / "graphs" / "expert_t0.yaml").read_text())
    for node in doc["nodes"]:
        node["path"] = str((REPO_ROOT / "graphs" / node["path"]).resolve())
        if node["id"] == "dora-genesis":
            node["env"] = {**(node.get("env") or {}), **env}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


# -- 1. the engine never leaks in from the ambient environment ------------


def test_engine_and_backend_are_scrubbed_from_the_ambient_environment():
    """CON-5, ADR-67: `harness fleet` builds its child env from
    scrub_bringup_env(os.environ), so an ambient AISLE_SIM_ENGINE=nexus in an
    operator's shell would silently swap the physics of a run whose git_sha,
    env_hash and graph_hash all attest clean. Both the engine and its backend
    are graph- or runner-owned, exactly like AISLE_PERCEPTION."""
    from aisle.harness.rollout import SCRUBBED_ENV, scrub_bringup_env

    assert "AISLE_SIM_ENGINE" in SCRUBBED_ENV
    assert "AISLE_SIM_BACKEND" in SCRUBBED_ENV
    kept = scrub_bringup_env(
        {"AISLE_SIM_ENGINE": "nexus", "AISLE_SIM_BACKEND": "webgpu", "AISLE_SEEDS": "0,1"}
    )
    assert kept == {"AISLE_SEEDS": "0,1"}


# -- 2. the graph declares the engine; the flag asserts it ----------------


def test_engine_check_takes_the_request_when_the_graph_declares_nothing(tmp_path):
    """ADR-67: an undeclared graph runs what the caller asked for, and a
    caller who asks for nothing gets the default engine."""
    from aisle.harness.rollout import engine_check

    graph = _graph_with_bridge_env(tmp_path, {})
    assert engine_check(REPO_ROOT, graph, None) == {
        "ok": True,
        "engine": "nexus",
        "declared": None,
    }
    assert engine_check(REPO_ROOT, graph, "nexus")["engine"] == "nexus"


def test_engine_check_honours_a_graph_declaration_the_cli_does_not_contradict(tmp_path):
    """ADR-67: the engine is declared on the bridge node like the perception
    rung, so a graph that names one is NOT silently overwritten with genesis
    when the runner is invoked without --sim-engine."""
    from aisle.harness.rollout import engine_check

    graph = _graph_with_bridge_env(tmp_path, {"AISLE_SIM_ENGINE": "nexus"})
    assert engine_check(REPO_ROOT, graph, None) == {
        "ok": True,
        "engine": "nexus",
        "declared": "nexus",
    }
    assert engine_check(REPO_ROOT, graph, "nexus")["engine"] == "nexus"


def test_engine_check_refuses_a_conflicting_request(tmp_path):
    """ADR-67, HAR-2: mirroring TC-9's perception assertion, a graph
    declaring one engine and a flag demanding another is refused at the
    `sim_engine` gate — overwriting would measure physics the graph hash does
    not attest."""
    from aisle.harness.rollout import engine_check

    graph = _graph_with_bridge_env(tmp_path, {"AISLE_SIM_ENGINE": "nexus"})
    refusal = engine_check(REPO_ROOT, graph, "genesis")
    assert refusal["ok"] is False and refusal["gate"] == "sim_engine"
    assert "nexus" in refusal["detail"] and "genesis" in refusal["detail"]
    assert "AISLE_SIM_ENGINE" in refusal["hint"]


def test_engine_check_refuses_an_unknown_declaration(tmp_path):
    """ADR-67 (TC-9's refuse-don't-guess rule): a typo must fail the gate
    rather than fall back to the default engine."""
    from aisle.harness.rollout import engine_check

    graph = _graph_with_bridge_env(tmp_path, {"AISLE_SIM_ENGINE": "bullet"})
    refusal = engine_check(REPO_ROOT, graph, None)
    assert refusal["ok"] is False and refusal["gate"] == "sim_engine"
    assert "unknown simulation engine" in refusal["detail"]
    blank_graph = _graph_with_bridge_env(tmp_path, {"AISLE_SIM_ENGINE": "  "}, "blank.yaml")
    blank = engine_check(REPO_ROOT, blank_graph, None)
    assert blank["ok"] is False


def test_instrumented_graph_refuses_rather_than_overwriting_a_declaration(tmp_path):
    """ADR-67: the last writer obeys the same rule as the gate. Before this,
    instrumentation stamped the requested engine over the bridge's own
    declaration, so a nexus graph ran on genesis without a word."""
    from aisle.harness.rollout import EngineConflict, instrumented_graph

    graph = _graph_with_bridge_env(tmp_path, {"AISLE_SIM_ENGINE": "nexus"})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(EngineConflict, match="nexus"):
        instrumented_graph(graph, REPO_ROOT, run_dir, sim_engine="genesis")
    out = instrumented_graph(graph, REPO_ROOT, run_dir, sim_engine="nexus")
    doc = yaml.safe_load(out.read_text())
    bridge = next(n for n in doc["nodes"] if n["id"] == "dora-genesis")
    assert bridge["env"]["AISLE_SIM_ENGINE"] == "nexus"


# -- 3. attested identity follows the resolved (engine, backend) pair -----


@pytest.mark.parametrize(
    ("engine", "backend", "device"),
    [
        ("genesis", "cpu", "cpu"),
        ("genesis", "metal", "mps"),
        ("nexus", "cpu", "cpu"),
        ("nexus", "metal", "metal"),
        ("nexus", "webgpu", "webgpu"),
    ],
)
def test_sim_device_follows_the_engine_and_backend(engine, backend, device):
    """CON-5: the attested device is derived from the resolved pair, not from
    the host OS. A Nexus webgpu run on Linux used to be attested `cpu`, which
    made GPU and CPU physics evidence indistinguishable in the manifest."""
    from aisle.harness.rollout import sim_device_for

    assert sim_device_for(engine, backend) == device


def test_resolve_sim_identity_refuses_an_engine_that_is_not_installed(monkeypatch):
    """CON-8, ADR-67: an engine whose package is missing refuses at the gate,
    before anything launches, instead of dying inside the bridge."""
    from aisle.harness import rollout as rollout_module

    monkeypatch.setattr("aisle.sim.engine_available", lambda engine: engine != "rapier")
    refused = rollout_module.resolve_sim_identity("sim", "rapier")
    assert refused["ok"] is False and refused["gate"] == "sim_engine"
    assert "'rapier' is not installed" in refused["detail"]
    assert rollout_module.resolve_sim_identity("sim", "genesis")["ok"] is True


def test_resolve_sim_identity_attests_the_nexus_gpu_device_on_linux(monkeypatch):
    """CON-5, ADR-67: the same rule end-to-end through the gate's resolver."""
    from aisle.harness import rollout as rollout_module

    monkeypatch.setattr(rollout_module.platform_module, "system", lambda: "Linux")
    monkeypatch.setattr("aisle.sim.engine_available", lambda engine: True)
    identity = rollout_module.resolve_sim_identity("sim", "nexus")
    assert identity["ok"] is True
    assert (identity["sim_backend"], identity["sim_device"]) == ("webgpu", "webgpu")
    genesis = rollout_module.resolve_sim_identity("sim", "genesis")
    assert (genesis["sim_backend"], genesis["sim_device"]) == ("cpu", "cpu")


# -- 4. budgets are engine-derived ----------------------------------------


def test_build_grace_and_pre_data_stall_are_engine_derived():
    """HAR-1, ADR-67: the 420 s build grace and the 600 s pre-data stall were
    both justified by the minutes-long Genesis build. Nexus builds a scene in
    seconds, so a wedged Nexus launch must clamp far sooner."""
    from aisle.harness.rollout import (
        GENESIS_BUILD_BUDGET_S,
        PRE_DATA_STALL_S,
        build_budget_s,
        pre_data_stall_s,
    )

    assert build_budget_s("genesis") == GENESIS_BUILD_BUDGET_S
    assert pre_data_stall_s("genesis") == PRE_DATA_STALL_S
    assert build_budget_s("nexus") < GENESIS_BUILD_BUDGET_S
    assert pre_data_stall_s("nexus") < PRE_DATA_STALL_S
    assert build_budget_s("nexus") > 0 and pre_data_stall_s("nexus") > 0


# -- 5. every sim-launching entry point can select the engine -------------


def _sim_engine_action(subparser_path: list[str]):
    """The --sim-engine action of a subcommand, found by walking argparse."""
    from aisle.harness.cli import build_parser

    parser = build_parser()
    for name in subparser_path:
        choices = next(
            action.choices
            for action in parser._actions
            if getattr(action, "choices", None) and isinstance(action.choices, dict)
        )
        parser = choices[name]
    return next(
        (action for action in parser._actions if action.option_strings == ["--sim-engine"]), None
    )


@pytest.mark.parametrize(
    "command",
    [
        ["rollout"],
        ["monolith", "run"],
        ["fault", "calibrate"],
        ["fleet"],
        ["skill", "register"],
    ],
)
def test_every_sim_launching_entry_point_shares_one_engine_flag(command):
    """ADR-67, CON-8: `harness rollout` was the only path that could select
    an engine, so every other launcher forced genesis. They now share one
    flag definition, which is also why no path can drift to a different
    engine set."""
    from aisle.sim import ENGINES

    action = _sim_engine_action(command)
    assert action is not None, f"{' '.join(command)} cannot select an engine"
    assert tuple(action.choices) == ENGINES
    assert action.default is None  # no assertion: the graph's declaration wins


def test_monolith_run_forwards_the_engine_to_the_runner(tmp_path, monkeypatch):
    """MON-3, ADR-67: the monolithic launcher is a sim-launching entry point;
    it must not pin its module to Genesis."""
    from aisle.harness import monolith as mono
    from aisle.harness import rollout

    observed = {}
    monkeypatch.setattr(mono, "check_module", lambda *args: {"ok": True})
    monkeypatch.setattr(mono, "stamp_graph", lambda *args: tmp_path / "stamped.yaml")
    monkeypatch.setattr(
        rollout, "rollout", lambda **kwargs: observed.update(kwargs) or {"ok": True}
    )
    mono.run(tmp_path, tmp_path / "m.py", [0], 1, run_id="engine-test", sim_engine="nexus")
    assert observed["sim_engine"] == "nexus"


def test_skill_registration_forwards_the_engine_to_the_eval_rollout(monkeypatch):
    """CAP-6, ADR-67: a skill's evalcard records a measured pass rate, so the
    eval must run on the engine the operator selected."""
    from aisle.harness import skill as skill_module

    observed = {}

    def fake_rollout(**kwargs):
        observed.update(kwargs)
        return {"ok": True, "pass1": 1.0}

    fake_skill = type(
        "S",
        (),
        {
            "eval_cfg": {
                "tier": "T0",
                "episodes": 1,
                "seeds": "0",
                "embodiment": "franka",
                "suite": "s",
            },
            "path": Path("."),
            "manifest": {"id": "x"},
        },
    )()
    monkeypatch.setattr(skill_module, "_eval_graph_path", lambda skill, root: Path("g.yaml"))
    rate = skill_module.run_skill_eval(fake_skill, Path("."), fake_rollout, "rid", "nexus")
    assert rate == 1.0
    assert observed["sim_engine"] == "nexus"


def test_fault_calibration_rung_passes_the_engine_to_the_rollout_argv(tmp_path, monkeypatch):
    """FLT-9, ADR-67: a calibration ladder measures severities against a
    clean baseline, so every rung must run the campaign's engine rather than
    the rollout default."""
    from aisle.harness import fault_calibration as fc

    calls = []

    class Completed:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Completed()

    monkeypatch.setattr(fc.subprocess, "run", fake_run)
    monkeypatch.setattr(fc, "materialize", lambda *args, **kwargs: {"receipt": "r"})
    outcome = fc.run_rung(
        root=tmp_path,
        worktree=tmp_path / "wt",
        clean_commit="abc",
        instance={"opaque_id": "i"},
        severity_index=0,
        clean_hash="h",
        graph_rel="graphs/expert_t1.yaml",
        seeds="0..1",
        tier="T1",
        embodiment="franka",
        perception="L1",
        run_id="rid",
        sim_engine="nexus",
    )
    assert outcome["result"]["ok"] is True
    rollout_argv = next(cmd for cmd in calls if "rollout" in cmd)
    assert rollout_argv[rollout_argv.index("--sim-engine") + 1] == "nexus"
    # no engine selected leaves the graph's own declaration in charge
    calls.clear()
    fc.run_rung(
        root=tmp_path,
        worktree=tmp_path / "wt2",
        clean_commit="abc",
        instance={"opaque_id": "i"},
        severity_index=0,
        clean_hash="h",
        graph_rel="graphs/expert_t1.yaml",
        seeds="0..1",
        tier="T1",
        embodiment="franka",
        perception="L1",
        run_id="rid2",
    )
    assert "--sim-engine" not in next(cmd for cmd in calls if "rollout" in cmd)


@pytest.mark.parametrize(
    "declared,requested,installed,expected",
    [
        ("nexus", "genesis", True, "graph declares engine nexus"),
        (None, "rapier", False, "'rapier' is not installed"),
    ],
)
def test_fleet_refuses_before_launch(
    tmp_path, monkeypatch, capsys, declared, requested, installed, expected
):
    """ADR-67, CON-8: fleet runs the same engine gate as rollout. A bridge's
    own AISLE_SIM_ENGINE beats the child env dora is given, so a conflicting
    flag would run one engine and report another; a missing engine would die
    inside the bridge. Both refuse before anything launches."""
    import json
    import subprocess
    import sys

    from aisle.harness import cli

    env = {} if declared is None else {"AISLE_SIM_ENGINE": declared}
    graph = _graph_with_bridge_env(tmp_path, env)
    monkeypatch.setattr("aisle.sim.engine_available", lambda engine: installed)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("fleet launched"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["harness", "fleet", "--graph", str(graph), "--agents", "2", "--episodes", "1"]
        + ["--seeds", "0", "--root", str(REPO_ROOT), "--out", str(tmp_path / "out")]
        + ["--sim-engine", requested],
    )
    assert cli.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["refused"]["gate"] == "sim_engine"
    assert expected in report["refused"]["detail"]
    assert not (tmp_path / "out").exists()


def test_fleet_report_records_the_engine_it_ran(tmp_path):
    """CON-5, ADR-67: `harness fleet` sets the engine in the child env after
    the scrub, so its report is where the run says which physics it measured
    — otherwise a fleet run carries no engine evidence at all."""
    from aisle.harness.fleet import run_fleet

    graph = REPO_ROOT / "graphs" / "expert_t0.yaml"
    out_dir = tmp_path / "fleet"

    def launch(graph_path):
        for agent in range(2):
            (out_dir / f"results_a{agent}.jsonl").write_text(
                '{"episode": 0, "seed": 0, "status": "success"}\n'
            )
        return lambda: 0

    report = run_fleet(
        graph,
        2,
        1,
        [0],
        out_dir,
        5.0,
        launch,
        root=REPO_ROOT,
        sim_engine="nexus",
        sim_backend="webgpu",
        sim_device="webgpu",
    )
    assert report["sim_engine"] == "nexus"
    assert report["sim_backend"] == "webgpu"
    assert report["sim_device"] == "webgpu"


# -- 6. the whole run: what is injected, and what is attested -------------


def _stub_run(tmp_path, monkeypatch, engine="genesis"):
    """A fake root whose dora launch is replaced by a results writer; returns
    (root, graph, spawned-env dict)."""
    import json

    from test_idea_gate import _fake_root

    from aisle.harness import rollout as rollout_module

    root = _fake_root(tmp_path)
    spawned: dict = {}

    def fake_spawn(exec_graph, run_dir, env, relaunch=0):
        spawned.update(env)
        results = Path(env["AISLE_RESULTS"])
        seeds = [int(s) for s in env["AISLE_SEEDS"].split(",")]
        with open(results, "w") as handle:
            for index, seed in enumerate(seeds):
                handle.write(
                    json.dumps(
                        {"episode": index, "seed": seed, "status": "success", "failure": None}
                    )
                    + "\n"
                )

        class FakeProc:
            """Exited: the runner then stops waiting as soon as the seeded
            rows are on disk, which keeps this a unit test."""

            pid = 0

            def poll(self):
                return 0

        return FakeProc()

    monkeypatch.setattr(rollout_module, "_spawn_dora", fake_spawn)
    monkeypatch.setattr(rollout_module, "_terminate", lambda proc: None)
    monkeypatch.setattr(
        rollout_module,
        "resolve_sim_identity",
        lambda extra, requested="genesis": {
            "ok": True,
            "sim_extra": extra,
            "sim_engine": engine,
            "sim_backend": "webgpu" if engine == "nexus" else "cpu",
            "sim_device": "webgpu" if engine == "nexus" else "cpu",
        },
    )
    return root, root / "graphs" / "expert_t0.yaml", spawned


def test_an_ambient_engine_never_reaches_the_launched_dataflow(tmp_path, monkeypatch):
    """CON-5, ADR-67: the leak this scrub closes. With AISLE_SIM_ENGINE=nexus
    exported in the shell, the run must still launch — and attest — the
    engine the gate resolved."""
    import json

    monkeypatch.setenv("AISLE_SIM_ENGINE", "nexus")
    monkeypatch.setenv("AISLE_SIM_BACKEND", "webgpu")
    root, graph, spawned = _stub_run(tmp_path, monkeypatch)
    from aisle.harness import rollout as rollout_module

    report = rollout_module.rollout(
        root=root,
        graph=graph,
        tier="T0",
        episodes=1,
        seeds=[0],
        reset_mode="teleport",
        verifier="oracle",
        run_id="ambient-engine",
        branch="b",
        no_idea_gate=True,
        env_baseline="local",
    )
    assert report["ok"] is True, report
    assert spawned["AISLE_SIM_ENGINE"] == "genesis"
    assert spawned["AISLE_SIM_BACKEND"] == "cpu"
    manifest = json.loads((root / "runs" / "ambient-engine" / "manifest.json").read_text())
    assert manifest["sim_engine"] == "genesis"


def test_the_run_attests_the_resolved_engine_and_its_build_grace(tmp_path, monkeypatch):
    """HAR-1, HAR-4, ADR-67: the manifest records the engine the gate
    resolved (a lost key must fail, never attest genesis), and the launch
    clamps against that engine's build grace."""
    import json

    from aisle.harness import rollout as rollout_module
    from aisle.harness.rollout import NEXUS_BUILD_BUDGET_S

    root, graph, _ = _stub_run(tmp_path, monkeypatch, engine="nexus")
    report = rollout_module.rollout(
        root=root,
        graph=graph,
        tier="T0",
        episodes=1,
        seeds=[0],
        reset_mode="teleport",
        verifier="oracle",
        run_id="nexus-grace",
        branch="b",
        no_idea_gate=True,
        env_baseline="local",
    )
    assert report["ok"] is True
    manifest = json.loads((root / "runs" / "nexus-grace" / "manifest.json").read_text())
    assert manifest["sim_engine"] == "nexus"
    assert manifest["sim_backend"] == "webgpu" and manifest["sim_device"] == "webgpu"
    assert manifest["build_grace_s"] == NEXUS_BUILD_BUDGET_S


def test_a_gate_without_an_engine_fails_instead_of_attesting_genesis(tmp_path, monkeypatch):
    """CON-5, HAR-4: `gates.get("sim_engine", "genesis")` turned a lost key
    into a genesis attestation on a run that may have been anything. The key
    is now required, so the failure is loud."""
    from aisle.harness import rollout as rollout_module

    root, graph, _ = _stub_run(tmp_path, monkeypatch, engine="nexus")
    real_run_gates = rollout_module.run_gates

    def gates_without_engine(*args, **kwargs):
        result = real_run_gates(*args, **kwargs)
        return {k: v for k, v in result.items() if k != "sim_engine"}

    monkeypatch.setattr(rollout_module, "run_gates", gates_without_engine)
    with pytest.raises(KeyError, match="sim_engine"):
        rollout_module.rollout(
            root=root,
            graph=graph,
            tier="T0",
            episodes=1,
            seeds=[0],
            reset_mode="teleport",
            verifier="oracle",
            run_id="no-engine",
            branch="b",
            no_idea_gate=True,
            env_baseline="local",
        )


def test_a_rollout_refuses_a_graph_declaring_another_engine(tmp_path, monkeypatch):
    """ADR-67, HAR-2: the refusal reaches the CON-8 report as its own gate,
    before any budget reservation or launch."""
    from aisle.harness import rollout as rollout_module

    root, graph, spawned = _stub_run(tmp_path, monkeypatch)
    doc = yaml.safe_load(graph.read_text())
    bridge = next(node for node in doc["nodes"] if node["id"] == "dora-genesis")
    bridge["env"] = {**(bridge.get("env") or {}), "AISLE_SIM_ENGINE": "nexus"}
    graph.write_text(yaml.safe_dump(doc, sort_keys=False))
    report = rollout_module.rollout(
        root=root,
        graph=graph,
        tier="T0",
        episodes=1,
        seeds=[0],
        reset_mode="teleport",
        verifier="oracle",
        run_id="conflict",
        branch="b",
        no_idea_gate=True,
        env_baseline="local",
        sim_engine="genesis",
    )
    assert report["ok"] is False
    assert report["refused"]["gate"] == "sim_engine"
    assert spawned == {}  # refused before launch


# -- 7. validation knows the variable exists ------------------------------


def test_validator_refuses_an_unknown_engine_on_the_bridge():
    """VAL-8's companion for ADR-67: the engine rides the bridge's env like
    the rung, so a typo must be a validation error rather than a launch-time
    surprise hours later."""
    from aisle.harness.validate import graph_sim_engine_errors, validate_nodes

    nodes = [{"id": "dora-genesis", "env": {"AISLE_SIM_ENGINE": "bullet"}}]
    errors = graph_sim_engine_errors(nodes, ["dora-genesis"])
    assert [e["code"] for e in errors] == ["SIM_ENGINE_UNKNOWN"]
    assert "bullet" in errors[0]["detail"] and "nexus" in errors[0]["hint"]
    reported, _ = validate_nodes(nodes, BRIDGE_MANIFESTS, set(), "franka", allow_unproven=True)
    assert "SIM_ENGINE_UNKNOWN" in [e["code"] for e in reported]


def test_validator_accepts_known_engines_and_an_absent_declaration():
    """ADR-67: every pre-ADR-67 graph declares nothing and MUST keep
    validating unchanged; a known engine is not an error either."""
    from aisle.harness.validate import graph_sim_engine_errors

    assert graph_sim_engine_errors([{"id": "dora-genesis"}], ["dora-genesis"]) == []
    for engine in ("nexus", " Genesis "):
        nodes = [{"id": "dora-genesis", "env": {"AISLE_SIM_ENGINE": engine}}]
        assert graph_sim_engine_errors(nodes, ["dora-genesis"]) == []
