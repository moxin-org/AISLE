"""Unit tests for the generated contributor inventory (CON-5, CON-8).

The documentation inventory is derived from executable repository sources and
checked in CI. These tests pin deterministic rendering and the write/check CLI
contract before the generator implementation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cli_helpers import REPO_ROOT, run_tool

pytestmark = pytest.mark.unit

TOOL = REPO_ROOT / "tools" / "docs_inventory.py"


def run_inventory(*args: str):
    return run_tool("docs_inventory.py", *args)


def inventory_module():
    """Import tools/docs_inventory.py for direct-call unit assertions."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import docs_inventory

    return docs_inventory


def tracked(pathspec: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", pathspec],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def write_markers(root: Path) -> None:
    """Minimal pyproject the generator's marker lookup accepts."""
    root.joinpath("pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers = ["unit: no sim"]\n', encoding="utf-8"
    )


def test_write_is_deterministic_and_covers_repository_surfaces(tmp_path: Path):
    """CON-5: identical repository inputs produce byte-identical graph,
    capability, CLI, ADR, and test inventories across invocations."""
    output = tmp_path / "project-inventory.md"

    first = run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--write")
    assert first.returncode == 0, first.stdout + first.stderr
    first_report = json.loads(first.stdout)
    first_bytes = output.read_bytes()

    second = run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--write")
    assert second.returncode == 0, second.stdout + second.stderr
    second_report = json.loads(second.stdout)
    assert output.read_bytes() == first_bytes
    assert second_report["sha256"] == first_report["sha256"]

    text = first_bytes.decode()
    assert "graphs/expert_t0.yaml" in text
    assert "registry/manifests/budget-guard.yaml" in text
    assert "`harness rollout`" in text
    assert "docs/decisions/ADR-M0.md" in text
    # suite DIRECTORIES, not modules — see test_module_churn_does_not_stale_the_inventory
    assert "`tests/unit`" in text
    assert "tests/unit/test_docs_inventory.py" not in text
    # tracked, not globbed: the generator documents COMMITTED surfaces, so a
    # dirty working tree must not move these numbers
    expected_counts = {
        "adrs": len(tracked("docs/decisions/ADR-*.md")),
        "capabilities": len(tracked("registry/manifests/*.yaml")),
        "graphs": len(tracked("graphs/*.yaml")),
    }
    assert {key: first_report["counts"][key] for key in expected_counts} == expected_counts
    assert first_report["counts"]["cli_commands"] >= 10
    assert first_report["counts"]["test_modules"] >= 60


def test_check_detects_missing_and_stale_output(tmp_path: Path):
    """CON-8: --check emits one JSON report and exits nonzero when the
    committed inventory is missing or differs from repository reality."""
    output = tmp_path / "project-inventory.md"

    missing = run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--check")
    missing_report = json.loads(missing.stdout)
    assert missing.returncode != 0
    assert missing_report["ok"] is False
    assert missing_report["reason"] == "missing"

    written = run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--write")
    assert written.returncode == 0
    output.write_text(output.read_text() + "stale\n")

    stale = run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--check")
    stale_report = json.loads(stale.stdout)
    assert stale.returncode != 0
    assert stale_report["ok"] is False
    assert stale_report["reason"] == "stale"


def test_check_accepts_fresh_output_and_stdout_is_json(tmp_path: Path):
    """CON-8: a fresh inventory passes with a single JSON object on stdout,
    no stdout prose, and exit zero iff ok is true."""
    output = tmp_path / "project-inventory.md"
    assert (
        run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--write").returncode == 0
    )

    checked = run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--check")
    report = json.loads(checked.stdout)
    assert checked.returncode == 0
    assert report["ok"] is True
    assert report["reason"] == "current"
    assert checked.stderr == ""


@pytest.mark.parametrize("mode", [(), ("--write", "--check")])
def test_exactly_one_mode_is_required(mode: tuple[str, ...]):
    """CON-8: ambiguous mutation/check intent is refused by argparse."""
    proc = run_inventory(*mode)
    assert proc.returncode != 0


def test_committed_inventory_matches_this_tree():
    """CON-5 self-consistency: docs/generated/project-inventory.md must
    describe THIS tree.

    Every other test here writes to tmp_path, so `pytest -m unit` went green
    on a stale committed inventory and the drift only surfaced in the ci/unit
    job's --check step — after push. tools/env_hash.py learned the same lesson
    (test_committed_hash_matches_this_tree); this is its docs counterpart."""
    proc = run_inventory("--root", str(REPO_ROOT), "--check")
    report = json.loads(proc.stdout)
    assert proc.returncode == 0, (
        f"committed inventory is {report.get('reason')}; "
        "run `uv run python tools/docs_inventory.py --write`"
    )
    assert report["ok"] is True


def test_untracked_residue_is_excluded_from_the_documented_surfaces(tmp_path: Path):
    """The gate documents COMMITTED surfaces: an untracked scratch test or an
    agent-authored graph is experiment residue, not documentation drift.

    `harness swap` rewrites graphs/ and `harness skill register` writes
    registry/manifests/ mid-session; globbing the working tree let that
    residue block every unrelated commit, and `--write` would commit it.

    Exercised in a THROWAWAY git repo under tmp_path: an earlier version of
    this test wrote and then unlinked fixed paths inside REPO_ROOT, which
    truncates and deletes a contributor's real file if the name collides and
    races a concurrent run (PR #141 review, P1). A unit test must never
    mutate the checkout it is running in."""
    module = inventory_module()
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    committed = graphs / "expert_demo.yaml"
    committed.write_text("nodes: []\n", encoding="utf-8")
    (graphs / "agent_scratch.yaml").write_text("nodes: []\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    # `git ls-files` reads the INDEX, so staging is enough to make a file
    # tracked — no commit, and therefore no committer identity, required
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "graphs/expert_demo.yaml"],
        check=True,
        capture_output=True,
    )

    tracked = module._tracked_paths(tmp_path)
    assert tracked == {"graphs/expert_demo.yaml"}
    assert module._select(graphs.glob("*.yaml"), tmp_path, tracked) == [committed]


def test_selection_falls_back_to_the_glob_outside_a_git_tree(tmp_path: Path):
    """Tracked-file selection is an exclusion of residue, not a hard git
    dependency: an export or tarball with no .git still renders."""
    module = inventory_module()
    (tmp_path / "only.yaml").write_text("nodes: []\n", encoding="utf-8")

    assert module._tracked_paths(tmp_path) is None
    assert module._select(tmp_path.glob("*.yaml"), tmp_path, None) == [tmp_path / "only.yaml"]


def test_adding_a_test_module_does_not_stale_the_inventory(tmp_path: Path):
    """Two independently-green PRs that each add a test file used to merge
    without conflict into a main whose inventory was stale — CI red on main.

    The rendered Tests section names suite DIRECTORIES, so module churn inside
    an existing suite cannot move the committed bytes."""
    output = tmp_path / "project-inventory.md"
    assert (
        run_inventory("--root", str(REPO_ROOT), "--output", str(output), "--write").returncode == 0
    )
    before = output.read_bytes()

    module = inventory_module()
    content, _ = module.render_inventory(REPO_ROOT)
    assert content.encode("utf-8") == before
    assert "### Test modules" not in content
    assert "| Test modules |" not in content


def test_check_survives_a_non_utf8_locale():
    """Reads and writes pin encoding='utf-8' while the digest is over UTF-8.

    Under LC_ALL=C the generator used to die with an opaque
    "'ascii' codec can't decode byte 0xe2" — the repo's graphs, ADRs and the
    inventory itself all carry em dashes — blocking tools/ci.sh with a message
    that says nothing about documentation drift."""
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--root", str(REPO_ROOT), "--check"],
        capture_output=True,
        text=True,
        env=env,
    )
    report = json.loads(proc.stdout)
    assert "error" not in report, report
    assert proc.returncode == 0, report


def test_a_non_adr_file_beside_the_adrs_is_ignored(tmp_path: Path):
    """A README, index or template in docs/decisions/ is not a malformed ADR.

    Globbing *.md made any such file raise "ADR has no level-one heading" and
    fail the whole gate for every contributor, on an error path that also
    dropped the reason/sha256/counts keys the JSON contract promises."""
    decisions = tmp_path / "docs" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / "ADR-1.md").write_text(
        "# ADR-1 — a decision\n\nStatus: ACCEPTED\n", encoding="utf-8"
    )
    (decisions / "README.md").write_text("## Index of decisions\n", encoding="utf-8")

    rows = inventory_module()._adr_inventory(tmp_path, None)
    assert [row["title"] for row in rows] == ["ADR-1 — a decision"]


def test_adr_status_is_the_first_status_line_only(tmp_path: Path):
    """The rendered legend promises "the literal first `Status:` line"; the
    continuation loop swept following prose and bullets into the cell."""
    decisions = tmp_path / "docs" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / "ADR-2.md").write_text(
        "# ADR-2 — scoped\n\nStatus: PROPOSED\nTask: T10. Specs: 090.\n- a bullet\n",
        encoding="utf-8",
    )

    rows = inventory_module()._adr_inventory(tmp_path, None)
    assert rows[0]["status"] == "PROPOSED"


def test_adr_inventory_rejects_duplicate_identity(tmp_path: Path):
    """Issue #144: two files must never render as the same decision id.

    A duplicate ADR-27 made every bare cross-reference ambiguous even though
    the generated table faithfully listed both files."""
    decisions = tmp_path / "docs" / "decisions"
    decisions.mkdir(parents=True)
    for name in ("ADR-27.md", "ADR-27-perception.md"):
        (decisions / name).write_text(
            "# ADR-27 — a decision\n\nStatus: ACCEPTED\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="duplicate ADR identity ADR-27"):
        inventory_module()._adr_inventory(tmp_path, None)


def test_adr_inventory_requires_declared_status(tmp_path: Path):
    """Issue #144: inventory generation fails instead of inferring status."""
    decisions = tmp_path / "docs" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / "ADR-1.md").write_text("# ADR-1 — a decision\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ADR has no declared Status"):
        inventory_module()._adr_inventory(tmp_path, None)


def test_adr_inventory_requires_filename_to_match_identity(tmp_path: Path):
    """Issue #144: the stable path and heading cannot name different ADRs."""
    decisions = tmp_path / "docs" / "decisions"
    decisions.mkdir(parents=True)
    (decisions / "ADR-28.md").write_text(
        "# ADR-27 — a decision\n\nStatus: ACCEPTED\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="filename identity ADR-28 does not match heading ADR-27"):
        inventory_module()._adr_inventory(tmp_path, None)


def test_suite_directories_are_directories(tmp_path: Path):
    """Grouping by parts[0] rendered a module sitting directly in tests/ as if
    the file itself were a suite directory."""
    tests = tmp_path / "tests"
    (tests / "unit").mkdir(parents=True)
    (tests / "test_root_level.py").write_text("", encoding="utf-8")
    (tests / "unit" / "test_nested.py").write_text("", encoding="utf-8")
    write_markers(tmp_path)

    suites, _ = inventory_module()._test_inventory(tmp_path, None)
    assert suites == ["tests", "tests/unit"]


def test_bridge_is_found_by_module_path_not_node_id(tmp_path: Path):
    """A renamed bridge node used to fall through to {} and publish
    "pharmacy/franka/L0 (default)" for a store/mobile/L2 graph. The rung is
    safety-relevant (TC-9), so a wrong one is worse than none."""
    graphs = tmp_path / "graphs"
    graphs.mkdir(parents=True)
    (graphs / "renamed.yaml").write_text(
        "nodes:\n"
        "  - id: sim-bridge\n"
        "    path: ../src/aisle/nodes/dora_genesis.py\n"
        "    env:\n"
        "      AISLE_SCENE: store\n"
        "      AISLE_EMBODIMENT: mobile\n"
        "      AISLE_PERCEPTION: L2\n"
        "    outputs: [poses]\n",
        encoding="utf-8",
    )
    (graphs / "bridgeless.yaml").write_text(
        "nodes:\n  - id: analysis-only\n    outputs: [report]\n", encoding="utf-8"
    )

    rows = {row["path"].name: row for row in inventory_module()._graph_inventory(tmp_path, None)}
    assert rows["renamed.yaml"]["scene"] == "store"
    assert rows["renamed.yaml"]["embodiment"] == "mobile"
    assert rows["renamed.yaml"]["perception"] == "L2"
    # no bridge => declare nothing rather than assert a default that is false
    assert rows["bridgeless.yaml"]["perception"] == "—"
    assert rows["bridgeless.yaml"]["scene"] == "—"


def test_graph_table_names_the_physics_engine(tmp_path: Path):
    """ADR-67, CON-5: a graph declaring `AISLE_SIM_ENGINE: nexus` runs
    different physics, which is a different environment. Without the engine
    column the generated inventory documented every graph as Genesis, and
    the Nexus package did not exist anywhere in the layout map."""
    graphs = tmp_path / "graphs"
    graphs.mkdir(parents=True)
    (graphs / "nexus.yaml").write_text(
        "nodes:\n"
        "  - id: sim-bridge\n"
        "    path: ../src/aisle/nodes/dora_genesis.py\n"
        "    env:\n"
        "      AISLE_SIM_ENGINE: nexus\n"
        "    outputs: [poses]\n",
        encoding="utf-8",
    )
    (graphs / "default.yaml").write_text(
        "nodes:\n  - id: sim-bridge\n    path: ../src/aisle/nodes/dora_genesis.py\n"
        "    outputs: [poses]\n",
        encoding="utf-8",
    )
    (graphs / "bridgeless.yaml").write_text(
        "nodes:\n  - id: analysis-only\n    outputs: [report]\n", encoding="utf-8"
    )

    rows = {row["path"].name: row for row in inventory_module()._graph_inventory(tmp_path, None)}
    assert rows["nexus.yaml"]["engine"] == "nexus"
    assert rows["default.yaml"]["engine"] == "nexus (default)"
    # no bridge => no engine declaration to report
    assert rows["bridgeless.yaml"]["engine"] == "—"

    text, _ = inventory_module().render_inventory(REPO_ROOT)
    assert "| Engine |" in text
    assert "sim engines" in text  # the layout map names the package


def test_cli_section_refuses_a_root_it_cannot_introspect(tmp_path: Path):
    """--root is honoured for graphs/manifests/ADRs/tests but the CLI table is
    built from the IMPORTED package, which --root cannot redirect.

    Silently describing the installed tree's CLI under another tree's
    inventory is wrong in both directions: it blocks a correct commit, or it
    certifies as current the very drift the gate exists to catch."""
    proc = run_inventory("--root", str(tmp_path), "--output", str(tmp_path / "out.md"), "--check")
    report = json.loads(proc.stdout)
    assert proc.returncode != 0
    assert report["ok"] is False
    assert "outside --root" in report["error"]


def test_layout_section_marks_which_paths_are_frozen():
    """The design doc's §8.0 tree said `env/` holds the bridge, scenes,
    verifier and reset, and that everything under it is frozen. Neither half
    is true any more: the code lives under `src/aisle/`, and the fence is
    `FROZEN_DIRS + FROZEN_FILES + FROZEN_GLOBS`, not a directory.

    Re-typing the real tree into prose would recreate the failure — a
    hand-maintained second copy of the fence is exactly how `src/aisle/mobility`
    stayed outside it (#189, ADR-33). So the layout is GENERATED, and the
    frozen marking is derived from `env_hash`'s constants rather than
    restated, so it cannot disagree with what is actually attested."""
    from docs_inventory import render_inventory

    text, _ = render_inventory(REPO_ROOT)
    assert "## Repository layout" in text
    # the fence is visible in the map, which is the property §8.0's `env/`
    # gave and the current structure lost
    assert "src/aisle/verifier" in text
    assert "frozen" in text.lower()


def test_layout_is_derived_from_env_hash_not_restated():
    """ADR-33's rule: never hand-maintain a second copy of the fence. Asserted
    BEHAVIOURALLY — every frozen path env_hash knows about must appear in the
    map, so a widening shows up automatically instead of needing a second edit
    someone forgets.

    (A source-inspection version of this test passed against a mutation that
    replaced the derivation with a hardcoded list, because the import line
    alone satisfied it. Set equality does not have that hole.)"""
    from docs_inventory import _layout_inventory
    from env_hash import FROZEN_DIRS, FROZEN_FILES, FROZEN_GLOBS

    rendered = " ".join(row["frozen"] for row in _layout_inventory(REPO_ROOT))
    for path in (*FROZEN_DIRS, *FROZEN_FILES, *FROZEN_GLOBS):
        assert f"`{path}`" in rendered, f"{path} is attested but missing from the layout map"
