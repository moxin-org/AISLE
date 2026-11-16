#!/usr/bin/env python3
"""Generate/check the source-derived contributor inventory (CON-5, CON-8).

The committed Markdown appendix is deterministic: it contains no wall time,
Git branch, or working-tree state. CI runs --check so graph, capability, CLI,
ADR, and test changes cannot silently leave the contributor reference stale.

Inputs are restricted to files git TRACKS. graphs/ and registry/manifests/ are
session-writable — `harness swap` rewrites graphs and `harness skill register`
writes manifests mid-experiment — so globbing the working tree would let
transient experiment residue turn this CI gate red on unrelated commits (and
`--write` would then commit that residue). Outside a git tree the glob is the
documented fallback.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from env_hash import FROZEN_DIRS, FROZEN_FILES, FROZEN_GLOBS  # noqa: E402

import aisle  # noqa: E402
from aisle.harness.cli import build_parser as build_harness_parser  # noqa: E402
from aisle.harness.registry import build_parser as build_registry_parser
from aisle.harness.validate import load_graph
from aisle.sim import DEFAULT_ENGINE  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("docs/generated/project-inventory.md")
BRIDGE_MODULE = "dora_genesis.py"


def _cell(value: object) -> str:
    """Escape a compact value for a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or "—"


def _link(label: str, target: str) -> str:
    return f"[{_cell(label)}]({_cell(target)})"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _tracked_paths(root: Path) -> set[str] | None:
    """Repo-relative posix paths git tracks, or None outside a git tree."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {name for name in proc.stdout.decode("utf-8").split("\0") if name}


def _select(paths: Iterable[Path], root: Path, tracked: set[str] | None) -> list[Path]:
    """Committed files only: an untracked graph/manifest/test is experiment
    residue, not a documented surface."""
    if tracked is None:
        return sorted(paths)
    return sorted(path for path in paths if _relative(path, root) in tracked)


def _is_bridge(node: dict) -> bool:
    """BY MODULE PATH first, not by node id: the id is the graph author's
    choice, and matching only "dora-genesis" used to fall through to `{}` and
    publish "pharmacy/franka/L0 (default)" for what may be a store/mobile/L2
    graph. The published rung is safety-relevant (TC-9)."""
    path = node.get("path")
    if isinstance(path, str) and path.endswith(BRIDGE_MODULE):
        return True
    return node.get("id") == "dora-genesis"


def _graph_inventory(root: Path, tracked: set[str] | None) -> list[dict]:
    rows = []
    for path in _select((root / "graphs").glob("*.yaml"), root, tracked):
        nodes, errors = load_graph(path)
        # nodes can come back non-None WITH structural errors (a node missing
        # its id), so the error list is the gate, not `nodes is None`.
        if nodes is None or errors:
            detail = errors[0].get("detail", errors[0]) if errors else "unreadable graph"
            raise ValueError(f"{path}: {detail}")
        node_ids = [node["id"] for node in nodes]
        bridge = next((node for node in nodes if _is_bridge(node)), None)
        env = bridge.get("env") if bridge else None
        if not isinstance(env, dict):
            env = {} if bridge else None
        rows.append(
            {
                "path": path,
                # no bridge => the graph declares no scene/embodiment/rung at
                # all; "—" says that, where a default would assert a falsehood
                "scene": env.get("AISLE_SCENE", "pharmacy (default)") if env is not None else "—",
                "scenario": env.get("AISLE_SCENARIO", "—") if env is not None else "—",
                "embodiment": (
                    env.get("AISLE_EMBODIMENT", "franka (default)") if env is not None else "—"
                ),
                "perception": (
                    env.get("AISLE_PERCEPTION", "L0 (default)") if env is not None else "—"
                ),
                # ADR-67: the engine is declared on the bridge like the rung,
                # and a run on another engine is another environment; without
                # this column every graph reads as one engine.
                "engine": (
                    env.get("AISLE_SIM_ENGINE", f"{DEFAULT_ENGINE} (default)")
                    if env is not None
                    else "—"
                ),
                "nodes": node_ids,
            }
        )
    return rows


def _capability_inventory(root: Path, tracked: set[str] | None) -> list[dict]:
    rows = []
    for path in _select((root / "registry" / "manifests").glob("*.yaml"), root, tracked):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("id"), str):
            raise ValueError(f"{path}: manifest must contain a string id")
        embodiment = manifest.get("embodiment", {})
        arms = embodiment.get("arm", []) if isinstance(embodiment, dict) else []
        if not isinstance(arms, list):
            arms = []
        provides = manifest.get("provides", [])
        if not isinstance(provides, list):
            provides = []
        rows.append(
            {
                "path": path,
                "id": manifest["id"],
                "provides": provides,
                "embodiment": arms,
                "safety": manifest.get("safety_class", "—"),
                "origin": manifest.get("origin", "—"),
                "source": manifest.get("source", "—"),
            }
        )
    return rows


def _parser_leaves(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...]
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    subparser_action = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    if subparser_action is None:
        return [(prefix, parser)]
    leaves = []
    for name, child in sorted(subparser_action.choices.items()):
        leaves.extend(_parser_leaves(child, (*prefix, name)))
    return leaves


def _parser_arguments(parser: argparse.ArgumentParser) -> list[str]:
    arguments = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction | argparse._SubParsersAction):
            continue
        if action.option_strings:
            arguments.append(action.option_strings[-1])
        else:
            arguments.append(f"<{action.dest}>")
    return arguments


def _cli_inventory(root: Path) -> list[dict]:
    # The CLI table is built by INTROSPECTING the imported package, which
    # --root cannot redirect. Refuse the mismatch rather than describe one
    # tree's CLI in another tree's inventory: a silent wrong verdict here
    # either blocks a correct commit or ships the drift this gate exists
    # to catch.
    package = Path(aisle.__file__).resolve().parent
    if root not in package.parents:
        raise ValueError(
            f"imported aisle package at {package} is outside --root {root}; "
            "run this generator against the tree whose package is installed "
            "(uv run python tools/docs_inventory.py ...)"
        )
    rows = [
        {
            "command": ("harness",),
            "arguments": [],
        }
    ]
    rows.extend(
        {
            "command": command,
            "arguments": _parser_arguments(parser),
        }
        for command, parser in _parser_leaves(build_harness_parser(), ("harness",))
    )
    rows.extend(
        {
            "command": ("python", "-m", "aisle.harness.registry", *command),
            "arguments": _parser_arguments(parser),
        }
        for command, parser in _parser_leaves(build_registry_parser(), ())
    )
    return rows


def _adr_inventory(root: Path, tracked: set[str] | None) -> list[dict]:
    rows = []
    # ADR-*.md, not *.md: an index, template or README living beside the ADRs
    # is not a malformed ADR, and used to fail the whole gate with no way out.
    for path in _select((root / "docs" / "decisions").glob("ADR-*.md"), root, tracked):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next(
            (line.removeprefix("# ").strip() for line in lines if line.startswith("# ")), None
        )
        if title is None:
            raise ValueError(f"{path}: ADR has no level-one heading")
        match = re.match(r"^(ADR-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)(?::|\s+[—-]\s+)", title)
        if match is None:
            raise ValueError(f"{path}: ADR heading has no parseable identity")
        identity = match.group(1)

        status = None
        for line in lines:
            if line.startswith("Status:"):
                # the LITERAL first Status: line, as the rendered legend
                # promises — continuation lines swept in trailing prose and
                # bullet lists that the table claimed were the status
                status = line.removeprefix("Status:").strip()
                break
        rows.append({"path": path, "identity": identity, "title": title, "status": status})

    identities: dict[str, Path] = {}
    for row in rows:
        path, identity = row["path"], row["identity"]
        if identity in identities:
            raise ValueError(
                f"{path}: duplicate ADR identity {identity}; first declared by "
                f"{identities[identity]}"
            )
        identities[identity] = path

    for row in rows:
        path, identity, status = row["path"], row["identity"], row["status"]
        if not status:
            raise ValueError(f"{path}: ADR has no declared Status")
        if path.stem != identity:
            raise ValueError(
                f"{path}: filename identity {path.stem} does not match heading {identity}"
            )
    return rows


def _test_inventory(root: Path, tracked: set[str] | None) -> tuple[list[str], list[str]]:
    modules = _select((root / "tests").rglob("test_*.py"), root, tracked)
    # Suite DIRECTORIES without per-file counts. An exhaustive module list (or
    # any count of it) makes two independently-green PRs that each add a test
    # merge cleanly into a main whose inventory is stale — CI red on main with
    # no conflict to warn anyone. Directories change rarely and meaningfully.
    suites = sorted({_relative(path.parent, root) for path in modules})

    with open(root / "pyproject.toml", "rb") as file:
        config = tomllib.load(file)
    configured = config["tool"]["pytest"]["ini_options"]["markers"]
    markers = [entry.split(":", 1)[0].strip() for entry in configured]
    return suites, markers


# What each top-level directory is FOR. Roles are editorial and change
# rarely; the tree itself and the frozen marking are both derived, so this
# map cannot drift into claiming a directory that does not exist.
_DIR_ROLES = {
    "analysis": "committed experiment findings and reports",
    "assets": "robot/scene assets (so101 URDF)",
    "docs": "guides, design doc, ADRs, this generated appendix",
    "env": "topic CONTRACT.md and guard limits (config, not code)",
    "graphs": "dataflow YAMLs + committed ADR-30 turn plans",
    "harness": "the research-agent contract and budget ledger",
    "registry": "capability schema and typed node manifests",
    "runs": "gitignored: traces, videos, run manifests",
    "skills": "registered agent-authored skills",
    "skills_pending_review": "recovered campaign skills, not registered",
    "specs": "numbered specs with MUST ids (000 = constitution)",
    "src": "the aisle package: scenes, bridge, sim engines, verifier, reset, harness, nodes",
    "templates": "scaffolding for new artifacts",
    "tests": "unit / sim / graph suites",
    "tools": "CI, env_hash, trace_check, campaign runners",
}


def _frozen_prefixes() -> list[str]:
    """Frozen paths, READ from env_hash's constants (ADR-33: never
    hand-maintain a second copy of the fence — that is exactly how
    src/aisle/mobility stayed outside it)."""
    return sorted({*FROZEN_DIRS, *FROZEN_FILES, *FROZEN_GLOBS})


def _layout_inventory(root: Path) -> list[dict]:
    """Top-level directories, their role, and which frozen paths each holds.

    The design doc's §8.0 tree is superseded: it put the bridge, scenes,
    verifier and reset under `env/` and made "frozen" a directory property.
    Both moved. This map is generated so it cannot go stale the way that one
    did, and it surfaces the fence per directory — the legibility §8.0's
    `env/` gave and the current layout lost."""
    frozen = _frozen_prefixes()
    rows = []
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        name = path.name
        held = [f for f in frozen if f == name or f.startswith(f"{name}/")]
        rows.append(
            {
                "dir": f"{name}/",
                "role": _DIR_ROLES.get(name, ""),
                "frozen": ", ".join(f"`{f}`" for f in held) if held else "—",
            }
        )
    return rows


def render_inventory(root: Path) -> tuple[str, dict[str, int]]:
    tracked = _tracked_paths(root)
    graphs = _graph_inventory(root, tracked)
    capabilities = _capability_inventory(root, tracked)
    cli_commands = _cli_inventory(root)
    adrs = _adr_inventory(root, tracked)
    test_suites, markers = _test_inventory(root, tracked)
    test_modules = _select((root / "tests").rglob("test_*.py"), root, tracked)

    lines = [
        "# Generated project inventory",
        "",
        "> Generated by `tools/docs_inventory.py`. Do not edit manually. Run",
        "> `uv run python tools/docs_inventory.py --write` after changing a source surface.",
        "",
        "This appendix is intentionally factual and source-derived. Qualitative maturity,",
        "architecture, and contribution guidance remain in the",
        "[contributor wiki](../contributor-wiki.md).",
        "",
        "## Repository layout",
        "",
        "The design doc's §8.0 tree is SUPERSEDED — it placed the bridge, scenes,",
        "verifier and reset under `env/` and made *frozen* a property of that",
        "directory. Packaging moved the code to `src/aisle/`, and ADR-33 moved the",
        "fence from a directory to a content set: the unit of the fence is what a",
        "result depends on, not where it lives.",
        "",
        "The frozen column is read from `tools/env_hash.py`'s constants, never",
        "restated here, so a widening appears automatically (ADR-33: a",
        "hand-maintained second copy of the fence is how `src/aisle/mobility`",
        "stayed outside it).",
        "",
        "| Directory | Role | Frozen paths it holds (CON-7) |",
        "|---|---|---|",
        *(
            f"| `{row['dir']}` | {row['role']} | {row['frozen']} |"
            for row in _layout_inventory(root)
        ),
        "",
        "## Snapshot counts",
        "",
        "| Surface | Count |",
        "|---|---:|",
        f"| Graphs | {len(graphs)} |",
        f"| Capability manifests | {len(capabilities)} |",
        f"| CLI command entries | {len(cli_commands)} |",
        f"| ADR files | {len(adrs)} |",
        "",
        "## Graphs",
        "",
        "| Graph | Scene/scenario | Embodiment | Perception | Engine | Nodes |",
        "|---|---|---|---|---|---:|",
    ]
    for row in graphs:
        relative = _relative(row["path"], root)
        lines.append(
            "| "
            + " | ".join(
                [
                    _link(relative, f"../../{relative}"),
                    _cell(f"{row['scene']} / {row['scenario']}"),
                    _cell(row["embodiment"]),
                    _cell(row["perception"]),
                    _cell(row["engine"]),
                    str(len(row["nodes"])),
                ]
            )
            + " |"
        )
    lines.extend(["", "### Graph node membership", ""])
    for row in graphs:
        relative = _relative(row["path"], root)
        lines.extend(
            [
                f"- **{relative}:** " + ", ".join(f"`{node}`" for node in row["nodes"]),
            ]
        )

    lines.extend(
        [
            "",
            "## Capability manifests",
            "",
            "| Capability | Provides | Arm profile(s) | Safety | Origin | Source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in capabilities:
        relative = _relative(row["path"], root)
        lines.append(
            "| "
            + " | ".join(
                [
                    _link(relative, f"../../{relative}"),
                    _cell(", ".join(row["provides"])),
                    _cell(", ".join(row["embodiment"])),
                    _cell(row["safety"]),
                    _cell(row["origin"]),
                    f"`{_cell(row['source'])}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## CLI commands",
            "",
            "| Command | Arguments/options |",
            "|---|---|",
        ]
    )
    for row in cli_commands:
        command = " ".join(row["command"])
        arguments = ", ".join(f"`{argument}`" for argument in row["arguments"]) or "—"
        lines.append(f"| `{_cell(command)}` | {arguments} |")

    lines.extend(
        [
            "",
            "## Architecture decision records",
            "",
            "Every ADR is validated to have a unique identity matching its filename and a",
            "declared `Status:`. The table renders the literal first status line without",
            "inference.",
            "",
            "| ADR | Title | Declared status |",
            "|---|---|---|",
        ]
    )
    for row in adrs:
        relative = _relative(row["path"], root)
        target = f"../decisions/{row['path'].name}"
        lines.append(
            f"| {_link(relative, target)} | {_cell(row['title'])} | {_cell(row['status'])} |"
        )

    lines.extend(
        [
            "",
            "## Tests",
            "",
            f"Configured pytest markers: {', '.join(f'`{marker}`' for marker in markers)}.",
            "",
            "Suite directories, not individual modules: an exhaustive module list goes stale",
            "on main whenever two PRs each add a test, with no merge conflict to catch it.",
            "",
            "| Suite directory |",
            "|---|",
        ]
    )
    for suite in test_suites:
        lines.append(f"| `{_cell(suite)}` |")
    lines.append("")

    counts = {
        "adrs": len(adrs),
        "capabilities": len(capabilities),
        "cli_commands": len(cli_commands),
        "graphs": len(graphs),
        "test_modules": len(test_modules),
    }
    return "\n".join(lines), counts


def _report(ok: bool, output: Path, content: str, counts: dict[str, int], reason: str) -> dict:
    return {
        "ok": ok,
        "output": str(output),
        "reason": reason,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the generated Markdown")
    mode.add_argument("--check", action="store_true", help="fail when generated Markdown drifts")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        content, counts = render_inventory(root)
        # encoding is PINNED on every read and write: the digest above is over
        # UTF-8, so a locale-default codec (LC_ALL=C) either kills the gate
        # with a codec error or writes bytes whose digest the same run
        # misreports.
        if args.write:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            report = _report(True, output, content, counts, "written")
        elif not output.exists():
            report = _report(False, output, content, counts, "missing")
        elif output.read_text(encoding="utf-8") != content:
            report = _report(False, output, content, counts, "stale")
        else:
            report = _report(True, output, content, counts, "current")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1

    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
