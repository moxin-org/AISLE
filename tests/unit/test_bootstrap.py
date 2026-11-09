"""Unit tests for the T01 repository bootstrap (CON-6, CON-12, CON-1, CON-2, CON-9)."""

import tomllib

import pytest
from cli_helpers import REPO_ROOT

pytestmark = pytest.mark.unit


def load_pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_dora_cli_and_api_release_match():
    """CON-3 / CON-5: nightly and both sim extras use one pinned Dora release."""
    import yaml

    project = load_pyproject()
    extras = project["project"]["optional-dependencies"]
    for extra in ("sim", "cuda"):
        assert "dora-rs==1.0.1" in extras[extra]
    assert "dora-rs" not in project["tool"]["uv"]["sources"]
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/nightly.yml").read_text())
    commands = [step.get("run", "") for step in workflow["jobs"]["sim-graph-accept"]["steps"]]
    assert "uv tool install dora-rs-cli==1.0.1" in commands
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    api = [package for package in lock["package"] if package["name"] == "dora-rs"]
    assert len(api) == 1
    assert api[0]["version"] == "1.0.1"
    assert "registry" in api[0]["source"]


def test_layout():
    """CON-6: the fixed repository layout exists."""
    required = [
        "specs",
        "src/aisle/scenes",
        "src/aisle/nodes",
        "src/aisle/verifier",
        "src/aisle/reset",
        "src/aisle/harness",
        "graphs",
        "registry/schema",
        "registry/manifests",
        "skills",
        "tests/unit",
        "tests/sim",
        "tests/graph",
        "tests/accept",
        "tools",
        "runs",
        "docs",
    ]
    missing = [d for d in required if not (REPO_ROOT / d).is_dir()]
    assert missing == []


def test_runs_gitignored():
    """CON-6: runs/ is gitignored."""
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "runs/" in gitignore


def test_pytest_markers():
    """CON-12: pytest markers unit, sim, graph, accept are registered and
    strict (unknown markers are errors)."""
    cfg = load_pyproject()["tool"]["pytest"]["ini_options"]
    registered = {m.split(":")[0].strip() for m in cfg["markers"]}
    assert {"unit", "sim", "graph", "accept"} <= registered
    assert "--strict-markers" in cfg["addopts"]


def test_python_version_and_layout_cfg():
    """CON-2: Python 3.11+, one workspace pyproject, packages under src/aisle."""
    project = load_pyproject()["project"]
    assert project["requires-python"].startswith(">=3.11")
    assert (REPO_ROOT / "src" / "aisle" / "__init__.py").exists()


def test_no_cuda_in_default_dependencies():
    """CON-1: CUDA-only dependencies MUST NOT enter the DEFAULT dependency
    set — checked over the resolved lock closure reachable from the
    declared default dependencies, so a transitive CUDA pull-in fails.
    CON-1 explicitly permits CUDA behind an optional extra: `cuda` is that
    extra and the only path by which a CUDA wheel may enter the lock. The
    `sim` extra resolves the CPU torch on Linux and stays clean."""
    forbidden = ("cuda", "nvidia", "cu11", "cu12")
    project = load_pyproject()["project"]
    for dep in project.get("dependencies", []):
        assert not any(k in dep.lower() for k in forbidden), dep

    with open(REPO_ROOT / "uv.lock", "rb") as f:
        lock = tomllib.load(f)
    # merge duplicate [[package]] entries (multi-platform locks repeat names)
    graph: dict[str, list[str]] = {}
    for p in lock.get("package", []):
        graph.setdefault(p["name"], []).extend(d["name"] for d in p.get("dependencies", []))
    roots = [
        dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        for dep in project.get("dependencies", [])
    ]
    missing_roots = [r for r in roots if r not in graph]
    assert not missing_roots, f"declared deps absent from lock graph: {missing_roots}"
    closure: set[str] = set()
    frontier = list(roots)
    while frontier:
        name = frontier.pop()
        if name in closure:
            continue
        closure.add(name)
        frontier.extend(graph.get(name, []))
    assert closure, "default dependency closure resolved to nothing — check lock parsing"
    for name in sorted(closure):
        assert not any(k in name.lower() for k in forbidden), name
    # CUDA/NVIDIA wheels are confined to the `cuda` extra, which CON-1
    # explicitly sanctions. The lock keys torch variants by version
    # (2.13.0+cpu vs 2.13.0+cu130), so the invariant is: the ONLY package
    # that may pull a forbidden wheel is a +cu torch. `sim` resolves the
    # +cpu variant on linux and therefore stays clean.
    cuda_parents = {
        (p["name"], p["version"])
        for p in lock.get("package", [])
        for d in p.get("dependencies", [])
        if any(k in d["name"].lower() for k in forbidden)
    }
    offenders = sorted(
        (name, version)
        for name, version in cuda_parents
        if not any(k in name.lower() for k in forbidden)
        and not (name == "torch" and "+cu" in version)
    )
    assert not offenders, f"CUDA wheels reachable outside the cuda extra: {offenders}"
    assert any(name == "torch" and "+cu" in version for name, version in cuda_parents), (
        "no CUDA torch in the lock — the cuda extra is inert"
    )


# ADR-70: engine wheels in the sim extra and deliberately absent from cuda.
SIM_ONLY_ENGINES = {"dimforge-nexus3d", "rapier3d"}


def test_sim_and_cuda_extras_route_linux_torch_to_disjoint_indexes():
    """CON-1: the portable and GPU simulation closures MUST bind Linux
    torch to their respective CPU/CUDA indexes; merely having both indexes
    somewhere in the lock does not prove either extra selects the right one.
    ADR-70: the published Nexus/rapier wheels ride the sim extra only, since
    no published Nexus wheel has CUDA; that is the extras' only difference."""
    pyproject = load_pyproject()
    extras = pyproject["project"].get("optional-dependencies", {})
    engines = {
        dep for dep in extras["sim"] if dep.split(";")[0].split("==")[0].strip() in SIM_ONLY_ENGINES
    }
    assert {dep.split("==")[0] for dep in engines} == SIM_ONLY_ENGINES
    assert set(extras["cuda"]) == set(extras["sim"]) - engines, (
        "cuda must carry the complete common sim stack; only the torch distribution "
        "and the sim-only engines differ"
    )
    sources = pyproject["tool"]["uv"]["sources"]["torch"]
    linux_bindings = {
        source["extra"]: source["index"]
        for source in sources
        if source.get("marker") == "sys_platform == 'linux'"
    }
    assert linux_bindings == {
        "sim": "pytorch-cpu",
        "cuda": "pytorch-cu130",
        # ADR-38: vla inference is CPU/MPS at bring-up; cuda conflicts declared
        "vla": "pytorch-cpu",
    }

    indexes = {i["name"]: i["url"] for i in pyproject["tool"]["uv"].get("index", [])}
    assert indexes[linux_bindings["sim"]].rstrip("/").endswith("/cpu")
    assert indexes[linux_bindings["cuda"]].rstrip("/").endswith("/cu130")

    with open(REPO_ROOT / "uv.lock", "rb") as f:
        lock = tomllib.load(f)
    root = next(package for package in lock["package"] if package["name"] == "aisle")
    locked = root["optional-dependencies"]
    sim_linux = [
        dep
        for dep in locked["sim"]
        if dep["name"] == "torch" and dep.get("source", {}).get("registry", "").endswith("/cpu")
    ]
    cuda_linux = [
        dep
        for dep in locked["cuda"]
        if dep["name"] == "torch" and dep.get("source", {}).get("registry", "").endswith("/cu130")
    ]
    assert len(sim_linux) == len(cuda_linux) == 1
    assert "+cpu" in sim_linux[0]["version"]
    assert "+cu130" in cuda_linux[0]["version"]

    torch_variants = {
        package["version"]: {dep["name"] for dep in package.get("dependencies", [])}
        for package in lock["package"]
        if package["name"] == "torch"
    }
    forbidden = ("cuda", "nvidia", "cu11", "cu12", "cu13")
    assert not any(
        token in dependency.lower()
        for dependency in torch_variants[sim_linux[0]["version"]]
        for token in forbidden
    )
    assert any(
        token in dependency.lower()
        for dependency in torch_variants[cuda_linux[0]["version"]]
        for token in forbidden
    )


def test_ci_script_gate_order():
    """CON-9: tools/ci.sh runs the local CI gates in the constitution's order:
    ruff format --check, ruff check, pytest -m unit, then trace_check."""
    lines = (REPO_ROOT / "tools" / "ci.sh").read_text().splitlines()
    script = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    order = [
        script.index("ruff format --check"),
        script.index("ruff check"),
        script.index("pytest -m unit"),
        script.index("trace_check.py"),
    ]
    assert order == sorted(order)
