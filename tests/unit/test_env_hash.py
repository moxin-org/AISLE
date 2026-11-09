"""Unit tests for tools/env_hash.py (CON-7, CON-5, CON-8).

CON-7: tools/env_hash.py fingerprints the frozen set so rollout can refuse
on mismatch. The set itself is FROZEN_DIRS + FROZEN_FILES + the
graphs/expert_*.yaml glob in that module — these tests read it from there
rather than restating it, so widening the fence cannot leave them stale.
"""

import json
import subprocess
from pathlib import Path

import pytest
from cli_helpers import REPO_ROOT, run_tool

pytestmark = pytest.mark.unit


def run_env_hash(*args: str) -> subprocess.CompletedProcess:
    return run_tool("env_hash.py", *args)


def make_root(tmp_path: Path) -> Path:
    """Minimal repo root containing the CON-7 frozen set plus non-frozen files."""
    for pkg in ("scenes", "verifier", "reset"):
        d = tmp_path / "src" / "aisle" / pkg
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")
    (tmp_path / "src" / "aisle" / "scenes" / "pharmacy.py").write_text("SHELF_LEVELS = 3\n")
    (tmp_path / "src" / "aisle" / "verifier" / "thresholds.toml").write_text("upright_deg = 30\n")
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "expert_t0.yaml").write_text("nodes: []\n")
    (tmp_path / "graphs" / "scratch.yaml").write_text("nodes: []\n")
    asset = tmp_path / "assets" / "so101"
    asset.mkdir(parents=True)
    (asset / "so101.urdf").write_text("<robot name='so101'/>\n")
    (tmp_path / "tools").mkdir()
    return tmp_path


def get_hash(root: Path) -> str:
    proc = run_env_hash("--root", str(root))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    return report["env_hash"]


def test_cli_json_stdout(tmp_path):
    """CON-8: env_hash emits a single JSON object on stdout, logs to stderr
    only, exit 0 iff ok."""
    root = make_root(tmp_path)
    proc = run_env_hash("--root", str(root))
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert len(report["env_hash"]) == 64
    assert int(report["env_hash"], 16) >= 0  # hex sha256


def test_deterministic(tmp_path):
    """CON-5: same input tree ⇒ same hash, across invocations and across
    freshly created identical trees (no time/inode dependence)."""
    root_a = make_root(tmp_path / "a")
    root_b = make_root(tmp_path / "b")
    assert get_hash(root_a) == get_hash(root_a)
    assert get_hash(root_a) == get_hash(root_b)


def test_content_change_changes_hash(tmp_path):
    """CON-7: a single mutated byte inside the frozen set changes env_hash."""
    root = make_root(tmp_path)
    before = get_hash(root)
    (root / "src" / "aisle" / "verifier" / "thresholds.toml").write_text("upright_deg = 31\n")
    assert get_hash(root) != before


def test_so101_asset_change_changes_hash(tmp_path):
    """CON-5, CON-7, SCN-4: the pinned SO-101 URDF and mesh closure are
    frozen inputs; changing an asset invalidates the environment hash."""
    root = make_root(tmp_path)
    before = get_hash(root)
    (root / "assets" / "so101" / "so101.urdf").write_text("<robot name='changed'/>\n")
    assert get_hash(root) != before


def test_rename_changes_hash(tmp_path):
    """CON-7: file paths are part of the fingerprint, not just contents."""
    root = make_root(tmp_path)
    before = get_hash(root)
    src = root / "src" / "aisle" / "scenes" / "pharmacy.py"
    src.rename(root / "src" / "aisle" / "scenes" / "pharmacy2.py")
    assert get_hash(root) != before


def test_non_frozen_files_ignored(tmp_path):
    """CON-7: files outside the declared frozen paths do not affect the hash."""
    root = make_root(tmp_path)
    before = get_hash(root)
    (root / "graphs" / "scratch.yaml").write_text("nodes: [changed]\n")
    (root / "src" / "aisle" / "scenes" / "__pycache__").mkdir()
    (root / "src" / "aisle" / "scenes" / "__pycache__" / "x.pyc").write_text("junk")
    assert get_hash(root) == before
    (root / "graphs" / "expert_t0.yaml").write_text("nodes: [changed]\n")
    assert get_hash(root) != before


def test_write_then_check_ok(tmp_path):
    """CON-7: --write commits tools/env_hash.json; --check passes while the
    frozen set is unchanged."""
    root = make_root(tmp_path)
    proc = run_env_hash("--root", str(root), "--write")
    assert proc.returncode == 0
    assert (root / "tools" / "env_hash.json").exists()
    check = run_env_hash("--root", str(root), "--check")
    report = json.loads(check.stdout)
    assert check.returncode == 0
    assert report["ok"] is True


def test_check_mismatch_fails(tmp_path):
    """CON-7: after a frozen-set edit, --check exits nonzero with ok=false
    so the rollout runner can refuse to launch."""
    root = make_root(tmp_path)
    run_env_hash("--root", str(root), "--write")
    (root / "src" / "aisle" / "verifier" / "thresholds.toml").write_text("upright_deg = 45\n")
    check = run_env_hash("--root", str(root), "--check")
    report = json.loads(check.stdout)
    assert check.returncode != 0
    assert report["ok"] is False


def test_check_corrupted_hash_file_reported_as_json(tmp_path):
    """CON-8: a corrupted tools/env_hash.json (invalid JSON or missing key)
    yields a JSON error report on stdout, not a Python traceback."""
    root = make_root(tmp_path)
    for bad in ("not json{", '{"wrong_key": 1}'):
        (root / "tools" / "env_hash.json").write_text(bad)
        check = run_env_hash("--root", str(root), "--check")
        report = json.loads(check.stdout)
        assert check.returncode != 0
        assert report["ok"] is False


def test_file_boundaries_are_unambiguous(tmp_path):
    """CON-7: content containing NUL bytes cannot make two different frozen
    trees hash equal (per-file digests frame each file's content)."""
    root_a = make_root(tmp_path / "a")
    root_b = make_root(tmp_path / "b")
    scenes_a = root_a / "src" / "aisle" / "scenes"
    scenes_b = root_b / "src" / "aisle" / "scenes"
    # Under naive path\0content\0 concatenation these two trees would feed
    # the hasher identical byte streams.
    (scenes_a / "a").write_bytes(b"b\0src/aisle/scenes/c\0d")
    (scenes_b / "a").write_bytes(b"b")
    (scenes_b / "c").write_bytes(b"d")
    assert get_hash(root_a) != get_hash(root_b)


def test_check_without_committed_hash_fails(tmp_path):
    """CON-8: --check with no committed tools/env_hash.json is an explicit
    error, not a silent pass."""
    root = make_root(tmp_path)
    check = run_env_hash("--root", str(root), "--check")
    report = json.loads(check.stdout)
    assert check.returncode != 0
    assert report["ok"] is False


def test_guard_and_limits_are_hashed(tmp_path):
    """SPEC 080 / CON-7 (PR review): the frozen safety artifacts —
    env/limits.toml and the budget-guard module — are part of the env
    hash; adding or changing either changes it."""
    root = make_root(tmp_path)
    base = get_hash(root)
    guard = root / "src" / "aisle" / "nodes" / "budget_guard.py"
    guard.parent.mkdir(parents=True)
    guard.write_text("GUARD = 1\n")
    with_guard = get_hash(root)
    assert with_guard != base
    limits = root / "env" / "limits.toml"
    limits.parent.mkdir()
    limits.write_text("[embodiment.franka]\n")
    with_limits = get_hash(root)
    assert with_limits != with_guard
    limits.write_text("[embodiment.franka]\nq = 1\n")
    assert get_hash(root) != with_limits


class TestTrustedBaseline:
    """ADR-21 (PR #24): --check --baseline <ref> defeats the regenerate-
    the-json attack — the baseline hash comes from the git object store at
    a protected ref the research agent cannot move, and the checker itself
    must match its blob there."""

    def _trusted_root(self, tmp_path: Path) -> Path:
        import shutil

        repo = Path(__file__).resolve().parents[2]
        root = make_root(tmp_path)
        shutil.copy(repo / "tools" / "env_hash.py", root / "tools" / "env_hash.py")
        (root / "harness").mkdir()
        (root / "harness" / "budget.toml").write_text(
            "[campaign]\ntokens = 5000000\nepisodes = 500\nwall_h = 40.0\n"
        )
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(tmp_path),
        }

        def git(*args):
            proc = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                env={**env, "PATH": __import__("os").environ["PATH"]},
            )
            assert proc.returncode == 0, (args, proc.stderr)

        git("init", "-q")
        assert run_env_hash("--write", "--root", str(root)).returncode == 0
        git("add", "-A")
        git("commit", "-qm", "baseline")
        git("update-ref", "refs/remotes/origin/main", "HEAD")
        return root

    def test_clean_tree_passes_trusted_check(self, tmp_path):
        root = self._trusted_root(tmp_path)
        proc = run_env_hash("--check", "--baseline", "origin/main", "--root", str(root))
        assert proc.returncode == 0, proc.stdout
        assert json.loads(proc.stdout)["baseline"] == "origin/main"

    def test_regenerated_local_json_does_not_bless_frozen_edits(self, tmp_path):
        """The PR #24 attack verbatim: edit frozen code, rerun --write —
        the LOCAL check passes, the TRUSTED check refuses."""
        root = self._trusted_root(tmp_path)
        (root / "src" / "aisle" / "verifier" / "thresholds.toml").write_text("upright_deg = 90\n")
        assert run_env_hash("--write", "--root", str(root)).returncode == 0
        local = run_env_hash("--check", "--root", str(root))
        assert local.returncode == 0  # the attack defeats the local check...
        trusted = run_env_hash("--check", "--baseline", "origin/main", "--root", str(root))
        assert trusted.returncode == 1  # ...but not the trusted one
        assert "diverges from origin/main" in json.loads(trusted.stdout)["error"]

    def test_tampered_checker_is_refused(self, tmp_path):
        """Rewriting the checker itself cannot bless anything: the trusted
        mode verifies tools/env_hash.py against its blob at the ref."""
        root = self._trusted_root(tmp_path)
        with open(root / "tools" / "env_hash.py", "a") as f:
            f.write("\n# patched\n")
        trusted = run_env_hash("--check", "--baseline", "origin/main", "--root", str(root))
        assert trusted.returncode == 1
        assert "not trusted" in json.loads(trusted.stdout)["error"]

    def test_missing_ref_is_an_explicit_error(self, tmp_path):
        root = self._trusted_root(tmp_path)
        proc = run_env_hash("--check", "--baseline", "origin/nope", "--root", str(root))
        assert proc.returncode == 1
        assert "origin/nope" in json.loads(proc.stdout)["error"]


def test_adr24_fingerprint_names_the_selection():
    """CON-5 as amended (ADR-24): the fingerprint hashes the lock AND the
    resolved selection — default vs --extra sim produce DIFFERENT
    identities from the same lock (the footgun the version map missed),
    and the same inputs reproduce the same fingerprint."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import current_selection, env_fingerprint

    lock = b"fake lock bytes"
    default = env_fingerprint(lock, current_selection([]))
    sim = env_fingerprint(lock, current_selection(["sim"]))
    assert default != sim
    assert sim == env_fingerprint(lock, current_selection(["sim"]))  # deterministic
    assert env_fingerprint(b"other lock", current_selection(["sim"])) != sim


def test_adr24_canonical_name_pep503():
    """PR #68 review (minor): every name join folds case and [-_.] runs."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import canonical_name

    assert canonical_name("Dora_YOLO") == "dora-yolo"
    assert canonical_name("genesis.world") == "genesis-world"
    assert canonical_name("torch") == "torch"


def test_adr24_direct_url_classification():
    """ADR-24 D2: editable/local-dir installs are unattestable; VCS needs
    a commit id (dora-rs's pinned-rev git install is legitimate); archive
    installs need a hash; index installs (no direct_url) are fine."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import classify_direct_url

    assert classify_direct_url({"vcs_info": {"vcs": "git", "commit_id": "abc"}}) is None
    assert "commit id" in classify_direct_url({"vcs_info": {"vcs": "git"}})
    assert "editable" in classify_direct_url({"dir_info": {"editable": True}})
    assert "local-directory" in classify_direct_url({"dir_info": {}})
    assert "hash" in classify_direct_url({"archive_info": {}})
    assert classify_direct_url({"archive_info": {"hashes": {"sha256": "x"}}}) is None


def test_adr24_registry_pip_dists_regex(tmp_path):
    """The trusted blob parses pip: sources without a yaml dependency —
    canonical names, quoted or bare, non-pip ignored."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import registry_pip_dists

    mdir = tmp_path / "registry" / "manifests"
    mdir.mkdir(parents=True)
    (mdir / "a.yaml").write_text("id: a\nsource: pip:Dora_YOLO\n")
    (mdir / "b.yaml").write_text("id: b\nsource: 'pip:dora-pose'\n")
    (mdir / "c.yaml").write_text("id: c\nsource: src/aisle/nodes/c.py\n")
    assert registry_pip_dists(tmp_path) == ["dora-pose", "dora-yolo"]


def _fake_md(monkeypatch, dists):
    """Install a fake importlib.metadata.distributions/distribution pair."""
    import importlib.metadata as md

    monkeypatch.setattr(md, "distributions", lambda: list(dists.values()))

    def one(name):
        for d in dists.values():
            if d.metadata.get("Name").lower().replace("_", "-") == name:
                return d
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "distribution", one)


class _FakeDist:
    def __init__(self, name, version, files, record_text="rec"):
        self.metadata = {"Name": name}
        self.version = version
        self.files = files
        self._record = record_text

    def read_text(self, fname):
        return self._record if fname == "RECORD" else None


def _hashed_file(tmp_path, name, content):
    import base64
    import hashlib as _hashlib

    path = tmp_path / name
    path.write_text(content)

    class FakeHash:
        mode = "sha256"
        value = (
            base64.urlsafe_b64encode(_hashlib.sha256(path.read_bytes()).digest())
            .rstrip(b"=")
            .decode()
        )

    class FakeFile:
        hash = FakeHash()

        def locate(self):
            return path

        def __str__(self):
            return name

    return FakeFile()


def test_adr24_verify_records_fails_closed(tmp_path, monkeypatch):
    """PR #69 review F1: the post-session audit verifies against the
    GATE-TIME inventory — removal, addition, version change, a changed
    RECORD (self-blessing), zero verifiable entries, and file mutation
    are each problems; the intact environment verifies."""
    import hashlib as _hashlib
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    import env_hash as eh

    good = _FakeDist("alpha", "1.0", [_hashed_file(tmp_path, "a.py", "ok")])
    expected = {
        "alpha": {
            "version": "1.0",
            "record_sha256": _hashlib.sha256(b"rec").hexdigest(),
        }
    }
    _fake_md(monkeypatch, {"alpha": good})
    assert eh.verify_records(expected)["ok"] is True

    # removed after the gate
    _fake_md(monkeypatch, {})
    report = eh.verify_records(expected)
    assert report["ok"] is False and any("removed" in p for p in report["problems"])

    # installed after the gate
    extra = _FakeDist("beta", "2.0", [_hashed_file(tmp_path, "b.py", "new")])
    _fake_md(monkeypatch, {"alpha": good, "beta": extra})
    report = eh.verify_records(expected)
    assert any("installed after the gate" in p for p in report["problems"])

    # RECORD changed (self-blessing attempt)
    blessed = _FakeDist("alpha", "1.0", [_hashed_file(tmp_path, "c.py", "evil")], "rec2")
    _fake_md(monkeypatch, {"alpha": blessed})
    report = eh.verify_records(expected)
    assert any("RECORD changed" in p for p in report["problems"])

    # zero verifiable entries
    empty = _FakeDist("alpha", "1.0", [])
    _fake_md(monkeypatch, {"alpha": empty})
    report = eh.verify_records(expected)
    assert any("zero hash-verifiable" in p for p in report["problems"])

    # mutated file under an intact RECORD
    mut_file = _hashed_file(tmp_path, "d.py", "before")
    (tmp_path / "d.py").write_text("after")
    mutated = _FakeDist("alpha", "1.0", [mut_file])
    _fake_md(monkeypatch, {"alpha": mutated})
    report = eh.verify_records(expected)
    assert any("does not match its RECORD hash" in p for p in report["problems"])


def test_adr24_pip_source_parser_parity():
    """PR #69 review F2: the trusted scanner and the validator's
    _pip_dist must be the SAME parser — decorated, pinned, case-varied,
    quoted, and indented forms all land on the canonical name."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import parse_pip_source

    from aisle.harness.registry import _pip_dist

    cases = [
        "pip:dora-yolo",
        "pip:Dora_YOLO[gpu]==1.2",
        "PIP:dora.pose",
        "pip: dora-ocr ",
        "pip:torch>=2.0; python_version>'3.10'",
    ]
    for source in cases:
        assert parse_pip_source(source) == _pip_dist({"source": source}), source
    assert parse_pip_source("src/aisle/nodes/x.py") is None
    assert _pip_dist({"source": "src/aisle/nodes/x.py"}) is None


def test_adr24_scanner_handles_indented_and_decorated_manifests(tmp_path):
    """PR #69 review F2 (the exact bypasses): indented `source:` lines and
    decorated pip values must land as canonical base names."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import registry_pip_dists

    mdir = tmp_path / "registry" / "manifests"
    mdir.mkdir(parents=True)
    (mdir / "a.yaml").write_text("id: a\nsource: pip:Dora_YOLO[gpu]==1.2\n")
    (mdir / "b.yaml").write_text("id: b\n  source: pip:dora_pose\n")
    (mdir / "c.yaml").write_text("id: c\nsource: 'pip:dora-ocr'\n")
    (mdir / "d.yaml").write_text("id: d\nsource: src/aisle/nodes/d.py\n")
    assert registry_pip_dists(tmp_path) == ["dora-ocr", "dora-pose", "dora-yolo"]


def test_adr24_selection_covers_abi_groups_and_tags():
    """PR #69 review F4: the selection names the full interpreter identity
    (version + ABI cache tag), the platform tag set, extras AND groups —
    and each axis moves the fingerprint."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import current_selection, env_fingerprint

    sel = current_selection(["sim"])
    assert set(sel) == {"python", "abi", "platform_tags", "extras", "groups"}
    assert sel["groups"] == ["dev"]  # uv's default sync includes dev
    assert sel["abi"].startswith("cpython-")
    lock = b"lock"
    base = env_fingerprint(lock, sel)
    assert env_fingerprint(lock, current_selection(["sim"], groups=[])) != base
    assert env_fingerprint(lock, current_selection([])) != base


def test_committed_hash_matches_this_tree():
    """CON-7 self-consistency: tools/env_hash.json must describe THIS tree.

    PR #122 landed a --write computed on a stale base (42 files, 0d68a166)
    while its own merged tree held 43 — from that commit until the fix,
    mainline could not pass its own trusted gate, and nothing in CI noticed
    because no test pinned committed-to-computed. This one does: any PR
    that changes the frozen set must run tools/env_hash.py --write in the
    SAME PR, on the FINAL tree — which is exactly the discipline CON-7
    demands, now enforced instead of requested."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(root / "tools" / "env_hash.py"), "--root", str(root)],
        capture_output=True,
        text=True,
    )
    computed = json.loads(proc.stdout)
    committed = json.loads((root / "tools" / "env_hash.json").read_text())
    assert computed["env_hash"] == committed["env_hash"], (
        f"frozen set changed without tools/env_hash.py --write: computed "
        f"{computed['env_hash'][:12]} ({computed['n_files']} files) vs committed "
        f"{committed['env_hash'][:12]} ({committed['n_files']}) — run the --write "
        "on the final tree and include it in this change (CON-7)"
    )
    assert computed["n_files"] == committed["n_files"]


def _sim_root(tmp_path: Path) -> Path:
    """make_root plus the ADR-67 engine realization package."""
    root = make_root(tmp_path)
    sim = root / "src" / "aisle" / "sim"
    sim.mkdir(parents=True)
    (sim / "__init__.py").write_text("ENGINES = ('genesis', 'nexus')\n")
    (sim / "nexus_backend.py").write_text("def build_scene():\n    return None\n")
    (sim / "nexus_physics.toml").write_text("[sim]\nsubsteps = 2\n")
    return root


def test_adr55_sim_engine_hash_covers_the_realization_the_fence_does_not(tmp_path):
    """CON-5, CON-7, ADR-67: the frozen set is engine neutral by decision
    (widening it would move every Genesis env_hash), so the engine
    realization gets its OWN digest. Two trees differing only in
    nexus_physics.toml's `substeps` share an env_hash — the exact hole the
    audit found — and must differ in sim_engine_hash."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import compute_env_hash, sim_engine_hash

    root = _sim_root(tmp_path)
    before_env, _ = compute_env_hash(root)
    before = sim_engine_hash(root, "nexus")
    assert before["engine"] == "nexus" and before["n_files"] == 3
    assert sim_engine_hash(root, "nexus") == before  # deterministic

    (root / "src" / "aisle" / "sim" / "nexus_physics.toml").write_text("[sim]\nsubsteps = 8\n")
    after_env, _ = compute_env_hash(root)
    assert after_env == before_env, "the solver settings are outside the fence by decision"
    assert sim_engine_hash(root, "nexus")["sim_engine_hash"] != before["sim_engine_hash"]


def test_adr55_sim_engine_hash_names_the_engine(tmp_path):
    """CON-5, ADR-67, ADR-70: the digest binds the engine name, so the same
    realization stepped by another engine is a different identity. The engine
    wheels come from the lock (ADR-70), so no build receipt is folded in."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import sim_engine_hash

    root = _sim_root(tmp_path)
    genesis = sim_engine_hash(root, "genesis")
    nexus = sim_engine_hash(root, "nexus")
    assert genesis["sim_engine_hash"] != nexus["sim_engine_hash"]
    assert set(nexus) == {"engine", "sim_engine_hash", "n_files"}


def test_adr55_attested_set_names_the_running_engines_distribution(tmp_path):
    """ADR-24 as amended by ADR-67: the attested sim core listed
    genesis-world only, so a Nexus run's provenance check never looked at
    the wheel whose solver produced the trajectories. The engine-neutral
    core stays; the running engine's dist is added to it."""
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import attested_set

    assert "dimforge-nexus3d" not in attested_set(tmp_path)
    assert "dimforge-nexus3d" not in attested_set(tmp_path, "genesis")
    nexus = attested_set(tmp_path, "nexus")
    assert "dimforge-nexus3d" in nexus
    assert {"genesis-world", "torch"} <= set(nexus)  # the neutral core is kept


def test_adr70_cli_attests_a_locked_engine_without_a_receipt(tmp_path):
    """CON-5, CON-8, ADR-70: Nexus and rapier come from the lock like Genesis,
    so the checker reports their digest without any build receipt, and a stale
    receipt left by the retired source builds changes nothing."""
    root = _sim_root(tmp_path)
    plain = json.loads(run_env_hash("--root", str(root)).stdout)
    proc = run_env_hash("--root", str(root), "--sim-engine", "nexus")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["env_hash"] == plain["env_hash"] and "sim" not in plain
    assert report["sim"]["engine"] == "nexus"
    assert len(report["sim"]["sim_engine_hash"]) == 64
    assert "build" not in report["sim"]

    (root / ".nexus-runtime-receipt.json").write_text(json.dumps({"version": "0.1.0"}))
    again = json.loads(run_env_hash("--root", str(root), "--sim-engine", "nexus").stdout)
    assert again["sim"] == report["sim"]


def _frozen_set():
    """The fence, read from the checker itself rather than restated."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import env_hash

    return env_hash.FROZEN_DIRS, env_hash.FROZEN_FILES


def _is_fenced(rel: str) -> bool:
    """Mirrors frozen_files() for PYTHON MODULE paths only.

    Deliberately omits that function's `graphs/expert_*.yaml` glob: every
    path fed here is derived from an `aisle.*` module name, so the glob can
    never match. Widen this if a future frozen glob ever covers .py files."""
    dirs, files = _frozen_set()
    return rel in files or any(rel.startswith(d + "/") for d in dirs)


#: Edges from fenced code to unfenced code that are NOT verdict paths, each
#: justified. An edge NOT listed here fails the closure test below.
_NOT_A_VERDICT_PATH = {
    # scenes/pharmacy._assert_reachable is a BUILD-TIME assertion (SCN-3:
    # "placements MUST be reachable -- assert at build time"). It imports the
    # planners inside that function; the guard never reaches it, calling only
    # load_physics and desk_scan_obstacles. Freezing the planners would drag
    # ik_trajectory, grasp_topdown and most of the tree in behind them.
    ("src/aisle/scenes/pharmacy.py", "src/aisle/nodes/ik_trajectory.py"),
    ("src/aisle/scenes/pharmacy.py", "src/aisle/nodes/grasp_topdown.py"),
}


def _first_party_imports(rel: str) -> set[str]:
    """Module names this file imports from `aisle`, at any nesting depth.

    Covers the four static forms, because three of them were missed by the
    first version and each is a way to pull in an unfenced verdict input:
    `from a.b import c`, `import a.b`, `from a import b` where `b` is a
    MODULE not a name (indistinguishable at the AST level, so both readings
    are returned and the caller drops the one that is not a file), and
    relative `from .b import c`.

    NOT covered, and it cannot be: `importlib.import_module` and
    `__import__` take runtime strings. A verdict input reached that way is
    invisible here — which is a reason not to reach one that way."""
    import ast

    tree = ast.parse((REPO_ROOT / rel).read_text())
    pkg = rel[len("src/") :].rsplit("/", 1)[0].replace("/", ".")
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("aisle"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against this file's package
                base = ".".join(pkg.split(".")[: len(pkg.split(".")) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if not module.startswith("aisle"):
                continue
            found.add(module)
            # `from aisle.nodes import dora_genesis` — the imported name may
            # itself be a module; the caller keeps only those that are files
            found.update(f"{module}.{a.name}" for a in node.names)
    return found


def _module_rel(mod: str) -> str | None:
    rel = "src/" + mod.replace(".", "/") + ".py"
    return rel if (REPO_ROOT / rel).is_file() else None


def test_the_guards_safety_inputs_are_all_fenced():
    """CON-7 (issue #189): the fence's unit is the SAFETY VERDICT, not the
    guard's module. budget_guard.py is plumbing around verdicts that live in
    other modules — so freezing the file while leaving those outside means a
    run's env_hash can stay identical while the velocity clamp, the keep-out,
    or a watchdog verdict changes underneath it.

    That was not hypothetical: PR #177 changed nav's stall/timeout budgets in
    `src/aisle/mobility/nav.py`, touched no other frozen file, and moved no
    hash.

    Walks the CLOSURE, not just the guard's direct imports. Depth 1 was the
    first version of this test and it missed `aisle.embodiment`, which the
    fenced `kinematics.py` reads for the TC-5 joint order that `so101_chain()`
    validates the guard's workspace FK against (issue #189 review). Any edge
    from fenced code to unfenced code must be listed in _NOT_A_VERDICT_PATH
    with a reason, or this fails."""
    frontier, seen, escapes = ["src/aisle/nodes/budget_guard.py"], set(), []
    while frontier:
        rel = frontier.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for mod in sorted(_first_party_imports(rel)):
            dep = _module_rel(mod)
            if dep is None:
                continue  # a package __init__, not a module file
            if _is_fenced(dep):
                frontier.append(dep)
            elif (rel, dep) not in _NOT_A_VERDICT_PATH:
                escapes.append(f"{rel} -> {dep}")
    assert len(seen) > 5, f"the walk collapsed to {seen}; this test has lost its subject"
    assert not escapes, (
        f"fenced code reaches UNFENCED code on a verdict path: {sorted(escapes)}. Those modules "
        "can change without moving env_hash. Fence them in tools/env_hash.py, or add the edge to "
        "_NOT_A_VERDICT_PATH with a reason it cannot affect a verdict."
    )


def test_the_frozen_set_covers_the_mobility_verdicts():
    """Issue #189, stated as the concrete claim rather than only derived:
    the MOB-3 verdict surface and the stamp trust boundary are inside."""
    for rel in (
        "src/aisle/mobility/guard.py",  # base_watchdog_reason, clamp_base_cmd
        "src/aisle/mobility/nav.py",  # the stall/timeout budgets PR #177 changed
        "src/aisle/topics.py",  # parse_sim_stamp, the BG-3 trust boundary
        "src/aisle/kinematics.py",  # SO-101 FK behind the workspace check
    ):
        assert (REPO_ROOT / rel).is_file(), rel
        assert _is_fenced(rel), f"{rel} decides a safety verdict but is outside the CON-7 fence"


def test_the_skill_gate_graphs_are_frozen():
    """CON-7 / ADR-36 (issue #228): `graphs/eval_*.yaml` are the exam an
    agent-authored skill sits to enter the registry — `skills/*/eval.yaml`
    names them, and `harness skill register` rolls out through them before
    writing a manifest. A skill gate the candidate can edit is not a gate,
    which is the same argument that put the verifier and reset in the
    frozen set at M0.

    Their committed ADR-30 turn plans go with them: since #219 those graphs
    are lockstep participants, and the barrier loads the plan at runtime, so
    the plan is executable scheduler topology for a measured run.

    `agent_campaign` is deliberately NOT here — it is the agent's own
    deliverable and freezing it would put CON-7 in conflict with the
    experiment."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from env_hash import frozen_files

    frozen = {p.relative_to(REPO_ROOT).as_posix() for p in frozen_files(REPO_ROOT)}
    gates = sorted(p.name for p in (REPO_ROOT / "graphs").glob("eval_*.yaml"))
    assert gates, "no eval_* graphs — this test went blind"
    for name in gates:
        assert f"graphs/{name}" in frozen, f"{name} is a skill gate but is not frozen"
        plan = f"graphs/turn_plans/{name.removesuffix('.yaml')}.json"
        if (REPO_ROOT / plan).is_file():
            assert plan in frozen, (
                f"{plan} is executable turn topology for a gate but is not frozen"
            )
    assert "graphs/agent_campaign.yaml" not in frozen, (
        "agent_campaign is the agent's own deliverable; freezing it makes CON-7 fight the "
        "experiment (ADR-36)"
    )
