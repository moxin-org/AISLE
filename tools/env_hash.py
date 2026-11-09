#!/usr/bin/env python3
"""env_hash: fingerprint the CON-7 frozen set (CON-5, CON-8).

Hashes the frozen set defined by FROZEN_DIRS + FROZEN_FILES below plus
graphs/expert_*.yaml — sorted relative paths + file contents; __pycache__
excluded — into one sha256. Read those constants, not this paragraph: an
enumeration here would be a second copy of the fence that goes stale, which
is how src/aisle/mobility stayed outside it (issue #189, ADR-33).

Modes: compute (default), --write (commit tools/env_hash.json),
--check (compare against the committed hash; rollout refuses on mismatch,
HAR-2). --check --baseline <git-ref> is the TRUSTED mode (PR #24, ADR-21):
the baseline hash is read from the git object store at <ref> (a protected
branch a research agent cannot move), and this checker itself must match
its blob at <ref> — regenerating the local json or editing the checker
after changing frozen code no longer blesses the change. JSON on stdout,
logs on stderr, exit 0 iff ok.

--sim-engine <engine> adds ADR-67's engine attestation to any mode: the
`sim` block (`sim_engine_hash` over src/aisle/sim/**) and that engine's own
distribution in the attested set. The digest is recorded beside the
engine-neutral frozen-set hash. Every engine's wheel comes from the lock
(ADR-70), so its provenance is the dist attestation's, as for Genesis.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

# THE RULE (issue #189): everything the budget guard reads to reach a
# SAFETY VERDICT is inside the fence, plus the scene/verifier/reset
# artifacts that define what a run is. "The guard's module" is not the
# unit — the guard is 700 lines of plumbing around verdicts that live
# elsewhere, and freezing the plumbing while leaving the verdicts outside
# is the hole this list closes.
#
# `tests/unit/test_env_hash.py::test_the_guards_safety_inputs_are_all_fenced`
# enforces it: every first-party module the guard imports must resolve
# inside. A new import fails that test rather than silently widening what
# can change without moving the hash.
FROZEN_DIRS = (
    "src/aisle/scenes",
    "src/aisle/verifier",
    "src/aisle/reset",
    # MOB-3 verdicts: base_watchdog_reason, clamp_base_cmd, valid_base_pose,
    # the keep-out geometry and the blind-drive predicates. Outside the
    # fence until issue #189 — PR #177 changed nav's stall/timeout budgets
    # here and moved no hash, so two runs with different failure conditions
    # attested as the same environment.
    "src/aisle/mobility",
    "assets/so101",
    "env",
)
# SPEC 080: the guard and its limits are frozen safety artifacts — a run's
# env_hash must change if either does. harness/budget.toml carries the
# campaign ceilings (ADR-21): budgets get the same tamper trust as limits.
# topics.py is the TC-2/BG-3 stamp trust boundary the guard reads on every
# message; turn_node.py and turns.py enforce the ADR-30 input/output closure
# around every lockstep guard verdict; kinematics.py is the SO-101 forward
# chain behind the workspace check (fk_ee_pose). These decide verdicts or
# whether they may be emitted, and none is a scene artifact.
# embodiment.py holds SO101_ARM_JOINTS, the TC-5 joint order `so101_chain()`
# refuses to build against a mismatched URDF — one hop behind the workspace
# check, and found only by following the fence's OWN imports rather than the
# guard's (issue #189 review).
#: Frozen by pattern rather than by name, because the set grows with the
#: corpus. `graphs/expert_*.yaml` are the measured expert graphs.
#: `graphs/eval_*.yaml` are the skill GATE — `skills/*/eval.yaml` names them
#: and `harness skill register` rolls out through one before writing a
#: manifest, so a candidate that can edit them can edit its own exam
#: (ADR-36, issue #228). `agent_campaign.yaml` is deliberately absent: it is
#: the agent's own deliverable and freezing it would put CON-7 in conflict
#: with the experiment.
#: The turn_plans entries are ADR-30 scheduler topology the barrier loads at
#: runtime — generated, but executable, so they are hashed beside the graphs
#: they compile from rather than treated as documentation.
FROZEN_GLOBS = (
    "graphs/expert_*.yaml",
    "graphs/eval_*.yaml",
    "graphs/turn_plans/expert_*.json",
    "graphs/turn_plans/eval_*.json",
)
FROZEN_FILES = (
    "src/aisle/nodes/budget_guard.py",
    "src/aisle/topics.py",
    "src/aisle/turn_node.py",
    "src/aisle/turns.py",
    "src/aisle/kinematics.py",
    "src/aisle/embodiment.py",
    "harness/budget.toml",
)
HASH_FILE = "tools/env_hash.json"
SELF = "tools/env_hash.py"


def frozen_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for d in FROZEN_DIRS:
        base = root / d
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for pattern in FROZEN_GLOBS:
        files.extend(p for p in root.glob(pattern) if p.is_file())
    files.extend(root / f for f in FROZEN_FILES if (root / f).is_file())
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def compute_env_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = frozen_files(root)
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        # fixed-length per-file digest: file boundaries stay unambiguous
        # even when contents contain NUL bytes
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def _git_blob(root: Path, ref: str, path: str) -> bytes | None:
    """The file's contents at <ref> from the git object store, or None."""
    proc = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, cwd=root)
    return proc.stdout if proc.returncode == 0 else None


def _baseline_hash(root: Path, ref: str) -> tuple[str | None, str | None]:
    """(committed env_hash at <ref>, error). TRUSTED mode (ADR-21): the
    baseline lives in git objects on a protected ref, not in the working
    tree — and this checker must match its own blob there, else a rewritten
    checker could bless anything."""
    self_blob = _git_blob(root, ref, SELF)
    if self_blob is None:
        return None, f"cannot read {SELF} at {ref!r} (not a git repo, or ref missing)"
    if self_blob != (root / SELF).read_bytes():
        return None, f"{SELF} diverges from {ref} — the gate tooling itself is not trusted"
    hash_blob = _git_blob(root, ref, HASH_FILE)
    if hash_blob is None:
        return None, f"cannot read {HASH_FILE} at {ref!r}"
    try:
        return json.loads(hash_blob)["env_hash"], None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, f"{HASH_FILE} at {ref} is corrupted"


# ---------------------------------------------------------------------------
# ADR-24: installed-environment attestation. Lives HERE — inside the
# self-verified checker blob — so an agent cannot rewrite the collector/
# comparator without tripping the checker's own baseline verification
# (PR #68 review D3). Identity comes from artifacts (the lock uv verified
# at install time), never from version probes.

# the attested set beyond registry pip: sources — the sim core whose code
# decides physics and gate verdicts (ADR-24 resolved question 1)
ATTESTED_SIM_CORE = ("genesis-world", "torch", "dora-rs", "pyarrow")
# ADR-67: the distribution that actually steps the scene, per engine. The
# core above is engine neutral, so a Nexus run used to attest genesis-world
# and say nothing about the wheel whose solver produced the trajectories;
# attested_set() adds the running engine's dist instead.
ENGINE_DISTS = {
    "genesis": ("genesis-world",),
    "nexus": ("dimforge-nexus3d",),
    # rapier steps the scene and the Nexus viewer renders it (ADR-68)
    "rapier": ("rapier3d", "dimforge-nexus3d"),
}
#: The engine realization package (ADR-67, ADR-68): the engine selector, the
#: Nexus and rapier backends, and their solver settings. Deliberately outside
#: FROZEN_DIRS (widening the fence moves every Genesis env_hash and breaks
#: the measured record's continuity), so it carries its own digest instead.
SIM_REALIZATION_DIR = "src/aisle/sim"
LOCK_FILE = "uv.lock"


def canonical_name(name: str) -> str:
    """PEP 503 canonicalization — every name join (lock, metadata,
    manifests) folds case and runs of [-_.] (PR #68 review, minor)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def current_selection(extras: list[str], groups: list[str] | None = None) -> dict:
    """The resolved environment descriptor the fingerprint names: the
    lock alone is selection-ambiguous (multi-platform entries; extras
    and groups share one lock — the default-vs-`--extra sim` footgun).
    PR #69 review F4: full interpreter identity (version + ABI cache
    tag), the platform tag set, extras AND dependency groups (uv's
    default sync includes the dev group)."""
    import platform as _platform
    import sysconfig

    return {
        "python": _platform.python_version(),
        "abi": sys.implementation.cache_tag,
        "platform_tags": sorted({sysconfig.get_platform(), _platform.machine(), sys.platform}),
        "extras": sorted(extras),
        "groups": sorted(groups if groups is not None else ["dev"]),
    }


def env_fingerprint(lock_bytes: bytes, selection: dict) -> str:
    """CON-5's fifth tuple component (ADR-24 resolved question 2)."""
    canon = json.dumps(selection, sort_keys=True).encode()
    return hashlib.sha256(lock_bytes + b"\0" + canon).hexdigest()


def parse_pip_source(value: str) -> str | None:
    """The ONE pip-source parser (PR #69 review F2): mirrors
    registry._pip_dist exactly — case-insensitive scheme, strip, cut at
    the first extras/specifier character, PEP 503 canonicalize — and a
    parity test pins the two implementations together. Returns the
    canonical distribution name, or None for non-pip values."""
    value = value.strip().strip("'\"")
    if value[:4].lower() != "pip:":
        return None
    name = value[4:].strip()
    for cut in "[=<>!~;@":
        name = name.split(cut, 1)[0]
    name = name.strip()
    return canonical_name(name) if name else None


def registry_pip_dists(root: Path) -> list[str]:
    """Canonical names of every registry pip: source, via the shared
    parser. Line-scanned (indentation allowed, quotes allowed) so the
    trusted blob stays stdlib-only; manifest sources are simple scalars
    (CAP-1-patterned)."""
    dists = set()
    manifest_dir = root / "registry" / "manifests"
    if manifest_dir.is_dir():
        for m in sorted(manifest_dir.glob("*.yaml")):
            for match in re.finditer(r"^[ \t]*source:[ \t]*(.+)$", m.read_text(), re.M):
                dist = parse_pip_source(match.group(1))
                if dist:
                    dists.add(dist)
    return sorted(dists)


def classify_direct_url(direct_url: dict) -> str | None:
    """PEP 610 provenance verdict for an attested dist: None = ok,
    else the problem. Editable/local-dir installs are unattestable;
    VCS installs need a commit id (dora-rs is a legitimate pinned-rev
    git dependency); archive installs need a hash."""
    dir_info = direct_url.get("dir_info")
    if dir_info is not None:
        kind = "editable" if dir_info.get("editable") else "local-directory"
        return f"{kind} install"
    vcs = direct_url.get("vcs_info")
    if vcs is not None:
        return None if vcs.get("commit_id") else "VCS install without a commit id"
    archive = direct_url.get("archive_info")
    if archive is not None and not (archive.get("hashes") or archive.get("hash")):
        return "archive install without a hash"
    return None


def attested_set(root: Path, engine: str | None = None) -> list[str]:
    """The dists whose provenance is checked. `engine` (ADR-67) adds the
    running engine's own distribution; None keeps the pre-ADR-67 set."""
    names = set(registry_pip_dists(root)) | set(map(canonical_name, ATTESTED_SIM_CORE))
    names |= {canonical_name(d) for d in ENGINE_DISTS.get(engine or "", ())}
    return sorted(names)


def sim_realization_files(root: Path) -> list[Path]:
    """Files under SIM_REALIZATION_DIR, sorted, __pycache__ excluded."""
    base = root / SIM_REALIZATION_DIR
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def sim_engine_hash(root: Path, engine: str) -> dict:
    """ADR-67 engine attestation: the identity env_hash cannot carry.

    One sha256 over the engine name and the realization package
    (`src/aisle/sim/**`, which holds nexus_physics.toml's substeps, PGS
    iterations, contact frequencies and friction rule). Two Nexus runs with
    different solver settings differ here while their env_hash is
    byte-identical; the manifest's `sim_engine` string alone could not tell
    them apart. The engine wheel itself is identified by the lock (ADR-70).
    Returns the facts for the run manifest: {"engine", "sim_engine_hash",
    "n_files"}; refusal policy is the caller's (CON-8)."""
    files = sim_realization_files(root)
    digest = hashlib.sha256()
    digest.update(engine.encode())
    digest.update(b"\0")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        # same framing as compute_env_hash: fixed-length per-file digests
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "engine": engine,
        "sim_engine_hash": digest.hexdigest(),
        "n_files": len(files),
    }


def environment_inventory() -> dict:
    """Gate-time capture for the post-session audit (PR #69 review F1/F3):
    EVERY installed distribution -> {version, record_sha256}. The RECORD
    content hash taken NOW, inside the trusted checker, is what the
    post-session verification trusts — a mid-session mutation of code
    plus RECORD cannot self-bless, and a removed/added distribution is a
    fail-closed mismatch. Full closure, not a bounded core: frozen code
    imports numpy et al., so 'identical tuples hold identical
    environments' requires the whole environment."""
    import importlib.metadata

    inventory: dict = {}
    for dist in importlib.metadata.distributions():
        name = canonical_name(dist.metadata.get("Name") or "")
        if not name:
            continue
        record = dist.read_text("RECORD")
        inventory[name] = {
            "version": dist.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
        }
    return dict(sorted(inventory.items()))


def dist_attestation(
    root: Path,
    baseline_ref: str | None,
    extras: list[str],
    groups: list[str] | None = None,
    engine: str | None = None,
) -> dict:
    """The ADR-24 D2 gate facts: fingerprint + problems (empty = attested)
    + the full environment inventory the post-session audit verifies
    against. Policy (refuse vs record) belongs to the caller; FACTS are
    computed here, inside the trusted blob."""
    import importlib.metadata

    problems: list[str] = []
    lock_path = root / LOCK_FILE
    lock_bytes = lock_path.read_bytes() if lock_path.is_file() else None
    if lock_bytes is None:
        problems.append(f"{LOCK_FILE} missing — the environment is unattestable")
    if baseline_ref and lock_bytes is not None:
        base_lock = _git_blob(root, baseline_ref, LOCK_FILE)
        if base_lock is None:
            problems.append(f"{LOCK_FILE} unreadable at {baseline_ref!r}")
        elif base_lock != lock_bytes:
            problems.append(f"{LOCK_FILE} diverges from {baseline_ref} (DIST_DRIFT)")
    selection = current_selection(extras, groups)
    fingerprint = env_fingerprint(lock_bytes, selection) if lock_bytes is not None else None
    if lock_bytes is not None:
        sync = subprocess.run(
            ["uv", "sync", "--locked", "--check"] + [f"--extra={e}" for e in sorted(extras)],
            capture_output=True,
            text=True,
            cwd=root,
        )
        if sync.returncode != 0:
            problems.append(
                "environment out of sync with the locked selection "
                f"(uv sync --locked --check: {(sync.stderr or sync.stdout).strip()[-200:]})"
            )
    for dist_name in attested_set(root, engine):
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue  # absence is INSTALL_MISSING's concern, not provenance's
        raw = dist.read_text("direct_url.json")
        if raw:
            try:
                verdict = classify_direct_url(json.loads(raw))
            except json.JSONDecodeError:
                verdict = "malformed direct_url.json"
            if verdict:
                problems.append(f"{dist_name}: {verdict}")
    return {
        "attested": not problems,
        "env_fingerprint": fingerprint,
        "selection": selection,
        "problems": problems,
        "inventory": environment_inventory(),
    }


def verify_records(expected_inventory: dict) -> dict:
    """ADR-24 D2 post-session audit, FAIL-CLOSED (PR #69 review F1):
    against the GATE-TIME inventory — a distribution that disappeared,
    appeared, changed version, changed its RECORD (self-blessing), has
    no verifiable hashed entries, or whose installed files no longer
    match that trusted RECORD, is each a problem. Full environment
    (F3): frozen code executes numpy and the transitive closure, so the
    identity claim covers everything installed."""
    import base64
    import importlib.metadata

    report: dict = {"ok": True, "verified": {}, "problems": []}
    live: dict = {}
    for dist in importlib.metadata.distributions():
        name = canonical_name(dist.metadata.get("Name") or "")
        if name:
            live[name] = dist
    for name in sorted(set(live) - set(expected_inventory)):
        report["problems"].append(f"{name}: installed after the gate (not in inventory)")
    for name, claim in sorted(expected_inventory.items()):
        dist = live.get(name)
        if dist is None:
            report["problems"].append(f"{name}: removed after the gate")
            continue
        if dist.version != claim.get("version"):
            report["problems"].append(
                f"{name}: version changed after the gate ({claim.get('version')} -> {dist.version})"
            )
            continue
        record = dist.read_text("RECORD")
        record_hash = hashlib.sha256(record.encode()).hexdigest() if record else None
        if record_hash != claim.get("record_sha256"):
            report["problems"].append(
                f"{name}: RECORD changed after the gate — a mutated RECORD "
                "cannot bless mutated code"
            )
            continue
        checked = mismatched = 0
        for f in dist.files or []:
            hash_spec = getattr(f, "hash", None)
            if hash_spec is None or hash_spec.mode != "sha256":
                continue
            try:
                data = f.locate().read_bytes()
            except OSError:
                report["problems"].append(f"{name}: {f} unreadable")
                continue
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            checked += 1
            if digest.decode() != hash_spec.value:
                mismatched += 1
                report["problems"].append(f"{name}: {f} does not match its RECORD hash")
        if checked == 0:
            report["problems"].append(f"{name}: zero hash-verifiable RECORD entries — unattestable")
        report["verified"][name] = {"files": checked, "mismatched": mismatched}
    report["ok"] = not report["problems"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--baseline",
        default=None,
        help="git ref holding the TRUSTED baseline (with --check); e.g. origin/main",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help=f"write hash to {HASH_FILE}")
    mode.add_argument("--check", action="store_true", help=f"compare against {HASH_FILE}")
    mode.add_argument(
        "--verify-records",
        action="store_true",
        help="ADR-24 audit: hash installed files of the attested set vs RECORD",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help="with --verify-records: path to the gate-time inventory JSON",
    )
    parser.add_argument(
        "--groups",
        action="append",
        default=None,
        help="declared dependency groups for the attestation selection",
    )
    parser.add_argument(
        "--extras",
        action="append",
        default=None,
        help="declared uv extras for the attestation selection (repeatable); "
        "with --check, enables the ADR-24 dist attestation",
    )
    parser.add_argument(
        "--sim-engine",
        default=None,
        help="ADR-67 engine (genesis|nexus|rapier): report the engine-specific "
        "sim_engine_hash and attest that engine's own distribution",
    )
    args = parser.parse_args()

    if args.verify_records:
        # PR #69 review F1: the audit self-verifies the checker first when
        # a baseline is given, and verifies against the GATE-TIME
        # inventory (fail-closed) — never against the live RECORDs alone
        if args.baseline:
            _, self_error = _baseline_hash(args.root, args.baseline)
            if self_error:
                print(json.dumps({"ok": False, "problems": [self_error]}))
                return 1
        if not args.expected:
            print(json.dumps({"ok": False, "problems": ["--expected inventory required"]}))
            return 1
        try:
            expected = json.loads(Path(args.expected).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"ok": False, "problems": [f"unreadable inventory: {exc}"]}))
            return 1
        report = verify_records(expected)
        print(json.dumps(report))
        return 0 if report["ok"] else 1

    env_hash, n_files = compute_env_hash(args.root)
    report: dict = {"ok": True, "env_hash": env_hash, "n_files": n_files}
    if args.sim_engine:
        # ADR-67: the digest is recorded beside the engine-neutral env_hash
        report["sim"] = sim_engine_hash(args.root, args.sim_engine)

    if args.write:
        hash_path = args.root / HASH_FILE
        hash_path.write_text(json.dumps({"env_hash": env_hash, "n_files": n_files}) + "\n")
        print(f"wrote {hash_path}", file=sys.stderr)
    elif args.check:
        committed = None
        if args.baseline:
            committed, error = _baseline_hash(args.root, args.baseline)
            if error is None and committed != env_hash:
                error = f"frozen set diverges from {args.baseline} (CON-7)"
        else:
            hash_path = args.root / HASH_FILE
            if not hash_path.exists():
                error = f"{HASH_FILE} not found"
            else:
                try:
                    committed = json.loads(hash_path.read_text())["env_hash"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    error = f"{HASH_FILE} is corrupted (expected JSON with an env_hash key)"
                else:
                    error = None if committed == env_hash else "frozen set changed (CON-7)"
        if error:
            report = {"ok": False, "env_hash": env_hash, "committed": committed, "error": error}
        elif args.baseline:
            report["baseline"] = args.baseline
        args.extras = [e for e in (args.extras or []) if e] if args.extras is not None else None
        if error is None and args.extras is not None:
            # ADR-24: attestation FACTS ride the same trusted-blob check;
            # refusal policy is the gate's (HAR-2)
            groups = [g for g in (args.groups or []) if g] if args.groups is not None else None
            report["dist"] = dist_attestation(
                args.root, args.baseline, args.extras, groups, args.sim_engine
            )

    print(json.dumps(report))
    if not report["ok"]:
        print(f"env_hash check failed: {report['error']}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
