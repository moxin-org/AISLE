"""Generic content-addressed fault injector and sealed-bank tooling
(FLT-1, FLT-2, FLT-3, FLT-5, FLT-7, FLT-11, FLT-13, FLT-14, FLT-16;
issue #348).

A production bank is evaluator-private content outside every worktree.
This public engine knows only schemas and operators: it materializes the
allowlisted targets from the repository into an evaluator-owned staging
tree, verifies each preimage hash, applies exactly one sham, single, or
coupled transaction atomically, refuses traversal, symlink escape, dirty
preimages, partial coupled changes, unknown operators, and any destination
inside the participant worktree, and emits postimage hashes for the sealed
ledger. Assignments derive deterministically from a hidden campaign seed
(FLT-5); the sealed ledger, reveal, and replay close the loop (FLT-13,
FLT-14); lifecycle moves monotonically (FLT-16).

Pure over paths and dicts (CON-12); no clock, no environment.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
from pathlib import Path
from typing import Any

BANK_SCHEMA = "aisle.fault-bank.manifest.v1"
# v2: FLT-8 visible-size pad, fixed mtime, visible_metadata receipts
INJECTOR_VERSION = "aisle.fault-injector.v2"
FAMILIES = ("perception", "decision", "motion", "schema_metadata", "clocking", "runtime")
PERSISTENCE = ("persistent", "intermittent")
REPAIR_CLASSES = ("novel_repair", "restoration", "diagnosis_only")
LIFECYCLE = ("draft", "calibration", "sealed", "scoring", "closed", "revealed", "retired")
OPERATORS = ("replace", "sham")
#: FLT-8 participant-visible normalization: every staged target (sham or
#: fault) is padded with a trailing comment block to exactly PAD_BYTES more
#: than its preimage, and stamped with one fixed mtime, so size, mode and
#: timestamp cannot separate sham from fault cells. A replacement that grows
#: a file by more than PAD_BYTES is refused rather than leaked.
PAD_BYTES = 512
STAGED_MTIME = 1_600_000_000  # 2020-09-13T12:26:40Z, fixed for every staged file
#: FLT-11 positive allowlist: participant-authored node sources only.
TARGET_ALLOWLIST = (
    "src/aisle/nodes/segmented_pose.py",
    "src/aisle/nodes/grasp_topdown.py",
    "src/aisle/nodes/ik_trajectory.py",
    "src/aisle/nodes/task_state_machine.py",
    "src/aisle/nodes/oracle_pose.py",
    "src/aisle/nodes/l2_pose.py",
    "src/aisle/nodes/label_reader.py",
)
#: FLT-11 exclusions whose pre/postimage must stay identical.
FROZEN_ASSETS = (
    "src/aisle/nodes/budget_guard.py",
    "src/aisle/nodes/dora_genesis.py",
    "src/aisle/nodes/so101_driver.py",
    "src/aisle/nodes/g1_driver.py",
    "src/aisle/nodes/turn_barrier.py",
    "src/aisle/verifier",
    "src/aisle/reset",
    "src/aisle/scenes",
    "env/limits.toml",
    "src/aisle/harness/trace_recorder.py",
    "src/aisle/harness/fault_injector.py",
)
INSTANCE_FIELDS = {
    "opaque_id",
    "family",
    "target",
    "operator",
    "persistence",
    "activation_rule",
    "severity_ladder",
    "expected_evidence",
    "degradation_metric",
    "repair_class",
    "safety_review",
    "calibration_state",
    "release_disposition",
}


class FaultInjectorError(Exception):
    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or []


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_bytes(path.read_bytes())
    digest = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        digest.update(p.relative_to(path).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def content_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


# ---------------------------------------------------------------- manifest


def clean_baseline_hash(root: Path) -> str:
    """Hash of every allowlisted target and every frozen asset at clean."""
    parts = {rel: sha256_path(root / rel) for rel in TARGET_ALLOWLIST + FROZEN_ASSETS}
    return content_hash(parts)


def residency_errors(bank_path: Path, root: Path, tool_roots: tuple[Path, ...] = ()) -> list[str]:
    """FLT-4 residency half: the private bank must live outside the
    participant worktree, outside every git worktree / object namespace
    the repository knows, outside HOME cache/temp views, and outside every
    allowed tool root. This proves WHERE the bytes are, not that a
    participant is denied reaching them — that denial is issue #353's
    confinement adapter, which the controller must run before a session."""
    import subprocess
    import tempfile

    bank = bank_path.expanduser().resolve()
    errors: list[str] = []
    if not bank.is_file():
        return [f"bank not found: {bank}"]
    forbidden: list[tuple[str, Path]] = [("participant worktree", root.resolve())]
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], capture_output=True, text=True, cwd=root
    )
    for line in listing.stdout.splitlines():
        if line.startswith("worktree "):
            forbidden.append(("git worktree", Path(line.split(" ", 1)[1]).resolve()))
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True, cwd=root
    ).stdout.strip()
    if common:
        forbidden.append(("git object namespace", (root / common).resolve()))
    home = Path.home().resolve()
    for name in (".cache", "Library/Caches", ".local/share/uv", ".claude"):
        forbidden.append((f"HOME cache view {name}", home / name))
    forbidden.append(("temp view", Path(tempfile.gettempdir()).resolve()))
    forbidden.extend(("tool root", p.resolve()) for p in tool_roots)
    for label, base in forbidden:
        if bank == base or base in bank.parents:
            errors.append(f"bank resides inside the {label}: {base}")
    return errors


def validate_manifest(manifest: dict) -> list[str]:
    """FLT-1 / FLT-2 / FLT-3 / FLT-11: fields, coverage, repair mix,
    allowlist. Errors block seal; an empty list is a valid bank."""
    errors = []
    for key in (
        "schema_version",
        "bank_id",
        "lifecycle_state",
        "clean_baseline_hash",
        "injector_version",
        "instances",
    ):
        if key not in manifest:
            errors.append(f"manifest missing {key}")
    if errors:
        return errors
    if manifest["schema_version"] != BANK_SCHEMA:
        errors.append("unsupported bank schema")
    if manifest["lifecycle_state"] not in LIFECYCLE:
        errors.append("unknown lifecycle state")
    if manifest["injector_version"] != INJECTOR_VERSION:
        errors.append("bank targets another injector version")
    ids = set()
    for inst in manifest["instances"]:
        missing = INSTANCE_FIELDS - set(inst)
        if missing:
            errors.append(f"instance {inst.get('opaque_id')} missing {sorted(missing)}")
            continue
        if inst["opaque_id"] in ids:
            errors.append(f"duplicate opaque id {inst['opaque_id']}")
        ids.add(inst["opaque_id"])
        if inst["family"] not in FAMILIES + ("sham", "coupled"):
            errors.append(f"unknown family {inst['family']}")
        if inst["persistence"] not in PERSISTENCE:
            errors.append(f"unknown persistence {inst['persistence']}")
        if inst["repair_class"] not in REPAIR_CLASSES:
            errors.append(f"unknown repair class {inst['repair_class']}")
        for edit in _edits(inst):
            if edit["target"] not in TARGET_ALLOWLIST:
                errors.append(f"target outside allowlist: {edit['target']}")
            if edit["operator"] not in OPERATORS:
                errors.append(f"unknown operator {edit['operator']}")
        if any(v in (None, "", "unresolved") for v in inst.values()):
            errors.append(f"unresolved field in {inst['opaque_id']}")
    real = [i for i in manifest["instances"] if i["family"] in FAMILIES]
    coupled = [i for i in manifest["instances"] if i["family"] == "coupled"]
    shams = [i for i in manifest["instances"] if i["family"] == "sham"]
    for family in FAMILIES:
        rows = [i for i in real if i["family"] == family]
        if not rows:
            errors.append(f"family missing: {family}")
        elif family in ("perception", "decision", "motion") and not any(
            len(i["severity_ladder"]) >= 2 for i in rows
        ):
            errors.append(f"family {family} needs multiple severity candidates")
    if not any(i["persistence"] == "intermittent" for i in real):
        errors.append("no intermittent activation instance")
    if not any(i["persistence"] == "persistent" for i in real):
        errors.append("no persistent activation instance")
    spanning = [i for i in coupled if len({e["target"] for e in _edits(i)}) >= 2]
    if len(spanning) < 2:
        errors.append("fewer than two coupled instances spanning distinct targets")
    if not shams:
        errors.append("no sham control")
    novel_families = {i["family"] for i in real if i["repair_class"] == "novel_repair"}
    if len(novel_families) < 2:
        errors.append("fewer than two novel-repair instances spanning distinct families")
    return errors


def _edits(instance: dict) -> list[dict]:
    op = instance["operator"]
    edits = op if isinstance(op, list) else [op]
    for edit in edits:
        edit.setdefault("target", instance["target"])
    return edits


# ---------------------------------------------------------------- staging


def _safe_join(base: Path, rel: str) -> Path:
    candidate = (base / rel).resolve()
    if base.resolve() not in candidate.parents:
        raise FaultInjectorError("path traversal refused", [rel])
    return candidate


def materialize(
    root: Path, staging: Path, instance: dict, *, clean_hash: str, severity_index: int = 0
) -> dict:
    """FLT-7: copy the allowlisted targets into an evaluator-owned staging
    tree, verify preimages, apply the transaction atomically, return the
    receipt with pre/postimage hashes. `severity_index` picks the rung of
    each edit's `replace` ladder when the edit carries one."""
    root, staging = root.resolve(), staging.resolve()
    if staging == root or root in staging.parents:
        raise FaultInjectorError(
            "staging destination inside the participant worktree", [str(staging)]
        )
    if clean_baseline_hash(root) != clean_hash:
        raise FaultInjectorError("dirty preimage: repository differs from the declared clean hash")
    staging.mkdir(parents=True, exist_ok=True)
    frozen_before = {rel: sha256_path(root / rel) for rel in FROZEN_ASSETS}
    edits = _edits(instance)
    plan: dict[str, tuple[bytes, bytes]] = {}  # target -> (preimage, postimage), edits composed
    for edit in edits:
        src = _safe_join(root, edit["target"])
        if src.is_symlink() or not src.is_file():
            raise FaultInjectorError("target must be a regular file", [edit["target"]])
        original = src.read_bytes()
        if edit.get("preimage_sha256") and sha256_bytes(original) != edit["preimage_sha256"]:
            raise FaultInjectorError("preimage hash mismatch", [edit["target"]])
        current = plan.get(edit["target"], (original, original))[1]
        if edit["operator"] == "sham":
            after = current
        elif edit["operator"] == "replace":
            text = current.decode()
            replacement = edit["replace"]
            if isinstance(replacement, list):
                replacement = replacement[min(severity_index, len(replacement) - 1)]
            if text.count(edit["find"]) != 1:
                raise FaultInjectorError("find pattern must occur exactly once", [edit["target"]])
            after = text.replace(edit["find"], replacement).encode()
        else:
            raise FaultInjectorError("unknown operator", [edit["operator"]])
        plan[edit["target"]] = (original, after)
    written = []
    visible: dict[str, dict] = {}
    try:
        for rel, (before, after) in plan.items():  # atomic: temp files then rename all
            dest = _safe_join(staging, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(pad_to_visible_size(before, after))
            src = _safe_join(root, rel)
            os.chmod(tmp, src.stat().st_mode & 0o777)
            os.utime(tmp, (STAGED_MTIME, STAGED_MTIME))
            written.append((tmp, dest))
        for tmp, dest in written:
            os.replace(tmp, dest)
        for rel in plan:
            visible[rel] = visible_metadata(_safe_join(staging, rel))
    except OSError as exc:
        for tmp, _dest in written:
            tmp.unlink(missing_ok=True)
        raise FaultInjectorError("materialization failed; rolled back", [str(exc)]) from exc
    frozen_after = {rel: sha256_path(root / rel) for rel in FROZEN_ASSETS}
    if frozen_after != frozen_before:
        raise FaultInjectorError("frozen asset changed during materialization")
    return {
        "opaque_id": instance["opaque_id"],
        "severity_index": severity_index,
        "staging": str(staging),
        "clean_baseline_hash": clean_hash,
        "edits": [
            {
                "target": rel,
                "preimage": sha256_bytes(b),
                "postimage": sha256_bytes(a),
                "changed": a != b,
            }
            for rel, (b, a) in plan.items()
        ],
        "frozen_assets_identical": True,
        "sham": all(a == b for b, a in plan.values()),
        # FLT-8: what a participant can observe about the staged files before
        # any outcome; the leakage probe classifies sham vs fault on these
        "visible_metadata": visible,
    }


def pad_to_visible_size(before: bytes, after: bytes) -> bytes:
    """FLT-8: the staged bytes of a target, padded with a trailing Python
    comment so `len(result) == len(before) + PAD_BYTES` whether the edit was
    a sham or a fault. Refuses an edit that grows the file past the pad."""
    room = len(before) + PAD_BYTES - len(after)
    if room < 2:
        raise FaultInjectorError("edit grows the target beyond the visible-size pad", [str(room)])
    tail = b"\n#" + b" " * (room - 2)
    return after + tail


def visible_metadata(path: Path) -> dict:
    """The FLT-8 feature allowlist for one staged file: size, mode bits and
    mtime — the metadata a participant sees without opening the file."""
    st = path.stat()
    return {"size_bytes": st.st_size, "mode": st.st_mode & 0o777, "mtime": int(st.st_mtime)}


def stage_graph(graph_path: Path, receipt: dict, root: Path, staging: Path, out: Path) -> str:
    """Rewrite node paths that resolve to a staged target so the run loads
    the staged copy; every other node keeps its repository source."""
    import yaml

    doc = yaml.safe_load(graph_path.read_text())
    staged = {edit["target"] for edit in receipt["edits"]}
    for node in doc.get("nodes", []):
        path = (graph_path.parent / node["path"]).resolve()
        try:
            rel = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if rel in staged:
            node["path"] = str((staging / rel).resolve())
        else:
            node["path"] = str(path)
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return sha256_path(out)


# ---------------------------------------------------------------- controller


def assign(hidden_seed: bytes, bank_id: str, block: str, session_id: str, cells: list[str]) -> dict:
    """FLT-5: deterministic opaque assignment; the proof is an HMAC that a
    revealed seed reproduces."""
    if not cells:
        raise FaultInjectorError("no planned cells")
    material = f"{bank_id}|{block}|{session_id}".encode()
    digest = hmac.new(hidden_seed, material, "sha256").hexdigest()
    index = int(digest, 16) % len(cells)
    return {"session_id": session_id, "block": block, "cell": cells[index], "proof": digest}


def commitment(
    manifest: dict, seed_commitment: str, protocol_hash: str, planned_cells: dict
) -> str:
    """FLT-5 pre-collection commitment without revealing identities."""
    return content_hash(
        {
            "bank_hash": content_hash(manifest),
            "assignment_algorithm": "hmac-sha256(seed, bank_id|block|session_id) mod cells",
            "seed_commitment": seed_commitment,
            "protocol_hash": protocol_hash,
            "planned_cells": planned_cells,
        }
    )


def advance(manifest: dict, new_state: str) -> dict:
    """FLT-16: lifecycle moves forward only, one state at a time."""
    current = LIFECYCLE.index(manifest["lifecycle_state"])
    if new_state not in LIFECYCLE or LIFECYCLE.index(new_state) != current + 1:
        raise FaultInjectorError(
            "lifecycle must advance monotonically", [manifest["lifecycle_state"], new_state]
        )
    if new_state == "sealed" and validate_manifest(manifest):
        raise FaultInjectorError("cannot seal an invalid bank", validate_manifest(manifest))
    return {**manifest, "lifecycle_state": new_state}


def ledger_row(assignment: dict, receipt: dict | None, *, campaign_id: str, status: str) -> dict:
    """FLT-13 sealed append-only record; the participant-facing view is
    only the opaque session identity."""
    return {
        "campaign_id": campaign_id,
        "session_id": assignment["session_id"],
        "block": assignment["block"],
        "opaque_assignment": content_hash(assignment["proof"]),
        "proof": assignment["proof"],
        "cell": assignment["cell"],
        "receipt": receipt,
        "status": status,
        "participant_view": {"session_id": assignment["session_id"]},
    }


def reveal(
    ledger: list[dict], manifest: dict, hidden_seed: bytes, bank_id: str, cells: list[str]
) -> dict:
    """FLT-14: verify every assignment reproduces from the revealed seed and
    map it to family, target, persistence, and repair class."""
    by_id = {i["opaque_id"]: i for i in manifest["instances"]}
    mapping, mismatches = [], []
    for row in ledger:
        expected = assign(hidden_seed, bank_id, row["block"], row["session_id"], cells)
        if expected["proof"] != row["proof"] or expected["cell"] != row["cell"]:
            mismatches.append(row["session_id"])
            continue
        inst = by_id.get(row["cell"])
        mapping.append(
            {
                "session_id": row["session_id"],
                "cell": row["cell"],
                "family": inst["family"] if inst else None,
                "targets": sorted({e["target"] for e in _edits(inst)}) if inst else [],
                "persistence": inst["persistence"] if inst else None,
                "repair_class": inst["repair_class"] if inst else None,
            }
        )
    return {"ok": not mismatches, "mapping": mapping, "commitment_mismatches": mismatches}


def replay(root: Path, staging: Path, manifest: dict, ledger: list[dict], clean_hash: str) -> dict:
    """FLT-14: reconstruct every recorded postimage and compare hashes."""
    by_id = {i["opaque_id"]: i for i in manifest["instances"]}
    drift = []
    for row in ledger:
        if not row.get("receipt"):
            drift.append(f"{row['session_id']}: no receipt")
            continue
        inst = by_id[row["cell"]]
        again = materialize(
            root,
            staging / row["session_id"],
            inst,
            clean_hash=clean_hash,
            severity_index=row["receipt"]["severity_index"],
        )
        if [e["postimage"] for e in again["edits"]] != [
            e["postimage"] for e in row["receipt"]["edits"]
        ]:
            drift.append(f"{row['session_id']}: postimage drift")
    shutil.rmtree(staging, ignore_errors=True)
    return {"ok": not drift, "drift": drift, "replayed": len(ledger)}
