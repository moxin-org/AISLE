"""Controller-only draft P3 acceptance arithmetic on pinned public calibration.

This is not a session launcher or admission audit. Live use still requires
candidate, isolation, budget, seed-role and retention checks by the controller.
Never expose this module, measurement bundles or its output through policy RPC.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from itertools import combinations
from pathlib import Path, PurePosixPath

import goal_chains
import yaml
from p3_calibration_evidence import audit_chain, screen
from p3_evidence import digest, read_policy


def require(condition, message):
    if not condition:
        raise ValueError(message)


def planned_seeds(seeds, count):
    return (
        isinstance(seeds, list)
        and len(seeds) == count
        and all(type(seed) is int and seed >= 0 and seed % 4 == 0 for seed in seeds)
        and len(set(seeds)) == count
    )


def chain_success(record):
    """Validate independent verdict facts; invocation returns never count as success."""
    require(isinstance(record, dict) and record.get("status") == "complete", "incomplete chain")
    seed = record.get("seed")
    require(type(seed) is int, "invalid chain seed")
    ids, verdicts = record.get("goal_ids"), record.get("verdicts")
    require(
        isinstance(ids, list)
        and len(ids) == 2
        and all(isinstance(g, str) and g for g in ids)
        and len(set(ids)) == 2
        and isinstance(verdicts, list)
        and len(verdicts) == 2,
        "expected two distinct goals and verdicts",
    )
    for goal, verdict in zip(ids, verdicts, strict=True):
        require(
            isinstance(verdict, dict)
            and verdict.get("verifier") == "oracle"
            and type(verdict.get("seed")) is int
            and verdict["seed"] == seed
            and verdict.get("goal_id") == goal
            and verdict.get("status") in ("success", "fail")
            and "failure" in verdict
            and (
                verdict["failure"] is None
                if verdict["status"] == "success"
                else isinstance(verdict["failure"], str) and bool(verdict["failure"])
            ),
            "invalid independent oracle verdict",
        )
    success = all(v["status"] == "success" for v in verdicts)
    require(record.get("success") is success, "aggregate disagrees with oracle verdicts")
    return success, sum(v["failure"] == "wrong_object" for v in verdicts)


def score_session(records, seeds):
    """Pure draft rule for controller-validated facts, not a provenance attestation.

    Missing/invalid evidence retains a null score. An oracle wrong_object verdict
    is never waived, including on the scripted initial delivery. Physical failure
    classifications remain the frozen verifier's responsibility.
    """
    require(planned_seeds(seeds, 8), "expected eight distinct corrected T4 seeds")
    require(isinstance(records, list), "chain outcomes must be a list")
    seen, errors, successes, wrong = set(), [], 0, 0
    for index, record in enumerate(records):
        try:
            require(isinstance(record, dict), "malformed chain")
            seed = record.get("seed")
            require(type(seed) is int and seed in seeds, "unplanned seed")
            require(seed not in seen, "duplicate seed")
            seen.add(seed)
            success, incidents = chain_success(record)
            successes += success
            wrong += incidents
        except ValueError as exc:
            errors.append(dict(record=index, reason=str(exc)))
    missing = [seed for seed in seeds if seed not in seen]
    if missing:
        errors.append(dict(reason="missing planned chains", seeds=missing))
    return dict(
        status="unscorable" if errors else "scored",
        success=None if errors else successes >= 6 and wrong == 0,
        successful_chains=None if errors else successes,
        wrong_object_events=None if errors else wrong,
        planned_seeds=seeds,
        outcomes=records,
        errors=errors,
    )


def calibration_feasibility(records, seeds):
    require(planned_seeds(seeds, 12), "expected twelve distinct corrected calibration seeds")
    require(screen(records, seeds)["eligible"], "expected complete qualified calibration")
    by_seed = {r["seed"]: r for r in records}
    histogram, passing = Counter(), 0
    for subset in combinations(seeds, 8):
        result = score_session([by_seed[s] for s in subset], list(subset))
        require(result["status"] == "scored", "expected complete valid calibration evidence")
        histogram[result["successful_chains"]] += 1
        passing += result["success"]
    return dict(
        format="aisle_p3_acceptance_feasibility",
        rule="draft: at least 6 of 8 complete two-goal successes and zero wrong_object verdicts",
        subset_count=sum(histogram.values()),
        passing_subsets=passing,
        successful_chain_histogram=dict(sorted(histogram.items())),
        chain_outcomes=[by_seed[s] for s in seeds],
        measured_sessions=0,
        session_integration_complete=False,
        interpretation=(
            "Exhaustive overlapping subsets of public calibration only; not independent "
            "sessions, a predicted agent success rate, a treatment effect or pilot admission."
        ),
    )


def archive_path(root, name):
    require(isinstance(name, str), "invalid archive path")
    relative = PurePosixPath(name)
    require(
        not relative.is_absolute()
        and name == relative.as_posix()
        and bool(relative.parts)
        and not {"..", "."} & set(relative.parts),
        "invalid archive path",
    )
    path = root / name
    require(
        path.resolve().is_relative_to(root.resolve())
        and all(
            not (root / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)
        ),
        "archive path escapes or uses symlink",
    )
    return path


def verify_archive(root, catalog_sha256):
    root = Path(root)
    catalog = archive_path(root, "archive-catalog.json")
    require(digest(catalog) == catalog_sha256, "archive catalog pin mismatch")
    files = json.loads(catalog.read_text())["files"]
    require(isinstance(files, list) and files, "empty archive catalog")
    names = set()
    for entry in files:
        name = entry["path"]
        path = archive_path(root, name)
        require(name not in names, "duplicate archive path")
        require(
            path.is_file()
            and type(entry["bytes"]) is int
            and path.stat().st_size == entry["bytes"]
            and digest(path) == entry["sha256"],
            "archive artifact mismatch",
        )
        names.add(name)
    return names


def read_calibration(archive, catalog_sha256, protocol_sha256):
    """Rebuild facts from retained sources, never merely trust qualification summaries."""
    root = Path(archive)
    names = verify_archive(root, catalog_sha256)

    def bound(name):
        require(name in names, "required evidence absent from pinned catalog")
        return archive_path(root, name)

    protocol_path = bound("batch/protocol.json")
    require(digest(protocol_path) == protocol_sha256, "protocol pin mismatch")
    protocol = json.loads(protocol_path.read_text())
    require(
        protocol["format"] == "aisle_p3_calibration_plan"
        and protocol["seed_role"] == "public calibration only; never pilot or confirmatory holdout"
        and protocol["engine"] == "genesis"
        and protocol["perception"] == "L1"
        and planned_seeds(protocol["seeds"], 12),
        "expected public L1 Genesis calibration protocol",
    )
    records = []
    for seed in protocol["seeds"]:
        prefix = f"batch/seed-{seed}/evidence"
        assessment = json.loads(bound(f"{prefix}/assessment.json").read_text())
        measurement = bound(f"{prefix}/measurement/manifest.json").parent
        policy = bound(f"{prefix}/policy/manifest.json").parent
        # Readers validate exact bundle contents and recompute derived data.
        # Every consumed file must also belong to the externally pinned catalog.
        for bundle in (measurement, policy):
            for path in bundle.rglob("*"):
                if path.is_file():
                    bound(path.relative_to(root).as_posix())
        chains, public = goal_chains.read_chains(measurement), read_policy(policy)
        source = bound(f"{prefix}/measurement/source/manifest.json")
        manifest = json.loads(source.read_text())
        run_prefix = f"runs/{protocol['run_id_prefix']}-seed{seed}"
        original = bound(f"{run_prefix}/manifest.json")
        require(digest(original) == digest(source), "measurement source manifest mismatch")
        require(
            manifest.get("git_sha") == protocol["instrument_commit"]
            and manifest.get("graph_hash") == protocol["graph_sha256"]
            and digest(bound(f"{run_prefix}/authored-graph.yaml")) == protocol["graph_sha256"]
            and manifest.get("env_fingerprint") == protocol["env_fingerprint"]
            and manifest.get("env_hash") == protocol["env_hash"]
            and manifest.get("sim_engine") == "genesis"
            and manifest.get("perception") == "L1"
            and manifest.get("env_attested") is True
            and manifest.get("seeds") == [seed]
            and manifest.get("relaunches") == 0
            and not manifest.get("wall_clamped"),
            "calibration instrument/environment provenance mismatch",
        )
        policy_manifest = public["manifest"]
        skill = public["journal"]["header"]["skill"]
        require(
            policy_manifest["source_git_sha"] == protocol["instrument_commit"]
            and policy_manifest["source_graph_sha256"] in manifest["exec_graph_hashes"]
            and policy_manifest["sim_engine"] == "genesis"
            and policy_manifest["perception"] == "L1"
            and skill["sources"] == protocol["skill_sources"],
            "policy instrument/skill provenance mismatch",
        )
        result = audit_chain(chains["episodes"], public["journal"], seed)
        result.update(
            source_manifest_sha256=digest(source),
            measurement_manifest_sha256=digest(measurement / "manifest.json"),
            policy_manifest_sha256=digest(policy / "manifest.json"),
            instrument_commit=protocol["instrument_commit"],
            graph_sha256=protocol["graph_sha256"],
            skill=skill,
            env_fingerprint=protocol["env_fingerprint"],
        )
        require(result == assessment, "retained assessment differs from source evidence")
        records.append(result)
    return records, protocol["seeds"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        records, seeds = read_calibration(args.archive, args.catalog_sha256, args.protocol_sha256)
        result = dict(
            ok=True,
            catalog_sha256=args.catalog_sha256,
            protocol_sha256=args.protocol_sha256,
            feasibility=calibration_feasibility(records, seeds),
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, zipfile.BadZipFile) as exc:
        result = dict(ok=False, error=str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
