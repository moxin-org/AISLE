"""Bounded offline execution replay (P2; CON-5/8, HAR-4/6).

P1 recordings currently establish resets and policy-visible requests, not skill
invocations. Explicit synthetic events exercise the proposed replay semantics.
No event authorizes actuation; no local completion establishes task delivery.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import demo_dataset as dataset

FORMAT = "aisle_execution_ledger"
VERSION = 1
KINDS = {"reset", "goal", "proposed", "started", "returned", "interrupted", "failed", "observed"}
TERMINAL = {"interrupted", "failed", "completed"}
COMMON = {"id", "kind", "sim_time_ns", "env_id", "episode"}
INVOCATION = {"goal_revision", "invocation_id", "skill"}
EXTRA = {
    "reset": set(),
    "goal": {"goal_revision", "instruction"},
    "proposed": INVOCATION | {"parameters", "completion"},
    "observed": INVOCATION | {"names", "positions", "units", "frame"},
    **{kind: INVOCATION for kind in ("started", "returned", "interrupted", "failed")},
}


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def finite_number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def vector(value) -> bool:
    return (
        isinstance(value, list) and 0 < len(value) <= 256 and all(finite_number(x) for x in value)
    )


def joint_names(value) -> bool:
    return (
        isinstance(value, list)
        and 0 < len(value) <= 256
        and all(nonempty(x) for x in value)
        and len(set(value)) == len(value)
    )


def validate_record(record: dict) -> None:
    if not isinstance(record, dict) or record.get("kind") not in KINDS:
        raise ValueError("unsupported evidence kind")
    kind = record["kind"]
    if set(record) != COMMON | EXTRA[kind]:
        raise ValueError(f"unexpected or missing evidence fields: {kind}")
    if not nonempty(record["id"]) or len(encoded(record)) > 65536:
        raise ValueError("invalid or oversized event identity/payload")
    for key in ("env_id", "episode", "sim_time_ns"):
        if type(record[key]) is not int or record[key] < 0:
            raise ValueError(f"invalid event {key}")
    if kind == "reset":
        return
    if kind == "goal":
        # None invalidates the old intent after a malformed captured request.
        if not nonempty(record["goal_revision"]) or (
            record["instruction"] is not None and not nonempty(record["instruction"])
        ):
            raise ValueError("invalid goal evidence")
        return
    if not nonempty(record["invocation_id"]) or not nonempty(record["goal_revision"]):
        raise ValueError("invalid invocation/goal identity")
    skill = record["skill"]
    if (
        not isinstance(skill, dict)
        or set(skill) != {"name", "version"}
        or any(v is not None and not nonempty(v) for v in skill.values())
    ):
        raise ValueError("invalid skill identity")
    if kind == "proposed":
        if not isinstance(record["parameters"], dict):
            raise ValueError("parameters must be an object")
        condition = record["completion"]
        if condition is not None and (
            not isinstance(condition, dict)
            or set(condition) != {"kind", "names", "target", "tolerance"}
            or condition["kind"] != "joint_position"
            or not vector(condition["target"])
            or not joint_names(condition["names"])
            or len(condition["names"]) != len(condition["target"])
            or not finite_number(condition["tolerance"])
            or not 0 < condition["tolerance"] <= 0.1
        ):
            raise ValueError("unsupported completion condition")
    if kind == "observed" and (
        not vector(record["positions"])
        or not joint_names(record["names"])
        or not nonempty(record["units"])
        or not nonempty(record["frame"])
    ):
        raise ValueError("invalid measured joint evidence")


def _replay(records, references, *, max_events, max_invocations):
    """Arrival-ordered replay; no inferred order at equal simulation timestamps."""
    if type(max_events) is not int or not 1 <= max_events <= 100000:
        raise ValueError("event limit must be in 1..100000")
    if type(max_invocations) is not int or not 1 <= max_invocations <= 10000:
        raise ValueError("invocation limit must be in 1..10000")
    if len(records) > max_events:
        raise ValueError("event limit exceeded")
    # Validate the whole input before exposing any replay result. Duplicate
    # identity is global, never evicted by resets, and bounded by max_events.
    identities, stamps = {}, {}
    for row in records:
        validate_record(row)
        fingerprint = digest(row)
        if row["id"] in identities and identities[row["id"]] != fingerprint:
            raise ValueError("conflicting event identity")
        identities[row["id"]] = fingerprint
        key = (row["env_id"], row["sim_time_ns"])
        # An old delayed event may have an earlier stamp; equality between
        # different events is still ambiguous and never sorted into certainty.
        if key in stamps and stamps[key] != row["id"]:
            raise ValueError("ambiguous timestamp between distinct events")
        stamps[key] = row["id"]

    states, clocks, goals, seen = {}, {}, {}, set()
    histories, timeline, duplicates = [], [], 0

    def invalidate(state, ref, reason, stamp):
        for entry in state["entries"]:
            entry["intent_current"] = False
            if entry["status"] not in TERMINAL:
                entry["status"] = (
                    "interrupted" if entry["started_ns"] is not None else "unconfirmed"
                )
                entry["reason"] = reason
                entry["updated_ns"] = stamp
                entry["evidence"].append(ref)

    def apply(row, ref):
        env, episode, stamp = (row[k] for k in ("env_id", "episode", "sim_time_ns"))
        kind = row["kind"]
        state = states.get(env)
        if kind == "reset":
            if state is None:
                if episode != 0 or stamp != 0:
                    return "missing_initial_reset"
            elif episode != state["episode"] + 1 or stamp <= clocks[env]:
                return "stale_reset"
            if state is not None:
                invalidate(state, ref, "reset", stamp)
                histories.append(copy.deepcopy(state))
            states[env] = {
                "env_id": env,
                "episode": episode,
                "reset_ns": stamp,
                "reset_evidence": ref,
                "goal_revision": None,
                "instruction": None,
                "goal_evidence": None,
                "entries": [],
                "skill_attribution": "unknown",
            }
            goals[env] = set()
            return None
        if state is None:
            return "missing_reset"
        if episode != state["episode"]:
            return "stale_episode"
        if stamp <= clocks[env]:
            return "noncausal_event"
        if kind == "goal":
            if row["goal_revision"] in goals[env]:
                return "reused_goal"
            invalidate(state, ref, "goal_changed", stamp)
            goals[env].add(row["goal_revision"])
            state.update(
                goal_revision=row["goal_revision"],
                instruction=row["instruction"],
                goal_evidence=ref,
            )
            return None
        if row["goal_revision"] != state["goal_revision"]:
            return "stale_goal"
        if state["instruction"] is None:
            return "unknown_goal"
        entry = next(
            (e for e in state["entries"] if e["invocation_id"] == row["invocation_id"]), None
        )
        if kind == "proposed":
            if entry is not None:
                return "reused_invocation"
            if len(state["entries"]) >= max_invocations:
                raise ValueError("invocation limit exceeded")
            known = all(nonempty(v) for v in row["skill"].values())
            state["entries"].append(
                {
                    "invocation_id": row["invocation_id"],
                    "env_id": env,
                    "episode": episode,
                    "goal_revision": row["goal_revision"],
                    "skill": copy.deepcopy(row["skill"]),
                    "parameters": copy.deepcopy(row["parameters"]),
                    "completion": copy.deepcopy(row["completion"]),
                    "proposed_ns": stamp,
                    "started_ns": None,
                    "updated_ns": stamp,
                    "status": "proposed" if known else "unconfirmed",
                    "reason": "intent_only" if known else "missing_skill_identity",
                    "establishes": "intent_only",
                    "delivery_completion": "unknown",
                    "intent_current": True,
                    "evidence": [ref],
                }
            )
            state["skill_attribution"] = "explicit_event_evidence"
            return None
        if entry is None:
            return "unknown_invocation"
        if row["skill"] != entry["skill"]:
            return "skill_mismatch"
        # Even a measurement that does not confirm the predicate makes older
        # arrivals stale. Unrelated episodes/goals/versions cannot move time.
        clocks[env] = stamp
        if not entry["intent_current"] or entry["status"] in TERMINAL:
            return "terminal_or_obsolete_invocation"
        if not all(nonempty(v) for v in entry["skill"].values()):
            return "missing_skill_identity"
        if kind == "started":
            if entry["status"] != "proposed":
                return "invalid_start"
            entry.update(
                status="started",
                started_ns=stamp,
                reason="execution_started",
                establishes="execution_reported_start",
            )
        elif kind in {"interrupted", "failed"}:
            if entry["started_ns"] is None:
                return "missing_start"
            entry.update(
                status=kind,
                reason=f"execution_{kind}",
                establishes="execution_reported_failure"
                if kind == "failed"
                else "execution_reported_interruption",
            )
        elif kind == "returned":
            entry.update(
                status="unconfirmed",
                reason="return_without_physical_confirmation",
                establishes="execution_reported_return",
            )
        elif kind == "observed":
            if entry["started_ns"] is None:
                return "missing_start"
            condition = entry["completion"]
            if condition is None:
                return "unknown_completion_condition"
            if (
                row["units"] != "rad"
                or row["names"] != condition["names"]
                or row["frame"] != "joint"
                or len(row["positions"]) != len(condition["target"])
            ):
                return "incompatible_observation"
            if not all(
                abs(a - b) <= condition["tolerance"]
                for a, b in zip(row["positions"], condition["target"], strict=True)
            ):
                return "condition_not_observed"
            entry.update(
                status="completed",
                reason="local_condition_observed",
                establishes="joint_configuration_within_tolerance",
            )
        entry["updated_ns"] = stamp
        entry["evidence"].append(ref)
        return None

    for row, ref in zip(records, references, strict=True):
        if row["id"] in seen:
            duplicates += 1
            continue
        seen.add(row["id"])
        reason = apply(row, ref)
        if reason is None:
            clocks[row["env_id"]] = row["sim_time_ns"]
        timeline.append(
            {
                "event_id": row["id"],
                "kind": row["kind"],
                "evidence": ref,
                "decision": "rejected" if reason else "applied",
                "reason": reason,
            }
        )
    return {
        "format": FORMAT,
        "version": VERSION,
        "limits": {"max_events": max_events, "max_invocations_per_episode": max_invocations},
        "active": [states[env] for env in sorted(states)],
        "histories": histories,
        "timeline": timeline,
        "duplicates": duplicates,
    }


def replay_events(document, *, max_events=10000, max_invocations=128):
    if (
        not isinstance(document, dict)
        or set(document) != {"format", "version", "provenance", "records"}
        or document["format"] != "aisle_execution_events"
        or type(document["version"]) is not int
        or document["version"] != VERSION
        or document["provenance"] != "synthetic"
        or not isinstance(document["records"], list)
    ):
        raise ValueError("expected version 1 explicitly synthetic event fixture")
    refs = [
        {"source": "records", "row": i, "sha256": digest(row)}
        for i, row in enumerate(document["records"])
    ]
    report = _replay(
        document["records"], refs, max_events=max_events, max_invocations=max_invocations
    )
    report.update(provenance="synthetic", source_sha256=digest(document))
    return report


def replay_dataset(root: Path, *, max_events=10000, max_invocations=128):
    """Read original requests, including ones with no subsequent command sample.

    Privileged episode goals/outcomes are validated by P1, but never copied to
    policy state or used to infer execution. Source rows retain their hashes.
    """
    data = dataset.read_dataset(root)
    manifest = data["manifest"]
    rows = {}
    for topic in ("reset_done", "target_request"):
        endpoint = manifest["endpoints"].get(topic)
        rows[topic] = []
        if endpoint is None:
            continue
        path = Path(endpoint)
        name = f"source/{endpoint}"
        if (
            path.parent != Path("traces")
            or path.suffix != ".arrow"
            or path.stem.rsplit("__", 1)[-1] != topic
            or name not in manifest["artifacts_sha256"]
        ):
            raise ValueError("invalid ledger source endpoint")
        rows[topic] = dataset.strict_stream(root / name, root)
        if any(r["topic"] != topic for r in rows[topic]):
            raise ValueError("ledger source topic mismatch")
    records, references = [], []
    resets = {}
    for row in rows["reset_done"]:
        if row["data"] == [1.0]:
            resets.setdefault(row["env_id"], []).append(row["sim_time_ns"])
    for topic, source in rows.items():
        for row in source:
            env, stamp = row["env_id"], row["sim_time_ns"]
            boundaries = resets.get(env, [])
            if topic == "reset_done":
                if row["data"] != [1.0]:
                    continue
                episode = boundaries.index(stamp)
                kind = "reset"
            else:
                # Reject rather than order a request on a reset boundary.
                if stamp in boundaries:
                    raise ValueError("ambiguous timestamp: request on reset boundary")
                episode = sum(t < stamp for t in boundaries) - 1
                if episode < 0:
                    raise ValueError("request before initial reset")
                kind = "goal"
            ref = {k: row[k] for k in ("source", "row", "seq", "sim_time_ns", "env_id")}
            ref["sha256"] = manifest["artifacts_sha256"][row["source"]]
            record = {
                "id": f"{row['source']}:{row['row']}:{row['seq']}",
                "kind": kind,
                "sim_time_ns": stamp,
                "env_id": env,
                "episode": episode,
            }
            if kind == "goal":
                request = dataset.instruction(row)
                record.update(
                    goal_revision=record["id"], instruction=request["text"] if request else None
                )
            records.append(record)
            references.append(ref)
    order = sorted(
        range(len(records)), key=lambda i: (records[i]["sim_time_ns"], records[i]["env_id"])
    )
    report = _replay(
        [records[i] for i in order],
        [references[i] for i in order],
        max_events=max_events,
        max_invocations=max_invocations,
    )
    report.update(
        provenance="retained_dataset",
        source_sha256=dataset.sha256(root / "manifest.json"),
        coverage={
            "episodes": len(data["episodes"]),
            "identified_invocations": 0,
            "skill_attribution": "unknown",
            "reason": "missing_versioned_invocation_events",
        },
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="validated P1 bundle")
    source.add_argument("--fixture", type=Path, help="explicitly synthetic normalized events")
    parser.add_argument(
        "--out", type=Path, help="new JSON report file; omit to return report on stdout"
    )
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--max-invocations", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        limits = {"max_events": args.max_events, "max_invocations": args.max_invocations}
        if args.out and args.out.exists():
            raise ValueError("output exists; refusing overwrite")
        if args.dataset:
            if args.out and args.out.resolve().is_relative_to(args.dataset.resolve()):
                raise ValueError("output must be outside source dataset")
            report = replay_dataset(args.dataset, **limits)
        else:
            if args.fixture.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("fixture exceeds 64 MiB")
            report = replay_events(json.loads(args.fixture.read_text()), **limits)
        result = {"ok": True, "report": report}
        if args.out:
            with args.out.open("x") as out:
                out.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
            result = {"ok": True, "output": str(args.out), "sha256": dataset.sha256(args.out)}
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
