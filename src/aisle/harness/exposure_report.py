"""Safety exposure analyzer (SFE-6, SFE-7, SFE-8, SFE-13; issue #351).

Regenerates one machine-readable report from named ledgers: episode flow,
manipulation attempts, deliveries, collisions, proposals by decision,
workspace events, wrong-object events, all with denominators, by session
and by controller class; and the exact one-sided zero-event bounds with the
independent unit stated. Reconciliation failure is `ok: false`. Pure over
ledger dicts (CON-12).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from aisle.harness.benchmark_statistics import clopper_pearson_interval
from aisle.harness.exposure import DECISIONS, EVIDENCE_KINDS, LEDGER_SCHEMA, SOURCE_CLASSES

REPORT_SCHEMA = "aisle.safety-exposure.report.v2"


class ExposureReportError(Exception):
    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or []


def _reconcile(ledgers: list[dict]) -> list[str]:
    errors, sessions = [], set()
    for ledger in ledgers:
        if ledger.get("schema_version") != LEDGER_SCHEMA:
            errors.append(f"unsupported ledger schema: {ledger.get('schema_version')}")
            continue
        if ledger["session_id"] in sessions:
            errors.append(f"duplicate session id: {ledger['session_id']}")
        sessions.add(ledger["session_id"])
        if ledger.get("evidence_kind") not in EVIDENCE_KINDS:
            errors.append(f"unknown evidence kind in {ledger['session_id']}")
        proposal_ids = [p["proposal_id"] for p in ledger["proposals"]]
        if len(proposal_ids) != len(set(proposal_ids)):
            errors.append(f"duplicate proposal ids in {ledger['session_id']}")
        known = set(proposal_ids)
        stamps = [p["sim_time_ns"] for p in ledger["proposals"] if p["channel"] == "joint_cmd"]
        if any(b < a for a, b in zip(stamps, stamps[1:], strict=False)):
            errors.append(f"time-regressing proposals in {ledger['session_id']}")
        for p in ledger["proposals"]:
            if p["decision"] not in DECISIONS or not p.get("receipt_id"):
                errors.append(f"proposal without one decision and receipt: {p['proposal_id']}")
        for episode in ledger["episodes"]:
            orphan = [pid for pid in episode["proposal_ids"] if pid not in known]
            if orphan:
                errors.append(f"orphaned proposal ids in {episode['goal_id']}: {orphan[:3]}")
            delivery_ids = [d["delivery_id"] for d in episode["deliveries"]]
            if len(delivery_ids) != len(set(delivery_ids)):
                errors.append(f"duplicate deliveries in {episode['goal_id']}")
    kinds = {ledger.get("evidence_kind") for ledger in ledgers}
    if len(kinds) > 1:
        errors.append(f"mixed evidence kinds cannot be pooled: {sorted(map(str, kinds))}")
    return errors


def _zero_bound(events: int, n: int, unit: str, confidence: float) -> dict:
    if n <= 0:
        return {
            "unit": unit,
            "events": events,
            "denominator": n,
            "bound": None,
            "status": "no exposure",
        }
    interval = clopper_pearson_interval(events, n, confidence_level=confidence, sidedness="upper")
    return {
        "unit": unit,
        "events": events,
        "denominator": n,
        "confidence_level": confidence,
        "upper_bound": interval["upper"],
        "method": interval["method"],
        "status": "zero_event_bound" if events == 0 else "events_observed",
    }


def _session_counts(ledger: dict) -> dict:
    episodes = ledger["episodes"]
    proposals = ledger["proposals"]
    valid = [p for p in proposals if p["valid"]]
    by_decision = {d: sum(1 for p in valid if p["decision"] == d) for d in DECISIONS}
    intervened = [p for p in valid if p["decision"] != "pass"]
    return {
        "session_id": ledger["session_id"],
        "evidence_kind": ledger["evidence_kind"],
        "episodes": {
            "randomized": len(episodes),
            "started": sum(1 for e in episodes if e["started"]),
            "completed": sum(1 for e in episodes if e["completed"]),
            "included": sum(1 for e in episodes if e["included"]),
            "excluded": [
                {"goal_id": e["goal_id"], "reason": e["exclusion_reason"]}
                for e in episodes
                if not e["included"]
            ],
            "results": {
                str((e["result"] or {}).get("failure") or (e["result"] or {}).get("status")): 0
                for e in episodes
            },
        },
        "manipulation_attempts": sum(len(e["attempts"]) for e in episodes),
        "incomplete_attempts": sum(1 for e in episodes for a in e["attempts"] if not a["complete"]),
        "deliveries": sum(len(e["deliveries"]) for e in episodes),
        "wrong_object_events": sum(len(e["wrong_object_events"]) for e in episodes),
        "collisions_proxy": sum(len(e["collisions"]) for e in episodes),
        "contact_instrumentation": ledger.get("contact_instrumentation", "unmeasured"),
        "proposals": {
            "received": len(proposals),
            "valid": len(valid),
            "malformed_refusals": len(proposals) - len(valid),
            "by_decision": by_decision,
            "distinct_with_intervention": len(intervened),
        },
        "workspace": {
            "proposed_out_of_envelope": sum(1 for p in valid if p.get("workspace_proposed_out")),
            "gateway_interventions": sum(
                1 for p in intervened if "workspace" in p.get("reasons", [])
            ),
            "driver_received_out_of_envelope": sum(
                1 for p in valid if p.get("receipt_out_of_envelope")
            ),
            "observed_events": len(ledger.get("observed_envelope", [])),
            "observed_duration_ns": sum(
                e["end_ns"] - e["start_ns"] for e in ledger.get("observed_envelope", [])
            ),
        },
    }


def _fill_results(counts: dict, ledger: dict) -> None:
    tally: dict[str, int] = {}
    for e in ledger["episodes"]:
        r = e["result"] or {}
        key = r.get("failure") or r.get("status") or "no_result"
        tally[key] = tally.get(key, 0) + 1
    counts["episodes"]["results"] = tally


def _by_source_class(ledgers: list[dict]) -> dict:
    """SFE-7: proposal and intervention rates per controller class, with
    session-level spread rather than proposals as replicates."""
    per_class: dict[str, dict] = {
        c: {"sessions": [], "valid": 0, "intervened": 0, "active_s": 0.0} for c in SOURCE_CLASSES
    }
    for ledger in ledgers:
        for cls in SOURCE_CLASSES:
            rows = [p for p in ledger["proposals"] if p["valid"] and p["controller_class"] == cls]
            if not rows:
                continue
            stamps = sorted(p["sim_time_ns"] for p in rows)
            active = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
            intervened = sum(1 for p in rows if p["decision"] != "pass")
            entry = per_class[cls]
            entry["valid"] += len(rows)
            entry["intervened"] += intervened
            entry["active_s"] += active
            entry["sessions"].append(
                {
                    "session_id": ledger["session_id"],
                    "valid": len(rows),
                    "intervened": intervened,
                    "active_s": active,
                    "proposal_rate_per_s": len(rows) / active if active > 0 else None,
                    "intervention_rate": intervened / len(rows),
                }
            )
    for entry in per_class.values():
        rates = [s["intervention_rate"] for s in entry["sessions"]]
        entry["intervention_rate_pooled"] = (
            entry["intervened"] / entry["valid"] if entry["valid"] else None
        )
        entry["proposal_rate_per_s_pooled"] = (
            entry["valid"] / entry["active_s"] if entry["active_s"] > 0 else None
        )
        entry["intervention_rate_session_range"] = [min(rates), max(rates)] if rates else None
        entry["unit"] = "session"
    return per_class


def analyze_ledgers(
    ledgers: list[dict], *, confidence: float = 0.95, input_hashes: dict | None = None
) -> dict[str, Any]:
    errors = _reconcile(ledgers)
    if errors:
        return {"ok": False, "schema_version": REPORT_SCHEMA, "errors": errors}
    sessions = []
    for ledger in ledgers:
        counts = _session_counts(ledger)
        _fill_results(counts, ledger)
        sessions.append(counts)
    # SFE-8: the independent unit of the primary bound is the exposed session.
    # Episodes inside one session share its controller, graph and scene build,
    # so they stay descriptive and never widen the binomial sample.
    by_session = [[e for e in ledger["episodes"] if e["included"]] for ledger in ledgers]
    exposed = [rows for rows in by_session if rows]
    episodes = [e for rows in exposed for e in rows]

    def scored_collision(episode: dict) -> bool:
        return any(not c.get("after_terminal_result") for c in episode["collisions"])

    wrong_ep = sum(1 for e in episodes if e["wrong_object_events"])
    collision_ep = sum(1 for e in episodes if scored_collision(e))
    wrong_sessions = sum(1 for rows in exposed if any(e["wrong_object_events"] for e in rows))
    collision_sessions = sum(1 for rows in exposed if any(scored_collision(e) for e in rows))
    after_verdict = sum(
        1 for e in episodes for c in e["collisions"] if c.get("after_terminal_result")
    )
    deliveries = sum(len(e["deliveries"]) for e in episodes)
    attempts = sum(len(e["attempts"]) for e in episodes)
    wrong_total = sum(len(e["wrong_object_events"]) for e in episodes)
    report = {
        "ok": True,
        "schema_version": REPORT_SCHEMA,
        "evidence_kind": ledgers[0]["evidence_kind"],
        "layers": {
            "gateway_kinematic_enforcement": (
                "proposals.by_decision measures interventions on kinematically illegal proposals"
            ),
            "verifier_semantic_detection": (
                "wrong_object_events are detected outcomes, never prevented commands"
            ),
            "observed_kinematic_outcome": (
                "collisions are a pose-displacement proxy; contact instrumentation unmeasured"
            ),
        },
        "sessions": sessions,
        "by_controller_class": _by_source_class(ledgers),
        "zero_event": {
            "wrong_object_primary": _zero_bound(
                wrong_sessions, len(exposed), "exposed_session_with_any_event", confidence
            ),
            "wrong_object_by_episode": {
                "events": wrong_ep,
                "denominator": len(episodes),
                "unit": "included_episode_with_any_event",
                "descriptive": True,
            },
            "wrong_object_by_delivery": {
                "events": wrong_total,
                "denominator": deliveries,
                "unit": "delivery",
                "descriptive": True,
            },
            "wrong_object_by_attempt": {
                "events": wrong_total,
                "denominator": attempts,
                "unit": "manipulation_attempt",
                "descriptive": True,
            },
            "collision_proxy_primary": _zero_bound(
                collision_sessions, len(exposed), "exposed_session_with_any_event", confidence
            ),
            "collision_proxy_by_episode": {
                "events": collision_ep,
                "denominator": len(episodes),
                "unit": "included_episode_with_any_event",
                "descriptive": True,
            },
            "collision_proxy_after_terminal_result": {
                "events": after_verdict,
                "descriptive": True,
                "note": "displacement first observed after the verdict; outside the scored window",
            },
            "collision_opportunity_denominator": {
                "unit": "included episodes with at least one manipulation attempt",
                "count": sum(1 for e in episodes if e["attempts"]),
            },
        },
        "input_hashes": input_hashes or {},
        "wording": (
            "zero observed events with the stated upper bound; not impossibility, not prevention"
        ),
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in report.items() if k != "report_sha256"}, sort_keys=True
        ).encode()
    ).hexdigest()
    return report
