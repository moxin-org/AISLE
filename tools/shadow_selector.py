"""P4 synthetic shadow replay, with no runtime or actuation authority (CON-5/8).

Only models interface decisions. Reliability and observations are fixture inputs,
not measured policy evidence. A proposal has never traversed the budget guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from execution_ledger import digest, finite_number, nonempty, validate_record

from aisle.nodes.vla_smolvla import VLA_STALE_NS, ChunkQueue

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = {
    "ik-transfer-v2": ("skill", "skills/ik-transfer-v2/ik_transfer_v2.py"),
    "vla-policy-smolvla": ("learned", "src/aisle/nodes/vla_smolvla.py"),
}
COMMON = {"id", "kind", "sim_time_ns", "env_id", "episode", "goal_revision"}
FIELDS = {
    "goal": {"instruction"},
    "observation": {"obs_ns"},
    "select": {"stage", "remaining_budget_ns", "applicable", "parameters"},
    "reply": {"request_id", "candidate", "version", "chunk"},
    "interrupt": {"request_id"},
    "boundary": set(),
}


def adapter_pin(name):
    """Pin manifest and direct adapter source; NOT a transitive policy/weight pin."""
    _, source = ADAPTERS[name]
    files = [f"registry/manifests/{name}.yaml", source]
    return "sha256:" + digest(
        {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    )


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def exact(value, fields):
    return isinstance(value, dict) and set(value) == set(fields)


def names(value):
    return (
        isinstance(value, list)
        and all(nonempty(x) for x in value)
        and len(set(value)) == len(value)
    )


def parameters_valid(name, value):
    if ADAPTERS[name][0] == "learned":
        return value == {}
    if not exact(value, {"grasp_pose"}):
        return False
    pose = value["grasp_pose"]
    return (
        isinstance(pose, list)
        and len(pose) == 7
        and all(finite_number(x) for x in pose)
        and abs(sum(x * x for x in pose[3:]) - 1) <= 1e-4
    )


def chunk_valid(chunk):
    return (
        isinstance(chunk, list)
        and 1 <= len(chunk) <= 128
        and all(
            exact(a, {"joints", "gripper"})
            and isinstance(a["joints"], list)
            and len(a["joints"]) == 7
            and all(finite_number(x) for x in a["joints"])
            and finite_number(a["gripper"])
            and 0 <= a["gripper"] <= 1
            for a in chunk
        )
    )


def validate(document):
    if (
        not exact(document, {"format", "version", "provenance", "candidates", "records"})
        or document["format"] != "aisle_shadow_events"
        or type(document["version"]) is not int
        or document["version"] != 1
        or document["provenance"] != "synthetic"
    ):
        raise ValueError("expected version 1 explicitly synthetic shadow fixture")
    # Reject NaN/Infinity even in otherwise uninterpreted payloads.
    source_hash = digest(document)
    candidates = document["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= len(ADAPTERS):
        raise ValueError("expected one or two allowlisted candidates")
    seen = set()
    for c in candidates:
        if (
            not exact(c, {"id", "version", "stages", "reliability", "max_attempts", "timeout_ns"})
            or not nonempty(c["id"])
            or c["id"] not in ADAPTERS
            or c["id"] in seen
            or c["version"] != adapter_pin(c["id"])
            or not names(c["stages"])
            or not c["stages"]
            or not integer(c["max_attempts"], 1)
            or not integer(c["timeout_ns"], 1)
        ):
            raise ValueError("invalid, duplicate, unpinned or non-allowlisted candidate")
        r = c["reliability"]
        if r is not None and (
            not exact(r, {"successes", "trials"})
            or not integer(r["trials"], 1)
            or not integer(r["successes"])
            or r["successes"] > r["trials"]
        ):
            raise ValueError("invalid reliability counts")
        seen.add(c["id"])
    records = document["records"]
    if not isinstance(records, list) or len(records) > 10000:
        raise ValueError("expected at most 10000 records")
    identities, previous = {}, -1
    for row in records:
        if not isinstance(row, dict) or not nonempty(row.get("id")):
            raise ValueError("invalid event identity")
        fingerprint = digest(row)
        if row["id"] in identities:
            if identities[row["id"]] != fingerprint:
                raise ValueError("conflicting event identity")
            continue
        identities[row["id"]] = fingerprint
        kind = row.get("kind")
        if not isinstance(kind, str):
            raise ValueError("invalid event kind")
        if kind in {"reset", "goal"}:
            validate_record(row)  # P2's explicit reset/goal evidence contract.
        elif kind not in FIELDS or not exact(row, COMMON | FIELDS[kind]):
            raise ValueError("unexpected event kind or fields")
        if (
            type(row["env_id"]) is not int
            or row["env_id"] != 0
            or not integer(row["episode"])
            or not integer(row["sim_time_ns"])
            or row["sim_time_ns"] <= previous
            or (kind != "reset" and not nonempty(row["goal_revision"]))
        ):
            raise ValueError("expected env 0, valid identity, strictly increasing arrival time")
        previous = row["sim_time_ns"]
        if kind == "observation" and not integer(row["obs_ns"]):
            raise ValueError("invalid observation timestamp")
        if kind == "select" and (
            not nonempty(row["stage"])
            or not integer(row["remaining_budget_ns"])
            or not names(row["applicable"])
            or not set(row["applicable"]) <= seen
            or not isinstance(row["parameters"], dict)
            or not set(row["parameters"]) <= seen
        ):
            raise ValueError("invalid selection context")
        if kind in {"reply", "interrupt"} and not nonempty(row["request_id"]):
            raise ValueError("invalid request identity")
        if kind == "reply" and (not nonempty(row["candidate"]) or not nonempty(row["version"])):
            raise ValueError("invalid reply adapter identity")
    return source_hash


class Shadow:
    """One offline environment; all time and policy context supplied by the fixture."""

    def __init__(self, candidates):
        self.candidates = sorted(candidates, key=lambda c: c["id"])
        self.queue = ChunkQueue()
        self.episode = -1
        self.goal = None
        self.goals = set()
        self.intent_ns = -1
        self.obs_ns = None
        self.observe_after = -1
        self.pending = None
        self.attempts = {}

    def cancel(self, now):
        previous = self.pending
        if previous and previous["kind"] == "skill" and previous["started"]:
            self.observe_after = now
        self.pending = None
        self.queue.flush()
        return previous["request_id"] if previous else None

    def apply(self, row):
        kind, now = row["kind"], row["sim_time_ns"]
        if kind == "reset":
            if row["episode"] != self.episode + 1 or (self.episode == -1 and now != 0):
                return dict(decision="rejected", reason="invalid_reset_boundary")
            cancelled = self.cancel(now)
            self.episode, self.goal, self.obs_ns = row["episode"], None, None
            self.goals, self.attempts = set(), {}
            self.intent_ns, self.observe_after = now, now
            return dict(decision="accepted", reason="reset", cancelled_request=cancelled)
        if row["episode"] != self.episode:
            return dict(decision="rejected", reason="wrong_episode")
        if kind == "goal":
            if row["goal_revision"] in self.goals:
                return dict(decision="rejected", reason="reused_goal_revision")
            cancelled = self.cancel(now)
            self.goals.add(row["goal_revision"])
            self.goal = row["goal_revision"] if row["instruction"] is not None else None
            self.obs_ns, self.attempts = None, {}
            self.intent_ns, self.observe_after = now, now
            return dict(decision="accepted", reason="goal", cancelled_request=cancelled)
        if self.goal is None or row["goal_revision"] != self.goal:
            return dict(decision="rejected", reason="wrong_goal")
        if kind == "observation":
            stamp = row["obs_ns"]
            if (
                stamp > now
                or stamp <= self.intent_ns
                or (self.obs_ns is not None and stamp <= self.obs_ns)
            ):
                return dict(decision="rejected", reason="invalid_observation_time")
            self.obs_ns = stamp
            return dict(decision="accepted", reason="observation_recorded")
        if kind == "select":
            return self.select(row)
        p = self.pending
        if kind in {"reply", "interrupt"}:
            if p is None:
                return dict(decision="rejected", reason="no_pending_request")
            if row["request_id"] != p["request_id"]:
                return dict(decision="rejected", reason="wrong_request")
        if kind == "interrupt":
            return dict(decision="hold", reason="interrupted", cancelled_request=self.cancel(now))
        if kind == "reply":
            if row["candidate"] != p["candidate"] or row["version"] != p["version"]:
                return dict(decision="rejected", reason="wrong_adapter")
            if p["kind"] == "skill":
                if not p["started"] or row["chunk"] is not None:
                    return dict(decision="rejected", reason="invalid_skill_return")
                return dict(
                    decision="hold",
                    reason="returned_unconfirmed",
                    cancelled_request=self.cancel(now),
                )
            if p["replied"]:
                return dict(decision="rejected", reason="duplicate_reply")
            if not chunk_valid(row["chunk"]):
                return dict(decision="rejected", reason="invalid_chunk")
            if not self.queue.offer(row["chunk"], p["obs_ns"], now):
                return dict(
                    decision="hold",
                    reason="stale_observation_hold",
                    cancelled_request=self.cancel(now),
                )
            p["replied"] = True
            return dict(decision="accepted", reason="chunk_queued")
        # All remaining validated events are motion boundaries.
        if p is None:
            return dict(decision="hold", reason="idle_hold")
        if p["kind"] == "skill" and p["started"]:
            return dict(decision="hold", reason="skill_in_progress")
        if now - p["obs_ns"] > VLA_STALE_NS:
            return dict(
                decision="hold", reason="stale_observation_hold", cancelled_request=self.cancel(now)
            )
        proposal = dict(
            request_id=p["request_id"],
            candidate=p["candidate"],
            version=p["version"],
            env_id=0,
            episode=self.episode,
            goal_revision=self.goal,
            obs_ns=p["obs_ns"],
            required_motion_route="budget_guard_inputs",
            actuation_authority=False,
        )
        if p["kind"] == "skill":
            p["started"] = True
            proposal.update(kind="skill_invocation", parameters=p["parameters"])
        else:
            action = self.queue.pop()
            if action is None:
                return dict(decision="hold", reason="awaiting_reply")
            proposal.update(kind="learned_action", action=action)
            if not self.queue.chunk:
                self.cancel(now)
        return dict(decision="proposed", reason="shadow_boundary", proposal=proposal)

    def select(self, row):
        if self.pending:
            return dict(decision="hold", reason="controller_busy")
        now = row["sim_time_ns"]
        if self.obs_ns is None or self.obs_ns <= self.observe_after:
            return dict(decision="hold", reason="observation_required")
        if now - self.obs_ns > VLA_STALE_NS:
            return dict(decision="hold", reason="stale_observation_hold")
        evaluations, eligible = [], []
        for c in self.candidates:
            name, r = c["id"], c["reliability"]
            reasons = []
            for rejected, reason in [
                (name not in row["applicable"], "not_applicable"),
                (row["stage"] not in c["stages"], "stage_mismatch"),
                (self.attempts.get(name, 0) >= c["max_attempts"], "attempts_exhausted"),
                (c["timeout_ns"] > row["remaining_budget_ns"], "insufficient_budget"),
                (r is None, "unknown_reliability"),
                (r is not None and 2 * r["successes"] < r["trials"], "below_reliability_floor"),
                (not parameters_valid(name, row["parameters"].get(name)), "invalid_parameters"),
            ]:
                if rejected:
                    reasons.append(reason)
            evaluations.append(dict(candidate=name, reasons=reasons))
            if not reasons:
                eligible.append(c)
        if not eligible:
            return dict(decision="hold", reason="no_eligible_candidate", candidates=evaluations)
        chosen = min(eligible, key=lambda c: (ADAPTERS[c["id"]][0] != "skill", c["id"]))
        name = chosen["id"]
        for evaluation in evaluations:
            if not evaluation["reasons"]:
                evaluation["reasons"] = [
                    "selected_fixed_rule"
                    if evaluation["candidate"] == name
                    else "lower_fixed_priority"
                ]
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.pending = dict(
            request_id=row["id"],
            candidate=name,
            version=chosen["version"],
            kind=ADAPTERS[name][0],
            obs_ns=self.obs_ns,
            deadline_ns=now + chosen["timeout_ns"],
            started=False,
            replied=False,
            parameters=row["parameters"][name],
        )
        return dict(
            decision="selected",
            reason="fixed_skill_first",
            selected=name,
            request_id=row["id"],
            candidates=evaluations,
        )


def replay(document):
    source_hash = validate(document)
    state = Shadow(document["candidates"])
    timeline, seen = [], set()
    for index, row in enumerate(document["records"]):
        if row["id"] in seen:
            result = dict(decision="rejected", reason="duplicate_event")
        else:
            seen.add(row["id"])
            # Every new arrival advances the watchdog, even a misattributed reply.
            # The deadline event is consumed as a hold; it cannot also select/move.
            if state.pending and row["sim_time_ns"] >= state.pending["deadline_ns"]:
                cancelled = state.cancel(row["sim_time_ns"])
                if row["kind"] in {"reset", "goal"}:
                    result = state.apply(row) | {"timed_out_request": cancelled}
                else:
                    result = dict(
                        decision="hold", reason="timeout_hold", cancelled_request=cancelled
                    )
            else:
                result = state.apply(row)
        timeline.append(
            dict(
                event_id=row["id"],
                row=index,
                event_sha256=digest(row),
                sim_time_ns=row["sim_time_ns"],
                **result,
                owner=state.pending["request_id"] if state.pending else None,
            )
        )
    return dict(
        ok=True,
        format="aisle_shadow_report",
        version=1,
        provenance="synthetic",
        source_sha256=source_hash,
        actuation_authority=False,
        guard_traversal="unverified",
        task_success=None,
        reliability_scope="synthetic_fixture_inputs",
        rule={
            "priority": ["skill", "learned"],
            "min_success_fraction": "1/2",
            "stale_ns": VLA_STALE_NS,
            "calibrated": False,
        },
        timeline=timeline,
        pending=state.pending,
        attempts=state.attempts,
    )


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    try:
        args = parser.parse_args(argv)
        if args.fixture.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("fixture exceeds 4 MiB")
        result = replay(json.loads(args.fixture.read_text()))
        if args.out:
            with args.out.open("x") as stream:
                stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
            result = dict(ok=True, output=str(args.out), source_sha256=result["source_sha256"])
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        result = dict(ok=False, error=str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
