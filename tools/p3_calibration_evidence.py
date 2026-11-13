"""Source-bound, controller-only P3 development calibration assessment."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import goal_chains
import yaml
from p3_evidence import digest, export_policy, write_json


def audit_chain(episodes, journal, seed):
    physical = [e for e in episodes if e["goals"]]
    if len(physical) != 1 or len(physical[0]["goals"]) != 2:
        raise ValueError("expected one complete two-goal chain")
    goals = physical[0]["goals"]
    if any(g["completeness"] != "complete" or g["verdict"] is None for g in goals):
        raise ValueError("incomplete scorer evidence")
    if any(g["goal"]["seed"] != seed or g["verdict"]["verifier"] != "oracle" for g in goals):
        raise ValueError("wrong seed or non-oracle scoring evidence")
    if goals[1]["goal"].get("expects_return") is not True:
        raise ValueError("second goal is not a return/redelivery task")
    goal_ids = {g["goal_id"] for g in goals}
    proposals = {e["invocation_id"]: e for e in journal["events"] if e["kind"] == "proposed"}
    if any(e["goal_revision"] not in goal_ids for e in proposals.values()):
        raise ValueError("invocation lacks a captured chain goal identity")
    started = {e["invocation_id"]: e for e in journal["events"] if e["kind"] == "started"}
    guarded = {e["invocation_id"] for e in journal["forwarded"]}
    witnessed = [proposals[i] for i in started if i in guarded]
    normal = "grasp-planner-topdown/grasp_pose"
    initial = [
        e
        for e in witnessed
        if e["goal_revision"] == goals[0]["goal_id"] and e["source"]["endpoint"] == normal
    ]
    returns = [
        e
        for e in witnessed
        if e["goal_revision"] == goals[1]["goal_id"]
        and e["source"]["endpoint"] == "return-planner/grasp_pose"
    ]
    redelivery = [
        e
        for e in witnessed
        if e["goal_revision"] == goals[1]["goal_id"] and e["source"]["endpoint"] == normal
    ]
    sequence = goals[0]["verdict"]["status"] == "success" and any(
        started[a["invocation_id"]]["event_seq"]
        < started[b["invocation_id"]]["event_seq"]
        < started[c["invocation_id"]]["event_seq"]
        for a in initial
        for b in returns
        for c in redelivery
    )
    return dict(
        seed=seed,
        status="complete",
        success=all(g["verdict"]["status"] == "success" for g in goals),
        goal_ids=[g["goal_id"] for g in goals],
        verdicts=[g["verdict"] for g in goals],
        failures=dict(
            Counter(g["verdict"]["failure"] for g in goals if g["verdict"]["status"] == "fail")
        ),
        proposed_invocations=len(proposals),
        started_invocations=len(started),
        guarded_invocations=len(set(started) & guarded),
        recovery_sequence=sequence,
    )


def screen(records, seeds):
    if len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds):
        raise ValueError("invalid planned seeds")
    seen = [r["seed"] for r in records]
    if len(seen) != len(set(seen)) or not set(seen) <= set(seeds):
        raise ValueError("duplicate or unplanned seed")
    missing = [s for s in seeds if s not in seen]
    failures = [r["seed"] for r in records if r["status"] != "complete"]
    if failures or missing:
        return dict(
            status="instrument_failure" if failures else "incomplete",
            eligible=False,
            successes=None,
            failures=None,
            unstarted_seeds=missing,
            invalid_seeds=failures,
        )
    if any(
        type(r.get("success")) is not bool or type(r.get("recovery_sequence")) is not bool
        for r in records
    ):
        raise ValueError("missing measured calibration outcome")
    successes = sum(r["success"] for r in records)
    eligible = (
        len(seeds) == 12 and 2 <= successes <= 10 and any(r["recovery_sequence"] for r in records)
    )
    return dict(
        status="eligible" if eligible else "ineligible",
        eligible=eligible,
        successes=successes,
        failures=len(seeds) - successes,
        unstarted_seeds=[],
        invalid_seeds=[],
    )


def assess_run(run, out, *, seed, instrument_commit, graph_sha256, env_fingerprint, skill_sources):
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not isinstance(env_fingerprint, str) or not env_fingerprint or not skill_sources:
        raise ValueError("calibration requires environment and skill source pins")
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        manifest.get("git_sha") != instrument_commit
        or manifest.get("graph_hash") != graph_sha256
        or digest(run / "authored-graph.yaml") != graph_sha256
    ):
        raise ValueError("calibration instrument/graph pin mismatch")
    if (
        manifest.get("env_fingerprint") != env_fingerprint
        or manifest.get("sim_engine") != "genesis"
        or manifest.get("perception") != "L1"
        or manifest.get("env_attested") is not True
        or manifest.get("seeds") != [seed]
        or manifest.get("relaunches") != 0
        or manifest.get("wall_clamped")
    ):
        raise ValueError("calibration run is not a complete attested L1 Genesis chain")
    out.mkdir(parents=True)
    try:
        goal_chains.export_chains(
            run,
            out / "measurement",
            endpoints={
                "reset_done": "reset__reset_done.arrow",
                "episode_goal": "rollout-client__episode_goal.arrow",
                "episode_result": "verifier-oracle__episode_result.arrow",
            },
        )
        chains = goal_chains.read_chains(out / "measurement")
        public = export_policy(run, out / "policy")
        if public["journal"]["header"]["skill"].get("sources") != skill_sources:
            raise ValueError("calibration skill source pin mismatch")
        result = audit_chain(chains["episodes"], public["journal"], seed)
        result.update(
            source_manifest_sha256=digest(run / "manifest.json"),
            measurement_manifest_sha256=digest(out / "measurement/manifest.json"),
            policy_manifest_sha256=digest(out / "policy/manifest.json"),
            instrument_commit=instrument_commit,
            graph_sha256=graph_sha256,
            skill=public["journal"]["header"]["skill"],
            env_fingerprint=manifest.get("env_fingerprint"),
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, zipfile.BadZipFile) as exc:
        result = dict(seed=seed, status="instrument_failure", success=None, error=str(exc))
        write_json(out / "assessment.json", result)
        raise
    write_json(out / "assessment.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--instrument-commit", required=True)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--env-fingerprint", required=True)
    parser.add_argument("--skill-sources", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = assess_run(
            args.run,
            args.out,
            seed=args.seed,
            instrument_commit=args.instrument_commit,
            graph_sha256=args.graph_sha256,
            env_fingerprint=args.env_fingerprint,
            skill_sources=json.loads(args.skill_sources.read_text()),
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, zipfile.BadZipFile) as exc:
        print(json.dumps(dict(ok=False, error=str(exc))))
        return 1
    print(json.dumps(dict(ok=True, assessment=result), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
