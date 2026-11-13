"""P3 measurement-only goal chains within physical reset episodes (CON-5/8, HAR-4/6).

This companion format retains privileged goals/verdicts and MUST NOT be supplied
to either policy arm. It does not change P1 or create skill invocation evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from demo_dataset import resolve_stream, sha256, strict_stream, write_json

from aisle.harness.trace_recorder import trace_row_digest

TOPICS = ("reset_done", "episode_goal", "episode_result")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def json_lines(path):
    text = path.read_text()
    require(not text or text.endswith("\n"), f"incomplete JSONL: {path.name}")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), f"non-object JSONL: {path.name}")
    return rows


def build(root, endpoints):
    """Derive from immutable source bytes, without consulting derived outcomes."""
    source = root / "source"
    streams = {}
    for topic in TOPICS:
        endpoint = endpoints.get(topic)
        if topic == "episode_result" and endpoint is None:
            streams[topic] = []
            continue
        require(
            isinstance(endpoint, str)
            and Path(endpoint).parent == Path("traces")
            and Path(endpoint).name.endswith(f"__{topic}.arrow"),
            "invalid source endpoint",
        )
        streams[topic] = strict_stream(source / endpoint, source)
    identities = {}
    for row in json_lines(source / "traces/identities.jsonl"):
        require(
            set(row) == {"format", "version", "endpoint", "row", "trace_row_sha256", "metadata"}
            and row["format"] == "aisle_trace_identity"
            and type(row["version"]) is int
            and row["version"] == 1
            and nonempty(row["endpoint"])
            and type(row["row"]) is int
            and row["row"] >= 0
            and isinstance(row["metadata"], dict),
            "invalid identity sidecar",
        )
        key = (row["endpoint"], row["row"])
        require(key not in identities, "duplicate identity sidecar row")
        identities[key] = row

    def identity(row):
        evidence = identities.get((Path(row["source"]).stem, row["row"]))
        require(
            evidence is not None and evidence["trace_row_sha256"] == trace_row_digest(row),
            "missing or conflicting source row identity",
        )
        return evidence["metadata"]

    def payload(row):
        value = json.loads(row["text"])
        require(isinstance(value, dict), "expected JSON object payload")
        return value

    def ref(row):
        return {key: row[key] for key in ("source", "row", "seq", "sim_time_ns", "env_id")}

    resets = {}
    for row in streams["reset_done"]:
        if row["data"] == [1.0]:
            identity(row)
            resets.setdefault(row["env_id"], []).append(row)
    require(
        resets and all(rows[0]["sim_time_ns"] == 0 for rows in resets.values()),
        "missing initial physical reset",
    )
    records = json_lines(source / "episodes.jsonl")
    outcomes = {}
    for record in records:
        require(
            type(record.get("episode")) is int
            and record["episode"] >= 0
            and nonempty(record.get("goal_id")),
            "outcome requires episode and goal identity",
        )
        require("env_id" in record or len(resets) == 1, "ambiguous outcome environment")
        env = record.get("env_id", next(iter(resets)))
        require(type(env) is int and env in resets, "invalid outcome environment")
        key = (env, record["goal_id"])
        require(key not in outcomes, "duplicate goal outcome")
        outcomes[key] = record

    goals, results = {}, {}
    for topic, target in (("episode_goal", goals), ("episode_result", results)):
        for row in streams[topic]:
            meta, value = identity(row), payload(row)
            goal_id = meta.get("goal_id", value.get("goal_id"))
            require(nonempty(goal_id), "missing captured goal identity")
            require(
                "goal_id" not in value or value["goal_id"] == goal_id,
                "payload and captured goal identity disagree",
            )
            key = (row["env_id"], goal_id)
            require(key not in target, "duplicate source goal identity")
            target[key] = (row, value)
    require(
        set(results) <= set(goals) and set(outcomes) <= set(results),
        "outcome or verdict has no matching source goal",
    )
    episodes, used = [], set()
    for env, boundaries in sorted(resets.items()):
        for episode, reset in enumerate(boundaries):
            start = reset["sim_time_ns"]
            end = boundaries[episode + 1]["sim_time_ns"] if episode + 1 < len(boundaries) else None
            episode_goals = [
                (key, row, goal)
                for key, (row, goal) in goals.items()
                if key[0] == env
                and start <= row["sim_time_ns"]
                and (end is None or row["sim_time_ns"] < end)
            ]
            episode_goals.sort(key=lambda item: item[1]["sim_time_ns"])
            chain, previous_result, seed = [], None, None
            for ordinal, (key, row, goal) in enumerate(episode_goals):
                used.add(key)
                outcome = outcomes.get(key)
                result = results.get(key)
                require(
                    type(goal.get("seed")) is int and type(goal.get("reset_sim_ns")) is int,
                    "goal lacks seed/start evidence",
                )
                if ordinal == 0:
                    seed = goal["seed"]
                    require(
                        goal["reset_sim_ns"] == start and not goal.get("expects_return", False),
                        "initial goal reset evidence conflicts with physical window",
                    )
                else:
                    require(
                        previous_result is not None
                        and goal.get("expects_return") is True
                        and goal["reset_sim_ns"] == previous_result["sim_time_ns"]
                        and row["sim_time_ns"] >= previous_result["sim_time_ns"],
                        "recovery start evidence conflicts with preceding verdict",
                    )
                require(goal["seed"] == seed, "goal seed changes within physical episode")
                if outcome is not None:
                    require(
                        outcome["episode"] == episode
                        and type(outcome.get("seed")) is int
                        and outcome["seed"] == seed,
                        "outcome episode/seed conflicts with physical window",
                    )
                    require(
                        outcome.get("recovery", False) is (ordinal > 0),
                        "missing or conflicting recovery marker",
                    )
                if result is not None:
                    result_row, verdict = result
                    require(
                        (
                            type(verdict.get("seed")) is int
                            or "seed" not in verdict
                            and verdict.get("verifier") == "realistic"
                        )
                        and verdict.get("status") in ("success", "fail")
                        and verdict.get("verifier") in ("oracle", "realistic")
                        and (
                            verdict.get("failure") is None
                            if verdict.get("status") == "success"
                            else nonempty(verdict.get("failure"))
                        ),
                        "invalid source verdict",
                    )
                    require(
                        row["sim_time_ns"] <= result_row["sim_time_ns"]
                        and (end is None or result_row["sim_time_ns"] < end),
                        "verdict outside its goal/physical episode",
                    )
                    require(
                        ("seed" not in verdict or verdict["seed"] == seed)
                        and verdict.get("goal_id") == key[1],
                        "verdict identity/seed conflict",
                    )
                    if outcome is not None:
                        require(
                            all(k in outcome and outcome[k] == v for k, v in verdict.items()),
                            "episode record differs from source verdict",
                        )
                    previous_result = result_row
                else:
                    result_row, verdict, previous_result = None, None, None
                chain.append(
                    dict(
                        goal_id=key[1],
                        ordinal=ordinal,
                        physical_reset_ns=start,
                        goal_reset_sim_ns=goal["reset_sim_ns"],
                        goal=goal,
                        goal_evidence=ref(row),
                        verdict=verdict,
                        verdict_evidence=ref(result_row) if result_row else None,
                        outcome=outcome,
                        completeness="complete"
                        if outcome is not None
                        else "missing_episode_record"
                        if verdict is not None
                        else "unscored",
                    )
                )
            episodes.append(
                dict(
                    env_id=env,
                    episode=episode,
                    window_ns=[start, end],
                    reset_evidence=ref(reset),
                    goals=chain,
                )
            )
    require(used == set(goals), "source goal outside physical reset windows")
    return episodes


def coverage(episodes):
    goals = [g for e in episodes for g in e["goals"]]
    return dict(
        physical_resets=len(episodes),
        goals=len(goals),
        verdict_records=sum(g["outcome"] is not None for g in goals),
        unscored_goals=sum(g["outcome"] is None for g in goals),
    )


def export_chains(run, out, endpoints=None):
    run, out = Path(run), Path(out)
    require(
        not out.exists() and not out.resolve().is_relative_to(run.resolve()),
        "output must be a new directory outside source run",
    )
    require(
        endpoints is None or isinstance(endpoints, dict) and set(endpoints) <= set(TOPICS),
        "invalid endpoint selection",
    )
    chosen = {t: resolve_stream(run, t, endpoints or {}) for t in TOPICS}
    require(
        chosen["reset_done"] is not None and chosen["episode_goal"] is not None,
        "required reset/goal endpoint missing",
    )
    selection = {t: str(p.relative_to(run)) if p else None for t, p in chosen.items()}
    paths = [
        run / "manifest.json",
        run / "episodes.jsonl",
        run / "traces/identities.jsonl",
        *[p for p in chosen.values() if p is not None],
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".goal-chains-", dir=out.parent) as temp:
        root = Path(temp) / "bundle"
        for path in paths:
            require(
                not path.is_symlink() and path.resolve().is_relative_to(run.resolve()),
                "source path escapes run",
            )
            target = root / "source" / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha256(path) == sha256(target), "source changed during snapshot")
        episodes = build(root, selection)
        write_json(root / "episodes.json", episodes)
        manifest = dict(
            format="aisle_goal_chains",
            version=1,
            policy_visible=False,
            evidence_kind="measurement_only_not_invocation_or_training_evidence",
            endpoints=selection,
            coverage=coverage(episodes),
            artifacts_sha256={
                str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
            },
        )
        write_json(root / "manifest.json", manifest)
        read_chains(root)
        root.rename(out)
    return manifest


def read_chains(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        isinstance(manifest, dict)
        and manifest.get("format") == "aisle_goal_chains"
        and type(manifest.get("version")) is int
        and manifest["version"] == 1
        and manifest.get("policy_visible") is False,
        "unsupported goal-chain format",
    )
    artifacts = manifest.get("artifacts_sha256")
    require(isinstance(artifacts, dict), "missing source artifact hashes")
    required = {
        "episodes.json",
        "source/manifest.json",
        "source/episodes.jsonl",
        "source/traces/identities.jsonl",
    }
    selection = manifest.get("endpoints")
    require(
        isinstance(selection, dict)
        and set(selection) == set(TOPICS)
        and all(
            isinstance(p, str) or t == "episode_result" and p is None for t, p in selection.items()
        ),
        "invalid source endpoints",
    )
    required.update(f"source/{p}" for p in selection.values() if p is not None)
    require(required <= set(artifacts), "missing source artifact hashes")
    for name, digest in artifacts.items():
        path = root / name
        require(
            path.resolve().is_relative_to(root.resolve())
            and path.is_file()
            and not path.is_symlink()
            and sha256(path) == digest,
            "source artifact hash mismatch",
        )
    episodes = json.loads((root / "episodes.json").read_text())
    require(episodes == build(root, selection), "derived chains differ from source evidence")
    require(manifest.get("coverage") == coverage(episodes), "coverage differs from source evidence")
    return dict(manifest=manifest, episodes=episodes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = export_chains(
            args.run, args.out, json.loads(args.endpoints.read_text()) if args.endpoints else None
        )
        result = dict(ok=True, output=str(args.out), coverage=manifest["coverage"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(ok=False, error=str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
