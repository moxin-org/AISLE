"""Skill registration (T18; design doc §8.4 item 1, §3 rule 3, §9.2;
CAP-5 as amended by the curated-core spec change).

A skill is a directory shipping everything the library needs (§9.2):
`skill.yaml` (a CAP-1 manifest, origin agent-authored, eval null until
registered), its node code / subgraph YAML, and `eval.yaml` — the
mini-rollout config that IS the skill's eval suite, whose graph MUST use
the candidate (a node with the skill's id).

Registration is a STAGED transaction (PR #30 review): the candidate
manifest — carrying a clearly-labelled provisional evalcard so CAP-6's
motion gate can pass during its own evaluation — is installed into the
registry FIRST, the whole registry is linted (CAP-2/3), the shipped eval
rollout runs against the STAGED candidate (an update evaluates the NEW
manifest, never the old one; a fresh candidate is discoverable instead
of MANIFEST_MISSING), and only a measured pass rate at or above the
skill's own `min_pass_rate` finalizes the evalcard. ANY failure rolls
the registry back to its prior state exactly.

That declared bar is the CANDIDATE's number, so it is floored by
`REGISTRY_MIN_PASS_RATE` (CAP-6, ADR-37): a skill may hold itself to
more than the registry demands, never to less.

Governance (§9.4 + CAP-5): curated-core ids are refused from the
single-sourced `registry/schema/curated_core.toml` REGARDLESS of current
file state (deleting a core manifest does not open its id); origin must
be agent-authored; the PR a human merges is the trust boundary.

Pure logic; the rollout runner is injected (CON-12/CON-5: unit tests
never touch sim, `last_run` comes from an injected clock). Every refusal
is a RegistrationError the CLI surfaces as {ok: false, error} (CON-8) —
malformed YAML, bad seed specs, and unparseable configs included.
"""

from __future__ import annotations

import math
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from aisle.harness.registry import lint, load_capability_schema, manifest_schema_errors
from aisle.harness.rollout import parse_seed_range

EVAL_REQUIRED = ("suite", "graph", "tier", "episodes", "seeds", "embodiment", "min_pass_rate")
CURATED_CORE = "registry/schema/curated_core.toml"

# CAP-6 (ADR-37, #243): the ABSOLUTE floor under every skill's self-declared
# `min_pass_rate`. Without it the exam is self-graded — a candidate shipping
# 0.0 registers at 0.0 and the gate reports ok, which is how `t2-scan-tsm`
# entered a campaign library. A skill MAY hold itself to more; never to less.
REGISTRY_MIN_PASS_RATE = 0.5


class RegistrationError(RuntimeError):
    """A refused registration — the reason is the message (CON-8: the CLI
    surfaces it as {ok: false, error})."""


@dataclass(frozen=True)
class Skill:
    path: Path
    manifest: dict
    eval_cfg: dict


def _load_yaml(path: Path, what: str) -> dict:
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as bad:
        raise RegistrationError(f"{what} is not valid YAML: {bad}") from bad
    if not isinstance(doc, dict):
        raise RegistrationError(f"{what} is not a mapping")
    return doc


def _parse_pass_rate(value: object, field: str) -> float:
    error = f"{field} invalid: expected a finite number in [0, 1]"
    try:
        rate = float(value)
    except (ValueError, TypeError, OverflowError) as bad:
        raise RegistrationError(error) from bad
    if isinstance(value, bool) or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise RegistrationError(error)
    return rate


def load_skill(skill_dir: Path) -> Skill:
    """Read the skill directory's manifest and shipped eval suite."""
    manifest_path = skill_dir / "skill.yaml"
    if not manifest_path.exists():
        raise RegistrationError(f"no skill.yaml in {skill_dir}")
    manifest = _load_yaml(manifest_path, "skill.yaml")
    eval_path = skill_dir / "eval.yaml"
    if not eval_path.exists():
        raise RegistrationError(
            "no eval.yaml: a skill SHIPS its eval suite (design doc §3 rule 3) — "
            "an unevaluated skill cannot be registered (CAP-6)"
        )
    eval_cfg = _load_yaml(eval_path, "eval.yaml")
    missing = [k for k in EVAL_REQUIRED if k not in eval_cfg]
    if missing:
        raise RegistrationError(f"eval.yaml missing keys: {missing}")
    # CON-8: malformed eval fields refuse cleanly, never traceback
    try:
        parse_seed_range(str(eval_cfg["seeds"]))
        int(eval_cfg["episodes"])
        declared = _parse_pass_rate(eval_cfg["min_pass_rate"], "eval.yaml min_pass_rate")
    except (ValueError, TypeError) as bad:
        raise RegistrationError(f"eval.yaml field invalid: {bad}") from bad
    # CAP-6: refuse HERE, before the eval rollout is spent — a candidate that
    # grades itself below the registry floor cannot be certified by any result.
    if declared < REGISTRY_MIN_PASS_RATE:
        raise RegistrationError(
            f"eval.yaml min_pass_rate {declared:.3f} is below the registry floor "
            f"{REGISTRY_MIN_PASS_RATE:.3f} (CAP-6) — a skill sets its own bar only "
            "ABOVE the floor; raise it, or raise the skill"
        )
    return Skill(path=skill_dir, manifest=manifest, eval_cfg=eval_cfg)


def curated_ids(root: Path) -> frozenset[str]:
    """The CAP-5 curated core, from its single Class-C source — never from
    mutable manifest-file presence (PR #30: deleting a core file must not
    open its id)."""
    path = root / CURATED_CORE
    if not path.exists():
        raise RegistrationError(f"{CURATED_CORE} missing — cannot establish the curated core")
    try:
        return frozenset(tomllib.loads(path.read_text())["core"])
    except (tomllib.TOMLDecodeError, KeyError) as bad:
        raise RegistrationError(f"{CURATED_CORE} unreadable: {bad}") from bad


def _eval_graph_path(skill: Skill, root: Path) -> Path:
    graph = root / str(skill.eval_cfg["graph"])
    if not graph.exists():
        raise RegistrationError(f"eval graph {skill.eval_cfg['graph']!r} not found")
    return graph


def validate_skill(skill: Skill, root: Path) -> None:
    """Schema + governance + candidate-binding checks, BEFORE any staging
    or eval spend."""
    manifest = skill.manifest
    if manifest.get("origin") != "agent-authored":
        raise RegistrationError(
            f"origin must be 'agent-authored' for registration, got "
            f"{manifest.get('origin')!r} — hub manifests are curated by hand (§9.4)"
        )
    schema = load_capability_schema(root)
    errors = manifest_schema_errors(schema, manifest)
    if errors:
        raise RegistrationError(f"manifest fails the capability schema (CAP-3): {errors[:3]}")
    skill_id = manifest["id"]
    if skill_id in curated_ids(root):
        raise RegistrationError(
            f"id {skill_id!r} is on the CURATED core list ({CURATED_CORE}) — "
            "a skill may never shadow a core capability (CAP-5)"
        )
    source = manifest.get("source", "")
    if not (root / source).exists() and not (skill.path / Path(source).name).exists():
        raise RegistrationError(f"skill source {source!r} not found")
    graph = _eval_graph_path(skill, root)
    # the eval suite must be BOUND to the candidate (PR #30): an unrelated
    # green graph must not be able to mint this skill's evalcard
    doc = _load_yaml(graph, "eval graph")
    node_ids = {n.get("id") for n in doc.get("nodes", []) if isinstance(n, dict)}
    if skill_id not in node_ids:
        raise RegistrationError(
            f"eval graph does not use the candidate: no node with id {skill_id!r} "
            f"(nodes: {sorted(i for i in node_ids if i)})"
        )


def _provisional_manifest(skill: Skill, now: str) -> dict:
    """The staged candidate: CAP-6 requires a non-null eval for motion
    skills DURING their own evaluation — the provisional card is clearly
    labelled and never survives a failed registration."""
    return {
        **skill.manifest,
        "eval": {
            "suite": f"{skill.eval_cfg['suite']} (provisional)",
            "pass_rate": 0.0,
            "last_run": f"provisional-{now}",
        },
    }


def _lint_or(root: Path, message: str) -> None:
    report = lint(root)
    if not report.get("ok"):
        raise RegistrationError(f"{message}: {report.get('errors', [])[:3]}")


def run_skill_eval(
    skill: Skill, root: Path, run_rollout, run_id: str, sim_engine: str | None = None
) -> float:
    """The shipped mini-rollout against the STAGED candidate: returns the
    measured pass rate (pass1). sim_engine (ADR-67) rides through to the
    runner so the eval is not pinned to genesis; None honours whatever the
    eval graph declares."""
    cfg = skill.eval_cfg
    report = run_rollout(
        root=root,
        graph=_eval_graph_path(skill, root),
        tier=str(cfg["tier"]),
        episodes=int(cfg["episodes"]),
        seeds=parse_seed_range(str(cfg["seeds"])),
        reset_mode=str(cfg.get("reset", "teleport")),
        verifier=str(cfg.get("verifier", "oracle")),
        run_id=run_id,
        branch="skill-eval",
        no_idea_gate=True,  # registration machinery, logged (ADR-22)
        embodiment=str(cfg["embodiment"]),
        env_baseline=str(cfg.get("env_baseline", "local")),
        sim_engine=sim_engine,
    )
    if not report.get("ok"):
        detail = report.get("refused") or report.get("error") or "no episodes"
        raise RegistrationError(f"eval run failed before scoring: {detail}")
    return _parse_pass_rate(report.get("pass1", 0.0), "measured pass_rate")


def register_skill(
    skill_dir: Path,
    root: Path,
    run_rollout,
    now: str,
    run_id: str | None = None,
    sandbox: bool = False,
    sim_engine: str | None = None,
) -> dict:
    """validate → STAGE → lint → eval → evalcard → final lint (§8.4);
    every failure rolls the registry back exactly. `sim_engine` (ADR-67)
    selects the physics engine the eval rollout runs on.

    `sandbox=True` (ADR-40, #265) takes the §9.4 lower rung: admit the id so
    graphs referencing it VALIDATE, while making no quality claim at all — no
    eval is run, no evalcard is written, and the entry is marked
    `trust_tier: sandbox`. This exists because ADR-37's floor, correctly,
    made certification require a measured 0.5 — and left an agent no way to
    declare a node it is still building, since `validate` refuses an unknown
    id with MANIFEST_MISSING. Campaign agents met that bind and worked around
    it by shipping `min_pass_rate: 0.0`, which is the hole ADR-37 closed.

    A sandbox entry can never hold `safety_class: motion` (§9.4's per-tier
    ceiling), never counts as a library skill, and is promoted only by
    re-registering WITHOUT this flag, which runs the real gate."""
    root = Path(root)
    skill = load_skill(Path(skill_dir))
    validate_skill(skill, root)

    if sandbox and skill.manifest.get("safety_class") == "motion":
        raise RegistrationError(
            "a sandbox registration may not hold safety_class 'motion' (§9.4 per-tier "
            "ceiling): an unproven node cannot command the arm. Register it for real "
            "once its eval passes the floor, or lower its safety class"
        )

    installed = root / "registry" / "manifests" / f"{skill.manifest['id']}.yaml"
    prior = installed.read_text() if installed.exists() else None
    if prior is not None:
        existing = yaml.safe_load(prior)
        if not isinstance(existing, dict) or existing.get("origin") != "agent-authored":
            raise RegistrationError(
                f"manifest id {skill.manifest['id']!r} already exists and is not a "
                "registered skill — refusing to replace it"
            )

    def rollback() -> None:
        if prior is None:
            installed.unlink(missing_ok=True)
        else:
            installed.write_text(prior)

    installed.write_text(yaml.safe_dump(_provisional_manifest(skill, now), sort_keys=False))
    try:
        _lint_or(root, "staged registry fails lint (CAP-2/3)")
        if sandbox:
            # no eval, no evalcard, no quality claim (ADR-40). The entry
            # exists so a graph naming this id validates; everything that
            # counts capabilities filters on the evalcard, so a null one
            # keeps it out of the library by construction rather than by
            # a second rule that could drift.
            final = {**skill.manifest, "eval": None, "trust_tier": "sandbox"}
            installed.write_text(yaml.safe_dump(final, sort_keys=False))
            _lint_or(root, "final registry fails lint (CAP-2/3)")
            return {
                "ok": True,
                "id": final["id"],
                "trust_tier": "sandbox",
                "pass_rate": None,
                "evalcard": None,
                "installed": str(installed),
                "eval_run_id": None,
                "governance": (
                    "SANDBOX: admitted for validation only, makes no quality claim, "
                    "cannot be reused or counted. Re-register without --sandbox to "
                    "run the real gate (§9.4)"
                ),
            }
        eval_run_id = run_id or f"skill-{skill.manifest['id']}-{now}-{uuid.uuid4().hex[:6]}"
        pass_rate = run_skill_eval(skill, root, run_rollout, eval_run_id, sim_engine=sim_engine)
        minimum = max(REGISTRY_MIN_PASS_RATE, float(skill.eval_cfg["min_pass_rate"]))
        if pass_rate < minimum:
            raise RegistrationError(
                f"measured pass_rate {pass_rate:.3f} below the skill's shipped "
                f"min_pass_rate {minimum:.3f} — not installed"
            )
        final = dict(skill.manifest)
        final["eval"] = {
            "suite": str(skill.eval_cfg["suite"]),
            "pass_rate": round(pass_rate, 4),
            "last_run": now,
        }
        # ADR-40: passing the real gate promotes off the sandbox rung — an
        # entry must never keep a `sandbox` marker it has outgrown
        final["trust_tier"] = "reviewed"
        installed.write_text(yaml.safe_dump(final, sort_keys=False))
        _lint_or(root, "final registry fails lint (CAP-2/3)")
    except BaseException:
        rollback()
        raise
    return {
        "ok": True,
        "id": final["id"],
        "trust_tier": "reviewed",
        "pass_rate": pass_rate,
        "evalcard": final["eval"],
        "installed": str(installed),
        "eval_run_id": eval_run_id,
        "governance": "open a PR with this manifest; a human merges it (§9.4)",
    }
