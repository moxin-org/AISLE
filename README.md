# AISLE — Agentic Infrastructure for Safe Learning and Execution

New contributor? Start with the
**[AISLE contributor wiki](docs/contributor-wiki.md)** for a source-linked
project overview, architecture, use cases, extension guide, code map, research
status, and known limitations.

**Student, or here for a hackathon?** The
**[technical report](docs/AISLE-technical-report.md)** doubles as the course
text (~50 pages). It assumes no background in dataflow runtimes, robot
learning, or experiment methodology. Start at §0, which gives three reading
paths; §5.5 dissects a real skill, §7.5 explains the three units of
measurement that people most often get wrong, and **Appendix C** is a graded
set of exercises and hackathon tracks — several of which are genuinely open
problems (**Appendix D** lists them with issue numbers).

**Visiting research group?** Read
**[`docs/lab-briefing-2026-09.md`](docs/lab-briefing-2026-09.md)** first: two
pages on what has been measured, what has not and why, and where a graduate
student can contribute this semester.

<!-- claim:typed-composition/readme -->
*AISLE is an **environment** in the reinforcement-learning sense — a simulated
world with tasks, dynamics, and a frozen scorer that agents act in — and the
**infrastructure** around it: the typed contract, registry, validator, safety
guard, and evidence harness. The name nods to the pharmacy aisle where the
first task family lives, but the scope is the substrate, not the store: the
bench suite (SPEC 300/310) is laboratory powder handling, and hardware
execution is the intended path.*

Agentic auto-research for robot manipulation on open infrastructure:
coding agents (Claude Code / Codex) compose and evolve **typed dora-rs
dataflows** against a **Genesis** physics scene, with frozen
verification/reset and scoped safety structure. The scene contract is
engine neutral; two optional engines run the same graphs for development,
[Nexus](docs/simulation-backends.md#nexus) on the GPU and
[rapier](docs/simulation-backends.md#rapier) on the CPU (`--sim-engine nexus|rapier`),
while Genesis stays the default and the only engine behind the measured
record.
<!-- claim:typed-dataflow-causal/readme -->
The **UNRUN confirmatory claim under test** is whether a
typed dataflow substrate makes agentic robotics faster, safer, more
auditable, and more reusable than script-level iteration — reproducible
on a MacBook.

## Research question

> Can AI coding agents autonomously build, diagnose, improve, reuse, and safely
> operate robotic systems when those systems are composed as typed dora
> dataflows?

AISLE does not treat a successful robot demo as sufficient evidence. The object
under study is the full engineering loop: an agent chooses and connects
capabilities, validates the graph, runs budgeted episodes, diagnoses typed
traces and failure classes, improves the system, and carries evaluated skills
into later tasks. The task result, research cost, safety events, graph and code
identity, environment, seeds, and admissibility audit are recorded together so
we can distinguish an attributable improvement from an easier seed, changed
scorer, environment drift, contamination, or extra compute.

The current model-light runtime is an experimental control, not the intended
limit of the architecture. VLA policies, world-model planners/environments, and
World Action Models (WAMs) can enter as typed, swappable nodes behind the same
action adapters, guard, verifier, and evidence contract. That makes “classical
pipeline vs. learned policy vs. predictive/hybrid system” a matched systems
experiment that the coding agent can itself propose and run.

Read the **[AISLE technical report](docs/AISLE-technical-report.md)** for the
full standalone treatment: architecture, determinism, evidence design, the
experimental program, results to date, threats to validity, and the staged
VLA/world-model/WAM agenda. The **[AISLE research program](docs/research-program.md)** gives the
technical-report framing: research object, falsifiable questions, why
experiments and evidence collection matter, claim discipline, and the staged
VLA/world-model/WAM agenda. The full original experiment design is
[`docs/Project_AISLE_Experiment_Design.md`](docs/Project_AISLE_Experiment_Design.md).
New to the repo? Start with `docs/getting-started.md` — for the
concepts behind it all (Physical AI, VLM/VLA/world models/WAMs,
sim-to-real, agentic auto-research), `docs/physical-ai-primer.md` — and
for the shorthand every other page uses (`CON-5`, `ADR-30`, `H3`, `A7`,
`T2`, `L1`, Class C, DoD, the frozen set),
**[`docs/glossary.md`](docs/glossary.md)**, which expands each identifier
and names the file that defines it.
The model-node tier (VLA policies, VLM verifier, world-model
environments) ran as **Phase 5** and is CLOSED at measured end-states —
the plan, its ratification, and the per-item ledger are
**[docs/next-phases.md](docs/next-phases.md)**. The paper distilling
the whole measured record is
**[docs/paper/aisle-paper.md](docs/paper/aisle-paper.md)** (v1.0).

<!-- current-status:canonical -->
## Status

**This table is the single current status page** (issue #142). Other overview
pages link here; protocol and evidence pages may retain dated summaries for
context, but must identify their snapshot and defer to this table on conflict.
Status as of **2026-09-14**, commit `fa339ee`. The measured record below is
the **2026-08-28** record (commit `93de5e0`): no hypothesis or ablation has
been measured since. What changed in between — the pre-registration program,
the pilot-first order, and the still-pending #347 pilot — is in
[Since 2026-08-28](#since-2026-08-28-the-pre-registration-program-and-the-pilot)
at the end of this section. Each row states the verdict its committed evidence
supports, with that evidence's own qualifications — a hypothesis with no
admissible data says so rather than reading as progress.

<!-- claim:development-ledger/readme -->
**SUPPORTED structural status, not a confirmatory result:** this is the dated
development-evidence ledger. Its empirical rows retain their own attestation,
sample, and scope qualifications; the ledger does not combine them into a
benchmark treatment effect.

### Phase status (design doc §8)

**Phase 2 (§8.3) and Phase 3 (§8.4) are both closed as of 2026-08-16.**
Full record: [`analysis/reports/phase2_phase3_report.md`](analysis/reports/phase2_phase3_report.md).

| Phase | DoD | Verdict |
|---|---|---|
| Phase 0 — world bring-up | SPEC 090 | signed off (M0) |
| Phase 1 — registry, validator, harness | validator + 12 manifests + H1 table | complete |
| **Phase 2** — the full autoresearch loop | 8 items | **complete** — pass@k curves T1/T2, verifier fidelity, iteration latency, A1/A3/A7, zero unclamped guard violations, post-mortems |
| **Phase 3** — skills, fleet, cross-embodiment | 6 items | **closed: 5 met, 1 NOT MET** — the skill library reached 3 evalcarded skills against a DoD of ≥5 at close (see the library note below: the two refused skills later cleared the floor) |
| **Phase 5** — the model tier (next-phases §5) | 5.1/5.1b/5.2/5.3 | **CLOSED 2026-08-27** at measured end-states; every remainder explicitly GPU- or design-gated ([ledger](docs/next-phases.md)) |
<!-- claim:hardware-so101-validation/readme -->
| Phase 6 — hardware | entry criteria + prep | **HARDWARE PENDING — prepared, not open**: SO-101 driver node (loopback-tested) + VER-8 hardware calibration parity landed (ADR-phase6-prep); entry gates unwaived (M5 met; M1 + judge fidelity gated) |

The one unmet row is stated plainly rather than rounded, and dated: at the
2026-08-16 close the library held `s1-driver-v2`, `s3-driver-v1`, and
**`ik-transfer-v2`** — the last a `safety_class: motion` trajectory skill an
agent authored against a trace-cited collision, evalcard 1.0, reviewed and
human-merged (#258) — while ADR-37's floor refused `t2-scan-pose` (0.33) and
`t2-scan-tsm` (0.0) on their own evalcards. **Post-close, both refused skills
cleared the floor** on pre-registered n=8 suites after the T2 breakthrough
work (0.5 each, trust tier `reviewed`, #304), bringing the library to
**five**; the closure verdict stands as dated. The reuse mechanism works in
both directions: `s3-driver-v1` appears verbatim in a desk deliverable
(retail→desk), and the registered T2 pair carried verified in-deliverable
reuse in the accumulation differential (#306).

`ik-transfer-v2` is also the §9.4 trust-tier path completing end to end:
authored by an agent in the governance-critical motion class, shipped with
its own eval suite and regression population, lost to the retention gap,
recovered and provenance-verified, reviewed, then human-merged into the
registry. Exercised once, on the class that matters most.

Exact graph/manifest/CLI/ADR catalogs are generated, never hand-counted:
[`docs/generated/project-inventory.md`](docs/generated/project-inventory.md).
Orientation for contributors: [`docs/contributor-wiki.md`](docs/contributor-wiki.md).

| Milestone | State |
|---|---|
| M0 — verified pharmacy-pick loop (SPEC 090) | signed off; expert graph 0.98 pass@1 over 50 seeds, with the milestone replicate independently re-satisfying the gate |
| H1 — zero-shot composition | measured, target not met: 40/40 schema-valid graphs, but 15% (claude) / 65% (codex) launch zero-shot; single dominant failure is uninstalled hub packages (`analysis/h1/`) |
| H2 — iteration to ≥90% | claude arm **met** held-out (1.0 pass@1); codex arm 0.875 held-out at N=8 (one `dropped`), with dev-side evidence of a ≥0.9 system — see `analysis/h2/` for the full verdict |
| H3 — skill accumulation | **UNDECIDED on both suites; no speedup measured.** Retail (S1→S3): `met: null`, every library-arm cell lost to drift. Desk (T1→T4, `analysis/h3/desk/`): `met: null` under strict admissibility, 13 caveats. The interpretable direction, stated with that caveat — on T4, the only tier where both arms produced clean first-success numbers, the ratio is **~1.03** (L 894 s vs W 872 s: parity, not ≤0.5); T2/T3 the library did not rescue what wiped sessions could not do either. **The finding is the ladder's difficulty spacing, not the library**: T1/T4 are easy for both arms (no headroom for a speedup) and T2/T3 are beyond both (no success to speed up), so the transfer curve never got a tier that could show an effect. The sharpest admissible follow-up (T2-only differential, #306): both arms 0.25 holdout, **library arm 35% cheaper** (451k vs 696k tokens) with verified reuse — accumulation bought economy, not ceiling. Skill reuse itself is verified live |
| H4 — hot-swap vs relaunch iteration | **measured at T0**, phase-randomized (ADR-h4 rev 2): hot-swap median iteration latency 32.4 s vs relaunch 41.8 s (ratio 1.29), n=6 per path, zero infra failures. Extremes overlap; no significance or equivalence claim at n=6. UNATTESTED dev measurement. **Scope note (H6 finding, measured 2/2):** live swap of a turn participant kills an ADR-30 lockstep dataflow — this table holds for pre-barrier free-run graphs only; turn-aware swap is a filed substrate follow-up (`analysis/h4/`) |
<!-- claim:safety-observed-outcomes/readme -->
| H5 — zero wrong-object under free iteration | **WEAKENED observational claim:** 0 wrong-object in the retained development denominator, including 224/224 episodes across the three H2 campaign runs (`analysis/h2/`) and subsequent campaign summaries. The repository does not yet expose one session-level, independently audited denominator spanning the historical “roughly 45 sessions” statement, so this is neither a prevention claim nor confirmatory safety evidence |
<!-- claim:live-fault-feasibility/readme -->
| H6 — agent operates a running system | **SUPPORTED as a bounded feasibility result, 3/3 cells** (2026-08-26, pre-registered ADR-h6 + five measured amendments): per fault tier (perception/decision/motion), an operator agent given only live evidence detected the induced degradation (299–447 s), localized the correct node with cited evidence, and restored 1.0 with a validated repair. Zero `wrong_object`, zero declared-graph guard bypass, no out-of-space action. n=1 per fault class; this is not a typed-evidence-versus-logs effect estimate (`analysis/h6/`) |
| Retail suite S1–S3 (mobile, long-horizon) | implemented: store scene, planogram verifier, mobility contract, S1 expert graph |
| Perception ladder L0/L1/L2 (TC-9) | implemented: L0 oracle poses, L1 segmentation + depth (`segmented-pose`), L2 RGB identity + same-stamp sensor-depth geometry (`l2-pose`); the rung rides the graph and is asserted per run (`--perception`) |
| Tier curves T1/T2 (Phase-2 DoD) | T1 expert **1.0** per rung. T2: the original expert measured **0.08** (the deliberate perception wall); the fleet-authored far-first read ladder broke through to **0.375** holdout (#299), and the registered stack holds **0.5** on the pre-registered n=8 suite across four re-measures. The residual failure class is traced to three named transit-collision mechanisms — one fixed (#314), one mitigated at the solve (#328, after a measured filter regression 0.5→0.375→0.5), one structural (tray-descent link sweep) (`analysis/t2/`, `analysis/t2_breakthrough/`, `analysis/transit_collisions/`). **0 `wrong_object` across every T2 episode ever run** |
| T2/T3 unsolved at session budgets | **standing challenge.** No campaign arm — desk-H3 either arm, A3, A4 — has solved T2 or T3 within a session budget. This is the tiers working as designed (§1 curriculum) and is the single biggest open scientific item |
| Realistic verifier (VER-5) | implemented (`src/aisle/verifier/realistic.py`, OWLv2 + rules, CPU-pinned); ADR at `docs/decisions/ADR-realistic-verifier.md` |
| VER-6 verifier fidelity | current VER-13 fusion recomputed over the same 31 recorded episodes: agreement **0.45**, false SUCCESS **0.00** (0/6), false FAIL **0.68** (17/25). The preserved first, pre-amendment measurement was 0.29 / 0.00 / 0.88 (`analysis/ver6-fidelity/`; current recomputation in SPEC 040 VER-13). Conservative, not yet interchangeable with the oracle |
| CON-5 reproducibility on S1 | **original violation dispositioned**: ADR-25 fixed and verified reset-anchored startup; ADR-26 defines full-episode outcomes as statistical under Metal noise. Issue #71 remains open for wall-coupled command/control timing and possible frozen-set retiming — per-seed outcome flips are not themselves a CON-5 violation |
| T4 increment two — post-delivery recovery | **first complete number** (#295–#303): dialogue-corrected misdeliveries 3/3; recovery chains 1/3 (one end-to-end return+redelivery success; one `not_returned` at budget; one collision traced to the tray-descent link sweep). Protocol/judging machinery 100%; the residual is manipulation (`analysis/t4_inc2/`) |
| M1 — learned policy value (Phase 5) | **the competence half is measured on-Mac**: zero-shot SmolVLA structurally impossible (uninitialized base normalizers); an 800-step LoRA validated the pipeline; live eval 0/8 with the mechanism = CPU latency vs ADR-38's staleness floor; the **lockstep-eval condition** (ADR-38 am. 1: sim time freezes during in-turn inference) then measured 0/8 with zero refusals and an inverted failure mix — competence, not latency, is the wall, so the next spend is training dose, per-dose measurable at ~17 min/episode (`analysis/m1/`) |
| VLM judge (Phase 5.2) | **five configurations measured and refused, zero false promotions** by a twice-hardened asymmetric gate (success-recall added after a constant-fail judge "passed" a failure-heavy holdout). 2B scales (0.2→0.6 agreement) but keeps the identity-free false-success class; label-reading prompts die on optics (21 px text, recall 0). Remainder: judged-frame design change or fine-tuning (`analysis/ver-vlm/`) |
| M3 — environment ladder (Phase 5.3) | **the swap works; the ranking question needs variance**: a v0 kinematic surrogate ran 16/16 agent-authored H1 graphs unmodified at ~100x Genesis speed; ranking agreement is honestly undecidable (13/16 identical Genesis scores + the pre-declared cartoon-vs-contact fidelity gap) — Spearman reported as undefined, never fabricated (`analysis/m3/`) |
| PW-0 — powder family feasibility (SPEC 300) | **ratified 2026-08-27**: GO for P0/P1 primitives only (MPM sand, ≤5k particles CPU-scored; Metal is nondeterministic + ~10% crashy, exploration only); PW-6 reframed best-effort (scoop CV ~88%); P2+ deferred pending a CUDA determinism spike; no repose-dependent scenes/verifiers (ADR-powder-spike, spec thresholds filled #322) |

### Ablations (design doc §6)

Every row is n-limited and says so. These are directional results from single
matched pairs, not powered comparisons.

| Ablation | Result |
|---|---|
| A1 — agent-composed vs expert graph | measured end-to-end (compose, launch, pass), with a composition that never launches scoring 0. An earlier draft conditioned on the 16/40 graphs that launched — selecting away the dominant failure mode A1 exists to measure — and was corrected. The T1 rerun is the attested cell of record (`analysis/a1/a1_table.md`; read its "what the records do NOT support" section first) |
| A3 — params-only vs params+code | **the constrained arm won on efficiency at equal quality** (pin `8af9b47a`): params-only reached first success in 9.8 min vs 13.8, spent **200k tokens vs 396k** (50% vs 99% of budget), 24 min vs 85 min wall, **1 dev rollout vs 4** — and both arms scored 1.0/1.0 held out with 0 `wrong_object`. Params-leak audit on arm P: clean. Reading: **schema-as-subsidy** where the registry already covers the task. n=1 per arm on the easiest tier (`analysis/a3/`) |
| A4 — Claude Code vs Codex | **both solve T1 outright**, 1.0/1.0 held out, 0 `wrong_object` (pin `cb814e12`, identical budgets/prompt/seeds). Codex reached first verified success sooner (**8.1 vs 9.7 min**) then kept iterating; Claude converged in 2 rollouts and stopped. End-to-end cost: **186k vs 364k tokens**, 36 vs 73 min. The difference is style, not capability; Claude's session was ~2× cheaper at equal quality. n=1/arm, lower bound (`analysis/a4/`). Kimi Code out of scope v1 |
| A5 — 1 vs 4 vs 8 agents (fleet scaling) | **throughput saturates at ~4 lanes on one host**: 1.6 → 4.1 → **4.3** successes/hour at N=1/4/8. Going 4→8 bought +5% throughput for 2× the agents and 2.2× the token burn. Per-agent latency degrades gracefully (median first success 10.5 → 14.1 → 18.0 min); **quality is contention-invariant** (holdout 1.0 on every lane); token super-linearity +22%/+31% per agent. 0 `wrong_object` across all 13 lanes. Deviation from §8.4.3 recorded in the ADR: lanes share the host with their own sim, not one batched bridge (`analysis/a5/`) |
| A6 — teleport vs behavioral reset | **teleporting hides a real task and a real cost.** Paired 10-episode T1 arms, seeds 0..9: teleport **1.00** pass@1 in 6.4 min; behavioral **0.80** (2 `never_grasped`) in 9.6 min, at **+19 s per episode**. Reset outcomes on the behavioral arm: 7 success / 3 audited `fallback: true` — i.e. the reset is itself a manipulation task that fails sometimes, which is exactly ENPIRE's claim and what a teleport inner loop conceals. 0 `wrong_object` in both arms (`analysis/a6/`) |
| A7 — oracle vs realistic verifier driving the loop | verifier-driven loop measured; budgets re-derived. See the VER-6 row for the fidelity number the ablation depends on, and note the rung caveat below |

### Governance findings (§8.4 review, 2026-08-15/16)

Running the Phase-3 agent-PR review to completion surfaced **three harness
defects and zero agent transgressions**. Recorded because that asymmetry is
itself the result — a review of agent-authored robot code produced no true
positives against the agents and three against the machinery that reviews them.

| Finding | Disposition |
|---|---|
| The skill eval floor was **self-graded** — `min_pass_rate` came from the candidate's own `eval.yaml`, so a skill shipping 0.0 registered at 0.0 and the gate reported `ok` | fixed: `REGISTRY_MIN_PASS_RATE = 0.5`, refused at load before an eval rollout is spent (**ADR-37**, #243) |
| Campaign deliverables were **never archived** — worktrees live under gitignored `runs/` with no retention step, so a campaign record could name a skill no reviewer could find | fixed forward: each session archives its working tree to `refs/campaign/<name>` (#245). The three affected skills were later recovered by hand from the campaign machine and provenance-verified (#252) |
| At perception rung **L2 the policy and the realistic judge would share the detector backbone**, so agreement would overstate independence | guarded: every fidelity report now carries a `backbone` verdict, persisted beside the rates, and fails closed on an unresolved rung (#248). **No reported number is affected** — every VER-6 measurement to date ran at L0, where the policy calls no detector |
| A proposed validator rule (VAL-9) banning policy nodes from importing the judge's detector | **dropped** (#244): the curated core does the same thing, so the rule would have rejected two frozen expert graphs — and the flagged skill never had the alleged import at all (#253) |

Signed review notes: [`analysis/reports/agent_pr_review_notes.md`](analysis/reports/agent_pr_review_notes.md).
The first pass could only review **2 of 5** skills, which any claim about
human-in-the-loop governance of agent-authored robot code has to carry.

### Since 2026-08-28: the pre-registration program and the pilot

Between the snapshot above and 2026-09-14 the repository gained 235
commits and **no new measured hypothesis or ablation result**. Nearly all of
that work is the machinery that would make the headline result defensible
when it runs. Stated as such:

| Item | State on 2026-09-14 |
|---|---|
| Pre-registration program, SPEC 400–540 | Tooling implemented (`harness freeze/stats/exposure/semantic/threat/fault/monolith/perception/hardware`, each with fixtures and self-tests). All fifteen specs are **PROPOSED, human-review gated**; none authorizes confirmatory collection. Nine are **parked** under ADR-66 (430, 450, 460, 470, 480, 510, 520, 530, 540) |
| Typed-vs-monolithic causal study (#347, SPEC 500) | **Unrun.** Confirmatory registration at v22 with all 7 integrity gates pending, two of them external reviews (#483 statistics, CON-14). Pilot v1 (2026-09-12) collected **zero sessions**: the executor admitted only the T1 oracle mode, both paired T1 surfaces put oracle state on the policy path, and the task band named no eligible stratum. A merged stack (#580–#588) bound a T1-L2 realistic pair as the instrument set; **pilot v8** (2026-09-13) is registered with collection pending on the session-bound confinement attestation (#589–#591). The task-band calibration (v13) records **BND-7 eligibility failed** for that candidate, 22 of 22 strata; any pilot on it carries that as a named limitation (`analysis/reports/cse_pilot_v1.md`, `analysis/freeze/cse-causal-study-pilot-v8/`) |
| Execution order and moratorium (ADR-66, accepted 2026-09-11) | Pilot-first: run #347's pilot, then H6, before any new spec, platform, or hypothesis; one measured result every two weeks or infrastructure PRs stop. CON-9 amended the same week: documentation-only commits run format, lint, and unit only. The renumbering note: the pilot-first decision is **ADR-66**; ADR-65 is the frontend-conformance ADR (#568) |
| Instrument ablations and pilots (synthetic or fixture; **not treatment effects**) | Semantic authorization (SPEC 480): held-plan replay, false-allow risk difference sensor_shield − no_shield **−0.93** (95% −0.97 to −0.82), n=60, same author for corpus and authorizer. Safety exposure (SPEC 470): guard-on **0/39** vs observe-only **32/39** at-risk traces, fake driver, no physics. Threat model (SPEC 460): **18/18** catalogued attacks blocked at the expected layer, one process; no boundary claim beyond that fixture. Fault bank (SPEC 450): v2 calibration gives every family ≥1 effective instance, leakage probe balanced accuracy 0.37 (p=0.80), bank **not sealed**. Perception audit (SPEC 490), second audit with the hardened auditor (2026-09-13, registered as calibration v11): **not eligible, 22 of 22 strata fail** on 320 scored evaluation frames; overhead stratum 214/320 (0.67), ibuprofen refuses 68 of 80, **0 wrong identity**, no threshold changed (`analysis/perception-audit/records/bnd-perception-corpus-03/`). Each README in `analysis/` states its own evidence kind |
| M3 environment ladder, v2 | **Spearman 0.746** ranking agreement between the kinematic surrogate and Genesis over 16 graphs × 8 seeds on a spread population (0.76 excluding the contact-geometry set); supersedes the "undecidable" v0 row above. UNATTESTED dev measurement, self-authored population (`analysis/m3/v2/`) |
| Local-model capability floor (qwen3:30b) | **0/5** schema-valid graphs, as pre-registered; two same-seed sessions bit-identical. UNATTESTED, n=5 (`analysis/local-arm/`) |
| Halt contract for motion nodes | ADR-64 (accepted): a halt ends the episode, no in-episode resume. The declared halt contract itself is #562, **parked** until SPEC 460 ratifies or Phase 6 opens |
| Related systems reviewed | PhyAgentOS (arXiv 2607.16636) and TypeGo (arXiv 2607.05482) assessed 2026-09-10/11; positioning in the paper's §7. PhyAgentOS runs on dora with the graph hidden behind an HTTP tool API — a shipped instance of this repository's monolithic control arm, which is the strongest external reason to run #347 |
| Open owner-side gates | External reviews not yet arranged: statistical (#483), terminology (#484), independent reproduction (#485), external benchmark user (#486). No in-house work can satisfy these |

## Quickstart

Use macOS arm64 or Linux CPU, Python >= 3.11, Git, rustup and
[uv](https://docs.astral.sh/uv/). Linux also needs the EGL/Mesa packages listed in
[getting started](docs/getting-started.md). From a fresh checkout:

```bash
uv sync --extra sim --locked   # plain `uv sync` REMOVES the sim extras
rustup toolchain install 1.97.1
export AISLE_DORA_PREFIX="$PWD/../aisle-dora-runtime"  # choose a new directory
uv run --extra sim --locked python tools/dora_runtime.py install --prefix "$AISLE_DORA_PREFIX"
uv run --extra sim --locked python tools/dora_runtime.py verify --prefix "$AISLE_DORA_PREFIX"
uv run --extra sim --locked python tools/quickstart.py --runtime-prefix "$AISLE_DORA_PREFIX"
```

### Choose a simulation backend

The `sim` extra installs Genesis, the default, and the optional Nexus and
rapier engines (on macOS arm64, Linux x86_64 and aarch64, and Windows x64).
Choose one with the same rollout flag:

```bash
# Default, measured backend
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local \
    --sim-engine genesis

# Optional GPU backend
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local \
    --sim-engine nexus

# Optional CPU solver with the Nexus renderer
uv run --extra sim --locked harness rollout --graph graphs/expert_t0.yaml --tier T0 \
    --episodes 2 --seeds 0..1 --no-idea-gate --env-baseline local \
    --sim-engine rapier
```

See the [simulation backend guide](docs/simulation-backends.md) for platform
support, backend overrides, direct graph declarations, and current
limitations. Nexus and rapier are development backends; do not compare
their results with the Genesis measured record.

The Python API remains pinned to 1.0.1; the CLI is built from the corrected,
immutable source revision in `dora-runtime.json`. CLI version output alone does
not identify that correction: the installer and quickstart verify its receipt.
The release CLI 1.0.1 has a reproduced timer-pressure event-loss defect (#516).
See [runtime installation](docs/benchmark/v1/dora-runtime.md) for identity checks
and [getting started](docs/getting-started.md) for manual rollouts and NVIDIA
setup. Never install with bare pip/conda.

The supported Linux clone and archive quickstarts passed in
[run 34082153283](https://github.com/moxin-org/aisle/actions/runs/34082153283).
These are development runs; the benchmark's independent-user, release and
physical-evidence gates remain outstanding.

## Repository map

```
CLAUDE.md          development-agent contract (read first if you are an agent)
specs/             numbered specs with MUST IDs (000 = constitution)
TASKS.md           implementation order + kickoff prompts
registry/          capability schema + typed node manifests
graphs/            expert + eval dataflows (T0..T4 desk, S1 retail,
                   VLA evals incl. the lockstep condition)
src/aisle/         scenes, bridge, verifier, reset, harness, mobility, nodes
                   (incl. world_model_env — the env-ladder surrogate — and
                   so101_driver — the Phase-6 hardware bridge, loopback-tested)
harness CLIs       `uv run harness {validate,rollout,traces,report,skill,swap,
                   probe,fleet,freeze,stats,exposure,semantic,threat,fault,
                   monolith,perception,hardware}` (the last nine are the
                   SPEC 400–540 pre-registration instruments)
tools/             CI, trace_check, env_hash, campaign runners
                   (h1..h4, h6, a3..a5, matched_campaign, pilot_*, m3_ranking,
                   judge_bench, vlm_judge, finetune_smolvla, hw_calibration,
                   spikes/)
tests/             unit / sim / graph markers; every MUST cited by a test
analysis/          committed experiment findings: hypotheses (h1..h4, h6),
                   ablations (a1..a6), tiers (t2, t2_breakthrough, t3, t4,
                   t4_inc2), m1, m3, ver-vlm, local-arm, transit_collisions,
                   ver6-fidelity, s1-determinism, postmortems, transcripts,
                   reports/; the pre-registration record: freeze/ (content-
                   addressed registrations), statistics, treatment-integrity,
                   monolithic-control, fault-bank, threat-model,
                   safety-exposure, semantic-authorization, perception-audit,
                   instrument-audit, claim-evidence
skills/            the registered library (5): s1-driver-v2, s3-driver-v1,
                   ik-transfer-v2, t2-scan-pose, t2-scan-tsm
docs/              guides, design doc, contributor wiki, lab briefing
                   (lab-briefing-2026-09.md), paper/ (v1.0), decisions/
                   (ADRs), generated/project-inventory.md
runs/              gitignored: traces, videos, run manifests
```

Structural counts (graphs, manifests, CLI commands, ADRs) are deliberately
absent here — they went stale faster than anyone noticed. The generated
appendix carries them and CI fails when it drifts.

## How development works

Spec-driven, tests-first: specs define WHAT with numbered MUSTs; every
MUST you implement needs a test citing its ID (`tools/trace_check.py`
enforces this in CI); agents implement tasks from `TASKS.md` under the
`CLAUDE.md` contract; humans review Class C paths (CODEOWNERS) and sign
milestones. Gates before every commit: `ruff format --check`,
`ruff check`, `pytest -m unit`, trace_check. Conventional commits; one
concern per PR.

The local CI script also checks requirement traceability, the generated
contributor inventory, and the committed frozen-environment hash.

The experiment's integrity rules are structural, not behavioral: the
environment/verifier/reset set is hash-frozen (rollouts refuse to start
on drift), `oracle_state` cannot be routed to policy nodes, all motion
passes through the budget guard, and research agents operate under a
separate contract (`harness/CLAUDE.research.md`) with idea-tree logging.

See `docs/development-workflow.md` for the full loop and
`docs/architecture.md` for a tour of the system.
