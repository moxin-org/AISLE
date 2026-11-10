# AISLE claim-to-evidence matrix

> Generated from `docs/claim-evidence.yaml` by `tools/claim_evidence.py`; do not edit.

## Release and architecture status

- Terminology review: `pending` (required before `public_benchmark_release`)
- Experimental unit: `coding_agent_session`
- Threat-model dependency: `#350`
- Trust zones: `frozen_evaluator`, `hidden_controller`, `mutable_participant`, `trusted_actuation`
- Inaccessible: ["Sealed task and fault contents held outside the participant workspace.", "Held-out assignments and controller secrets not mounted for the participant."]
- Forbidden: ["Direct actuation outside declared participant interfaces.", "Editing frozen evaluator artifacts or inspecting sealed answers."]

## Publication purpose boundary

| Surface | Purpose | In scope | Out of scope |
|---|---|---|---|
| `technical_report` | Preserve the complete systems and historical project record. | ["complete_project_record", "historical_development_results", "teaching_and_system_detail"] | ["focused_confirmatory_headline", "causal_superiority_without_control"] |
| `focused_paper` | Test typed versus monolithic engineering and typed evidence versus conventional logs under frozen confirmatory protocols. | ["focused_confirmatory_headline", "scoped_safety_boundary", "independent_reproduction"] | ["historical_development_results", "complete_project_record"] |

## Claim index

| Claim | Type | Status | Environment | Attestation |
|---|---|---|---|---|
| [`campaign-registrations`](#campaign-registrations) | `structural` | `supported` | agnostic | {"rationale": "Manifests re-hash clean with seed sources withheld.", "status": "repository_verified"} |
| [`development-ledger`](#development-ledger) | `structural` | `supported` | agnostic | {"rationale": "The catalog checker enforces the one-source marker and dated snapshot links.", "status": "repository_verified"} |
| [`evidence-attestation`](#evidence-attestation) | `structural` | `supported` | agnostic | {"rationale": "The frozen inventory and guard dependencies are tested from tracked sources.", "status": "repository_verified"} |
| [`external-reproduction`](#external-reproduction) | `reproducibility` | `unrun` | simulation_and_hardware | {"rationale": "Self-runs are not independent reproduction.", "status": "not_applicable"} |
| [`hardware-so101-validation`](#hardware-so101-validation) | `future` | `hardware_pending` | hardware | {"rationale": "Simulation and loopback tests are not physical evidence.", "status": "not_applicable"} |
| [`live-fault-feasibility`](#live-fault-feasibility) | `empirical` | `supported` | simulation | {"rationale": "Cell records and scorer are retained; no independent reproduction exists.", "status": "repository_recorded"} |
| [`safety-exposure-hardware`](#safety-exposure-hardware) | `future` | `hardware_pending` | hardware | {"rationale": "Simulated contacts, gateway receipts, and held-command timing are not physical exposure.", "status": "not_applicable"} |
| [`safety-exposure-ledger`](#safety-exposure-ledger) | `empirical` | `supported` | simulation | {"rationale": "Ledgers, reports, manifests, and regeneration commands are tracked; raw traces are retained privately with hashes in the ledger.", "status": "repository_verified"} |
| [`safety-identity-authorization`](#safety-identity-authorization) | `future` | `unrun` | simulation_and_hardware | {"rationale": "The authorizer, permit gateway, and synthetic held-plan replay are tracked (SPEC 480); no simulation or hardware transition has been authorized by them.", "status": "repository_verified"} |
| [`safety-kinematic`](#safety-kinematic) | `structural` | `supported` | agnostic | {"rationale": "Guard implementation and fuzzed command tests are tracked.", "status": "repository_verified"} |
| [`safety-observed-outcomes`](#safety-observed-outcomes) | `empirical` | `weakened` | simulation | {"rationale": "Some component campaigns are retained; the aggregate denominator is not unified.", "status": "mixed_historical"} |
| [`safety-semantic`](#safety-semantic) | `structural` | `supported` | simulation | {"rationale": "Verifier source and a wrong-object regression test establish detection behavior.", "status": "repository_verified"} |
| [`safety-topology`](#safety-topology) | `structural` | `supported` | agnostic | {"rationale": "Source and adversarial topology tests establish the declared-path rule.", "status": "repository_verified"} |
| [`semantic-authorization-hardware`](#semantic-authorization-hardware) | `future` | `hardware_pending` | hardware | {"rationale": "The synthetic replay and any simulation result are bounded to their rendering and adapter envelope.", "status": "not_applicable"} |
| [`session-statistics`](#session-statistics) | `structural` | `supported` | agnostic | {"rationale": "Tracked implementation and exact-method tests establish the capability.", "status": "repository_verified"} |
| [`typed-composition`](#typed-composition) | `structural` | `supported` | agnostic | {"rationale": "Tracked implementation and independently invoked unit tests establish the scope.", "status": "repository_verified"} |
| [`typed-dataflow-causal`](#typed-dataflow-causal) | `causal` | `unrun` | simulation | {"rationale": "No confirmatory treatment records exist.", "status": "not_applicable"} |
| [`typed-evidence-causal`](#typed-evidence-causal) | `causal` | `unrun` | simulation | {"rationale": "No confirmatory comparator records exist.", "status": "not_applicable"} |

## campaign-registrations

Every planned simulation campaign (safety-exposure ablation, semantic-authorization held plans, fault-bank calibration, non-oracle task band, typed-versus-monolithic causal study, fault-evidence study) has a content-addressed registration whose manifest binds hypotheses, endpoints, decision rules, exclusions, instrument set, seed commitment, budgets, integrity gates, and analysis code, and none is frozen because their human review gates are pending.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "not_applicable", "environment": "agnostic", "perception": "not_applicable", "platform": "repository-supported Python hosts", "task": "campaign pre-registration"}
- Experimental unit and sample: {"rationale": "A registry property, not an outcome.", "value": "not_applicable"}; {"rationale": "The committed manifests are checked by unit tests.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No sampling.", "value": "not_applicable"}
- Attestation: {"rationale": "Manifests re-hash clean with seed sources withheld.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/harness/freeze.py", "rationale": "Builds and checks the content-addressed manifests.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_freeze_registry.py::test_committed_registrations_check_clean_with_withheld_seeds", "path": "tests/unit/test_freeze_registry.py", "rationale": "Every committed registration checks clean and none claims a freeze.", "scope": "agnostic"}]
- Counterevidence: ["A registration is a hashing step, not evidence of any outcome."]
- Limitations: ["The confirmatory power inputs are assumptions until pilots exist.", "Freezing requires the pending CON-14 and STA-12 reviews."]
- Allowed wording: {"focused_paper": "Pre-registrations are recorded but not yet frozen.", "readme": "Campaign registrations exist and are pending review; none authorizes scored collection.", "technical_report": "Content-addressed registrations bind each campaign's protocol; all are pending review."}
- Headline markers: []

## development-ledger

README contains the sole canonical dated project-status ledger and keeps development verdict qualifications visible.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "not_applicable", "environment": "agnostic", "perception": "not_applicable", "platform": "repository documentation", "task": "project documentation status"}
- Experimental unit and sample: {"rationale": "This is a documentation-structure claim.", "value": "not_applicable"}; {"rationale": "No experimental sample is used.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Sampling uncertainty does not apply.", "value": "not_applicable"}
- Attestation: {"rationale": "The catalog checker enforces the one-source marker and dated snapshot links.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "README.md", "rationale": "Contains the canonical marker and dated status ledger.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_claim_evidence.py::test_only_one_canonical_current_status_is_allowed", "path": "tests/unit/test_claim_evidence.py", "rationale": "Rejects a second canonical status declaration.", "scope": "agnostic"}]
- Counterevidence: ["The ledger contains heterogeneous historical evidence and is not a pooled benchmark result."]
- Limitations: ["Individual rows retain their own evidentiary strength and may predate SPEC 410."]
- Allowed wording: {"focused_paper": "Historical status is supplemental and defers to the README ledger.", "readme": "The README is the canonical dated development-status ledger.", "technical_report": "Dated report snapshots defer to the README status ledger."}
- Headline markers: [{"marker": "development-ledger/readme", "path": "README.md"}]

## evidence-attestation

The harness checks tracked frozen-artifact identities and records admissibility metadata for measured runs.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "any participant", "environment": "agnostic", "perception": "all", "platform": "repository-supported hosts", "task": "frozen evaluator integrity"}
- Experimental unit and sample: {"rationale": "This is an implemented integrity mechanism.", "value": "not_applicable"}; {"rationale": "No sampled session population is asserted.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Sampling uncertainty does not apply to code behavior.", "value": "not_applicable"}
- Attestation: {"rationale": "The frozen inventory and guard dependencies are tested from tracked sources.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "tools/env_hash.py", "rationale": "Computes and checks frozen artifact hashes.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_env_hash.py::test_the_guards_safety_inputs_are_all_fenced", "path": "tests/unit/test_env_hash.py", "rationale": "Verifies that guard inputs remain in the frozen set.", "scope": "agnostic"}]
- Counterevidence: ["Start-time hashes detect tracked artifact drift but do not prevent every runtime side channel."]
- Limitations: ["This is tamper evidence and admissibility checking, not process isolation."]
- Allowed wording: {"focused_paper": "Frozen evaluator artifacts are hash-checked; process isolation is not claimed.", "readme": "Frozen tracked artifacts are hash-checked and recorded for admissibility.", "technical_report": "The harness provides scoped hash attestation of tracked artifacts."}
- Headline markers: [{"marker": "evidence-attestation/paper-contributions", "path": "docs/paper/aisle-paper.md"}]

## external-reproduction

A party outside the AISLE authorship chain independently reproduces the frozen benchmark and its confirmatory analysis from archived inputs.

- Type / status: `reproducibility` / `unrun`
- Scope: {"agent_model": "independently selected supported agent", "environment": "simulation_and_hardware", "perception": "protocol-defined", "platform": "independently provisioned environment", "task": "frozen public benchmark release"}
- Experimental unit and sample: independent_reproduction_attempt; {"rationale": "No external reproduction attempt has been completed.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No reproduction outcome exists.", "value": "not_applicable"}
- Attestation: {"rationale": "Self-runs are not independent reproduction.", "status": "not_applicable"}
- Evidence: [{"rationale": "Issue #355 remains open.", "value": "not_applicable"}]
- Counterevidence: ["Local CI and author-run regeneration do not satisfy independence."]
- Limitations: ["Public reproducibility wording remains blocked until a retained external record exists."]
- Allowed wording: {"focused_paper": "UNRUN: no external reproducibility result is reported.", "readme": "UNRUN: external reproduction has not been completed.", "technical_report": "UNRUN: local regeneration is not independent reproduction."}
- Headline markers: [{"marker": "external-reproduction/paper-abstract", "path": "docs/paper/aisle-paper.md"}]

## hardware-so101-validation

The complete AISLE benchmark protocol has been validated on a physical SO-101 robot with retained calibration, telemetry, and safety records.

- Type / status: `future` / `hardware_pending`
- Scope: {"agent_model": "protocol-defined", "environment": "hardware", "perception": "hardware-calibrated rung", "platform": "physical SO-101 equipment", "task": "SO-101 physical validation checklist under issue #356"}
- Experimental unit and sample: physical_robot_session; {"rationale": "No physical-robot session has been executed.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No hardware observation exists.", "value": "not_applicable"}
- Attestation: {"rationale": "Simulation and loopback tests are not physical evidence.", "status": "not_applicable"}
- Evidence: [{"rationale": "Hardware is unavailable; issue #356 remains externally blocked.", "value": "not_applicable"}]
- Counterevidence: ["Driver loopback and simulation calibration parity do not constitute robot trials."]
- Limitations: ["No physical performance, safety, or sim-to-real claim is allowed."]
- Allowed wording: {"focused_paper": "HARDWARE PENDING: the physical validation cell is unrun.", "readme": "HARDWARE PENDING: preparation exists, but no physical trial has run.", "technical_report": "HARDWARE PENDING: simulation and loopback are not physical evidence."}
- Headline markers: [{"marker": "hardware-so101-validation/readme", "path": "README.md"}]

## live-fault-feasibility

In three retained development cells, one per perception, decision, and motion fault tier, an agent using live evidence detected the fault, localized the designated node, and restored the measured task outcome.

- Type / status: `empirical` / `supported`
- Scope: {"agent_model": "recorded operator-agent sessions", "environment": "simulation", "perception": "fault-dependent", "platform": "development Mac campaign host", "task": "H6 induced-fault development cells F1, F2, and F3"}
- Experimental unit and sample: coding_agent_session; 3 sessions; one per fault tier; no repeated-session population inference
- Uncertainty: Descriptive 3/3 existence result; no interval used for a population claim
- Attestation: {"rationale": "Cell records and scorer are retained; no independent reproduction exists.", "status": "repository_recorded"}
- Evidence: [{"kind": "protocol", "path": "docs/decisions/ADR-h6-operation-protocol.md", "rationale": "Declares cell scoring, evidence set, and amendments.", "scope": "simulation"}, {"kind": "raw_record", "path": "analysis/h6/records/F1/cell.json", "rationale": "Retained perception-fault cell record.", "scope": "simulation"}, {"kind": "raw_record", "path": "analysis/h6/records/F2/cell.json", "rationale": "Retained decision-fault cell record.", "scope": "simulation"}, {"kind": "raw_record", "path": "analysis/h6/records/F3/cell.json", "rationale": "Retained motion-fault cell record.", "scope": "simulation"}, {"kind": "analyzer", "path": "tools/h6_campaign.py", "rationale": "Implements cell and campaign verdict derivation.", "scope": "agnostic"}]
- Counterevidence: ["There is no conventional-logs comparator and only one session per fault tier."]
- Limitations: ["This supports only the recorded 3/3 feasibility statement.", "It does not support a typed-evidence advantage or general repair rate."]
- Allowed wording: {"focused_paper": "SUPPORTED bounded feasibility; not a typed-evidence treatment effect.", "readme": "SUPPORTED bounded feasibility: the three retained H6 cells met their criteria.", "technical_report": "SUPPORTED bounded feasibility in 3/3 development cells, without comparison."}
- Headline markers: [{"marker": "live-fault-feasibility/readme", "path": "README.md"}, {"marker": "live-fault-feasibility/paper-abstract", "path": "docs/paper/aisle-paper.md"}, {"marker": "live-fault-feasibility/paper-contributions", "path": "docs/paper/aisle-paper.md"}]

## safety-exposure-hardware

Physical exposure ledgers, held-command timing, and contact instrumentation on the SO-101 station reproduce the simulation safety layers.

- Type / status: `future` / `hardware_pending`
- Scope: {"agent_model": "protocol-defined", "environment": "hardware", "perception": "hardware-calibrated rung", "platform": "physical SO-101 equipment", "task": "SO-101 station exposure ledger under SFE-15 and issue #356"}
- Experimental unit and sample: physical_robot_session; {"rationale": "No physical exposure ledger exists.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No hardware observation exists.", "value": "not_applicable"}
- Attestation: {"rationale": "Simulated contacts, gateway receipts, and held-command timing are not physical exposure.", "status": "not_applicable"}
- Evidence: [{"rationale": "Hardware is unavailable; the schema, analyzer, fake-driver ablation, and dry-run commands exist in simulation form only.", "value": "not_applicable"}]
- Counterevidence: ["The fake-driver ablation contains no physics and no contact instrument."]
- Limitations: ["No physical safety claim of any layer is allowed."]
- Allowed wording: {"focused_paper": "HARDWARE PENDING: physical exposure is unmeasured.", "readme": "HARDWARE PENDING: no physical exposure ledger exists.", "technical_report": "HARDWARE PENDING: simulation and fake-driver ledgers must not be relabeled as physical exposure."}
- Headline markers: []

## safety-exposure-ledger

The SPEC 470 exposure ledger regenerates from retained simulation traces and reports, for two instrument sessions (expert_t1 and expert_t2, 16 development episodes), 16 manipulation attempts, 8 deliveries, 24838 valid proposals with 23 clamps, and zero wrong-object events in 2 exposed sessions, an exact one-sided 95% upper bound of 0.776 on the session unit.

- Type / status: `empirical` / `supported`
- Scope: {"agent_model": "expert graphs, no coding agent", "environment": "simulation", "perception": "L1 and L2", "platform": "development Mac campaign host", "task": "T1 and T2 expert graphs on development seeds 0..7"}
- Experimental unit and sample: exposed_session; 2 exposed sessions holding 16 included episodes; instrument pilot, not a campaign.
- Uncertainty: Exact one-sided 95% Clopper-Pearson upper bound 0.776 on sessions with any wrong-object event; the 0/16 episode count is descriptive.
- Attestation: {"rationale": "Ledgers, reports, manifests, and regeneration commands are tracked; raw traces are retained privately with hashes in the ledger.", "status": "repository_verified"}
- Evidence: [{"kind": "raw_record", "path": "analysis/safety-exposure/records/pooled-report.json", "rationale": "Pooled SFE-6 report with denominators, exclusions, and bounds.", "scope": "simulation"}, {"kind": "raw_record", "path": "analysis/safety-exposure/records/sfe-exposure-pilot-01/episodes.jsonl", "rationale": "The T1 session's episode outcomes.", "scope": "simulation"}, {"kind": "analyzer", "path": "src/aisle/harness/exposure.py", "rationale": "Derives the ledger from Arrow traces under frozen exposure rules.", "scope": "agnostic"}]
- Counterevidence: ["Two sessions bound nothing tighter than about 78 percent; episodes within a session are nested, not independent (SFE-8, issue", "Collision rows are a pose-displacement proxy; contact instrumentation is unmeasured."]
- Limitations: ["Simulation evidence only (SFE-15); no physical ledger exists.", "Development seeds, not a held-out distribution."]
- Allowed wording: {"focused_paper": "Zero observed wrong-object events in 2 sessions (16 episodes), session-level upper bound 0.776; not prevention.", "readme": "The exposure ledger reports zero wrong-object deliveries in 2 development sessions (16 episodes) with a session-level upper bound of 0.776.", "technical_report": "A regenerable exposure ledger with explicit denominators exists for two development sessions."}
- Headline markers: []

## safety-identity-authorization

An identity-aware authorization boundary prevents commands for an unauthorized object from reaching actuation.

- Type / status: `future` / `unrun`
- Scope: {"agent_model": "any participant", "environment": "simulation_and_hardware", "perception": "identity-capable rung", "platform": "future ratified boundary", "task": "wrong-object intervention pending issue #352"}
- Experimental unit and sample: coding_agent_session; {"rationale": "The authorization intervention has not run.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No authorization effect estimate exists.", "value": "not_applicable"}
- Attestation: {"rationale": "The authorizer, permit gateway, and synthetic held-plan replay are tracked (SPEC 480); no simulation or hardware transition has been authorized by them.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/harness/semantic_shield.py", "rationale": "Identity-aware authorizer and permit-consuming gateway (SEM-2 to SEM-7).", "scope": "agnostic"}, {"kind": "raw_record", "path": "analysis/semantic-authorization/records/sem-held-plan-adversarial-v2/result.json", "rationale": "Synthetic three-arm replay over 60 held plans; internal consistency of the mechanism with its declared semantics, not a prevention effect.", "scope": "agnostic"}]
- Counterevidence: ["Existing verifier detection occurs after observation and is not command authorization.", "The synthetic corpus, its expected decisions, and the authorizer share an author."]
- Limitations: ["Zero observed wrong-object events cannot establish this prevention claim.", "No sensor-backed simulation trial and no hardware trial has run; see semantic-authorization-hardware."]
- Allowed wording: {"focused_paper": "UNRUN: no semantic authorization effect is reported.", "readme": "UNRUN: identity-aware semantic authorization is not yet demonstrated.", "technical_report": "UNRUN: verifier detection is not identity authorization."}
- Headline markers: []
- Safety category: `identity_authorization`

## safety-kinematic

The budget guard clamps or holds malformed, stale, out-of-limit, and out-of-workspace commands according to its configured envelope.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "any declared motion producer", "environment": "agnostic", "perception": "not_applicable", "platform": "supported embodiments", "task": "supported command envelopes"}
- Experimental unit and sample: {"rationale": "This is a deterministic command-transform property.", "value": "not_applicable"}; {"rationale": "Unit and fuzz tests exercise the transformation contract.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Sampling uncertainty does not apply to the bounded code property.", "value": "not_applicable"}
- Attestation: {"rationale": "Guard implementation and fuzzed command tests are tracked.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/nodes/budget_guard.py", "rationale": "Implements configured command clamping and holds.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_guard_clamping.py::test_fuzzed_commands_never_crash_and_always_legal", "path": "tests/unit/test_guard_clamping.py", "rationale": "Exercises malformed and extreme commands against the legal envelope.", "scope": "agnostic"}, {"kind": "raw_record", "path": "analysis/safety-exposure/ablation/sfe-held-command-ablation-v2/result.json", "rationale": "Fixed-proposal ablation on a fake driver: 0/39 at-risk traces with any driver-received violation under guard_on versus 32/39 observe-only.", "scope": "simulation"}, {"kind": "analyzer", "path": "src/aisle/harness/held_command.py", "rationale": "Replays byte-identical proposals through both enforcement modes.", "scope": "agnostic"}]
- Counterevidence: ["Kinematic clamping does not establish semantic object authorization."]
- Limitations: ["The held-command ablation is fake-driver evidence under the tested limits; physical exposure remains hardware-pending."]
- Allowed wording: {"focused_paper": "The guard enforces configured command bounds on declared paths.", "readme": "The budget guard clamps commands to its configured kinematic envelope.", "technical_report": "Kinematic enforcement is separate from semantic safety."}
- Headline markers: []
- Safety category: `kinematic_enforcement`

## safety-observed-outcomes

Historical development summaries report zero wrong-object events in retained denominators, but the project-wide session denominator is not yet represented by one independently audited raw table.

- Type / status: `empirical` / `weakened`
- Scope: {"agent_model": "heterogeneous historical sessions", "environment": "simulation", "perception": "heterogeneous", "platform": "historical development hosts", "task": "heterogeneous historical development campaigns"}
- Experimental unit and sample: coding_agent_session; Historical summaries include 224 H2 episodes and later campaign denominators.
- Uncertainty: {"rationale": "No unified session-level raw table supports a project-wide interval yet.", "value": "not_applicable"}
- Attestation: {"rationale": "Some component campaigns are retained; the aggregate denominator is not unified.", "status": "mixed_historical"}
- Evidence: [{"kind": "raw_record", "path": "analysis/safety-exposure/records/pooled-report.json", "rationale": "SPEC 470 exposure report over the two instrument sessions (16 development episodes): 0 wrong-object events in 2 exposed sessions, exact one-sided 95% upper bound 0.776 on the session unit (SFE-8). It is the first audited denominator, not the project-wide historical one.", "scope": "simulation"}, {"kind": "analyzer", "path": "src/aisle/harness/exposure_report.py", "rationale": "Regenerates the report from the retained ledgers with reconciliation.", "scope": "agnostic"}]
- Counterevidence: ["Event-free development observations do not prove prevention.", "Historical statements used episode and approximate session denominators inconsistently.", "The audited ledger covers 16 development episodes, not the historical session set."]
- Limitations: ["The allowed claim is observational and simulation-scoped.", "Issue #351's exposure ledger and held-command ablation now exist for two sessions; the historical denominator remains unaudited.", "Physical exposure ledgers are absent; see safety-exposure-hardware."]
- Allowed wording: {"focused_paper": "WEAKENED historical observation, not confirmatory safety evidence.", "readme": "WEAKENED: retained development summaries report zero events; prevention is unproven.", "technical_report": "WEAKENED observational record with no project-wide session interval."}
- Headline markers: [{"marker": "safety-observed-outcomes/readme", "path": "README.md"}, {"marker": "safety-observed-outcomes/paper-contributions", "path": "docs/paper/aisle-paper.md"}]
- Safety category: `observed_outcomes`

## safety-semantic

The verifier implements semantic detection of supported wrong-object observations and emits the corresponding failure outcome.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "not_applicable", "environment": "simulation", "perception": "oracle and recorded realistic-verifier paths", "platform": "verifier-supported hosts", "task": "supported desk and retail verifier states"}
- Experimental unit and sample: {"rationale": "This is a verifier implementation property.", "value": "not_applicable"}; {"rationale": "The row does not estimate detector population performance.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Fidelity uncertainty belongs to separate empirical rows.", "value": "not_applicable"}
- Attestation: {"rationale": "Verifier source and a wrong-object regression test establish detection behavior.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/verifier/realistic.py", "rationale": "Implements realistic-verifier semantic observations and latch behavior.", "scope": "simulation"}, {"kind": "test", "node": "tests/unit/test_realistic_judge.py::test_a_wrong_object_seen_by_the_OVERHEAD_still_fails_the_episode", "path": "tests/unit/test_realistic_judge.py", "rationale": "Confirms a supported wrong-object observation produces failure.", "scope": "simulation"}]
- Counterevidence: ["Detection fidelity is imperfect and does not authorize the commanded target."]
- Limitations: ["Semantic prevention is not attributed to the verifier or kinematic guard."]
- Allowed wording: {"focused_paper": "The verifier emits semantic failures on supported observations.", "readme": "The verifier detects supported wrong-object observations; prevention is not claimed.", "technical_report": "Semantic detection is distinct from kinematic enforcement and authorization."}
- Headline markers: []
- Safety category: `semantic_detection`

## safety-topology

The validator requires every declared motion-command path in an accepted graph to traverse the configured budget guard.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "any graph author", "environment": "agnostic", "perception": "all", "platform": "validator-supported hosts", "task": "validated declared graphs"}
- Experimental unit and sample: {"rationale": "This is a structural graph property.", "value": "not_applicable"}; {"rationale": "The validator checks paths rather than sampled sessions.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Sampling uncertainty does not apply.", "value": "not_applicable"}
- Attestation: {"rationale": "Source and adversarial topology tests establish the declared-path rule.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/harness/validate.py", "rationale": "Implements motion-path gate validation.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_validator.py::test_motion_gate_is_topological", "path": "tests/unit/test_validator.py", "rationale": "Distinguishes topology from node naming and rejects an ungated path.", "scope": "agnostic"}]
- Counterevidence: ["The validator does not model arbitrary process, socket, device, or driver side channels."]
- Limitations: ["Wider bypass claims defer to issue #350."]
- Allowed wording: {"focused_paper": "Declared topology is gated; a process-wide bypass boundary is not claimed.", "readme": "Accepted declared graph paths route motion through the budget guard.", "technical_report": "The validator establishes declared graph-path gating only."}
- Headline markers: [{"marker": "safety-topology/technical-report", "path": "docs/AISLE-technical-report.md"}, {"marker": "safety-topology/paper-abstract", "path": "docs/paper/aisle-paper.md"}, {"marker": "safety-topology/paper-contributions", "path": "docs/paper/aisle-paper.md"}]
- Safety category: `graph_topology`

## semantic-authorization-hardware

A sensor-backed identity adapter on physical hardware meets the pre-registered false-allow, false-block, latency, bypass, and feasibility criteria.

- Type / status: `future` / `hardware_pending`
- Scope: {"agent_model": "protocol-defined", "environment": "hardware", "perception": "hardware sensor adapter", "platform": "physical SO-101 equipment", "task": "sensor-backed semantic authorization under SEM-14 and issue #356"}
- Experimental unit and sample: physical_robot_session; {"rationale": "No calibrated physical trial has retained the underlying observations.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No hardware observation exists.", "value": "not_applicable"}
- Attestation: {"rationale": "The synthetic replay and any simulation result are bounded to their rendering and adapter envelope.", "status": "not_applicable"}
- Evidence: [{"rationale": "Hardware adapter contract and dry-run fixtures may validate schema and refusal paths only.", "value": "not_applicable"}]
- Counterevidence: ["Oracle-arm success in simulation cannot justify deployability or physical prevention."]
- Limitations: ["H5 stays narrowed to measured layers and zero observed events until these criteria are met."]
- Allowed wording: {"focused_paper": "HARDWARE PENDING: no deployable identity source is demonstrated.", "readme": "HARDWARE PENDING: no physical semantic-authorization trial has run.", "technical_report": "HARDWARE PENDING: sensor-backed authorization is unmeasured on hardware."}
- Headline markers: []

## session-statistics

AISLE implements session-level binary, continuous, survival, power, and zero-event analyses with preserved assignments and exclusions.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "any recorded treatment arm", "environment": "agnostic", "perception": "not_applicable", "platform": "repository-supported Python hosts", "task": "benchmark session analysis"}
- Experimental unit and sample: coding_agent_session; {"rationale": "This row concerns implemented analysis behavior, not a treatment result.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Each future campaign supplies its own estimate and interval.", "value": "not_applicable"}
- Attestation: {"rationale": "Tracked implementation and exact-method tests establish the capability.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/harness/benchmark_statistics.py", "rationale": "Implements session flow, effects, intervals, power, and release gates.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_benchmark_statistics.py::test_binary_effect_uses_sessions_reports_strata_and_artifact_claim_guard", "path": "tests/unit/test_benchmark_statistics.py", "rationale": "Checks session effects, strata, uncertainty, and artifact claim guard.", "scope": "agnostic"}]
- Counterevidence: ["An implemented analyzer does not make any confirmatory campaign result exist."]
- Limitations: ["Independent statistical review remains open under STA-12."]
- Allowed wording: {"focused_paper": "Analysis tooling exists, while confirmatory effects remain unrun.", "readme": "AISLE implements session-level benchmark statistics; no campaign effect is implied.", "technical_report": "The common analyzer implements registered session-level methods."}
- Headline markers: []

## typed-composition

AISLE implements a typed capability registry and static graph validator that reject incompatible declared graph structures with stable errors.

- Type / status: `structural` / `supported`
- Scope: {"agent_model": "any participant using the repository interface", "environment": "agnostic", "perception": "all declared rungs", "platform": "repository-supported Python hosts", "task": "shipped AISLE graph corpus"}
- Experimental unit and sample: {"rationale": "This is a structural property of tracked code and fixtures.", "value": "not_applicable"}; {"rationale": "Structural source and corpus tests are not sampled sessions.", "value": "not_applicable"}
- Uncertainty: {"rationale": "Sampling uncertainty does not apply to the structural claim.", "value": "not_applicable"}
- Attestation: {"rationale": "Tracked implementation and independently invoked unit tests establish the scope.", "status": "repository_verified"}
- Evidence: [{"kind": "source", "path": "src/aisle/harness/validate.py", "rationale": "Implements schema, topology, embodiment, and stable-error validation.", "scope": "agnostic"}, {"kind": "test", "node": "tests/unit/test_validator.py::test_every_shipped_graph_validates", "path": "tests/unit/test_validator.py", "rationale": "Exercises every shipped tracked graph through the validator.", "scope": "agnostic"}]
- Counterevidence: ["Graph validity does not establish that external packages are installed or a graph launches."]
- Limitations: ["The claim concerns declared structures, not arbitrary participant process behavior.", "Runtime success and causal engineering benefit require separate evidence."]
- Allowed wording: {"focused_paper": "The implemented substrate statically validates declared typed graphs.", "readme": "AISLE implements typed capability composition and static graph validation.", "technical_report": "Tracked tests support the declared typed-graph validation contract."}
- Headline markers: [{"marker": "typed-composition/readme", "path": "README.md"}, {"marker": "typed-composition/paper-abstract", "path": "docs/paper/aisle-paper.md"}, {"marker": "typed-composition/paper-contributions", "path": "docs/paper/aisle-paper.md"}]

## typed-dataflow-causal

Typed dataflow engineering improves coding-agent outcomes relative to an equal-capability monolithic interface.

- Type / status: `causal` / `unrun`
- Scope: {"agent_model": "randomized coding-agent sessions", "environment": "simulation", "perception": "frozen by the future protocol", "platform": "frozen campaign platform", "task": "non-oracle benchmark task band pending issue #346"}
- Experimental unit and sample: coding_agent_session; {"rationale": "Confirmatory sessions have not been collected.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No treatment estimate exists before the controlled campaign.", "value": "not_applicable"}
- Attestation: {"rationale": "No confirmatory treatment records exist.", "status": "not_applicable"}
- Evidence: [{"rationale": "Issue #347 remains dependent on controls and protocol freeze.", "value": "not_applicable"}, {"kind": "protocol", "path": "analysis/freeze/cse-causal-study-pilot-v1/declaration.json", "rationale": "CSE-8 power-input source (ADR-66 pilot tier): the pilot registration cse-causal-study-pilot-v1 (purpose pilot, fresh seed commitment, 6 sessions per arm). Zero sessions were collected; see the report.", "scope": "simulation"}, {"kind": "protocol", "path": "analysis/reports/cse_pilot_v1.md", "rationale": "CSE-8 power inputs as recorded by the pilot: control success = not collected (0 of 0), effect direction = unobserved, per-session tokens/wall/rollouts = none, validate-fix cycles = none. The confirmatory sample size stays an assumption-based 49 per arm (control 0.40, risk difference 0.25) until a pilot runs.", "scope": "simulation"}, {"kind": "raw_record", "path": "analysis/freeze/cse-causal-study-pilot-v1/executor-preflight.json", "rationale": "Refusal receipts for why the pilot could not collect with the current infrastructure (T1-only development mode, oracle state on the policy path, no eligible BND stratum).", "scope": "simulation"}, {"kind": "protocol", "path": "analysis/freeze/cse-causal-study-pilot-v8/declaration.json", "rationale": "CSE-8 power-input source, second registration (ADR-66 pilot tier, 2026-09-13): fresh seed commitment, instrument set fixed to the T1-L2 realistic pair on Codex 0.153.4, sealed balanced assignments, session success = 28 of 32 controller-run held-out oracle successes, 6 sessions per arm. Collection pending; no session has run.", "scope": "simulation"}]
- Counterevidence: ["Historical single-session ablations do not isolate typed dataflow structure.", "The monolithic equal-capability control exists (SPEC 440 surface, docs/monolithic/) and passed an 8/8 seeds 0..7 engineering shakeout, but expert parity is blocked on independent authorship (MON-9) and no agent session has used it."]
- Limitations: ["No superiority, equivalence, or non-inferiority conclusion is currently allowed.", "The 2026-09-12 pilot registration (cse-causal-study-pilot-v1) collected zero sessions; every CSE-8 power input is still an assumption and the pilot's gates are all pending.", "The 2026-09-13 registration (cse-causal-study-pilot-v8) is pending collection; its BND-12 gate is recorded failed (BND-7, not perception-eligible) and every other gate is pending, so any result it yields is a named-limitation pilot input, never a claim."]
- Allowed wording: {"focused_paper": "UNRUN: no typed-dataflow treatment effect is reported.", "readme": "UNRUN: typed-versus-monolithic superiority is the confirmatory claim under test.", "technical_report": "UNRUN: historical development results do not establish the causal effect."}
- Headline markers: [{"marker": "typed-dataflow-causal/readme", "path": "README.md"}, {"marker": "typed-dataflow-causal/paper-abstract", "path": "docs/paper/aisle-paper.md"}, {"marker": "typed-dataflow-causal/paper-introduction", "path": "docs/paper/aisle-paper.md"}, {"marker": "typed-dataflow-causal/paper-contributions", "path": "docs/paper/aisle-paper.md"}]

## typed-evidence-causal

Typed runtime evidence improves hidden-fault localization and repair relative to conventional logs under a sealed fault assignment.

- Type / status: `causal` / `unrun`
- Scope: {"agent_model": "randomized coding-agent sessions", "environment": "simulation", "perception": "fault-stratified", "platform": "frozen campaign platform", "task": "sealed live-fault bank pending issue #348"}
- Experimental unit and sample: coding_agent_session; {"rationale": "Comparator sessions have not been collected.", "value": "not_applicable"}
- Uncertainty: {"rationale": "No session-level treatment estimate exists.", "value": "not_applicable"}
- Attestation: {"rationale": "No confirmatory comparator records exist.", "status": "not_applicable"}
- Evidence: [{"rationale": "Issue #349 awaits the sealed bank, comparator, and protocol freeze.", "value": "not_applicable"}]
- Counterevidence: ["The H6 feasibility cells exposed typed evidence to every participant and had no logs-only arm."]
- Limitations: ["Feasibility cannot be relabeled as a comparative localization benefit."]
- Allowed wording: {"focused_paper": "UNRUN: no typed-evidence localization effect is reported.", "readme": "UNRUN: typed evidence versus conventional logs has no treatment result.", "technical_report": "UNRUN: H6 is feasibility evidence, not a logs comparator."}
- Headline markers: [{"marker": "typed-evidence-causal/paper-abstract", "path": "docs/paper/aisle-paper.md"}, {"marker": "typed-evidence-causal/paper-contributions", "path": "docs/paper/aisle-paper.md"}]
