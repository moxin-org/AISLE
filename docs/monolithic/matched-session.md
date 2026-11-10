# Matched engineering session controller

Work in progress for [#519](https://github.com/heyong4725/aisle/issues/519).
The source-checkout entry point is `tools/matched_campaign.py`. It currently
supports admission and an unscored engineering process attempt. Full MON-12 tool,
run, evaluator and guard evidence integration is still under implementation.
Neither a successful admission nor `ok: true` from an engineering attempt is a
baseline result or authorization to collect a pilot or confirmatory session.

Run from the source checkout's locked environment:

```sh
uv run --extra sim --locked --no-sync python tools/matched_campaign.py admit \
  --request /absolute/controller/admission-request.json \
  --output /absolute/controller/plan.json
uv run --extra sim --locked --no-sync python tools/matched_campaign.py run \
  --request /absolute/controller/run-request.json \
  --output /absolute/controller/attempt-001 \
  --arm typed --session-id engineering-001 --purpose engineering
```

These commands emit one JSON object and exit zero only when its `ok` is true.
Use a new plan filename or attempt directory each time. Outputs must be outside
participant views and attempt evidence must also be outside participant scratch.
The controller refuses overwriting existing evidence or resuming an attempt.
The request files and capability records are controller inputs; participants
must not author or modify them.

The admission request is a JSON object containing:

- `root`: absolute controller source root. Its controller and runner bytes must
  match the executing checkout. The real treatment table and interface checks
  must pass.
- `visible_roots`: absolute `typed` and `monolithic` view paths, disjoint from
  each other and the controller root.
- `candidates`: both complete SPEC 420 treatment manifest inputs, with each
  assignment's arm matching its key and exact arm-specific editable grants.
- `confinement`: per-arm `scratch` and `policy` objects. Policy fields use the
  canonical `MacOSPolicy` representation. Actual compiled hashes must match
  the manifests, and the controller checks equivalent authority across arms.
- Optional `private_roots`: named controller-private directory paths, such as
  `session_evidence` and `capability_evidence`. Confinement is required. Roots
  must be canonical existing directories, explicitly hidden from both arms,
  and disjoint from other roles, controller source and every participant grant.
  Only these exact verified paths compare by role; undeclared differences remain
  significant. Original paths stay in the hashed plan and are rechecked before
  execution and during postflight admission.
- `ambient`: per-arm `environment` and `record` from
  `build_declared_environment`. Generated private paths and concrete hashes
  are rechecked; environment inheritance is unavailable in this entry point.
- `launches`: per-arm objects containing `argv`, including a canonical absolute
  executable path. Both argument lists must match. The executable's bytes must
  match the manifest's agent CLI hash. Each manifest must also declare a finite
  positive `budget.wall_ceiling_s`.
- Optional `prompt_row`: the explicit MON-1 documentation-row id described in
  [ADR-62](../decisions/ADR-62.md). This binds document availability and bytes;
  delivered prompt binding remains part of the unfinished integration.

Confinement, ambient and launch bindings are optional for inspection-only
admission, but all three are required before the process runner will start.
An admitted inspection plan is not necessarily executable.

The run request contains `root`, `visible_roots`, the complete retained `plan`,
`profile_path`, the complete capability `attestation` object, and
`hidden_access_log`: either the path of a controller-supplied log or
`{"collect": "macos-sandbox-reports"}`. With the latter the controller streams
the macOS sandbox's own denial reports (`log stream`, sender `Sandbox`) for the
whole launch window, keeps the reports whose process name is an executable
admitted by the arm, validator or any prepared worker policy, classifies each
target against the arm's readable and hidden roots, and retains the result as
`hidden-access-log.json` beside a `hidden-access-collection.json` record (the
window, the selection, counts and pids, never targets) under
`access-collection/` in the attempt directory (TRT-6). `adapter_active` is
true only when the verified wrapped command runs under the real
`sandbox-exec`, so a collected run under any other adapter is retained and
excluded by postflight (`confinement_inactive`), not refused at admission.
The kernel does not report every denial it enforces: on the development host
it stopped reporting denials by third-party binaries (the admitted uv
interpreter, node) after a burst of sessions while still reporting those of
Apple-signed ones and still enforcing all of them. `complete` therefore rests
on positive controls, not on stream liveness alone: at the start and at the
end of the window the controller runs the arm's own admitted executable
(interpreter, else shell, else cat) under the arm's profile against a fresh
controller-owned canary under `/private/tmp/aisle-canary-*` whose read the
profile denies, and requires a file-read denial from that executable on the
exact canary. The collection record lists selected process names the markers
did not probe; any such name leaves `complete` false. A
missing marker, a stream that started late, died, or had to be killed, or a
reader failure all leave `complete` false and the attempt excluded
(`hidden_access_log_incomplete`); a policy that admits no interpreter, shell
or cat cannot run its markers and refuses collection, as does a run whose
workers are provisioned dynamically (no worker policy to select). Allows are
never reported, so `visible_allows` stays zero, and selection is by process
name because the kernel's report carries no ancestry: the controller runs one
session at a time and no capability audit during a session, and the retained
pids let a reviewer cross-check the selection. Distinct executables that share
one process name also cannot be distinguished by this sensor, so an independent
coverage check is still required before treating the log as complete study
evidence. These are stated limitations. The supplied capability must match the bound adapter,
compiled profile and current adapter bytes. This capability is limited to
unscored engineering work; it does not establish the independent study gates.
The runner uses the plan's environment, zero prior spend and the declared token
and wall ceilings. It preserves stdout, stderr, token samples and the underlying
process record, including on a nonzero process exit. Final snapshots and
postflight failures remain in `matched-session.json` alongside the original
launcher failure.

The `attestation` must have been produced under the arm's exact confinement
policy, or the launch wrapper refuses it: `python -m
aisle.harness.treatment_confinement audit-macos --policy <policy.json>
--output <attestation.json>` runs the full deny/allow matrix under that policy
(its sentinels are placed inside the policy's own visible, output and hidden
roots and removed afterwards) and retains the profile hash and policy id the
wrapper demands (TRT-5/TRT-7). The policy must admit the audit's probes
(`/bin/bash`, `/bin/cat`, the Apple developer Git, and either `/usr/bin/nc`
or a Python interpreter for the Unix-socket case) and must not admit its
unlisted-executable control (`/usr/bin/printf`). By default each sentinel goes
into the last directory of the matching root list; `--sentinel-visible`,
`--sentinel-output` and `--sentinel-hidden` name controller-private
directories instead so the audit never writes into another arm's readable
view. The retained report replaces hidden-root paths with their digests
(`policy_id` still binds the full policy). A sentinel left behind by an audit
process that died uncleanly is reclaimed on the next run; any other occupant
refuses. Two network policies exist ([ADR-69](../decisions/ADR-69.md)).
`deny-external` allows no network. `loopback` pins one `loopback_port` and
compiles to `(allow network-outbound (remote tcp "localhost:<port>"))`: TCP
only, because `remote ip` also admits UDP. SBPL's `localhost` covers every
address the host owns, LAN and link-local included, so the grant is exclusive
only while the controller holds the port on the dual-stack wildcard and every
assigned address (`relay_reservation.py`). Under `loopback` the audit holds the
port exactly that way and requires `loopback_tcp_read` (the relay canary),
`foreign_loopback_tcp_read`, `external_tcp_read` and `udp_send` (each refused
with "Operation not permitted"), `unrestricted_udp_baseline`, and
`reservation_intruder_bind` (no unsandboxed bind of the port on any covered
address or wildcard under any reuse-flag combination).

A worker or validator policy admits only an interpreter and cannot run the
audit's probes. `--widen-probes` audits it under the policy plus exactly those
probes and the read roots they load from. The attestation binds the session
profile and records the additions under `probe_widening`; the launch wrapper
recomputes the audited profile from the session policy and the host's Apple Git.
`attest-many --policies <dir> [--widen-probes] [--jobs N]` audits each
`<name>.policy.json` concurrently, serializing policies that share a sentinel
home. It writes `<name>.attestation.json` beside each policy, validates
existing attestations before skipping them, and names each policy it cannot
attest (CON-8). An optional `<name>.sentinels.json` names controller-private
sentinel homes. The attestation is an unsigned controller-owned document:
recomputation guards against tool misuse, not against an author who can edit
that document.

A `loopback` arm must launch the App Server with an owned `provider` whose
`relay_port` equals the policy's `loopback_port`, and may not use
`mcp_harness` or `code_mode_host`: both listen on ephemeral ports a one-port
grant cannot reach, so admission refuses them. At launch the controller
reserves the port on every host address (refusing the session if any bind
fails) and retains `relay-reservation.json` (port, per-family counts and an
address-set digest, never addresses). The relay serves from the reservation's
`127.0.0.1` socket and requires a fresh per-session `X-Aisle-Relay-Token`.
Codex reads it from `AISLE_RELAY_KEY` through
`model_providers.aisle_authorized.env_http_headers`, so only the variable's
name appears in argv. The value is a controller-only addition to the
frontend's environment: `invocation.json` hashes the declared environment and
lists `controller_environment_keys`. Codex's default shell-environment
excludes (`*KEY*`) keep it out of agent commands' environment, but the agent
(or any same-user process) can still read it from Codex's process arguments
through `KERN_PROCARGS2`, which the sandbox does not mediate. The token
excludes other users' processes and untargeted traffic, not the agent; see
ADR-69. Before admission, nothing a caller does fails the session: stalls,
resets, tokenless requests and slot floods end only that connection.
The controller re-enumerates host addresses every second and at teardown; any
change fails the session as an unattested environment. This remains
engineering infrastructure: the pilot still requires an attested session,
exact-context admission and the registration's other gates before
collection.

Deterministic integration tests run real child processes for both arms through
an explicitly synthetic pass-through adapter. Those tests cover the process
boundary and failure retention. Their synthetic access logs and capability
records are not evidence of operating-system confinement. Dedicated campaign
access, external confinement, expert parity, protocol freeze, statistical review
and evaluator separation remain separate prerequisites.

`matched-session.json` now includes `common_evidence` under the versioned
`aisle.matched-common-evidence.v1` schema. Its assignment is tied to the verified
arm manifest, and its budget section separates declared limits from observed
usage. Unobserved values are `null`, not zero. The validation property references
one explicit MON-1 declaration; monolithic absence of typed validation is not
reported as a missing typed-validator result.

The envelope references retained transcript hashes and names collections that
have not yet been collected. Its `complete` marker currently remains false:
global frontend-tool coverage, total simulator work, and complete session
reconciliation remain unverified. File
presence alone will not establish complete evidence. Malformed process telemetry
and symlinked retained artifacts produce infrastructure exclusions. Non-finite
process records are retained as rejected diagnostic text so they cannot break
serialization of the attempt record.

Each audited run entry preserves its verdict, returned result, process status,
preparation diagnostics, and the retained stdout path alongside its collection.
Normal gate refusals therefore retain their explanation even when no simulation
manifest exists. A successful evidence collection does not change the run verdict.

For an executable attempt, `launch.json`, `launch-profile.sb` and
`capability.json` retain the command, exact profile and capability inputs. The
wrapper uses the retained profile itself. Their hashes are included with the
process streams in the attempt record.

A controller-selected engineering run can be retained separately with:

```sh
uv run --extra sim --locked --no-sync python tools/matched_campaign.py collect-run \
  --request /absolute/controller/collection-request.json \
  --output /absolute/controller/retained-run-001
```

The request contains `source` (the canonical run directory) and `run_id` (the
expected manifest identity). The collector copies every regular file under
`raw/`, retains hashes, checks source stability, and writes `run-evidence.json`.
It refuses symlinks, overlapping source/output directories and reused outputs.
Missing files, invalid JSON or mismatched identities leave a failed collection
record with the raw files copied before the failure.

Collection success does not change episode outcomes. The evaluator section
retains the declared verifier, planned seeds and observed seeds. Its `complete`
flag reports whether the ordered observed seed list matches the plan; an
unfinished run remains visibly incomplete. Guard summaries are derived from the
retained Arrow streams, with malformed streams rejected. Absent guard evidence
stays `not_collected`.

This command does not establish that a run belonged to a particular fresh agent
session, authenticate an independent evaluator, or change the run's campaign
purpose. The session tool dispatcher still needs to bind these collections into
the common envelope. Collection records always remain ineligible for estimates.

The engineering `check` entry point invokes the actual T1/Franka arm check:

```sh
uv run --extra sim --locked --no-sync python tools/matched_campaign.py check \
  --request /absolute/controller/check-request.json \
  --output /absolute/controller/check-attempt-001 \
  --arm monolithic --session-id engineering-001
```

The request contains `root`, `visible_roots`, `plan`, `python` (the absolute
harness interpreter path), `profile_path` and `attestation`. The manifest must
admit `harness.check` in `policy.allowed_external_tools`, the interpreter's
`harness-python` entry in `runtime_binaries`, and positive `tool_ceiling` and
`tool_wall_ceiling_s` limits in `budget`. The interpreter must also have an
execution grant in the admitted confinement policy. The supplied profile and
adapter must match the admitted capability bindings.

The controller fixes the command: typed calls `harness validate` on its graph;
monolithic calls `harness monolith check` on its module. It supplies neither an
`--allow-unproven` override nor arbitrary caller arguments. It permits changes
only to the active arm's previously declared editable files; the fresh-launch
preflight remains strict. Invocation arguments, stdout, stderr, exit status and
hashes are retained with started/finished entries in `tool-events.jsonl`.
Negative harness verdicts remain tool results. Launch failures, timeouts,
inconsistent verdicts and exhausted budgets remain infrastructure exclusions.

Tool counts and wall spend are enforced within one `ToolController` instance.
The standalone command creates one engineering attempt and does not establish a
persistent campaign ledger. Linking a live session's request channel, tool
controller and collected runs remains part of #519's unfinished integration.
Tests use a synthetic pass-through adapter to exercise the real arm commands;
they do not establish external confinement or a ready research-agent runtime.

For a live engineering session with `harness.check` admitted, the `run` entry
point now hosts one tool service alongside the agent process. Both arm launch
bindings must include the same `tool_python` path, whose bytes match the admitted
`harness-python` runtime. Provision an empty, canonical `HOME/tool-channel`
directory for each arm before launch. Existing request files prevent reuse.

A client can call `request_check(channel, timeout_s=30)` from
`aisle.harness.matched_tool_service`. The channel accepts only versioned `check`
requests with unique 32-digit hexadecimal ids. It accepts no argv, file selectors,
policy changes or budget overrides. The controller reads bounded regular files
through an anchored directory descriptor and refuses symlinks and hard links.
Responses use exclusive creation, so existing entries cannot redirect controller
writes. The session controller's budget and journal persist across client calls.

The participant-writable response is a convenience for the client, not an audit
attestation. Request copies, request-to-attempt links, the authoritative tool
journal and `tool-service.json` remain in the controller's evidence directory.
The request-to-attempt link is flushed before publishing a response, so a blocked
response cannot erase the observed invocation. A service error or pending request
at shutdown invalidates the engineering session while retaining process evidence.

The service dispatches checks and registered development runs. To admit runs,
declare `harness.run` and supply a `development` object at admission with
`schema_version: aisle.matched-development.v1`, `purpose: expert_parity`,
`tier: T1`, `embodiment: franka`, `verifier: oracle`, `reset: teleport`,
distinct nonnegative integer `seeds`, positive integer `run_ceiling` and
`episode_ceiling`, and finite positive `timeout_s`. The episode ceiling must
cover the seed list. `development.timeout_s` is passed to the ordinary rollout
as its execution budget. The enclosing run-controller transaction also provisions
workers and retains evidence after rollout shutdown, and uses only the remaining
admitted `tool_wall_ceiling_s`. It is not cut off at the inner rollout deadline;
a shorter remaining tool budget still wins. Setup and collection therefore need
space within the declared tool budget. A forced outer timeout remains an
infrastructure exclusion, and incomplete worker evidence does not pass auditing.
This development label does not certify expert parity.

A `development` object may instead use `schema_version:
aisle.matched-development.v2`, which names one `candidate` id from the
controller-owned `docs/monolithic/candidates.json` and binds that table's
SHA-256 as `candidates_sha256`, alongside the same `seeds`, `run_ceiling`,
`episode_ceiling` and `timeout_s`. Admission resolves the candidate's tier,
embodiment, perception rung, verifier, reset mode, graph, turn plan and
monolithic module from the table and retains them in the plan; a stale table
hash, an unknown id, a free launch field, or a candidate whose artifacts are
not all present in the controller tree refuses admission. The v1 form remains
the fixed T1/franka/oracle/teleport protocol. Every arm launch reads its
surface from the retained development form: the typed validation snapshot
records the candidate's graph, turn plan and allowlist, and the execution
bundle, graph stage, worker hosts and rollout transport bind those recorded
paths (never a literal); the monolithic launcher receives the candidate's
module, template graph, verifier and reset mode, and refuses any other value.
Both arms receive the same verifier, reset mode, embodiment, seeds and
budgets. Naming a candidate does not
qualify it: the declared `t1-l2-realistic` row is refused until its monolithic
pair exists, and admission never substitutes an easier task.
Clients use `request_run(channel, timeout_s=...)`; they cannot alter these inputs.
Run and episode reservations persist across requests in the controller instance.

Each run uses the actual arm launcher with the admitted parameters and a run ID
derived from session, plan, arm and attempt. After the child stops, the controller
retains its run directory even when the process crashed, timed out or emitted
invalid JSON. On a wait failure, both configured run controllers and direct arm
tool launches observe descendants before reaping the owned process, check their
process identities, and terminate observed children even if they started separate
process sessions. The cleanup report is retained in the child process record or
the direct tool attempt. It refuses to signal a reaped
controller PID and bounds process observation and the final liveness check.
This remains operational cleanup: a child reparented before observation may be
absent, and the report explicitly sets `exhaustive: false`. It does not establish
independent process containment or make an incomplete run admissible.
Missing or malformed episode evidence remains an exclusion with
the partial raw files preserved. Session auditing checks the collection report,
run identity, complete file index and raw file hashes, including nested files.

Run-evidence collection executes in an isolated controller child with the
remaining admitted tool wall budget. Blocking collection is stopped and reaped
with bounded cleanup grace; timeout or cancellation retains an invalid terminal
attempt and any partial files. The `run-collector` directory records the fixed
invocation, output streams, process result and cleanup. A partial directory is
never treated as completed collection. The journal audit checks these receipts
and their artifact hashes, while independent containment and study admission
still require their separate evidence.

Successful paired simulation, complete aggregation and final session-evidence
validation remain unfinished. Live child-process tests exercise both arms with explicit
engineering fixtures and synthetic adapters; they do not authorize a coding-agent
campaign or establish external confinement.

Session finalization now audits a retained tool journal before using its metrics.
It checks paired start/finish events, session/plan/arm identities, attempt sequence,
content identities, matching `attempt.json` records and all declared artifact
hashes. Every observed child process must have its invocation, profile, stdout
and stderr. A returned verdict must match the retained stdout. Hosted sessions
also check service identity and request-to-attempt links against retained requests.
A failed audit invalidates the engineering session and leaves the audited tool
metrics unknown; the audit itself is retained as `tool-audit.json`.

The common budget section names the verified quantities
`controller_tool_requests`, `controller_tool_processes` and
`controller_tool_wall_s`. They cover this controller's request channel only.
Overall agent `tool_calls` remains unknown because frontend-native actions still
need verified telemetry and linkage. The complete common-evidence marker remains
false pending that work and complete rollout aggregation.

Every executable launch binding must also include `system_prompt_arg`, a positive
index into its `argv`. The selected argument's UTF-8 SHA-256 must equal the arm
manifest's `prompts.system_sha256`. The controller validates this before launch
and retains the mapping under `launch.json.system_prompt_delivery`. Engineering
process tests verify that the child receives those bytes. The receipt explicitly
leaves `provider_role_verified` false: argument delivery does not attest to a
frontend's interpretation or to the provider's effective system instructions.

Launch bindings also require `research_contract_arg`, a distinct positive argv
index. For a plain-text contract, its argument must hash to
`prompts.research_contract_sha256`. When `prompt_row` declares arm-specific
documents, pass an ordered JSON array of objects with `path` and `text` keys,
using exactly the row's document paths and UTF-8 contents. Serialize with
`json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))`. This verified
bundle is the sole permitted difference in launch arguments between arms.
`launch.json.research_contract_delivery` retains the mapping and encoding;
`frontend_interpretation_verified` remains false.

Frontend observations are retained in `frontend-tools.json`, bound to the
controller-retained transcript hash. `frontend_tool_calls` counts distinct
observed tool IDs; pending calls and their source line numbers remain available.
Codex start/update/completion events share one identity, while completed-only
file-change events count once. Claude tool-use/result blocks are paired by ID.
Malformed or unidentified events, or unverified stream completion, leave the
usable count unknown. Missing or invalid event or Claude content-block types,
and unsupported item lifecycle events, invalidate telemetry instead of silently
counting zero calls. These observations do not establish full frontend coverage
or linkage to controller requests; `tool_calls` remains unknown.

The supported event shapes are based on the official
[Codex exec events](https://github.com/openai/codex/blob/main/codex-rs/exec/src/exec_events.rs)
and [Claude Agent SDK message types](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py).
Pinned frontend-version conformance remains necessary before study collection.

An executable engineering plan may declare positive integer
`budget.frontend_tool_ceiling`. The matched launcher then attaches an incremental
observer to the supervisor's live stdout pipe. More distinct observed calls than
that ceiling stop the process group with `frontend_tool_budget`; malformed
observed telemetry causes an infrastructure exclusion. Start/update/completion
records are processed once and do not multiply a call. `frontend-live.json`
retains the live observations and ceiling even after a failed process. The
existing supervisor's default behavior is unchanged when no guard is supplied.
This is reactive enforcement of reported events, not proof that unreported work
was prevented. Complete tool coverage and controller-request linkage remain
unverified.

### Engineering worker configuration

The monolithic CLI accepts a controller-owned worker configuration through
`--worker-config /absolute/path/config.json --worker-config-sha256 <sha256>`
on `check` and `run`. Both arguments are required together. The configuration
uses `aisle.monolith.worker-config.v1`, `purpose: expert_parity`, an embodiment,
the exact `module_sha256`, an absolute `output_root`, and a `launch` object with
the worker bundle/manifest, policy, profile, adapter evidence, interpreter/hash,
declared environment/record, pipe timeout, primitive-call cap and handle cap.
It is an engineering configuration, not a study admission or attestation.

A check reserves `<output_root>/check`; runtime node construction reserves
`<output_root>/run`. Existing evidence is never resumed. Use a fresh bound
configuration for each tool attempt. Configuration bytes and authored source
must still match their hashes. The stamped graph carries the configuration path
and hash only to the monolithic broker. That broker keeps the primitive objects
and action validation in its process while executing authored code in the worker.

The matched-session controller can provision these configurations from private
worker-preparation declarations, as described below. This integration does not
establish complete descendant-resource enforcement, frontend coverage, or the
independent evidence required to complete #519.

The admitted typed-check path accepts both `tool_runtime` and `typed_validation`
in the admission request. `typed_validation` uses schema
`aisle.typed-validation-binding.v1` and contains `bundle`, `bundle_manifest`,
`snapshot_storage`, `policy`, `profile_path`, `attestation`, `python`,
`python_sha256`, `environment`, and `environment_record`. These are fixed
controller inputs; tool requests cannot replace them. The separate validator
profile grants read access to the bundle and snapshot storage, private-HOME
writes, and the bound runtime. Controller and participant source roots remain
hidden. The adapter receipt must match that exact profile and the admitted
adapter binary; synthetic test receipts do not establish actual confinement.
The admitted launch resolves the interpreter to its canonical executable and
imports validator dependencies from the verified runtime tree roots, after the
validator bundle. These roots must contain the dependencies as importable
packages. They are added directly to the import path without processing their
`.pth` files; candidate snapshots are not import roots.

With this binding, `ToolController.check()` copies the complete typed deliverable
and pinned registry dependencies into a fresh snapshot and invokes the normal
validator through the separate adapter. Successive calls retain distinct
snapshots under the same profile. A controller claims empty snapshot storage for
its session; pre-existing storage is refused, and later calls check the retained
owner. The tool journal binds the nested validator/runtime evidence as well as
its usual stdout, stderr, profile and invocation files. Fresh plan admission also
requires empty generated controller/validator HOME layouts and empty snapshot
storage. Active-plan verification permits state created by the owning arm while
continuing to require fresh private state for the inactive arm. Direct construction
of a runtime-bound tool controller checks the fresh plan for either arm before
accepting requests. A runtime-bound typed
check without a validation binding is refused. Earlier engineering fixtures
without these bindings retain their existing check path; that path is not a
claim of confined typed execution or campaign readiness.

Dynamic typed run evidence includes the complete indexed typed-provider tree,
including preparation that precedes a rollout-gate refusal. The journal checks
each successful stage's snapshot, stage identity, location and file hashes
against the sealed run and retained inventory. This is retention and binding
evidence, not a substitute for worker protocol audits or simulation results.

For controller-provisioned worker runs, the run request may include
worker_preparations, a private list with one entry per reserved development run.
A typed entry is the per-launch list of node-to-worker declarations accepted by
run_with_workers; a monolithic entry is its single worker launch declaration.
The controller copies this list before serving requests. Each run reserves its
budget, selects the next entry, retains its exact bytes and SHA-256, and prepares
the current authored sources. A failed preparation consumes that entry. An empty
or exhausted list refuses the run. Participant requests still specify only the
operation, with no worker paths, profiles, seeds or gate overrides.

These declarations must contain actual verified capability bindings. This input
path does not generate arbitrary new worker profiles or independent study
attestations. Supplying preparations enables the ordinary tool-service run path. Provider
templates below construct typed stages from the validated current-source snapshot
and monolithic workers from the sealed current module. Complete successful-session
coverage remains unverified.

Worker runtime imports: each runtime tree root is also an explicit Python import
root. Package collections must therefore place their importable packages directly
under a declared root. After verifying the complete runtime inventory, both worker
launchers pass the roots in sorted order to the fixed bootstrap, after the curated
worker bundle. The bootstrap inserts these paths directly into sys.path; it does
not use site.addsitedir or process .pth files for these added roots. The retained
launch command records the paths, and the typed worker evidence audit reconstructs
the same command. A resolved base interpreter consequently needs an admitted
package collection containing NumPy/PyArrow; package availability in an operator's
virtualenv alone does not establish worker runtime readiness.

Typed event metadata uses a data envelope with a separate, explicit datetime
timestamp field. Dora's incoming timestamp is restored as a built-in datetime
with its offset, microseconds and fold preserved; ordinary metadata remains
data. The codec refuses malformed or conflicting timestamp declarations and
never imports peer-selected types. The real Dora worker graph covers a stamped
observation followed by a turn, numeric/null payloads, and host acknowledgements.

Dynamic typed run configuration may select a provider instead of a prebuilt stage
list. Its exact fields are snapshot, snapshot_record, validation_output,
allocation_root, timeout_s, and max_calls. The run entry derives the worker runtime,
interpreter and adapter from the bound run, and retains provider evidence under
run-controller/typed-provider beside the private configuration. Every stage request
rechecks that configuration and the participant bytes, then verifies each generated
host's run bindings. A failed check makes that provider unusable for later launches.
The ordinary session controller constructs this declaration from the template
described below. This interface alone does not establish total descendant resource
enforcement.

For ordinary typed run requests, a controller-owned worker_preparations entry may
contain a provider template with allocation_root, timeout_s and max_calls. The
controller reserves the run, checks fresh allocation and disjoint
participant/controller/runtime state, performs normal current-source validation, and seals the
snapshot and validation locations into the private run configuration. The run child
then allocates stages on demand. Each reserved run needs its own fresh allocation
root. These values are controller inputs; the participant's run request remains
parameter-free. Failed validation consumes the reservation and retains diagnostics.

The monolithic run child also accepts a provider declaration with allocation_root,
timeout_s, max_primitive_calls, max_handles and module_sha256. It rechecks the sealed
module and run configuration around actual worker capability provisioning and
source preparation. A private binding receipt links the resulting worker-config
path/hash to the original run-config and module hashes. The journal auditor checks
that binding plus the generated worker's runtime, adapter, allocation and limits.
For ordinary monolithic run requests, the worker_preparations entry contains a
provider template with allocation_root, timeout_s, max_primitive_calls and
max_handles. After reserving the run, the controller checks the same allocation
and budget constraints as the typed arm and captures the current module without
executing it. It seals that module hash into the child declaration; capability
provisioning occurs inside the supervised run child. The journal checks the
captured module hash and the complete indexed inventory of both dynamic provider
and prepared-input directories. Added, missing or altered retained files are
refused. Raw capability retention does not itself prove every observation's
semantic validity or complete descendant resource enforcement.

The opt-in `tests/graph/test_matched_session_simulation.py` exercises ordinary
filesystem run requests through the session controller and dynamic worker
provisioning to real Dora/Genesis episodes for both arms. Select it explicitly
with `AISLE_MATCHED_SESSION_SIM=1` in a macOS simulation environment. It requires
the normal protected-baseline, distribution, open-idea, and budget gates; these
runs consume the ordinary run ledger and retain their outcomes. The frontend is
a deterministic Python fixture, not a coding agent. Actual capability probes
exercise the selected sandbox profiles, but the access log is deliberately
incomplete. The test requires successful run evidence and an excluded outer
session, with no eligibility for study estimates. It does not establish paired
expert parity, full tool coverage, or external confinement.

## Held-out evaluation of a finished session

```sh
uv run --extra sim --locked --no-sync python tools/matched_campaign.py heldout-eval \
  --request /absolute/controller/heldout-request.json
```

The request names `session_dir` (an intact completed attempt with
`matched-session.json`, `admission.json` and the retained `final/` snapshot),
`controller_root`, the registration's `manifest` (its `seed_commitment` is the
only commitment accepted), the private `seeds_source` and `salt_source`, the
frozen `rule` (`{"kind": "oracle_successes_at_least", "threshold": N}`, with
`1 <= N <= seeds`) and `timeout_s`. The evidence always lands at
`<session_dir>/heldout/heldout-evidence.json`; an existing `heldout/` refuses.
The controller refuses a session that is not an intact completed engineering
session, a drifted final snapshot, seeds that do not match the registered
commitment or overlap the session's development seeds, and an unresolved rule.
It rebuilds the submitted system from the controller's committed tree (git HEAD,
recorded as `controller_commit`) with the final deliverable overlaid, runs it
through the arm's own launcher as a separate process rooted at that view on the
private seeds with the realistic verifier driving the loop and the oracle as
the held-out scorer (BND-3), reads the oracle verdicts by goal id across every
trace directory of the run, derives STA-10 exposure counts from the run's
SPEC 470 ledger, and writes the evidence (per-seed verdicts by index, success
count, exposure, the rule and the decision). A launcher refusal or
infrastructure failure leaves `ok: false` so the session is excluded rather
than scored; a deliverable that fails to launch is a failed session. Seed
values are never written (CSE-9). This is the CSE-2 session outcome for the
ADR-66 pilot tier; it does not authorize scored collection.

The request names `session_dir` (a finished attempt with `matched-session.json`,
`admission.json` and the retained `final/` snapshot), `controller_root`, the
private `seeds_source` and `salt_source`, the registration's `seed_commitment`,
the frozen `rule` (`{"kind": "oracle_successes_at_least", "threshold": N}`) and
`timeout_s`. The controller refuses an unfinished session, a drifted final
snapshot, seeds that do not match the commitment or overlap the session's
development seeds, and an existing output. It rebuilds the submitted system as
the controller's tracked tree with the final deliverable overlaid, runs it
through the arm's own launcher on the private seeds with the realistic verifier
driving the loop and the oracle as the held-out scorer (BND-3), and writes
`heldout-evidence.json` (per-seed oracle verdicts by index, success count, the
rule and the decision). Seed values are never written (CSE-9). This is the CSE-2
session outcome for the ADR-66 pilot tier; it does not authorize scored
collection.

## Pilot assignments and records

`tools/pilot_assignments.py create --campaign <id> --protocol <SPEC 400
protocol> --seed <private 256-bit hex file> --ledger <ledger.json>` seals a
balanced plan whose arms and size come from the protocol (one typed and one
monolithic assignment per temporal block, TRT-8/CSE-7) and writes only its
public commitment; `reveal` appends exactly the next assignment after
re-verifying the history. `tools/pilot_records.py --protocol <SPEC 400 protocol>
--ledger <ledger.json> --sessions <root with one directory per session id>
--agent-system <id> --task <candidate> --output records.json` re-verifies every
ledger assignment against the sealed commitment and derives one
`aisle.stats.records.v1` row per assignment, started or not: outcomes come only
from the session's retained `heldout/heldout-evidence.json`, which must name the
session, arm, plan and final snapshot hashes; costs are the executor's observed
budgets (`null` when unobserved, never zero); validate-fix cycles are the
executed `check` tool attempts; exposure is the held-out run's SPEC 470 counts
(STA-10). The lifecycle follows the executor's own classification: an
`engineering_execution` session is a completed, included session, and any other
retained session is `infrastructure_excluded` with the executor's retained
reason (CSE-14). Named limitation: the executor retains no structured budget
stop (a session at its ceiling is an exclusion under MON-8/MON-12), so no row is
censored under CSE-13 until the runner retains one. A session directory outside
the ledger, a completed session without held-out evidence, or evidence that
belongs to another session refuses (STA-3/STA-11/CSE-8).

The `t1-l2-realistic` candidate uses the paired
[public observation graphs and pinned cache preparer](public-observations.md).
Bind the selected candidate table and worker environment before admission;
old receipts do not qualify a changed graph or model-cache context.
