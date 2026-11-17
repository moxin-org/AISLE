# P3 launch preflight and mock evidence sessions

The measured P3 pilot remains incomplete. The owner approved a USD 200 hard
ceiling but deferred API invocation; this preparation makes no model calls.
P4 remains on hold. The closed qualification batches remain closed.

## Check admission before freezing

The qualification at `77966866fe9c` stopped before launching simulation because
its branch had no OPEN idea record (HAR-8). Environment attestation and graph
validation had passed, but those checks did not cover every rollout gate.

`tools/p3_preflight.py` now checks the proposal's exact source revision, branch,
graph and source hashes, requires the branch's OPEN idea, checks the paired
Dora CLI/Python API, and calls the actual rollout `run_gates` with one captured
graph, one requested episode, the protected baseline and no idea bypass.
Fingerprint drift and downstream refusals remain failures. A consumed or
closed proposal is refused. It creates no batch, launches no simulation,
and always reports `collection_authorized: false`.

Run it from the pinned integration tree with its attested simulation environment:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/aisle/.venv \
  uv run --no-sync python /path/to/tools/p3_preflight.py --proposal proposal.json
```

The proposal must name `instrument_worktree`, `instrument_commit`,
`instrument_branch`, `graph`, `graph_sha256`, `protected_environment_baseline`,
`engine`, `env_hash`, `env_fingerprint` and the `library_hashes`, `skill_sources`
and `assessment_sources` maps. Its status is `prepared_not_authorized` or
`approved_for_preflight`. This status permits a check, not collection.
An OPEN idea must be recorded through `harness report log` in the actual
instrument worktree. A preflight pass does not override campaign scope,
stop rules, outcome qualification, or the need for fresh launch-time gates.

## Controller-owned evidence interface

`tools/p3_mock_session.py` adds a controller-owned dispatcher over the existing
P3 evidence exporter. The controller binds a session id, arm, evidence manifest
hash and call budget. Requests contain only `operation` and `arguments`.
Requests cannot override the arm, session or evidence bundle.

Both arms can query original journal records, Arrow rows, the artifact inventory
and bounded file chunks. Only `history` can request the derived invocation
view. Every admitted call retains its request and response, including refusals.
Refusals consume the call budget. Exhaustion retains a stop event and closes
the session. A closed output directory cannot be reused. Source hashes are
checked for each response; a changed manifest or forbidden scorer path is
refused. Backend exceptions do not expose controller filesystem paths.

The mock runner exercises up to six pairs of fresh interface sessions over one
explicitly supplied evidence bundle. It checks common raw response parity and
the history grant, keeps failed and unstarted slots, and stops on the first
failed check. This is a fixed synthetic exercise, not randomization or pilot
assignment. Reports always retain zero measured sessions, zero provider calls,
zero spend, `p3_complete: false`, and `p4_on_hold: true`.

```sh
UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked python tools/p3_mock_session.py \
  --bundle /path/to/exported-policy-evidence --out /path/to/new-mock-output
```

The displayed USD 200 ceiling records the current owner decision. It is not a
live billing meter: this tool has no live mode or provider dispatch path.

## Boundaries still required for the measured pilot

Do not give the dispatcher object, tools directory or controller filesystem to
participant code. This in-process interface is not an OS sandbox. The isolated
participant and generated-worker boundary, qualified task/action surface,
session-level acceptance feasibility, final provider/model and cost reservation,
frozen paired protocol, actual pilot and analysis remain outstanding. Existing
typed/monolithic launchers cannot be relabeled as history/trace arms.

Mock checks do not establish physical completion, policy competence, a history
benefit or isolation against generated code. TC-8's oracle-only scoring rule
continues to govern any future measured outcome.

Tests cite the existing HAR-2/4/6/8, CON-5/8 and TC-8 constraints. This adds no
spec requirement and changes no frozen source, runtime graph or registration.

## Isolated local mock clients

`tools/p3_isolated_mock.py` runs the evidence interface through fresh Python
processes under the existing macOS confinement adapter. Each mock participant
produces a small worker script; the controller captures its bounded source
bytes after the participant exits and launches it in a separate fresh workspace.
Both arms use identical scripts and raw operations. The controller alone binds
the arm, session, immutable evidence manifest and call budget.

The real OS capability audit must pass before each launch. Each process gets
only its script, the recorded standard-library runtime and its own private
HOME. Network access and reads of the evidence bundle, controller repository,
scorer and peer files are denied; evidence is available only through the bounded
RPC interface. The child inherits no operator credentials or extra descriptors.
Hard CPU, individual-file-size, core-dump and no-fork limits are installed before
authored code starts. These limits do not constitute a general disk quota or a
resource-isolation claim for a future arbitrary coding-agent session.

The controller retains OS audit results, runtime and source identities, exact
RPC frames including incomplete messages, stderr, refusals, process termination
and postflight verification. A failed check stops the mock suite and preserves
all planned slots. Raw journal records, Arrow rows, artifact inventory and a
bounded artifact chunk must match across each pair. An attempted arm override
is refused in both arms; only the bound history arm receives the derived view.

Use an output path outside the controller repository and input evidence tree:

```sh
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=/path/to/aisle/.venv-unit \
  uv run --no-sync python tools/p3_isolated_mock.py \
  --bundle /path/to/exported-policy-evidence \
  --out /private/tmp/aisle-p3-isolation-check --blocks 1
```

This command exercises four unscored processes per block: participant and
generated worker in each arm. It makes no provider calls and cannot launch a
robot policy. Passing these mock checks does not select the final task/action
surface, prove session-level acceptance feasibility, or admit the measured
pilot. The live task-specific worker integration, provider/model decision and
frozen measured protocol remain separate work. P4 stays on hold.
