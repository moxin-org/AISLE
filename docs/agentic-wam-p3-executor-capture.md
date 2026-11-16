# P3 executor capture repair

This repair supplies actual invocation evidence for P3 calibration. It does
not establish task eligibility, physical recovery, a completed pilot, or a
history-interface benefit. The measured pilot remains unrun.

## Runtime behavior

The IK executor now accepts indexed `grasp_pose_N` inputs. The existing T4
increment-two graph routes the return planner to `grasp_pose_2`; previously
that input was silently ignored. Indexed inputs preserve the executor's
one-plan-at-a-time rule and destination metadata. An arriving request cannot
preempt an executing plan.

The L1 pose node retains the accepted request's `goal_id`/`request_id` while
waiting for paired frames. Rejected requests do not relabel the pending
request. Reset clears the identity, and a later request without an ID stays
unknown. Image timestamps still identify the observation used for the pose.

Set `AISLE_EXECUTION_CAPTURE=1` on a rollout to request executor capture. The
instrumented graph binds the exact source hashes, input edges, node ID,
run/relaunch identity and output path for the built-in IK executor and
`skills/ik-transfer-v2/ik_transfer_v2.py`. Other controllers are not silently
labelled as these skills. Authored graphs cannot supply capture configuration.
Each relaunch gets separate files under its trace directory's `executions/`.

The executor records proposals, acceptance/rejection, the first actual
transport emission, each command, reset interruption/cancellation and return.
Read-ladder retries belong to the same requested invocation; each command
records its internal attempt and stage. A plan being constructed is not a
start. A failed raw send does not establish emission. Command capture occurs
after the turn wrapper supplies the authoritative output timestamp.

The Arrow schema is unchanged. The recorder's row-hashed identity sidecar
retains the invocation ID, source version, goal ID and original command
endpoint/sequence on emitted and guard-forwarded commands. The guard already
preserves extra metadata, so its policy and implementation are unchanged.

## Evidence limits

`read_capture` checks closure, byte integrity and executor-local lifecycle
order. Equal simulation timestamps retain their observed local `event_seq`;
no artificial nanoseconds are added. This does not establish a global order
between different nodes and does not change P2's synthetic replay format.

`verify_capture` additionally checks the retained instrumented-graph binding,
resolves every referenced request/observation/command against raw Arrow rows,
checks captured request goal IDs and emitted-command identities, and joins
guarded commands only through explicit ancestry metadata. Its caller must
supply the config and guard endpoints from the retained graph. Missing or
ambiguous source rows fail verification. Timing alone cannot prove forwarding.
These checks establish consistency of controller-owned evidence, not
cryptographic authenticity against someone who can rewrite all sources.

A transport send is not a guard decision. A guard-forwarded target is not
measured movement. An executor return is not a delivered or returned object.
`delivery_completion` remains `unknown`; only independent scorer outcomes
count as task success (TC-8). Open invocations and proposals remain explicit.
An incomplete capture must not be silently excluded from calibration results.

## Candidate scope and remaining work

The original P3 plan tests equal information with different evidence access;
it does not require L2. The first calibration candidate remains the existing
L1 T4 return-and-redelivery task. ADR-32 permits the scripted human's observed
post-delivery feedback. This is a narrower, declared L1 study, not a claim of
non-oracle perception or deployment competence. The earlier readiness draft's
proposed L2 restriction is not a prerequisite for that contrast.

The [policy evidence adapter](agentic-wam-p3-policy-evidence.md) now supplies
an explicit endpoint allowlist, common raw bundle and derived history query.
The eventual participant session adapter still must enforce equal source access. Scorer goals, oracle state and final scoring records must stay outside
them. The full goal-chain measurement bundle is privileged. Capture files
alone do not enforce access control. Development calibration must qualify the
current task before a pilot protocol can be frozen or paid sessions launched.
The provider/model and hard dollar ceiling are still unresolved.

Requirements: HAR-3/4/6, TC-2/6/7/8, BG-1, CON-5; ADR-30/32. No frozen graph,
scene, guard, verifier, reset, topic contract, or P2 format changes.

The shared rollout source and regenerated treatment/expert identities also
require pending CSE registration successors: v31 and pilot v17. They preserve
the prior protocols, seed commitments and pending gates, retain the historical
registrations unchanged, and add the optional capture helper's source hash.
They confer no collection authorization and contain no new measured results.
