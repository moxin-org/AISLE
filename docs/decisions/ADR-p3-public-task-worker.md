# ADR-p3-public-task-worker — Public T4 requests across an isolated worker boundary

Status: implementation interpretation of TC-6/7/8, VAL-2/6 and the existing
P3 draft; no frozen graph, verifier, specification or measured protocol change.

The qualified development graph's state machine consumes oracle results to
close a goal. Its existing input stream cannot be exposed to an authored
worker, and merely deleting that input leaves subsequent recovery requests
ignored. A controller-owned boundary instead opens revisions from public
`human_msg` requests, preserves any outstanding motion across a new request,
and releases motion authority only on the matching fixed executor's goal and
request identities. Resets and fresh controller-supplied worker binding IDs
fence old completions. Public confirmation is required before motion, but
choosing the medicine and ordering return/delivery requests remains the
worker's responsibility. The boundary supplies neither the private final
target nor a physical success label. A completion says only that the executor
returned. This preserves the independent scorer and the legitimate public
recovery dialogue described by VER-3/ADR-32; it does not turn the L1 task into
a non-oracle competence claim.

The initial integration uses the existing verified macOS launcher with a
controller-supplied service wrapper. Both arms get the same public task calls,
raw evidence and fixed-library request authority; only the existing derived
history grant differs. Source capture, process reaping, private state,
confinement probes and transcript retention remain required. The executable
mock uses real isolated participant and generated-worker processes with a
synthetic transport. Its reports explicitly leave guard traversal unverified
and physical scoring null. Early worker exit, refusal and transport failure
remain retained facts; this unscored mock does not decide whether a future
measured session has an authored failure or an infrastructure exclusion.
That distinction needs the live adapter's independent evidence.

A reviewed live transport still must bind these calls to the fixed planners,
executor and budget guard, own an outer deadline for blocking host iteration,
bind a single candidate source across acceptance chains, and retain private
scoring and resource/provenance checks. No physical qualification or provider
invocation is authorized by this implementation. The completed qualification
batch stays closed, and the USD 200 ceiling and API deferral remain in force.
