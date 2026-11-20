# P3 private Dora host

`aisle.harness.p3_dora_host` lets the existing isolated worker launcher retain
child-process ownership while Dora handles task transport in another process.
The controller wraps `TaskSession` around `TaskLink`; the trusted host wraps
`DoraTaskTransport` around its lockstep `Node`. Authored code continues to see
only the original framed task/evidence RPC.

The caller creates a fresh, short Unix socket path and supplies a private token,
session ID, candidate SHA-256 and a timeout of at most 300 seconds. All controller
socket operations share one absolute deadline. Partial, oversized, ambiguous,
wrong-session or malformed messages fail closed. On timeout, the existing worker
supervisor kills and reaps its owned child even if the host remains blocked in
`Node`. The caller also owns graph startup, timeout, cleanup and retained evidence.
Socket removal checks the original filesystem identity before unlinking it.

If macOS refuses a worker process-group signal, the supervisor retains an
explicit cleanup failure and does not claim success. It still tries to terminate
and reap the owned child directly, with a five-second cleanup bound. Unresolved
group ownership remains visible in the terminal record.

The trusted host entry point accepts `--config` and `--config-sha256`. Its exact
JSON configuration binds the socket, private token, session and candidate,
producer epoch, timeout, output receipt and fixed adapter source hashes. Config
and source checks precede Dora attachment and repeat at normal completion.
Outputs remain the fixed public task requests. The receipt reports zero measured
sessions and leaves guard traversal unverified.

The existing launcher's service factory is called after the actual OS audit.
A graph owner can start the trusted host there, accept its authenticated channel
and provide the task service before the confined worker is spawned. Private
configuration, sockets, graph files and receipts must remain outside every
worker-visible root; the native acceptance fixture tests direct socket denial.
Use a fresh unpredictable token for each real launch. Fixed tokens in tests
are synthetic fixtures, not production credentials.

`tests/accept/test_p3_dora_host_acceptance.py` exercises actual confined history
and trace workers through native Dora using synthetic reset, dialogue and
executor completion. A third case leaves the host waiting indefinitely for a
new turn; the controller times out and reaps the worker. The fixture separately
cleans the graph and checks for remaining processes scoped to its directory.
It does not start physics or call a provider. The [decision record](decisions/ADR-p3-private-dora-host.md)
explains why ownership stays in the controller.

This component does not admit a production graph or a measured P3 session.
Fixed-skill guard capture, private chain shutdown, feedback/retry accounting,
candidate identity across acceptance chains, protocol resources and independent
scoring remain to be integrated. Transport exhaustion alone is not a successful
physical outcome or an infrastructure-versus-agent classification.
