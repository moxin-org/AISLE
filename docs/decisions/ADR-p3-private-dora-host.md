# ADR-p3-private-dora-host — Keep worker ownership outside blocking Dora callbacks

Status: engineering interpretation of HAR-4 and TRT-5/6/7/9/10; no production
graph, measured-session admission, protocol freeze or collection authorization.

The existing P3 launcher reaps its confined child after RPC timeout, but a
trusted in-process `Node` callback can block that supervisor. A private Unix
socket separates the task controller from the trusted Dora host. The controller
keeps the child process and its pipes; every host exchange uses the same absolute
deadline. Thus a stalled Dora iterator becomes a failed controller call and the
existing launcher can reap its child. The host binds the session, candidate source
and a fresh controller-supplied token; authored code receives none of the socket,
token or host objects. Its verified confinement denies direct socket access.
Exact config and adapter-source hashes are checked before attaching Dora and
again at normal completion. Only the fixed public task transport is exposed.

The launcher's existing trusted service hook runs after confinement admission,
so it can start and own the graph after the slow OS audit, without a duplicate
preprovisioning mechanism. The caller must retain graph startup/termination
evidence and clean up that graph separately. The native acceptance fixture does
this for both arms and for an intentionally blocked host, checking the worker's
terminal receipt and absence of matching graph processes. Operational graph
cleanup is not a claim of exhaustive OS containment. Production graph routes,
guard capture, orderly private chain termination, feedback/retry accounting and
session scoring remain separate requirements. No simulator or provider runs in
these checks, and the completed qualification batch remains closed.

A regression check exposed macOS refusing a process-group signal after child
exit. The supervisor now retains that cleanup failure, attempts direct cleanup
of the owned child within five seconds, and refuses a successful terminal result
when group cleanup remains unresolved. A signal error must not erase the receipt.
