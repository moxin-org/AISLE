# P3 public task transport

`aisle.harness.p3_dora_transport.DoraTaskTransport` connects the decoded
`TaskSession` interface to the trusted lockstep `Node`. It accepts only the
three high-level outputs `robot_msg`, `target_request` and `return_request`.
The host owns `turn_done`; direct actuator and reset outputs are refused.

The constructor requires an exact producer `turn_epoch` and a lockstep node
with those outputs plus `turn_done`, without wall-output grants. Arrow reset
messages must be `UInt32[1]=1`. Dialogue, executor completion and guard violation
messages use the fixed nodes' single-value Utf8 JSON representation. The
adapter rejects duplicate JSON fields, nonfinite constants, oversized values,
unknown inputs and invalid stamps. Only the required stamp and correlation
fields reach the task service; seed and other private metadata are discarded.
`TaskBoundary` performs the separate projection of public dialogue fields and
enforces confirmation, revision and motion-request authority.

The adapter reads each complete turn before returning its events, placing
reset before same-turn dialogue. Sends then occur while the trusted `Node`
still owns that open turn. Moving to the next turn emits the host's normal
acknowledgement. Public clock events occur at one-second simulation intervals
after reset. Wall ticks are ignored. Producer epoch changes and duplicate or
backward coordinator turns terminate the adapter. Only explicit closure of the
coordinator's input is ordinary transport exhaustion; STOP, ERROR and bare EOF
are failures.

`tests/accept/test_p3_dora_transport_acceptance.py` runs a three-turn exchange
over native Dora with synthetic reset, dialogue and executor completion. It
checks Arrow values, private-field projection, request correlation and host
watermarks under an external 30-second timeout. It starts no simulator or
provider. The result remains `physical_completion: unknown`.

This is a transport component, not a live P3 launcher. It does not verify
graph routes or capture actual guard traversal. The caller must supply an
outer process deadline because a trusted Node iterator may block. Source and
confinement preprovisioning, production graph admission, feedback/retry
accounting, private scoring and session resource checks remain required.
The [design decision](decisions/ADR-p3-dora-task-transport.md) records these
limits and the reset-order interpretation.
