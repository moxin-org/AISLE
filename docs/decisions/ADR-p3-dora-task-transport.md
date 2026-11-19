# ADR-p3-dora-task-transport — Public task RPC over lockstep Arrow transport

Status: engineering implementation of TC-2/6/7 and VAL-2/6; no protocol freeze,
frozen graph edit or physical collection authorization.

P3's controller-owned task boundary consumes decoded public events, while the
fixed library speaks Arrow through the lockstep `Node` wrapper. The trusted
adapter accepts only reset, dialogue, executor completion and guard violation
inputs, and only confirmation/delivery/return outputs. It binds one producer
epoch, validates correlation and wire payloads, strips private metadata, and
leaves output stamping and turn acknowledgements with the host. Because `Node`
yields a turn's inputs in lexical topic order, the adapter collects the complete
turn and delivers reset first: otherwise `human_msg` can precede `reset_done`
on a valid initial turn. Actions still execute within that open control turn.
Public one-second ticks derive from simulation time after reset; wall timer
events never grant action authority. Replayed turns, producer drift and malformed
traffic close the adapter. Explicit coordinator-input closure is transport
exhaustion, not evidence of physical success or an accepted session.

The native transport test uses synthetic task traffic with a hard external
process deadline. This does not admit a production graph or prove guard
traversal. Production integration still needs preprovisioned worker confinement,
an outer deadline covering blocking host callbacks, graph/source/capture checks,
feedback and retry accounting, private acceptance scoring and measured-session
admission. Provider calls remain deferred and completed physical batches closed.
