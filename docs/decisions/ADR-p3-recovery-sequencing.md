# ADR-p3-recovery-sequencing — Wait for correlated executor completion

Status: proposed implementation interpretation of TC-2/7, HAR-3 and ADR-32;
no spec, topic payload, frozen graph, scorer or guard change.

The fixed development calibration at `3738fe0` retained twelve complete
T4/L1 chains. Every return proposal was rejected as `controller_busy`:
the state machine emitted return and delivery requests together, while the
executor rejects overlapping plans. Three chains passed the independent
physical scorer, but none demonstrated the required execution sequence.
That batch remains ineligible and is not replaced by this repair. The
[calibration report](../../analysis/aw0_p3/development_calibration_3738fe0.json)
contains all twelve outcomes and content-addressed references to the retained
raw runs; those files are under the main checkout's ignored `runs/` directory.

The state machine now gives each T4 motion request a deterministic request ID.
The executor echoes the accepted request's goal and request IDs on `plan_done`,
using the current completion transport stamp. Rejected proposals and sensor
metadata cannot change that identity. A confirmed recovery waits for any
remaining initial motion, then requests the return, then requests redelivery
only after the matching return completion. Duplicate, unrelated and obsolete
completions cannot release a newer action. Completion still means executor
return, never physical task success. The independent scorer remains unchanged.

ADR-32's initial human request opens a new physical episode; its recovery
request carries `return_med` and continues the same physical episode. Only
the latter preserves a pending prior motion across a goal result. A new
initial request clears that pending identity; a closed goal clears queued
recovery intent. This is scoped to the existing scripted T4 protocol, not a
general reset or arbitrary-goal scheduling API. Non-T4 retry emissions retain
their existing behavior. New physical qualification requires a separate
source pin and declared calibration scope; these tests do not qualify a task
or complete the measured P3 pilot.
