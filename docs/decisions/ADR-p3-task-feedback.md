# ADR-p3-task-feedback — Controller-owned P3 task feedback

Status: implementation decision; measured P3 admission remains pending.

TC-7 requires goal-correlated feedback, while TRT-5/10 keeps the independent
oracle and transport authority outside authored code. The private controller
therefore emits `episode_feedback` after each accepted public event and action,
using the accepted simulation clock and the current public dialogue goal.
The existing transport's simulation ticks provide the 1 Hz cadence. Phases
describe confirmation and outstanding requests only; executor return never
becomes a physical success label. A send failure closes the task service and
retains the failure. The worker still has only confirm/deliver/return authority.

For the recovery chain, the first return and first delivery under a new public
goal are distinct initial subtask requests. Each further request of either kind
increments a shared repeat counter, limited to eight; changing medicine cannot
reset it. This conservatively limits requested repeats without interpreting
private verdicts. It does not establish that a retry was conditioned on a
physical failure, prove a skill invocation, or compute HAR-3 pass@8. Those
claims require independent execution and scoring evidence. Dialogue corrections
have a separate TC-7 counter and never consume a repeat. New public goals reset
these per-goal counters; the session-wide motion allowance remains consumed.

Native Dora tests cover the feedback wire, host watermarks and actual confined
workers, including a stalled-host timeout. They launch no simulator or provider.
Production graph admission, guard capture, private chain shutdown and measured
session admission remain separate work.
