# P3 policy evidence and history interface

This adapter prepares the same evidence for both P3 arms. It is not a campaign
launcher, a task-eligibility result, or a measured pilot. It currently supports
the first declared candidate: T4 return and redelivery with L1 perception.

The controller exports one completed launch at a time:

```sh
uv run --no-sync python tools/p3_evidence.py export \
  --run runs/<run-id> --out runs/<public-bundle> --launch 0
uv run --no-sync python tools/p3_evidence.py raw \
  --bundle runs/<public-bundle> --endpoint ik-trajectory/joint_cmd \
  --offset 0 --limit 100
uv run --no-sync python tools/p3_evidence.py history \
  --bundle runs/<public-bundle>
```

The exporter verifies the retained execution graph against the rollout
manifest and checks the executor journal against the exact trace rows and
command ancestry. It copies only the declared policy endpoints, executor
journal, allowed identity metadata, overhead video and retained camera arrays.
The bundle omits oracle state, scorer goals, scorer verdicts, rollout outcome
records, the private run manifest and the executable graph. L1 segmentation
is explicitly allowed sensor information for this candidate; this is not an
L2 or non-oracle-perception study. Scripted human feedback is permitted under
ADR-32.

Both arms receive the identical raw bundle and its artifact hashes. The raw
query returns original Arrow values, row identities and captured metadata.
The history query groups the same executor events by invocation and reports
state, parameters, stages, emitted commands and explicitly linked guarded
commands. It neither invents observations nor counts executor return as
physical completion. All delivery completion fields remain unknown. The
independent scorer and goal-chain measurement bundle remain controller-only.

The reader checks the artifact inventory, hashes, journal field allowlist and
camera array types. Unknown annotations and extra files are refused. Archived
relaunches resolve relative to the archive, not the original host's absolute
capture paths. These checks rely on a trusted controller and pinned producer
code; consistent hashes alone do not authenticate fabricated source data.

The participant boundary still has to be enforced by the eventual session
adapter. Giving agents unrestricted host access plus a filtered directory
would not isolate hidden scoring data. The controller must expose the raw
query to both arms and the history query only as the treatment, retain every
request and response, and keep private storage outside participant authority.
The trace arm may organize its allowed evidence itself. The contrast is the
provided interface, not a claim that the control is unable to derive history.

## Development calibration

Qualification uses public development seeds, a fixed graph/library and one
engine. The proposed T4 set is 200, 204, …, 244, all from the corrected-seed
stratum. Run at most twelve chains with two scorer goals each, retain every
attempt and stop on an instrument failure. No tuning or replacement occurs
inside the batch. Admission requires at least two complete successes and two
complete failures, complete invocation provenance, and a reconstructable
initial delivery/return/redelivery sequence. A valid but ineligible T4 batch
permits considering the proposed T2 candidate; it does not qualify T2 by
analogy or permit silently extending the T4 sample.

`tools/p3_calibration_evidence.py` checks one recorded chain against the
instrument/graph pin and trusted L1 Genesis attestation, then creates separate
private measurement and public policy bundles. A runtime failure, wall clamp,
missing source or incomplete verdict remains an instrument failure, never a
zero-valued task result. Its screen keeps all planned slots, refuses duplicate
or unplanned seeds, and treats missing slots as incomplete. An ordered motion
sequence counts as a recovery attempt only when the initial delivery is
independently scored successful; it does not override a failed final goal.

The calibration protocol must pin the tested executor commit, authored graph,
source/library hashes, seed set, environment fingerprint, wall limits and
selection rule before launch. A two-hour development-calibration ceiling does
not authorize paid model sessions. The provider/model, dollar cap, participant
adapter, final matched protocol and actual pilot remain outstanding.

Requirements: CON-5/8, HAR-4/6, TC-8/9. No existing P1/P2 formats, measured
outcomes, frozen runtime paths or study-arm names are changed.
