# Native Nano validation — 2026-11-17

This is an experimental Nano profile, separate from AISLE's frozen pharmacy
benchmark and registered skills. All robot execution uses native Rust processes
and Dora 1.0.1 on Apple Silicon Metal. Codex CLI executed and independently
audited navigation and pickup episodes. No real robot was driven.

The room is 12 × 10 m with a sofa, coffee table, chairs, kitchen island,
bookshelf, divider, pickup surface and colored cubes. Geometry is rigid and
simplified. Navigation uses a static furniture map and simulator localization.
RGB identifies the red target; synchronized depth and calibrated camera poses
provide geometry. The policy has no object-truth topic, but this is not an
adversarial filesystem sandbox. Self-collision is disabled in the inherited
Nano model. It is not hardware calibration, SLAM or open-vocabulary perception.

## Retained experiments

Every attempt has a separate run directory. Failures were not replaced by
successful controller status. The original experiments and the subsequent
control/contact/fixture changes are not a single fixed-treatment comparison.

| Run family | Outcome and reason |
|---|---|
| Pickup probes 01, 07 | IK placement/approach outside solver tolerance; failed before manipulation |
| Pickup probes 02, 03 | Unbounded imported gripper motor and target accumulation ejected the cube; 03 repeated the same binary |
| Pickup probe 04 | Reduced gripper torque; cube slipped |
| Pickup probe 05 | 6 cm cube experiment did not fit the grasp; missing depth delivery also left an incomplete evaluation |
| Pickup probes 06, 08 | Returned to 4 cm cube; bounded 0.30 Nm grip still slipped; 06 recovered an injected depth omission |
| Pickup probe 09 | Local right pickup/place passed, 9.47 cm maximum lift, zero furniture contacts; the early provenance incorrectly said 0.30 Nm, while the hashed binary used the URDF 1 Nm limit |
| Codex navigation v1 | Incomplete: missing depth delivery stalled the observation turn |
| Codex navigation v2 | Arrived, but failed for furniture-contact samples; old GPU contact audit could read stale allocated slots |
| Navigation probe 03 | Arrived, but collided with the pickup station placed beside the navigation goal |
| Codex v3 navigation | Passed: 4.995 m sampled route, 18.95 mm endpoint error, zero furniture contacts, injected depth delivery recovered |
| Codex v3 local left pickup | Passed: 9.30 cm lift, 2.6 s supported above 7 cm, final finger-free stable placement 4.4 s |
| Integrated right pickup v3 | Failed: 8.24 cm peak lift, but cube slipped before the hold requirement |
| Integrated right pickup v4 | Passed: 2.155 m sampled route, 8.60 cm lift, zero furniture contacts; final placement and normal Dora exit verified |
| Codex navigation v4 | Passed: 4.987 m route, 33.66 mm final error, stationary arrival, zero furniture contacts, two delivery retries |
| Codex integrated left pickup v4 | Passed: 2.164 m route, 9.75 cm lift, 5.4 s supported lift and final 7 s placement with no robot contact |
| Complex right pickup v4 | Passed: 5.933 m route around furniture to (2.3, 3.9), 9.37 cm lift, stable placement, zero furniture contacts, 756 frames per camera |
| Final local-origin right pickup | Failed: 3.30 cm peak lift, then slipped; 167 frames, zero furniture contacts, capture and normal process completion passed |
| Complex right repeat | Passed again: 5.933 m route, 9.37 cm lift, supported zone entry and terminal released placement; zero furniture contacts, 756 frames per camera, normal Dora exit |
| Final entry-transition judge run | Passed with the final judge active: same 5.933 m route and 9.37 cm lift, zero furniture contacts, complete capture and normal Dora exit |

The v4 controller uses forward/turn motion for the solid-wheel model, resets
wheel targets at every observation, aligns the jaws to the cube, rejects IK
branch jumps above 1.5 rad, and limits loaded arm motion to 0.20 rad/s. Grippers
use the actual 1 Nm URDF effort bound with 0.10 rad position-target lead.
Navigation-only runs keep the pickup station at a fixed room location; pickup
runs approach it along its docking axis.

## Evidence and criteria

Physics runs at 100 Hz; control and three-camera capture are 5 Hz. Head and both
wrist cameras retain RGB, depth, segmentation, extrinsics and timestamps in raw
NPZ plus three replay MP4s. Segmentation and cube truth are recording/judge data,
not policy inputs. The independent Codex v3 audit decoded all six videos, checked
all 383 raw-frame hashes and all 137 referenced mesh hashes, and reconstructed
100 Hz same-step two-finger contacts. See [its report](codex-v3.md).
The [v4 Codex report](codex-v4.md) checks final stopped arrival, all-robot contact
release, wheel-target reset, 903 raw frames and six fully decoded videos.
The [complex-route audit](codex-complex.md) independently verifies the 5.933 m
integrated right-arm episode: 5.2 s supported lift, supported placement-zone
entry, 4.2 s terminal robot-free placement, all 756 raw hashes and all three videos.
Both subsequent repeats pass the Rust evidence acceptance. The last repeat runs
the final entry-transition judge during execution. Its judge trace, all 756 raw
NPZ frames and all three videos are byte-identical to the independently audited
complex-route recording; copied verdict metrics match exactly. These repeats
do not establish universal GPU determinism or a measured success distribution.
The 151.2 s recording took 287.72 and 244.42 wall seconds respectively to produce,
so this configuration does not demonstrate real-time capture.

Final acceptance requires a stopped arrival window, at least 7 cm lift supported
by both fingers for 0.4 s, supported entry into the placement zone, then no robot
link contacting the cube and 1 s stable placement. Arrival and placement must
still hold at termination. Furniture contact fails the episode. The final
Rust evidence test recomputes these criteria from retained samples; it also
checks guard coverage, clocks, graph isolation, executable/scene hashes and
three-camera dataset metadata. A normal Dora exit and complete capture are
required in authoritative `rollout.json`. Capture completion is never task success.

The simulation does not weld or attach the cube, teleport the root during a
rollout, or use target poses as achieved-motion evidence. A single missing depth
delivery is retried without advancing physics twice. The numerical seed currently
changes target x by only ±6 mm; these tests are not a statistical robustness claim.
The final local-origin failure also shows that a successful room episode does
not establish grasp robustness at every base/arm configuration. These failures
remain part of the acceptance evidence rather than being relabeled as success.

## Review

A separate Claude CLI reviewer inspected the Codex-authored implementation under
CON-16. High findings about one-finger release and latched historical success
were fixed with regression tests. Follow-up review confirmed those fixes plus
wheel windup, IK continuity, capture/result separation, normal process shutdown,
Rust CI and simulator-reported actuator settings. Its additional accidental-drop
case now has a rejecting regression test. Final review found that a later
two-finger touch could retroactively qualify an unsupported drop. The regression
first reproduced this bug; the judge now requires support on the actual outside-to-
inside zone transition. Both arms' retained integrated traces pass under this
final rule; a fresh complex-room run also passes with it active. The original
on-run judges were less strict. Remaining limits are the coarse arm workspace
bounds, no self-collision or adversarial oracle sandbox, and incomplete pure
state-machine unit coverage beyond the actual dropped-message experiment.

The original crate location inside the Python package caused test fixtures to
copy and fingerprint mutable Cargo outputs. The crate now lives at `native/nano`,
outside the Python source tree. The old broad-gate failures are retained, and
the full unit gate completed after this correction (results below). A reproduced one-ULP
JSON metadata copy discrepancy is fixed with `serde_json/float_roundtrip`; old
run artifacts were not rewritten. The Rust re-evaluation tolerates at most
1e-12 m of JSON float noise, with exact Boolean and count comparisons.

## Gate results

- Native Nano: 15 Rust tests passed; formatting and clippy with warnings denied
  passed. The normally ignored physical-evidence test was explicitly run against
  both integrated arm episodes, the audited complex episode and its two repeats;
  all five pass the final judge.
- Studio adapter: 17 native runtime tests passed.
- Python formatting/lint, generated documentation inventory and requirement
  trace check passed (372 requirements in the legacy Python scanner). The claim
  inventory is current; its existing CLM-12 release blocker remains.
- The full legacy unit run completed: **4,917 passed, 4 failed, 425 deselected**
  in 38 minutes 55 seconds. Two failures were stale documentation inventory;
  regeneration fixes them. Two were app-server response-deadline timeouts.
  All four exact failures subsequently passed with `pytest -m unit --lf`
  (4 passed in 1.95 s). The full inventory and app-server module rechecks also
  passed (21 and 7 tests respectively). A clean aggregate pass is not inferred
  from these rechecks.
- The full Genesis sim/graph gate did not pass: the reset-anchored bridge test
  timed out after 420 seconds without its capture-completion sentinel; the run
  was interrupted after that failure. A targeted guard/recorder/model-cache
  recheck passed all 10 selected tests. Native Nexus physical acceptance above
  does not replace this legacy Genesis gate or authorize a release.

Full transcripts, raw datasets and unsanitized logs remain local. This document
contains no personal filesystem paths or access credentials.
