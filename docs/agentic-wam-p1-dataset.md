# P1 offline demonstration dataset

P1 repairs the exporter/loader contract identified by the
[P0 audit](agentic-wam-p0-audit.md). It prepares offline data for the
[adoption plan](agentic-wam-adoption-plan.md); it does not implement AW-0,
train a model, or certify skill execution. The implementation is Class A:
tools, tests and documentation, under CON-5/8/12, HAR-4/6 and TC-1/5.

## Export and inspect

Run against a completed, immutable recording. The output directory must be
new and outside the source run:

```bash
uv run --no-sync python tools/demo_export.py \
  --run runs/t1-genesis-l2-spot --out /tmp/aisle-p1-dataset
```

The command returns a single JSON result with coverage and exclusions. An
export can succeed with zero usable samples: successful export means that
the diagnostic bundle was built, not that training or benchmark admission
was approved. Missing required run metadata, ambiguous producers, invalid
timestamps/sequences, invalid Arrow schemas and incomplete Arrow streams
return an error and leave no published dataset.

The bundle contains `manifest.json`, `episodes.json`, the exact selected
source streams, the original run manifest and episode outcomes, captured
graphs when present, overhead video and raw overhead NPZ captures. Artifact
hashes cover every bundled file except the manifest itself. Exporter source
hashes record the transformation implementation. Hashes establish content
identity, not independent attestation or authenticity.

`demo_dataset.read_dataset(Path(...))` verifies the version, artifact hashes,
source row identities and values, causal bounds and split consistency. A
bundle can be exported again from its `source/` directory with the same
contract and endpoint selections; it needs no original run directory.
Assets with privileged depth/segmentation data remain in the source archive
for audit. The training adapter reads only the `rgb` array and the explicit
policy-visible instruction, state and forwarded target fields.

For multiple producers, provide `--endpoints endpoints.json`, containing
entries such as `{"joint_state": "bridge__joint_state.arrow"}`. The reset
service's `reset__reset_done.arrow` takes precedence over the bridge's copy;
other ambiguous topics require selection. Raw captures do not encode a
producer, so selecting one of multiple overhead camera producers still
cannot make their raw files eligible.

## Explicit action interpretation

No action contract is guessed from vector width or embodiment. Without
`--contract contract.json`, values remain available for analysis with
`unknown_action_contract` exclusions. A contract declares:

- `embodiment` and the training action projection;
- ordered joint `names`, `units` and `frame` for `state` and `forwarded`;
- the gripper channel's names, units and frame;
- `position_target` semantics for forwarded and gripper commands.

The optional contract is labeled **operator declared**, not recovered trace
metadata. It must agree with the source manifest's embodiment. The supported
SO-101 trainer contract is exposed as `demo_dataset.SO101_CONTRACT`: six
joint coordinates in official TC-5 order and radians, a separate normalized
closed-fraction gripper command, and five forwarded arm targets plus that
gripper command as the six training targets. This describes the simulator
data interpretation, not an automatically verified hardware calibration.
Use it only after checking the recording's source revision and graph.

To materialize that contract for a suitable recording:

```bash
uv run --no-sync python -c 'import json, sys; sys.path.insert(0, "tools"); from demo_dataset import SO101_CONTRACT; print(json.dumps(SO101_CONTRACT, indent=2))' > /tmp/aisle-so101-contract.json
uv run --no-sync python tools/demo_export.py \
  --run /path/to/completed-so101-run --out /tmp/aisle-so101-dataset \
  --contract /tmp/aisle-so101-contract.json
```

The SO-101 loader rejects Franka, unsupported format versions, missing or
different contracts, invalid dimensions and nonfinite values before loading
a model. It no longer truncates arbitrary vectors or substitutes zeros.
Normalization statistics are generated only from eligible successes in the
train split, with a standard-deviation floor of `1e-6`.

## Alignment and evidence boundaries

Each command retains its producer, topic, original file row, sequence,
simulation timestamp, environment and payload. Dropping one sample cannot
renumber another command. Records retain proposed joint/gripper targets,
forwarded targets and measured state separately. The proposed record is the
latest strictly earlier proposal; its relationship is explicitly **not a
verified command pair**. It cannot support a per-command clamp attribution.
The bundled streams retain all proposal rows, including equal-time rows.
Forwarded targets are never described as achieved physical states.

Episodes map contiguous per-environment episode ordinals to successful
reset-service boundaries. Shuffled, missing initial or ambiguous environment
identities are refused. The final episode without a closing reset remains
in the analysis bundle but is excluded. Commands outside all recorded
episode windows are counted as unassigned and retained in the source stream.
This is the existing reset-ordinal convention, not recovered request-ID
metadata; a historical run with unrecorded internal resets needs recapture.

All joins stay in one episode and environment, and use observations strictly
earlier than the command. Independent producers have no recorded total
ordering at equal timestamps. Commands at reset boundaries are excluded;
same-time instruction or gripper changes are explicitly ambiguous. No
future state or previous-episode frame is used as a fallback.

Instructions come from `target_request` evidence. Each revision carries the
request's source row and sequence; a mid-episode correction changes the
revision. A malformed request excludes subsequent samples until a new valid
request arrives. `episode_goal` is archived but never used as a policy-input
fallback: at T4 it may disclose the final corrected target before the human
says it. Absent policy-visible requests stay unknown, never “requested box.”

Skill name, version, invocation, parameters, branch, stage and execution
span are unknown. A graph reference or nearby proposal cannot recover those
labels. Recorded outcomes are retained verifier evidence, not a new physical
verification. Source manifests preserve engine and environment provenance;
export does not upgrade local/development recordings to trusted baselines.

## Pixels, completeness and splits

Raw captures follow the existing recorder convention
`traces/frames/overhead/<20-digit-sim-time-ns>.npz`. Exact timestamps identify
RGB pixels only for one camera producer and one environment, with unique
camera timestamps, valid uint8 RGB arrays and consistent resolution. Decode
failures and resolution changes are excluded. Sparse capture means a command
whose selected latest camera row has no raw asset is excluded; the exporter
does not quietly substitute an older captured image.

Video is decoded sequentially when OpenCV is installed. Declared, decoded
and camera-row counts and shape changes are reported. Without the optional
decoder, the manifest says `decoder_unavailable`. Matching counts leave the
mapping `unverified_ordinal`; video never provides training pixels in v1.
Neither closed Arrow streams nor decoded videos prove that the runtime
recorded every event. Missing final resets, invalid timestamps and truncated
files are treated separately from semantic capture identity.

Split assignment precedes sampling. The first eight hex digits of the
SHA-256 of `task_family`, modulo ten, select train (0–7), validation (8) or
test (9). An explicit episode family overrides the run family. Unknown
families share one conservative `unknown` group; they are not split by
episode number or inferred from tier. Thus a small or single-family dataset
may have empty evaluation splits. Do not manufacture a holdout from nearby
frames to fill them. Reusing this policy across runs keeps a declared family
together; meaningful family taxonomy remains the dataset curator's job.

All outcomes are exported. The trainer sees only
`success_and_eligible_and_train_split`; failures and excluded successes stay
available for reliability analysis. The existing trainer still repeats a
single target across its action chunk and fills two camera inputs with zeros.
P1 does not turn that into future-action supervision or synchronized
multi-camera training; the training record now states the repeated-target
semantics explicitly.

## Evidence and validation

The compact [P1 report](../analysis/aw0_p1/dataset_checks.json) separates a
retained stock-controller recording from synthetic fixtures. Full bundles
are generated locally rather than committed as duplicate video/Arrow data.
The synthetic fixture demonstrates usable training tuples, deterministic
export, re-export from bundled sources, original command identity, distinct
proposal/forwarded/state values, exact RGB identity and train-only statistics.
It is not a measured robot success or skill execution.

| Dataset | Episodes | Commands | Eligible samples | Success-only train view |
| --- | ---: | ---: | ---: | ---: |
| Retained stock-controller Franka recording | 12 | 15,864 | 0 | 0 |
| Synthetic SO-101 fixture | 2 | 4 | 2 | 1 |

The retained run has 10,205 decoded frames, matching both the container and
camera-row counts, but its mapping remains unverified. All 15,864 commands
lack a declared action contract; 15,855 select an unverified ordinal image
and nine have no same-episode frame. Exclusion counts overlap. There are
450 equal-time gripper ambiguities and one command in the final episode
without a closing reset. No source command is unassigned. The first
episode's recovered instruction names **amoxicillin**, not a generic target.
Two exports produced identical manifests and artifact hashes; reading the
bundle verified the referenced Arrow rows. The SO-101 loader refused it as
Franka data before model initialization.

Focused tests cover the P0 regressions, corrections and equal-time ambiguity,
episode/environment isolation, dimensions/order/units/frame incompatibility,
missing streams/reset, malformed and partial files, duplicate/nonmonotonic
timestamps, failed decode/resolution changes, legacy/video mapping rejection,
hash and row identity checks, all-outcome retention and grouped splits.

Validation completed on 2026-11-11:

- Manual diff review covered causal joins, source and pixel identity,
  instruction leakage, action interpretation, partial files and compatibility.
- Repository-wide `ruff format --check .` and `ruff check .`: passed.
- Focused regression suite: **41 passed**. Traceability check: passed with
  no uncovered requirement IDs or errors.
- `UV_PROJECT_ENVIRONMENT=.venv-unit uv run --locked pytest -m unit`:
  **4,619 passed, 419 deselected**, 1,632.01 seconds, exit 0.
- Real-data repeated export and synthetic re-export from bundled sources:
  identical manifests and artifact hashes; source-row, RGB-identity and
  train-only statistics checks passed. Exporter and retained input hashes
  match the committed evidence report.

The full gate used the documented lean unit environment outside the sandbox
because existing process-cleanup tests need `/bin/ps`. No test was disabled
or weakened. No simulation, training, paid campaign, GPU allocation or runtime
instrumentation was part of this work.

The next capture change remains separately reviewed: explicit
camera-row identity plus versioned skill invocation/stage events. P2 can
consume the offline format, but cannot claim skill-history validation from
these legacy unknown labels.
