# ADR-73 — experimental PEAR-style inference runtime

Status: IMPLEMENTED for local bring-up; not registered or benchmark accepted.
Scope approved on 2026-11-16: runtime inference first, without RL training;
subsequently extended at the user's explicit request to the Nano's two arms.

## Context

PEAR uses repeated RGB/proprioception observations and model-produced pose
actions. AISLE already has an RGB policy slot, robot kinematics, a motion
budget guard, a turn barrier, and an independent verifier. We can reuse those
interfaces without replacing the research agent or executing generated code.

The specs do not define a PEAR provider or an unevaluated motion skill's
registration. CAP-6/CAP-7 require measured evidence for registration; a
stand-in model's transport test cannot supply that evidence. VAL-2 explicitly
permits non-attesting bring-up graphs to run directly outside the validator.

## Decision

1. Add an experimental, unregistered `pear-policy` node and a separate
   `python -m aisle.harness.pear prepare` entry point. Generate a new graph
   and turn plan from the existing SmolVLA topology. Do not change the
   registry, frozen graphs, scenes, reset service, verifier, or guard.
2. The provider receives same-stamp overhead/wrist RGB, measured arm/finger
   positions, FK tool pose, the instruction, bounded conversation history,
   and execution feedback. No oracle/object poses, segmentation, depth,
   scene configuration, or executable code enter the provider request.
   The verifier's episode result only stops the controller; its contents
   are not supplied to the model.
3. Parse one final fenced JSON action with centimeter translation, additive
   fixed-axis XYZ Euler increments in radians, and gripper-width delta in
   centimeters. Reject malformed, duplicate, nonfinite, oversized, ambiguous,
   or truncated answers. Use existing IK, joint/workspace limits, measured
   settling, and the existing independent budget guard for execution.
4. Keep robot-specific kinematics and action contracts behind the common
   controller lifecycle. Franka uses seven arm joints and linear fingers.
   Nano loads its actual 17-channel URDF, five-axis arm chains and angular
   grippers; its three cameras and independent guard have their own graph.
   Nano commands xyz plus pitch-joint sum and wrist roll, without an invented
   sixth independent orientation axis. Grippers use radians, not a fabricated
   linear width calibration. Both prompts are PEAR-style adapters, not a claim
   of compatibility with every published PEAR embodiment prompt.
5. Separate `lockstep-eval` (synchronous inference inside a real AISLE turn)
   from `async-live` (one background request; reject stale or displaced
   observations). Both generated graphs retain the turn barrier. Only the
   former waits for inference before closing the turn. Keep guard wall-time
   budgets unchanged. A larger barrier liveness watchdog is not an extension
   of the robot's motion budget.
6. Record model identity/revision, condition, input PNGs/messages, request
   hash, final answer, usage, and acceptance/rejection. Mark all records
   `attested: false`. Reset/new goals invalidate pending responses. Task
   success remains the independent verifier's decision.

## Consequences and validation

This implements an inference boundary, not PEAR weights, RL, an MLX vision
backend, navigation, or a measured grasping policy. Nano now has a measured
Nexus arm-loop check with a stand-in model, not an evaluated PEAR grasping policy.
The server must support image Chat Completions with final content separate
from reasoning. A base Qwen text-only benchmark does not establish that.

Tests cite CON-1/2/4/5/7/8, TC-1/2/3/6/7/8/9, BG-1/2/3, and VAL-5/6.
Unit tests cover the parser, units, observation boundary, controller
lifecycle, measured IK, motion bounds, and generated topology. Live HTTP
and Dora tests use explicitly labeled stand-ins with the real IK, guard,
and turn barrier. They establish transport/control integration, not model
quality or physical contact. Promotion requires real checkpoint evaluation
and normal registry/validator acceptance; no synthetic evalcard is created.

Nano's fixture constrains the root and three wheel joints; all original arm
geometry and effort limits are retained. This is explicit in its run manifest
and derived URDF: continuous wheel/floor contacts under a fixed root were
unstable on the current Nexus solver. The adapter uses FK from measured joints
and a bounded feedback servo, including small motor load error; it does not
inject commanded joint positions as measured state. Its run ends at a time
limit without a task-success verdict. Tool-mount origins and nominal camera
optics are not claims of calibrated contact frames or hardware intrinsics.

## Codex runtime provider (2026-11-16)

At the user's request to connect Codex and close the loop, add an explicit
`--provider codex` option alongside the existing Chat Completions provider.
The Codex app-server receives the same RGB/proprioception/action-history
envelope. Its final answer passes through the unchanged embodiment parser,
IK, measured-state controller, and independent command guard.

Use a fresh ephemeral thread for each bounded-history inference. Disable
execution environments (`environments: []`), dynamic tools, configured MCP
servers, apps/plugins, shell/web, hooks, memories and project instructions.
Check the reported model identity and disabled MCP inventory before starting
the turn. Reject unexpected tool activity, approvals, incomplete turns, or
cross-thread/turn events. A temporary cwd outside the repositories avoids
project discovery. Per-process overrides preserve the user's global settings
and supported CLI authentication. No credential files are read by AISLE.

The actual local check used CLI 0.169.0, `gpt-6.1-terra`, medium reasoning,
three Nexus-rendered Nano RGB cameras, and measured URDF joint state. Codex
identified the cube colors, chose the right arm, requested a 3 mm probe, then
received fresh RGB and measured motion before deciding the next action.
This supersedes the earlier *stand-in-only* runtime status for Codex, not for
the PEAR checkpoint. The Nano remains a fixed-base arm fixture with no grasp
or navigation success claim. Codex does not expose a seed through this API;
traces mark that limitation and remain non-attesting under CON-5 / VAL-2.

See [the runtime guide](../pear-runtime.md) for setup and limitations.

## Local OminiX-MLX checkpoint provider (2026-11-16)

The user requested local inference through OminiX-MLX with actual testing.
The companion OminiX Qwen3.5/3.8 crate now has a Rust vision tower, explicit
multimodal rotary positions, image-embedding insertion and a persistent
generation worker. A loopback Python HTTP adapter uses the upstream processor
and tokenizer; both neural towers execute in Rust/MLX. No remote inference
fallback is used.

Pin `mit-han-lab/Qwen3.8-27B-PEAR` to
`65176e435e5fa80a059ebfafaf8215e6ad00a83e`. Verify both source shards against
upstream SHA-256, convert the language weights to affine INT4/group 64, and keep
the vision weights unquantized BF16. Preserve the original processor, tokenizer
and chat template after conversion: the converter's custom processor serializer
is not interchangeable with Transformers' loader. Record source/derived hashes,
processor asset hashes and the actual Rust worker binary hash.

Use float32 computation for both towers, after merging text/image embeddings
in the checkpoint embedding dtype. This avoids accidental precision changes
between the first decoder block and later blocks and makes comparison with the
independent MLX reference meaningful. Vision uses learned-position interpolation,
spatial RoPE, independent attention per image, and the checkpoint's patch merger.
The language decoder uses compressed interleaved T/H/W positions for images and
continues from their position maximum during generation.

The real PEAR checkpoint passed red/blue RGB counterfactuals and reference
comparison, including a six-image Nano observation/history request. Float32
vision-feature RMS errors were below 0.000006 and decoder-logit RMS errors below
0.00024; top tokens matched in all three checks. These are numerical/runtime
checks, not a reproduction of published PEAR task-success results.

Add explicit HTTP `--no-thinking` / `--max-tokens` options to the generated
graph and manifest. Defaults remain unchanged. The local smoke condition uses
thinking off and 512 tokens; the server preserves truncation status and separates
reasoning from final content. AISLE still rejects incomplete, malformed, stale,
unreachable or out-of-bounds actions. The Nano fixture and independent guard
remain in place; frozen graphs, registry, benchmark declarations and evalcards
are not changed by this extension (CON-5/7, TC-9, BG-1/2, VAL-2/5/6).

`tests/accept/test_pear_ominix_runtime.py` requires the verified local checkpoint
and checks RGB-driven small probes, fresh feedback, measured motion, held joints
and joint limits. See the runtime guide for the command and scope. Navigation,
contact/grasp/placement and hardware validation remain separate work.

The opt-in local Nano test passed with the Studio URDF hash
`a03319b78dcf338cbd64bed7bdc2062066af6f8a189cdb2cb67f9ba1e9b6b627`:
four accepted <=3 mm actions, three completed probes reflected in subsequent
RGB/proprioception, 9.28 mm measured displacement, and 598 bounded guarded
commands. Calls took 45–120 seconds as history grew. This establishes the
local model/control boundary in paused simulation, not real-time capability
or task success.

## Single-A6000 endpoint preparation (2026-11-16)

At the user's request, prepare the same pinned checkpoint in an isolated uv
environment on their CUDA host, with vLLM 0.34.0 and bitsandbytes 0.50.2.
Keep CUDA dependencies outside AISLE's default environment (CON-1/2). Use
one explicit GPU UUID, parallel sizes of 1, a loopback endpoint and an SSH
tunnel. Verify source hashes and record runtime/GPU identity. The launcher
refuses insufficient free GPU memory and does not stop unrelated workloads.

Record HTTP client wall time separately from server-reported usage, using
an injected monotonic clock (CON-5). Add an opt-in real-checkpoint CUDA/Nano
acceptance check for the same RGB boundary, feedback and joint guards
(TC-9, BG-1/2). A single-GPU CUDA/BF16 arithmetic check passed; model inference
and the Nano acceptance run remain pending because all eight A6000 cards
became occupied during installation. No CUDA model-performance claim is made.

## Upstream basis

Inspected PEAR revision `d604856a1b16fdccd47e22a27d4dd935e3a62014`:

- [Deployment instructions](https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/examples/models/qwen3_8_27b_pear/deploy/README.md)
- [Action parser](https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/deploy/station/agent_loop/action_parser.py)
- [YAM pose prompt](https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/deploy/station/control/robots/yam/prompts/bimanual_pose.md)
- [Inference client](https://github.com/mit-han-lab/pear/blob/d604856a1b16fdccd47e22a27d4dd935e3a62014/pear/deploy/station/agent_loop/clients/vllm.py)
