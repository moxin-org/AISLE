# Nano sorts laundry and toys

Drive a simulated Adora Nano through a furnished 12 × 10 m apartment, watch its three cameras in DoRobot Studio, and check where each object ended up. The clothes are folded rigid bodies. The basket rims, shelves and low tables participate in the physics simulation.

| Workstation | Object | Destination |
|---|---|---|
| Hall bench | Folded red shirt | Hall laundry basket |
| Play corner | Blue building brick | Playroom coffee table |
| Reading nook | Green stacking block | Lounge coffee table |
| Bedroom bench | Folded yellow shorts | Bedroom laundry basket |

## Install once on Apple Silicon

Start from the AISLE repository root. Reuse an installed matching Relax runner, Dora CLI and simulator where available. Otherwise get the pinned sources below. The development tools require Git, Rust/Cargo, Xcode command-line tools, ripgrep, patch and unzip. The Studio runtime also requires its pinned shader toolchain and FFmpeg.

```sh
deps="$HOME/.local/share/aisle-household"
mkdir -p "$deps"
git clone https://github.com/octos-org/octos.git "$deps/octos"
git -C "$deps/octos" checkout 5fb009a7d6ba1bff8264236428fee48968505f0e
"$deps/octos/scripts/jit/setup-relax.sh"

git clone https://github.com/dorobot-org/dorobot-studio.git "$deps/studio"
git -C "$deps/studio" checkout de6ba8d4807b021292d305e30f6a5d989fd130cc
uv tool install dora-rs-cli==1.0.1
```

Use the Studio runtime from that revision with the `aisle-node` subcommand. If you need to build the simulator dependency, follow the [FFmpeg setup](https://github.com/dorobot-org/dorobot-studio/blob/de6ba8d4807b021292d305e30f6a5d989fd130cc/tools/release/README.md#build) followed by the [runtime build](https://github.com/dorobot-org/dorobot-studio/blob/de6ba8d4807b021292d305e30f6a5d989fd130cc/runtime/README.md#build). The checkout includes the URDF and its meshes.

```sh
export AISLE_NANO_MODEL="$deps/studio/data/adora_nano/adora_nano.urdf"
export AISLE_NANO_SIMULATOR="$deps/studio/runtime/target/release/dorobot-nano-runtime"
export AISLE_DORA="$(command -v dora)"
"$AISLE_DORA" --version
```

## Run and watch

To check the assets and plan routes between all four stations offline, generate an episode in a fresh directory under an existing parent. Keep its path short so the simulator can bind its Unix socket.

```sh
examples/nano-household/run.sh --prepare /tmp/nano-plan
```

For a live run, supply your DeepSeek API credential in `DEEPSEEK_API_KEY` through your shell's secret manager, or point `AISLE_OCTOS_PROFILE` to a private Octos profile. The planner sends the three camera images and task state to DeepSeek. Each run makes metered model calls. The example starts the Rust planner, policy, guard, control and receipt nodes through JIT. First startup can take several minutes.

```sh
# Optional: open Studio in an isolated JIT session. Omit for a headless run.
export AISLE_STUDIO_JIT="$deps/studio/tools/run_jit.sh"
# Optional when the Studio FFmpeg libraries are installed.
export AISLE_STUDIO_VIDEO=1
examples/nano-household/run.sh /tmp/nano-live
```

Studio's Live page reads `/tmp/nano-live/preview.sock` during the run. To connect later, launch Studio with `--live-socket /tmp/nano-live/preview.sock --screen live`. After capture, open `/tmp/nano-live/dataset` in Replay. Nexus records RGB, depth, segmentation, camera poses, base and joint state, and contacts every 200 ms of simulation time. Wall-clock playback and preview rates depend on the machine.

## Understand and reproduce a result

The [Dora graph](dataflow.yaml) connects two control loops. Octos sends the head and both wrist images, state, available skills and feedback to `deepseek-flash`. The reply selects a bounded subgoal. Local Rust code validates the reply and implements navigation, picking and placing using a known furniture map, simulated base odometry, color and depth perception, URDF inverse kinematics and gripper trajectories. The guard is the only node that can send commands to Nexus. The model runs on subgoal requests. It does not generate wheel or joint targets.

The launcher writes the resolved graph, station coordinates, furniture and policy settings into each output directory. Set `AISLE_HOUSEHOLD_START_AT='bedroom bench'` to change the initial station. An existing compatible `aisle-nano` binary can optionally serve the guard, control and receipt roles via `AISLE_NANO_HELPER`. Use the default JIT nodes for a source checkout.

```sh
tools/aisle-jit nano run audit-household --output /tmp/nano-live
```

Read `household-result.json` for the per-object verdict and frame count. Each delivery requires initial shelf contact, a lift of at least 4 cm, contact with two robot links while lifted, release, destination support, and ten settled samples inside the destination. The overall result also requires a terminal simulator acknowledgement and zero robot–furniture contact samples. The command exits nonzero on a failed mission. DeepSeek choices and physics outcomes can vary between runs. A completed capture can be replayed even when a delivery fails.
