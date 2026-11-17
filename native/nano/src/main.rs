use aisle_nano::{
    kinematics::Robot,
    runtime::{self, Config},
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(about = "AISLE experimental Nano: Rust perception, kinematics and Dora rollout")]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Node {
        role: String,
    },
    Model {
        #[arg(long)]
        model: PathBuf,
    },
    Rollout {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        simulator: PathBuf,
        #[arg(long)]
        dora: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long,default_value="pickup",value_parser=["inspect","navigate","pickup"])]
        task: String,
        #[arg(long,default_value="right",value_parser=["left","right"])]
        arm: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 700)]
        frames: u64,
        #[arg(long, default_value_t = 1.6, allow_hyphen_values = true)]
        goal_x: f64,
        #[arg(long,default_value_t=-1.2,allow_hyphen_values=true)]
        goal_y: f64,
        #[arg(long, default_value_t = 1800)]
        wall_seconds: u64,
        #[arg(long)]
        drop_depth_once: Option<u64>,
        #[arg(long, default_value_t = 15)]
        grace_seconds: u64,
    },
}
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}
fn hash(path: &std::path::Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
fn asset_hashes(model: &std::path::Path) -> Result<std::collections::BTreeMap<String, String>> {
    let robot = urdf_rs::read_file(model)?;
    let mut files = std::collections::BTreeMap::from([("model.urdf".to_owned(), hash(model)?)]);
    let root = model.parent().context("Model directory")?;
    for link in robot.links {
        for geometry in link
            .visual
            .iter()
            .map(|v| &v.geometry)
            .chain(link.collision.iter().map(|c| &c.geometry))
        {
            if let urdf_rs::Geometry::Mesh { filename, .. } = geometry {
                let name = filename
                    .strip_prefix("package://adora_nano/")
                    .unwrap_or(filename);
                if !files.contains_key(name) {
                    files.insert(name.to_owned(), hash(&root.join(name))?);
                }
            }
        }
    }
    Ok(files)
}
fn run() -> Result<serde_json::Value> {
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(e)
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            return Ok(json!({"ok":true,"help":e.to_string()}));
        }
        Err(e) => return Err(anyhow::anyhow!("{e}")),
    };
    match args.command {
        Action::Node { role } => {
            runtime::node(&role)?;
            Ok(json!({"ok":true,"node":role}))
        }
        Action::Model { model } => {
            let r = Robot::read(&model)?;
            let q = vec![0.; 17];
            let mut links = serde_json::Map::new();
            for link in [
                "left_tool_mount",
                "right_tool_mount",
                "left_wrist_housing",
                "right_wrist_housing",
                "head_tilt",
            ] {
                links.insert(
                    link.into(),
                    json!(r.fk(link, &q)?.to_homogeneous().as_slice()),
                );
            }
            Ok(
                json!({"ok":true,"joints":r.names,"zero_link_transforms_column_major":links,"urdf_sha256":hash(&model)?}),
            )
        }
        Action::Rollout {
            model,
            simulator,
            dora,
            output,
            task,
            arm,
            seed,
            frames,
            goal_x,
            goal_y,
            wall_seconds,
            drop_depth_once,
            grace_seconds,
        } => {
            ensure!(!output.exists(), "Use a new output directory");
            ensure!(
                frames > 0 && frames <= 10000 && wall_seconds > 0 && wall_seconds <= 7200,
                "Invalid rollout budget"
            );
            ensure!(grace_seconds <= 60, "Invalid shutdown grace");
            ensure!(goal_x.is_finite() && goal_y.is_finite(), "Invalid goal");
            let model = model.canonicalize()?;
            let simulator = simulator.canonicalize()?;
            let dora = dora.canonicalize()?;
            Robot::read(&model)?;
            let version = Command::new(&dora).arg("--version").output()?;
            ensure!(
                version.status.success()
                    && String::from_utf8_lossy(&version.stdout).trim() == "dora-cli 1.0.1",
                "Dora CLI 1.0.1 required"
            );
            fs::create_dir_all(&output)?;
            let output = output.canonicalize()?;
            let goal = [goal_x, goal_y];
            let station = if task == "navigate" {
                [1.6, -1.2]
            } else {
                goal
            };
            let objects = runtime::room(seed, station, &arm);
            let map = runtime::map(&objects);
            // The same route the policy drives from the scene origin.
            runtime::route(&map, &task, [0., 0.], goal)?;
            let config = Config {
                model: model.clone(),
                output: output.clone(),
                map,
                goal,
                drop: [
                    goal_x + 0.34,
                    goal_y + if arm == "left" { 0.23 } else { -0.23 },
                    0.20,
                ],
                task: task.clone(),
                arm: arm.clone(),
                seed,
                max_frames: frames,
            };
            runtime::write(&output.join("policy.json"), &config)?;
            runtime::write(&output.join("scene.json"), &objects)?;
            // Freeze the actual executables before hashing/spawning. A concurrent
            // rebuild must never silently change an in-flight experiment.
            fs::create_dir(output.join("bin"))?;
            let exe = output.join("bin/aisle-nano");
            fs::copy(std::env::current_exe()?.canonicalize()?, &exe)?;
            let frozen_simulator = output.join("bin/dorobot-nano-runtime");
            fs::copy(&simulator, &frozen_simulator)?;
            let simulator = frozen_simulator;
            let env = json!({"AISLE_NANO_CONFIG":output.join("policy.json"),"AISLE_NANO_OUTPUT":output,"AISLE_NANO_SCENE":output.join("scene.json"),"AISLE_NANO_MODEL":model,"AISLE_NANO_ARM":arm,"AISLE_NANO_DROP_DEPTH_ONCE":drop_depth_once.map_or_else(||"none".to_owned(),|n|n.to_string())});
            let mut graph = json!({"nodes":[
                {"id":"sim","path":simulator,"args":"aisle-node","env":env,"inputs":{"command":"guard/joint_cmd_safe","done":"policy/done","judge_ack":"judge/ack"},"outputs":["state","truth","finished","rgb_head","depth_head","pose_head","rgb_left_wrist","depth_left_wrist","pose_left_wrist","rgb_right_wrist","depth_right_wrist","pose_right_wrist"]},
                {"id":"policy","path":exe,"args":"node policy","env":env,"inputs":{"state":"sim/state","rgb":"sim/rgb_head","depth":"sim/depth_head","camera":"sim/pose_head","finished":"sim/finished"},"outputs":["joint_cmd","done"]},
                {"id":"guard","path":exe,"args":"node guard","env":env,"inputs":{"state":"sim/state","command":"policy/joint_cmd","finished":"sim/finished"},"outputs":["joint_cmd_safe"]},
                {"id":"judge","path":exe,"args":"node judge","env":env,"inputs":{"truth":"sim/truth","done":"policy/done","finished":"sim/finished"},"outputs":["ack"]}
            ]});
            // Explicit queues; one observation turn is in flight. No silent latest-wins mode.
            for n in graph["nodes"].as_array_mut().unwrap() {
                for (_, source) in n["inputs"].as_object_mut().unwrap() {
                    *source = json!({"source":source.clone(),"queue_size":8});
                }
            }
            let graph_path = output.join("dataflow.yaml");
            fs::write(&graph_path, serde_yaml::to_string(&graph)?)?;
            runtime::write(&output.join("asset-manifest.json"), &asset_hashes(&model)?)?;
            runtime::write(
                &output.join("provenance.json"),
                &json!({"profile":"adora_nano_experimental_v1","seed":seed,"platform":std::env::consts::OS,"arch":std::env::consts::ARCH,"aisle_head_at_last_build":env!("AISLE_SOURCE_REVISION"),"policy_sha256":hash(&exe)?,"simulator_sha256":hash(&simulator)?,"model_sha256":hash(&model)?,"dora_sha256":hash(&dora)?,"assets_sha256":hash(&output.join("asset-manifest.json"))?,"cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.lock"))),"scene_sha256":hash(&output.join("scene.json"))?,"graph_sha256":hash(&graph_path)?,"fault_injection":{"drop_head_depth_once":drop_depth_once}}),
            )?;
            let log = fs::File::create(output.join("dora.log"))?;
            // Dora runs in its own process group, so the terminal's Ctrl-C never
            // reaches it: the supervisor must stop that group itself.
            unsafe {
                libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t);
                libc::signal(libc::SIGTERM, interrupt as *const () as libc::sighandler_t);
            }
            let mut child = Command::new(dora)
                .arg("run")
                .arg(&graph_path)
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log))
                .process_group(0)
                .spawn()?;
            let start = Instant::now();
            let group = -(child.id() as i32);
            let mut finished = None;
            let mut interrupted = false;
            loop {
                if let Some(status) = child.try_wait()? {
                    finished = Some(status.success());
                    break;
                }
                interrupted = INTERRUPTED.load(Ordering::SeqCst);
                if interrupted || start.elapsed().as_secs() >= wall_seconds {
                    break;
                }
                thread::sleep(Duration::from_millis(200));
            }
            if finished != Some(true) {
                // Only the owned Dora process group. SIGINT lets native nodes finish;
                // whatever outlives the grace period is killed, even after Dora exits.
                let alive = || unsafe { libc::kill(group, 0) } == 0;
                unsafe {
                    libc::kill(group, libc::SIGINT);
                }
                let deadline = Instant::now() + Duration::from_secs(grace_seconds);
                while alive() && Instant::now() < deadline {
                    child.try_wait()?;
                    thread::sleep(Duration::from_millis(100));
                }
                if alive() {
                    unsafe {
                        libc::kill(group, libc::SIGKILL);
                    }
                }
                child.wait()?;
            }
            ensure!(!interrupted, "Rollout interrupted; processes stopped");
            let result = fs::read(output.join("result.json"))
                .context("No independent result; inspect dora.log")?;
            let mut result: serde_json::Value = serde_json::from_slice(&result)?;
            let capture = fs::read(output.join("capture-complete.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
            result["capture_complete"] = json!(
                capture
                    .as_ref()
                    .is_some_and(|v| v["ok"] == true && v["terminal_acknowledged"] == true)
            );
            result["dora_exited_successfully"] = json!(finished == Some(true));
            // A policy that stopped on its own error is not a measured physical failure.
            result["policy_fault"] = json!(
                fs::read(output.join("policy-status.json"))
                    .ok()
                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                    .and_then(|status| runtime::policy_fault(&status))
            );
            result["ok"] = json!(
                result["ok"] == true
                    && result["capture_complete"] == true
                    && result["policy_fault"].is_null()
                    && finished == Some(true)
            );
            result["success"] = json!(result["ok"] == true && result["success"] == true);
            let mut provenance: serde_json::Value =
                serde_json::from_slice(&fs::read(output.join("provenance.json"))?)?;
            let info: serde_json::Value =
                serde_json::from_slice(&fs::read(output.join("scene-info.json"))?)?;
            provenance["simulator_settings"] = info["simulation_settings"].clone();
            runtime::write(&output.join("provenance.json"), &provenance)?;
            if output.join("dataset/meta").is_dir() {
                runtime::write(&output.join("dataset/meta/aisle-result.json"), &result)?;
                fs::copy(
                    output.join("provenance.json"),
                    output.join("dataset/meta/aisle-provenance.json"),
                )?;
            }
            result["output"] = json!(output);
            result["wall_seconds"] = json!(start.elapsed().as_secs_f64());
            runtime::write(&output.join("rollout.json"), &result)?;
            Ok(result)
        }
    }
}
fn main() {
    match run() {
        Ok(v) => {
            println!("{v}");
            if v["ok"] != true {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("{}", json!({"ok":false,"error":format!("{e:#}")}));
            std::process::exit(1);
        }
    }
}
