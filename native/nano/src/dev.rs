//! Source-only development tools; no hardware or Dora command authority.
mod household_audit;
use aisle_nano::{kinematics::Robot, navigation::base_command, workflow};
use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{io::Write, path::PathBuf, time::Duration};

#[derive(Parser)]
#[command(
    about = "Nano JIT development: edit shared Rust logic without rebuilding; no robot output"
)]
struct Args {
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    AuditHousehold {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        label_replay: bool,
    },
    Household {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        simulator: PathBuf,
        #[arg(long)]
        helper: Option<PathBuf>,
        #[arg(long)]
        start_at: Option<String>,
    },
    Model {
        #[arg(long)]
        model: PathBuf,
    },
    CheckFlow {
        #[arg(long)]
        path: PathBuf,
    },
    /// Re-evaluate the real navigation controller while editing navigation.rs.
    WatchNavigation {
        #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
        target_x: f64,
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        target_y: f64,
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        x: f64,
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        y: f64,
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        yaw: f64,
        #[arg(long, default_value_t = 500)]
        interval_ms: u64,
        /// Zero watches until interrupted; otherwise emit exactly this many samples.
        #[arg(long, default_value_t = 0)]
        samples: u64,
    },
}

fn run() -> Result<Value> {
    let args = match Args::try_parse() {
        Ok(v) => v,
        Err(e)
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            return Ok(json!({"ok":true,"help":e.to_string()}));
        }
        Err(e) => anyhow::bail!("{e}"),
    };
    match args.command {
        Action::AuditHousehold {
            output,
            label_replay,
        } => {
            let result = household_audit::audit(&output)?;
            if label_replay {
                household_audit::label_replay(&output, &result)?;
            }
            Ok(result)
        }
        Action::Household {
            output,
            model,
            simulator,
            helper,
            start_at,
        } => {
            let launcher = PathBuf::from(
                std::env::var_os("AISLE_JIT_LAUNCHER")
                    .ok_or_else(|| anyhow::anyhow!("run through tools/aisle-jit"))?,
            );
            aisle_nano::household::prepare(
                &output,
                &model,
                &simulator,
                helper.as_deref(),
                &launcher,
                start_at.as_deref(),
            )
        }
        Action::Model { model } => {
            let robot = Robot::read(&model)?;
            let q = vec![0.; robot.names.len()];
            Ok(json!({"ok":true,"development":true,"joints":robot.names,
                "left_tool":robot.fk("left_tool_mount", &q)?.to_homogeneous().as_slice(),
                "right_tool":robot.fk("right_tool_mount", &q)?.to_homogeneous().as_slice()}))
        }
        Action::CheckFlow { path } => workflow::check(&path),
        Action::WatchNavigation {
            target_x,
            target_y,
            x,
            y,
            yaw,
            interval_ms,
            samples,
        } => {
            ensure!(
                (10..=60000).contains(&interval_ms),
                "interval must be 10..60000 ms"
            );
            let mut sequence = 0_u64;
            loop {
                let (forward, turn) = base_command([target_x, target_y], [x, y, yaw])?;
                println!(
                    "{}",
                    json!({"ok":true,"development":true,"robot_output":false,
                    "pid":std::process::id(),"sequence":sequence,"forward":forward,"turn":turn})
                );
                std::io::stdout().flush()?;
                sequence += 1;
                if samples != 0 && sequence >= samples {
                    break;
                }
                std::thread::sleep(Duration::from_millis(interval_ms));
            }
            Ok(json!({"ok":true,"development":true,"samples":sequence}))
        }
    }
}

fn main() {
    match run() {
        Ok(v) => {
            println!("{v}");
            if v["success"] == false {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("{}", json!({"ok":false,"error":e.to_string()}));
            std::process::exit(1);
        }
    }
}
