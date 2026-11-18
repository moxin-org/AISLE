use anyhow::{Result, bail, ensure};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Action,
    Subgoal,
}

impl Mode {
    pub fn topic(self) -> &'static str {
        match self {
            Self::Action => "action_msg",
            Self::Subgoal => "subgoal_msg",
        }
    }
    pub fn instruction(self) -> &'static str {
        match self {
            Self::Action => {
                "Adapter contract: return one final fenced json Nano action. ALL fields required: left_arm and right_arm each {x,y,z,pitch,roll}, left_gripper and right_gripper each {angle}. xyz are increments in CENTIMETERS, pitch/roll in radians. The EUCLIDEAN translation norm sqrt(x*x+y*y+z*z) must be <=0.3 cm per arm (not per axis). At most one arm may move. Pitch/roll norm <=0.2 rad. Both gripper increments must be zero. Base/head hold. This mode permits non-contact probes only. If uncertain, return all zeros. Do not claim task success."
            }
            Self::Subgoal => {
                "Adapter contract: output exactly ONE subgoal JSON object, with no motor coordinates or task-success claim. Allowed shapes: {\"kind\":\"navigate\",\"target\":\"visible destination\"}; {\"kind\":\"pick\",\"object\":\"visible object\",\"arm\":\"left\" or \"right\"}; {\"kind\":\"place\",\"target\":\"destination\",\"arm\":\"left\" or \"right\"}; {\"kind\":\"observe\",\"target\":\"what needs observation\"}; {\"kind\":\"hold\",\"reason\":\"why\"}. Local Dora navigation/manipulation policies execute subgoals and report feedback. Select only the next step from current RGB, measured state and feedback. Use hold when uncertain. Only the independent verifier can establish task success."
            }
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    x: f64,
    y: f64,
    z: f64,
    pitch: f64,
    roll: f64,
}
impl Arm {
    fn moving(&self) -> bool {
        [self.x, self.y, self.z, self.pitch, self.roll]
            .iter()
            .any(|x| *x != 0.)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            [self.x, self.y, self.z, self.pitch, self.roll]
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite action"
        );
        ensure!(
            self.x.hypot(self.y).hypot(self.z) <= 0.3 + 1e-12,
            "translation norm exceeds 0.3 cm"
        );
        ensure!(
            self.pitch.hypot(self.roll) <= 0.2 + 1e-12,
            "rotation norm exceeds 0.2 rad"
        );
        Ok(())
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Gripper {
    angle: f64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Action {
    left_arm: Arm,
    right_arm: Arm,
    left_gripper: Gripper,
    right_gripper: Gripper,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Side {
    Left,
    Right,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Subgoal {
    Navigate { target: String },
    Pick { object: String, arm: Side },
    Place { target: String, arm: Side },
    Observe { target: String },
    Hold { reason: String },
}

pub fn validate_answer(answer: &str, mode: Mode) -> Result<Value> {
    ensure!(
        answer.len() <= 65536 && !answer.contains("<think>"),
        "invalid final answer"
    );
    let trimmed = answer.trim();
    let raw = if let Some((_, tail)) = trimmed.split_once("```json") {
        ensure!(trimmed.matches("```").count() == 2, "ambiguous JSON fences");
        tail.strip_suffix("```")
            .ok_or_else(|| anyhow::anyhow!("answer must end with JSON fence"))?
            .trim()
    } else if mode == Mode::Subgoal {
        trimmed
    } else {
        bail!("expected one final fenced json action")
    };
    match mode {
        Mode::Action => {
            let a: Action = serde_json::from_str(raw)?;
            a.left_arm.validate()?;
            a.right_arm.validate()?;
            ensure!(
                !(a.left_arm.moving() && a.right_arm.moving()),
                "only one arm may move"
            );
            ensure!(
                a.left_gripper.angle == 0. && a.right_gripper.angle == 0.,
                "grippers must hold in probe mode"
            );
            Ok(serde_json::to_value(a)?)
        }
        Mode::Subgoal => {
            let g: Subgoal = serde_json::from_str(raw)?;
            let text = match &g {
                Subgoal::Navigate { target }
                | Subgoal::Place { target, .. }
                | Subgoal::Observe { target } => target,
                Subgoal::Pick { object, .. } => object,
                Subgoal::Hold { reason } => reason,
            };
            ensure!(
                !text.trim().is_empty() && text.len() <= 256 && !text.chars().any(char::is_control),
                "invalid subgoal label"
            );
            Ok(serde_json::to_value(g)?)
        }
    }
}
