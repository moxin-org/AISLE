//! External skill boundaries around the native RGB/IK policy. No judge truth input.
use crate::{
    kinematics::Robot,
    navigation::distance,
    runtime::{Config, Policy},
};
use anyhow::{Result, ensure};
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Skill {
    Navigate { target: String },
    Pick { object: String, arm: String },
    Place { target: String, arm: String },
    Observe { target: String },
    Hold { reason: String },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub skill_id: u64,
    pub skill: Skill,
    pub status: String,
    pub reason: String,
}
struct Active {
    id: u64,
    skill: Skill,
    settling: Option<u64>,
}
pub struct Controller {
    policy: Policy,
    station: [f64; 2],
    active: Option<Active>,
    interrupted: Option<Skill>,
    hold_q: Option<Vec<f64>>,
    picked: bool,
    placed: bool,
    last_id: Option<u64>,
    household: Vec<crate::household::Station>,
    docked: Option<usize>,
    selected: Option<usize>,
    delivered: Vec<usize>,
}
fn valid(state: &[f64]) -> Result<()> {
    ensure!(
        state.len() == 24 && state.iter().all(|x| x.is_finite()),
        "invalid state"
    );
    ensure!(
        (state[3..7].iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 0.01,
        "invalid quaternion"
    );
    Ok(())
}
impl Controller {
    pub fn new(config: Config, robot: Robot) -> Result<Self> {
        let station = config.goal;
        let household = if config.task == "household" {
            serde_json::from_slice(&std::fs::read(config.output.join("stations.json"))?)?
        } else {
            Vec::new()
        };
        let mut policy = Policy::new(config, robot)?;
        policy.external_skill = true;
        Ok(Self {
            policy,
            station,
            active: None,
            interrupted: None,
            hold_q: None,
            picked: false,
            placed: false,
            last_id: None,
            household,
            docked: None,
            selected: None,
            delivered: Vec::new(),
        })
    }
    pub fn active(&self) -> bool {
        self.active.is_some()
    }
    pub fn pose_index(&self) -> usize {
        self.policy.pose_index
    }
    pub fn placed(&self) -> bool {
        if self.household.is_empty() {
            self.placed
        } else {
            self.delivered.len() == self.household.len()
        }
    }
    pub fn phase(&self) -> &str {
        if self.active.is_some() {
            &self.policy.stage
        } else if self.placed {
            "placed"
        } else if self.picked {
            "pick_trajectory_complete"
        } else {
            "idle"
        }
    }
    pub fn at_station(&self, base: [f64; 2]) -> bool {
        distance(base, self.station) < 0.08
    }
    pub fn capabilities(&self) -> serde_json::Value {
        if !self.household.is_empty() {
            let mut allowed = Vec::new();
            if self.picked {
                if let Some(i) = self.docked {
                    allowed.push(Skill::Place {
                        target: self.household[i].destination.clone(),
                        arm: self.policy.config.arm.clone(),
                    });
                }
            } else if let Some(i) = self.docked.filter(|i| !self.delivered.contains(i)) {
                allowed.push(Skill::Pick {
                    object: self.household[i].object.clone(),
                    arm: self.policy.config.arm.clone(),
                });
            } else {
                for (_, s) in self
                    .household
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !self.delivered.contains(i))
                {
                    allowed.push(Skill::Navigate {
                        target: s.station.clone(),
                    });
                }
            }
            return serde_json::json!({"available_subgoals":allowed,"stations":self.household.iter().map(|s|serde_json::json!({"station":s.station,"object":s.object,"destination":s.destination,"goal_xy":s.goal})).collect::<Vec<_>>(),"completed_trajectories":self.delivered.iter().map(|&i| &self.household[i].object).collect::<Vec<_>>(),"docked_at":self.docked.map(|i|&self.household[i].station),"phase":self.phase()});
        }
        serde_json::json!({"navigation_targets":["station","start"],"pick_object":"red cube","place_target":"placement_zone","arm":self.policy.config.arm,"phase":self.phase(),"pick_trajectory_complete":self.picked,"placed":self.placed,"holding_is_not_grasp_verification":true,"interrupted_skill":self.interrupted})
    }
    pub fn start(&mut self, id: u64, skill: Skill, seq: u64, state: &[f64]) -> Result<()> {
        valid(state)?;
        if let Some(a) = &self.active {
            ensure!(
                a.id == id && a.skill == skill,
                "another skill active or reused ID changed"
            );
            return Ok(());
        }
        ensure!(self.last_id.is_none_or(|old| id > old), "retired skill ID");
        match &skill {
            Skill::Navigate { target } => {
                ensure!(
                    target == "station"
                        || target == "start"
                        || self.household.iter().any(|s| &s.station == target),
                    "unknown navigation target"
                );
                ensure!(
                    !self.picked
                        && !matches!(
                            self.interrupted,
                            Some(Skill::Pick { .. } | Skill::Place { .. })
                        ),
                    "cannot navigate with unresolved manipulation"
                );
                let is_pick_station = if self.household.is_empty() {
                    target == "station"
                } else {
                    let i = self
                        .household
                        .iter()
                        .position(|s| &s.station == target)
                        .ok_or_else(|| anyhow::anyhow!("unknown household station"))?;
                    ensure!(!self.delivered.contains(&i), "station already serviced");
                    ensure!(
                        self.docked
                            .is_none_or(|old| self.delivered.contains(&old) || old == i),
                        "finish current station first"
                    );
                    if self.selected != Some(i) {
                        self.policy.depart_from = self.docked.map(|old| self.household[old].goal);
                    }
                    self.selected = Some(i);
                    let s = &self.household[i];
                    self.station = s.goal;
                    self.policy.config.drop = s.drop;
                    self.policy.item_color = s.color;
                    self.policy.item_height = s.size[2];
                    self.placed = false;
                    true
                };
                self.policy.config.goal = if is_pick_station {
                    self.station
                } else {
                    [0., 0.]
                };
                self.policy.config.task = if is_pick_station {
                    "pickup"
                } else {
                    "navigate"
                }
                .into();
                self.policy.route.clear();
                self.policy.waypoint = 0;
                self.policy.stage = "navigate".into();
            }
            Skill::Pick { object, arm } => {
                ensure!(
                    arm == &self.policy.config.arm
                        && if self.household.is_empty() {
                            object == "red cube"
                        } else {
                            self.docked.is_some_and(|i| {
                                !self.delivered.contains(&i) && &self.household[i].object == object
                            })
                        },
                    "unsupported object/arm"
                );
                ensure!(!self.picked && !self.placed, "pickup already completed");
                ensure!(
                    self.at_station([state[0], state[1]]),
                    "navigate to station before picking"
                );
                if self.interrupted.as_ref() != Some(&skill) {
                    self.policy.poses.clear();
                    self.policy.pose_index = 0;
                    self.policy.pose_offset = 0;
                    self.policy.stage = "observe".into();
                }
            }
            Skill::Place { target, arm } => {
                ensure!(
                    arm == &self.policy.config.arm
                        && if self.household.is_empty() {
                            target == "placement_zone"
                        } else {
                            self.docked
                                .is_some_and(|i| &self.household[i].destination == target)
                        },
                    "unsupported placement/arm"
                );
                ensure!(
                    self.picked && !self.placed,
                    "pick trajectory must complete before place"
                );
                if !self.household.is_empty() && self.empty_closed_gripper(state)? {
                    self.picked = false;
                    self.interrupted = None;
                    anyhow::bail!(
                        "gripper closed past the object width; observe and pick the item again"
                    );
                }
                if self.interrupted.as_ref() != Some(&skill) {
                    self.policy.prepare_place(state, seq)?;
                }
            }
            Skill::Observe { target } => ensure!(
                ["scene", "station", "red cube", "placement_zone"].contains(&target.as_str()),
                "unknown observation target"
            ),
            Skill::Hold { reason } => ensure!(
                !reason.is_empty() && reason.len() <= 256,
                "invalid hold reason"
            ),
        }
        self.policy.stage_frame = seq;
        if !matches!(skill, Skill::Observe { .. } | Skill::Hold { .. }) {
            self.interrupted = None;
        }
        self.last_id = Some(id);
        if self.hold_q.is_none() {
            self.hold_q = Some(state[7..].to_vec());
        }
        self.active = Some(Active {
            id,
            skill,
            settling: None,
        });
        Ok(())
    }
    fn empty_closed_gripper(&self, state: &[f64]) -> Result<bool> {
        let grip = self.policy.robot.arm(&self.policy.config.arm)?[5];
        // Household props have a 40 mm jaw span. The loaded simulated motor stops near 0.35 rad.
        Ok(state[7 + grip] > 0.62)
    }
    pub fn hold(&self, state: &[f64]) -> Result<Vec<f64>> {
        valid(state)?;
        let mut v = vec![0.; 17];
        if let Some(q) = &self.hold_q {
            for i in 3..17 {
                v[i] = ((q[i] - state[i + 7]) * 1.0).clamp(-0.20, 0.20);
            }
        }
        let carrying = self.picked
            || self.active.as_ref().is_some_and(|a| {
                matches!(a.skill, Skill::Pick { .. }) && self.policy.stage == "settle"
            });
        if !self.household.is_empty()
            && carrying
            && let Some(q) = &self.hold_q
        {
            let grip = self.policy.robot.arm(&self.policy.config.arm)?[5];
            v[grip] = ((q[grip] - state[grip + 7]) * 2.).clamp(-0.45, 0.45);
        }
        Ok(v)
    }
    pub fn cancel(&mut self, state: &[f64], reason: &str) -> Result<Option<Outcome>> {
        valid(state)?;
        let Some(a) = self.active.take() else {
            return Ok(None);
        };
        let mut q = if matches!(a.skill, Skill::Observe { .. } | Skill::Hold { .. }) {
            self.hold_q.clone().unwrap_or_else(|| state[7..].to_vec())
        } else {
            state[7..].to_vec()
        };
        // Preserve closure when cancelling a loaded/closing manipulation.
        if matches!(a.skill, Skill::Pick { .. } | Skill::Place { .. })
            && matches!(self.policy.stage.as_str(), "manipulate" | "settle")
        {
            let grip = self.policy.robot.arm(&self.policy.config.arm)?[5];
            if let Some(goal) = self.policy.poses.get(
                self.policy
                    .pose_index
                    .min(self.policy.poses.len().saturating_sub(1)),
            ) {
                q[grip] = goal[grip];
            }
        }
        self.hold_q = Some(q);
        if !matches!(a.skill, Skill::Observe { .. } | Skill::Hold { .. }) {
            self.interrupted = Some(a.skill.clone());
        }
        Ok(Some(Outcome {
            skill_id: a.id,
            skill: a.skill,
            status: "cancelled".into(),
            reason: reason.into(),
        }))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        seq: u64,
        t: u64,
        state: &[f64],
        rgb: &[u8],
        depth: &[f32],
        camera: &Isometry3<f64>,
        w: usize,
        h: usize,
    ) -> Result<(Vec<f64>, Option<Outcome>)> {
        valid(state)?;
        if self.active.is_none() {
            return Ok((self.hold(state)?, None));
        }
        // A loose item can still land on its intended support. Re-observe it before
        // commanding another pickup. The independent recorder remains the physical judge.
        if self.policy.stage == "observe"
            && self
                .active
                .as_ref()
                .is_some_and(|a| matches!(a.skill, Skill::Pick { .. }))
            && let Some(i) = self.docked
            && !self.household.is_empty()
        {
            let station = &self.household[i];
            let [x, y, rx, ry] = station.region;
            let hits = crate::perception::detect(
                rgb,
                depth,
                w,
                h,
                h as f64 / (2. * 35_f64.to_radians().tan()),
                station.color,
                t,
                t,
                camera,
            )?;
            let ready = hits.iter().find(|hit| {
                hit.pixels >= 30
                    && hit.extent.max() < 0.14
                    && (hit.position.x - x).abs() < rx - 0.018
                    && (hit.position.y - y).abs() < ry - 0.018
                    && (station.surface_z - 0.003
                        ..station.surface_z
                            + station.size.iter().copied().fold(0., f64::max)
                            + 0.008)
                        .contains(&hit.position.z)
            });
            if let Some(hit) = ready {
                let center = [hit.position.x, hit.position.y, hit.position.z];
                let same = self.policy.target.is_some_and(|last| {
                    last.into_iter()
                        .zip(center)
                        .all(|(a, b)| (a - b).abs() < 0.005)
                });
                let a = self.active.as_mut().unwrap();
                if !same || a.settling.is_none() {
                    self.policy.target = Some(center);
                    a.settling = Some(seq);
                }
                if seq - a.settling.unwrap() >= 8 {
                    let a = self.active.take().unwrap();
                    self.delivered.push(i);
                    self.picked = false;
                    self.placed = true;
                    self.interrupted = None;
                    return Ok((
                        self.hold(state)?,
                        Some(Outcome {
                            skill_id: a.id,
                            skill: a.skill,
                            status: "already_placed".into(),
                            reason: format!(
                                "RGB-D observed {} stationary at {} after release",
                                self.household[i].object, self.household[i].destination
                            ),
                        }),
                    ));
                }
                return Ok((self.hold(state)?, None));
            }
            self.active.as_mut().unwrap().settling = None;
        }
        let a = self.active.as_ref().unwrap();
        if !self.household.is_empty()
            && self.empty_closed_gripper(state)?
            && (matches!(a.skill, Skill::Pick { .. }) && self.policy.stage == "settle"
                || matches!(a.skill, Skill::Place { .. })
                    && self.policy.stage == "manipulate"
                    && self.policy.pose_index < 2)
        {
            let mut outcome = self
                .cancel(
                    state,
                    "gripper closed past loaded aperture; re-observe and retry pickup",
                )?
                .unwrap();
            outcome.status = "failed".into();
            self.picked = false;
            self.interrupted = None;
            self.policy.stage = "observe".into();
            return Ok((self.hold(state)?, Some(outcome)));
        }
        let passive = matches!(a.skill, Skill::Observe { .. } | Skill::Hold { .. });
        let reached = match a.skill {
            Skill::Navigate { .. } => self.policy.stage != "navigate",
            Skill::Pick { .. } | Skill::Place { .. } => self.policy.stage == "settle",
            _ => true,
        };
        if reached {
            if self.active.as_ref().unwrap().settling.is_none() {
                let q = if !passive
                    && !self.policy.poses.is_empty()
                    && !matches!(self.active.as_ref().unwrap().skill, Skill::Navigate { .. })
                {
                    self.policy.poses.last().unwrap().clone()
                } else {
                    state[7..].to_vec()
                };
                if !passive || self.hold_q.is_none() {
                    self.hold_q = Some(q);
                }
                self.active.as_mut().unwrap().settling = Some(seq);
            }
            let v = self.hold(state)?;
            let settle_frames = if !self.household.is_empty()
                && matches!(self.active.as_ref().unwrap().skill, Skill::Pick { .. })
            {
                3
            } else {
                8
            };
            if seq - self.active.as_ref().unwrap().settling.unwrap() >= settle_frames {
                let a = self.active.take().unwrap();
                match a.skill {
                    Skill::Pick { .. } => self.picked = true,
                    Skill::Navigate { .. } if self.selected.is_some() => {
                        self.docked = self.selected
                    }
                    Skill::Place { .. } => {
                        self.placed = true;
                        if let Some(i) = self.docked {
                            self.delivered.push(i);
                            self.picked = false;
                        }
                    }
                    _ => {}
                }
                return Ok((
                    v,
                    Some(Outcome {
                        skill_id: a.id,
                        skill: a.skill,
                        status: "completed".into(),
                        reason: "post-trajectory hold interval completed; physical success is judge-only"
                            .into(),
                    }),
                ));
            }
            return Ok((v, None));
        }
        match self.policy.step(seq, t, state, rgb, depth, camera, w, h) {
            Ok((v, _, _)) => Ok((v, None)),
            Err(e) => {
                let mut result = self.cancel(state, &e.to_string())?.unwrap();
                result.status = "failed".into();
                // A perception/IK error can be retried with a fresh observation and plan.
                if self.policy.stage == "observe" {
                    self.interrupted = None;
                }
                Ok((self.hold(state)?, Some(result)))
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn loaded(stage: &str) -> (Controller, Vec<f64>, usize) {
        let model = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/robots/adora_nano/model.urdf");
        let robot = Robot::read(&model).unwrap();
        let grip = robot.arm("right").unwrap()[5];
        let config = Config {
            model,
            output: std::env::temp_dir(),
            map: crate::navigation::Map {
                bounds: [-6., 6., -5., 5.],
                clearance: 0.55,
                obstacles: vec![],
            },
            goal: [0., 0.],
            drop: [0.34, -0.23, 0.2],
            task: "pickup".into(),
            arm: "right".into(),
            seed: 0,
            max_frames: 1000,
        };
        let mut c = Controller::new(config, robot).unwrap();
        let mut s = vec![0.; 24];
        s[6] = 1.;
        c.start(
            1,
            Skill::Pick {
                object: "red cube".into(),
                arm: "right".into(),
            },
            0,
            &s,
        )
        .unwrap();
        let mut pose = vec![0.; 17];
        pose[grip] = 0.75;
        c.policy.poses = vec![pose.clone(); 4];
        c.policy.pose_index = if stage == "settle" { 4 } else { 2 };
        c.policy.stage = stage.into();
        c.hold_q = Some(pose);
        (c, s, grip)
    }
    /// TC-7/8: preserve closure across the pick barrier and report a fully closed empty grasp.
    #[test]
    fn household_grip_at_handoff() {
        let (mut c, mut state, grip) = loaded("settle");
        c.household = crate::household::stations().unwrap();
        state[7 + grip] = 0.347;
        let (command, outcome) = c
            .tick(1, 0, &state, &[], &[], &Isometry3::identity(), 1, 1)
            .unwrap();
        assert!(outcome.is_none());
        assert!((command[grip] - 0.45).abs() < 1e-10);
        state[7 + grip] = 0.70;
        let (_, outcome) = c
            .tick(2, 1, &state, &[], &[], &Isometry3::identity(), 1, 1)
            .unwrap();
        assert_eq!(outcome.unwrap().status, "failed");
        assert!(!c.picked && !c.placed());
    }
    /// TC-7/BG-2: interruption survives passive skills; a loaded gripper never opens.
    #[test]
    fn cancel_observe_cancel_retains_closure_and_resume_phase() {
        for stage in ["manipulate", "settle"] {
            let (mut c, s, grip) = loaded(stage);
            c.cancel(&s, "pause").unwrap();
            assert!(c.hold(&s).unwrap()[grip] > 0.);
            c.start(
                2,
                Skill::Observe {
                    target: "scene".into(),
                },
                1,
                &s,
            )
            .unwrap();
            c.tick(2, 0, &s, &[], &[], &Isometry3::identity(), 1, 1)
                .unwrap();
            c.cancel(&s, "pause observation").unwrap();
            assert!(c.hold(&s).unwrap()[grip] > 0.);
            assert!(
                c.start(
                    3,
                    Skill::Navigate {
                        target: "start".into()
                    },
                    3,
                    &s
                )
                .is_err()
            );
            c.start(
                4,
                Skill::Pick {
                    object: "red cube".into(),
                    arm: "right".into(),
                },
                3,
                &s,
            )
            .unwrap();
            assert_eq!(c.policy.stage, stage);
            assert_eq!(c.policy.poses.len(), 4);
        }
    }
    /// TC-7/8: pick completion is a barrier, not automatic place or episode success.
    #[test]
    fn pick_completes_without_starting_place() {
        let (mut c, s, grip) = loaded("settle");
        assert!(
            c.tick(1, 0, &s, &[], &[], &Isometry3::identity(), 1, 1)
                .unwrap()
                .1
                .is_none()
        );
        let (_, result) = c
            .tick(10, 0, &s, &[], &[], &Isometry3::identity(), 1, 1)
            .unwrap();
        assert_eq!(result.unwrap().status, "completed");
        assert!(!c.active());
        assert!(c.picked);
        assert!(!c.placed());
        assert!(c.hold(&s).unwrap()[grip] > 0.);
        assert_eq!(c.policy.poses.len(), 4);
    }
}
