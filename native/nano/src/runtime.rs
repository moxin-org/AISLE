use crate::{
    kinematics::Robot,
    navigation::{Box2, Map, distance},
    perception::{Color, detect},
};
#[cfg(feature = "dora")]
use crate::{
    oracle::{Feed, Input, Step},
    safety::Guard,
};
use anyhow::{Context, Result, bail, ensure};
#[cfg(feature = "dora")]
use dora_node_api::futures::{StreamExt, executor::block_on};
#[cfg(feature = "dora")]
use dora_node_api::{DoraNode, Event, Metadata, MetadataParameters, Parameter};
use nalgebra::Matrix3;
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[cfg(feature = "dora")]
use std::io::Write;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub model: PathBuf,
    pub output: PathBuf,
    pub map: Map,
    pub goal: [f64; 2],
    pub drop: [f64; 3],
    pub task: String,
    pub arm: String,
    pub seed: u64,
    pub max_frames: u64,
}
pub fn write(path: &Path, v: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(v)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}
pub fn room(seed: u64, goal: [f64; 2], arm: &str) -> Vec<Value> {
    let mut objects = Vec::new();
    let mut add = |name: &str, size: [f64; 3], position: [f64; 3], color: [f64; 3], fixed: bool| {
        objects.push(json!({"name":name,"size":size,"position":position,"color":[color[0],color[1],color[2],1.],"fixed":fixed}))
    };
    add(
        "floor",
        [12., 10., 0.1],
        [0., 0., -0.0843],
        [0.65, 0.62, 0.57],
        true,
    );
    for (n, s, p) in [
        ("north_wall", [12., 0.15, 2.7], [0., 5., 1.31]),
        ("south_wall", [12., 0.15, 2.7], [0., -5., 1.31]),
        ("east_wall", [0.15, 10., 2.7], [6., 0., 1.31]),
        ("west_wall", [0.15, 10., 2.7], [-6., 0., 1.31]),
    ] {
        add(n, s, p, [0.8, 0.83, 0.85], true);
    }
    add(
        "sofa_seat",
        [2.6, 0.9, 0.4],
        [0., 2.6, 0.3],
        [0.2, 0.31, 0.43],
        true,
    );
    add(
        "sofa_back",
        [2.6, 0.22, 0.9],
        [0., 3.1, 0.6],
        [0.2, 0.31, 0.43],
        true,
    );
    for x in [-1.3, 1.3] {
        add(
            &format!("sofa_arm_{x}"),
            [0.22, 0.9, 0.7],
            [x, 2.6, 0.45],
            [0.2, 0.31, 0.43],
            true,
        );
    }
    add(
        "coffee_table",
        [1.4, 0.8, 0.08],
        [0., 1.1, 0.5],
        [0.38, 0.23, 0.12],
        true,
    );
    for x in [-0.6, 0.6] {
        for y in [0.8, 1.4] {
            add(
                &format!("coffee_leg_{x}_{y}"),
                [0.08, 0.08, 0.5],
                [x, y, 0.21],
                [0.19, 0.14, 0.09],
                true,
            );
        }
    }
    add(
        "kitchen_island",
        [1.2, 2., 0.9],
        [-3.4, 0.3, 0.4157],
        [0.45, 0.45, 0.42],
        true,
    );
    add(
        "bookshelf",
        [0.35, 2., 1.6],
        [4.8, 2., 0.7657],
        [0.32, 0.2, 0.12],
        true,
    );
    add(
        "room_divider",
        [0.18, 1.2, 1.5],
        [0.85, -2.5, 0.7157],
        [0.7, 0.72, 0.74],
        true,
    );
    for (i, p) in [[-2.5, -2.5], [3.5, 3.2], [-3.2, 3.6]].iter().enumerate() {
        add(
            &format!("chair_{i}"),
            [0.55, 0.55, 0.7],
            [p[0], p[1], 0.3157],
            [0.3, 0.34, 0.25],
            true,
        );
    }
    let sign = if arm == "left" { 1. } else { -1. };
    add(
        "pickup_table",
        [0.36, 0.70, 0.04],
        [goal[0] + 0.48, goal[1] + sign * 0.23, 0.16],
        [0.25, 0.28, 0.3],
        true,
    );
    let jitter = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1) >> 32) % 1000) as f64
        / 1000.
        * 0.012
        - 0.006;
    add(
        "cube_6",
        [0.04; 3],
        [goal[0] + 0.41 + jitter, goal[1] + sign * 0.23, 0.205],
        [0.92, 0.10, 0.07],
        false,
    );
    add(
        "cube_blue",
        [0.04; 3],
        [goal[0] + 0.53, goal[1] + sign * 0.45, 0.205],
        [0.10, 0.45, 0.95],
        false,
    );
    add(
        "cube_green",
        [0.04; 3],
        [goal[0] + 0.56, goal[1] + sign * 0.04, 0.205],
        [0.10, 0.8, 0.25],
        false,
    );
    for (i, p) in [[3.2, -0.6], [3.4, 0.8], [-2., -3.8], [2.7, 2.2]]
        .iter()
        .enumerate()
    {
        add(
            &format!("floor_cube_{i}"),
            [0.08; 3],
            [p[0], p[1], 0.006],
            [0.95, 0.7, 0.10],
            false,
        );
    }
    objects
}
pub fn map(objects: &[Value]) -> Map {
    Map {
        bounds: [-5.9, 5.9, -4.9, 4.9],
        clearance: 0.55,
        obstacles: objects
            .iter()
            .filter(|o| o["fixed"] == true && o["name"] != "floor")
            .map(|o| Box2 {
                center: [
                    o["position"][0].as_f64().unwrap(),
                    o["position"][1].as_f64().unwrap(),
                ],
                half: [
                    o["size"][0].as_f64().unwrap() / 2.,
                    o["size"][1].as_f64().unwrap() / 2.,
                ],
                clearance: if o["name"] == "pickup_table" {
                    Some(0.27)
                } else {
                    None
                },
            })
            .collect(),
    }
}
/// The base route for a task. Pickup docks facing forward: turning beside the
/// low work surface sweeps a folded shoulder into it even when the base fits.
pub fn route(map: &Map, task: &str, start: [f64; 2], goal: [f64; 2]) -> Result<Vec<[f64; 2]>> {
    if task == "pickup" && distance(start, goal) > 0.15 {
        let pre = [goal[0] - 0.7, goal[1]];
        let mut route = map.plan(start, pre).context("Pickup docking point")?;
        route.extend(map.plan(pre, goal)?);
        Ok(route)
    } else {
        map.plan(start, goal)
    }
}
/// The reason a policy stopped on its own error, from its retained status.
pub fn policy_fault(status: &Value) -> Option<String> {
    (status["fault"] == true).then(|| status["reason"].as_str().unwrap_or("unknown").to_owned())
}
#[cfg(feature = "dora")]
pub(crate) fn stamp(m: &Metadata) -> Result<(u64, u64)> {
    let n = |key| match m.parameters.get(key) {
        Some(Parameter::Integer(n)) if *n >= 0 => Ok(*n as u64),
        _ => bail!("Missing or invalid {key}"),
    };
    ensure!(
        m.parameters.get("env_id") == Some(&Parameter::Integer(0)),
        "Wrong environment"
    );
    n("seq")?;
    Ok((n("observation_id")?, n("sim_time_ns")?))
}
#[cfg(feature = "dora")]
fn send(node: &mut DoraNode, id: &str, p: MetadataParameters, v: Vec<f32>) -> Result<()> {
    node.send_output(id.to_owned().into(), p, v)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
#[cfg(feature = "dora")]
fn decode32(data: &dora_node_api::DoraArray) -> Result<Vec<f64>> {
    let v: &[f32] = data.try_into().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(v.iter().map(|x| *x as f64).collect())
}
/// A row-major 4x4 camera pose; rejects anything that is not a rigid transform.
pub fn transform(v: &[f64]) -> Result<Isometry3<f64>> {
    ensure!(
        v.len() == 16 && v.iter().all(|x| x.is_finite()),
        "Invalid camera transform"
    );
    let rotation =
        Matrix3::from_row_slice(&[v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]]);
    ensure!(
        (rotation.transpose() * rotation - Matrix3::identity()).norm() < 0.001
            && (rotation.determinant() - 1.).abs() < 0.001,
        "Invalid camera rotation"
    );
    Ok(Isometry3::from_parts(
        Translation3::new(v[3], v[7], v[11]),
        UnitQuaternion::from_matrix(&rotation),
    ))
}
#[cfg(feature = "dora")]
pub fn node(role: &str) -> Result<()> {
    match role {
        "supervisor" => return crate::supervisor::run(),
        "control" => return crate::control::run(),
        _ => {}
    }
    let config: Config = serde_json::from_slice(&fs::read(
        std::env::var_os("AISLE_NANO_CONFIG").context("Config missing")?,
    )?)?;
    let (mut node, mut events) = DoraNode::init_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let robot = Robot::read(&config.model)?;
    match role {
        "policy" => {
            let mut policy = Policy::new(config.clone(), robot)?;
            let mut state = None;
            let mut rgb = None;
            let mut depth = None;
            let mut camera = None;
            let mut cached: Option<(u64, u64, Vec<f32>, bool, i64)> = None;
            while let Some(e) = block_on(events.next()) {
                if let Event::Error(ref error) = e {
                    bail!("Dora input error: {error}");
                }
                if let Event::Input { id, data, metadata } = e {
                    let (seq, t) = stamp(&metadata)?;
                    if id.as_str() == "finished" {
                        break;
                    }
                    let delivery = match metadata.parameters.get("seq") {
                        Some(Parameter::Integer(n)) => *n,
                        _ => unreachable!(),
                    };
                    if let Some((old, clock, command, done, sent)) = &mut cached {
                        if seq < *old {
                            continue;
                        }
                        if seq == *old {
                            ensure!(t == *clock, "Changed observation clock");
                            if delivery > *sent {
                                if *done {
                                    node.send_output(
                                        "done".into(),
                                        metadata.parameters.clone(),
                                        vec![seq],
                                    )
                                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                                } else {
                                    send(
                                        &mut node,
                                        "joint_cmd",
                                        metadata.parameters.clone(),
                                        command.clone(),
                                    )?;
                                }
                                *sent = delivery;
                            }
                            continue;
                        }
                    }
                    let mut receipt = fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(config.output.join("policy-inputs.jsonl"))?;
                    writeln!(
                        receipt,
                        "{}",
                        json!({"input":id.as_str(),"seq":seq,"sim_time_ns":t})
                    )?;
                    match id.as_str() {
                        "state" => {
                            state = Some((seq, t, decode32(&data)?, metadata.parameters.clone()))
                        }
                        "rgb" => {
                            let a: &[u8] =
                                (&data).try_into().map_err(|e| anyhow::anyhow!("{e}"))?;
                            rgb = Some((seq, t, a.to_vec()));
                        }
                        "depth" => {
                            let a: &[f32] =
                                (&data).try_into().map_err(|e| anyhow::anyhow!("{e}"))?;
                            depth = Some((seq, t, a.to_vec()));
                        }
                        "camera" => {
                            let a: &[f64] =
                                (&data).try_into().map_err(|e| anyhow::anyhow!("{e}"))?;
                            camera = Some((seq, t, transform(a)?));
                        }
                        _ => bail!("Unknown policy input {id}"),
                    }
                    if let (Some(s), Some(r), Some(d), Some(c)) = (&state, &rgb, &depth, &camera) {
                        ensure!(
                            s.0 == r.0
                                && r.0 == d.0
                                && d.0 == c.0
                                && s.1 == r.1
                                && r.1 == d.1
                                && d.1 == c.1,
                            "Mismatched observation packets"
                        );
                        let dimension = |name: &str| match s.3.get(name) {
                            Some(Parameter::Integer(n)) if *n > 0 && *n <= 4096 => Ok(*n as usize),
                            _ => bail!("Missing image dimension {name}"),
                        };
                        let response = policy.step(
                            s.0,
                            s.1,
                            &s.2,
                            &r.2,
                            &d.2,
                            &c.2,
                            dimension("w")?,
                            dimension("h")?,
                        );
                        let (cmd, done, reason, fault) = match response {
                            Ok((cmd, done, reason)) => (cmd, done, reason, false),
                            Err(e) => (vec![0.; 17], true, format!("{e:#}"), true),
                        };
                        write(
                            &config.output.join("policy-status.json"),
                            &json!({"frame":s.0,"stage":policy.stage,"pose_index":policy.pose_index,"done":done,"fault":fault,"reason":reason,"target_rgb_m":policy.target,"command":cmd}),
                        )?;
                        let done = done || s.0 >= config.max_frames;
                        let command: Vec<f32> = cmd.into_iter().map(|x| x as f32).collect();
                        cached = Some((s.0, s.1, command.clone(), done, delivery));
                        if done {
                            node.send_output("done".into(), s.3.clone(), vec![s.0])
                                .map_err(|e| anyhow::anyhow!("{e}"))?;
                        } else {
                            send(&mut node, "joint_cmd", s.3.clone(), command)?;
                        }
                        state = None;
                        rgb = None;
                        depth = None;
                        camera = None;
                    }
                }
            }
        }
        "guard" => {
            let mut guard = Guard::new(robot);
            let mut state = None;
            let mut command = None;
            let mut cached: Option<crate::delivery::CachedCommand> = None;
            let mut last_delivery = -1;
            while let Some(e) = block_on(events.next()) {
                if let Event::Error(ref error) = e {
                    bail!("Dora input error: {error}");
                }
                if let Event::Input { id, data, metadata } = e {
                    let key = stamp(&metadata)?;
                    if id.as_str() == "finished" {
                        break;
                    }
                    let delivery = match metadata.parameters.get("seq") {
                        Some(Parameter::Integer(n)) => *n,
                        _ => unreachable!(),
                    };
                    if let Some(c) = &cached {
                        if key.0 < c.id || (key.0 == c.id && id.as_str() == "state") {
                            continue;
                        }
                        if id.as_str() == "command"
                            && let Some(safe) = c.replay(key.0, key.1, &decode32(&data)?)?
                        {
                            if delivery > last_delivery {
                                send(
                                    &mut node,
                                    "joint_cmd_safe",
                                    metadata.parameters.clone(),
                                    safe.iter().map(|x| *x as f32).collect(),
                                )?;
                                last_delivery = delivery;
                            }
                            continue;
                        }
                    }
                    if id.as_str() == "state" {
                        state = Some((key, decode32(&data)?));
                    } else if id.as_str() == "command" {
                        command = Some((key, decode32(&data)?, metadata.parameters));
                    }
                    if let (Some((s, q)), Some((c, v, p))) = (&state, &command) {
                        ensure!(s == c && q.len() == 24, "Guard state/command mismatch");
                        let safe = guard.apply(s.1, &q[7..], v)?;
                        cached = Some(crate::delivery::CachedCommand::new(
                            s.0,
                            s.1,
                            v.clone(),
                            safe.clone(),
                        ));
                        last_delivery = delivery;
                        send(
                            &mut node,
                            "joint_cmd_safe",
                            p.clone(),
                            safe.iter().map(|x| *x as f32).collect(),
                        )?;
                        let mut f = fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(config.output.join("guard.jsonl"))?;
                        writeln!(
                            f,
                            "{}",
                            json!({"seq":s.0,"stamp":s.1,"clamped":safe!=*v,"proposed":v,"applied":safe})
                        )?;
                        state = None;
                        command = None;
                    }
                }
            }
        }
        "judge" => {
            let mut feed = Feed::new(config.goal, config.drop);
            let mut done = None;
            let mut trace = fs::File::create(config.output.join("judge-trace.jsonl"))?;
            while let Some(e) = block_on(events.next()) {
                if let Event::Error(ref error) = e {
                    bail!("Dora input error: {error}");
                }
                if let Event::Input { id, data, metadata } = e {
                    let (seq, t) = stamp(&metadata)?;
                    let step = match id.as_str() {
                        "finished" => {
                            done = Some(seq);
                            break;
                        }
                        "done" => {
                            done = Some(seq);
                            continue;
                        }
                        "state" => feed.offer(seq, t, Input::State(&decode32(&data)?))?,
                        "truth" => {
                            let v: &[f64] =
                                (&data).try_into().map_err(|e| anyhow::anyhow!("{e}"))?;
                            feed.offer(seq, t, Input::Truth(v))?
                        }
                        _ => bail!("Unknown judge input {id}"),
                    };
                    if step == Step::Pending {
                        continue;
                    }
                    if let Step::Observed(sample) = step {
                        writeln!(trace, "{}", serde_json::to_string(&sample)?)?;
                    }
                    write(&config.output.join("judge-progress.json"), &feed.verdict())?;
                    // State and truth of one observation carry the same receipt metadata.
                    node.send_output("ack".into(), metadata.parameters, vec![seq])
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                }
            }
            let verdict = feed.verdict().context("No physics evidence")?;
            let complete = done.is_some() && done == feed.last();
            let success = crate::oracle::success(&config.task, complete, &verdict);
            write(
                &config.output.join("result.json"),
                &json!({"ok":complete,"success":success,"task":config.task,"verdict":verdict,"seed":config.seed,"oracle":"independent physics contacts/poses; no controller success input","limitations":["experimental Nano profile, not frozen benchmark acceptance","static map and simulator localization, not SLAM","color-specific RGB perception, not open vocabulary","simulation model, not hardware calibration"]}),
            )?;
        }
        _ => bail!("Unknown node role"),
    }
    Ok(())
}

pub(crate) struct Policy {
    pub(crate) config: Config,
    pub(crate) robot: Robot,
    pub(crate) stage: String,
    pub(crate) route: Vec<[f64; 2]>,
    pub(crate) waypoint: usize,
    pub(crate) stage_frame: u64,
    pub(crate) target: Option<[f64; 3]>,
    pub(crate) item_color: Color,
    pub(crate) item_height: f64,
    pub(crate) depart_from: Option<[f64; 2]>,
    transit: Option<Map>,
    pub(crate) poses: Vec<Vec<f64>>,
    pub(crate) pose_index: usize,
    pub(crate) external_skill: bool,
    pub(crate) pose_offset: usize,
}
impl Policy {
    pub(crate) fn new(config: Config, robot: Robot) -> Result<Self> {
        let transit = (config.task == "household").then(|| config.map.transit());
        Ok(Self {
            transit,
            config,
            robot,
            stage: "navigate".into(),
            route: Vec::new(),
            waypoint: 0,
            stage_frame: 0,
            target: None,
            item_color: Color::Red,
            item_height: 0.04,
            depart_from: None,
            poses: Vec::new(),
            pose_index: 0,
            external_skill: false,
            pose_offset: 0,
        })
    }
    pub(crate) fn prepare_place(&mut self, state: &[f64], seq: u64) -> Result<()> {
        let base = Isometry3::from_parts(
            Translation3::new(state[0], state[1], state[2]),
            UnitQuaternion::new_normalize(Quaternion::new(state[6], state[3], state[4], state[5])),
        );
        let drop = base
            .inverse_transform_point(&Vector3::from(self.config.drop).into())
            .coords;
        let dir = base.rotation.inverse()
            * Vector3::y()
            * if self.config.arm == "left" { 1. } else { -1. };
        let arm = self.robot.arm(&self.config.arm)?;
        let link = format!("{}_wrist_housing", self.config.arm);
        let tcp = Vector3::new(0.012, 0., -0.12);
        let initial = &state[7..];
        // Keep the same shoulder/elbow branch over the reachable work surface.
        // A high transfer point near the workspace boundary can force an elbow flip.
        let solve = |clearance: f64| -> Result<Vec<Vec<f64>>> {
            let mut previous = initial.to_vec();
            let mut poses = Vec::new();
            for (p, grip) in [
                (drop + Vector3::z() * clearance, 0.75),
                (drop + Vector3::z() * 0.015, 0.75),
                (drop + Vector3::z() * 0.015, -0.2),
                (drop + Vector3::z() * clearance, -0.2),
            ] {
                let start = self
                    .robot
                    .fk(&link, &previous)?
                    .transform_point(&tcp.into())
                    .coords;
                let count = if self.transit.is_some() {
                    ((p - start).norm() / 0.022).ceil().max(1.) as usize
                } else {
                    1
                };
                let mut q = previous.clone();
                for step in 1..=count {
                    let point = start + (p - start) * (step as f64 / count as f64);
                    let next = self.robot.ik_axis(
                        &self.config.arm,
                        &link,
                        tcp,
                        point,
                        Vector3::x(),
                        Some(dir),
                        &q,
                    )?;
                    crate::safety::validate_arm_transition(&q, &next, &arm[..5])?;
                    q = next;
                }
                crate::safety::validate_arm_transition(&previous, &q, &arm[..5])?;
                q[arm[5]] = grip;
                previous = q.clone();
                poses.push(q);
            }
            Ok(poses)
        };
        let poses = if self.transit.is_some() {
            let mut selected = None;
            for clearance in [0.055, 0.075, 0.045, 0.10] {
                if let Ok(p) = solve(clearance) {
                    selected = Some(p);
                    break;
                }
            }
            selected.context("no continuous placement trajectory above this surface")?
        } else {
            solve(0.10)?
        };
        self.poses = poses;
        self.pose_index = 0;
        self.pose_offset = 4;
        self.stage = "manipulate".into();
        self.stage_frame = seq;
        write(&self.config.output.join("place-plan.json"), &self.poses)?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        seq: u64,
        t: u64,
        state: &[f64],
        rgb: &[u8],
        depth: &[f32],
        camera: &Isometry3<f64>,
        width: usize,
        height: usize,
    ) -> Result<(Vec<f64>, bool, String)> {
        ensure!(
            state.len() == 24 && state.iter().all(|v| v.is_finite()),
            "Invalid state"
        );
        let q = &state[7..];
        let norm = state[3..7].iter().map(|v| v * v).sum::<f64>();
        ensure!((norm - 1.).abs() < 0.01, "Invalid base quaternion");
        let mut v = vec![0.; 17];
        // Aim the head at the work surface through real joint motors. The
        // teleoperation park posture looks sideways and cannot see this table.
        let pan = if self.config.arm == "left" {
            0.62
        } else {
            -0.62
        };
        v[3] = ((pan - q[3]) * 2.).clamp(-0.5, 0.5);
        v[4] = ((0.30 - q[4]) * 2.).clamp(-0.5, 0.5);
        if self.config.task == "inspect" {
            return Ok((v, seq >= 2, "inspection".into()));
        }
        let base = Isometry3::from_parts(
            Translation3::new(state[0], state[1], state[2]),
            UnitQuaternion::new_normalize(Quaternion::new(state[6], state[3], state[4], state[5])),
        );
        let yaw = base.rotation.euler_angles().2;
        if self.stage == "navigate" {
            if let Some(origin) = self.depart_from {
                // Reverse straight out before rotating the extended wrist past a workstation.
                if state[0] > origin[0] - 0.8 {
                    v[..3].copy_from_slice(&wheel_rates(
                        -0.095,
                        0.,
                        (-1.4 * yaw).clamp(-0.25, 0.25),
                    ));
                    return Ok((v, false, "backing out of station".into()));
                }
                let mut error = 0.0_f64;
                for (i, park) in self
                    .robot
                    .arm(&self.config.arm)?
                    .into_iter()
                    .zip([0.2, 0., 0.35, -1., 0., 0.])
                {
                    let e = park - q[i];
                    error = error.max(e.abs());
                    v[i] = (e * 1.6).clamp(-0.40, 0.40);
                }
                if error > 0.035 {
                    return Ok((v, false, "stowing arm".into()));
                }
                v[5..].fill(0.);
                self.depart_from = None;
            }
            if self.route.is_empty() {
                let start = [state[0], state[1]];
                // Reach the low work surface facing forward. Turning beside it
                // sweeps a folded shoulder into the table even when the base fits.
                let facing_dock = (self.config.goal[1] - start[1]).abs() < 0.025
                    && (0.0..0.70).contains(&(self.config.goal[0] - start[0]))
                    && yaw.abs() < 0.04
                    && self.config.map.clear_segment(start, self.config.goal);
                if self.transit.is_none() {
                    self.route =
                        route(&self.config.map, &self.config.task, start, self.config.goal)?;
                } else if self.config.task == "pickup"
                    && distance(start, self.config.goal) > 0.15
                    && !facing_dock
                {
                    let pre = [self.config.goal[0] - 0.7, self.config.goal[1]];
                    self.route = if let Some(transit) = &self.transit {
                        transit.transit_route(start, pre)?
                    } else {
                        self.config.map.plan(start, pre)?
                    };
                    self.route
                        .extend(self.config.map.plan(pre, self.config.goal)?);
                } else {
                    self.route = self.config.map.plan(start, self.config.goal)?;
                }
                write(&self.config.output.join("route.json"), &self.route)?;
            }
            let next = self.route[self.waypoint];
            let error = distance([state[0], state[1]], next);
            let intermediate = self.waypoint + 1 < self.route.len();
            if error < if intermediate { 0.10 } else { 0.035 } {
                if self.waypoint + 1 < self.route.len() {
                    self.waypoint += 1;
                    return Ok((v, false, "next waypoint".into()));
                } else if yaw.abs() < 0.03 {
                    self.stage = "observe".into();
                    self.stage_frame = seq;
                    if self.config.task == "navigate" {
                        self.stage = "settle".into();
                        return Ok((v, false, "stopping at navigation goal".into()));
                    }
                }
            }
            if self.stage == "navigate" {
                // The imported collision meshes have no independent omni rollers.
                // Use their validated forward/turn modes, not ideal lateral drive.
                let (forward, turn) =
                    crate::navigation::base_command(next, [state[0], state[1], yaw])?;
                v[..3].copy_from_slice(&wheel_rates(forward, 0., turn));
            }
        } else if self.stage == "observe" {
            if (q[3] - pan).abs() > 0.04 || (q[4] - 0.30).abs() > 0.04 {
                ensure!(
                    seq - self.stage_frame < 60,
                    "Head failed to aim at pickup surface"
                );
                return Ok((v, false, "aiming head".into()));
            }
            let hits = detect(
                rgb,
                depth,
                width,
                height,
                height as f64 / (2. * 35f64.to_radians().tan()),
                self.item_color,
                t,
                t,
                camera,
            )?;
            let hit = hits
                .iter()
                .find(|h| h.extent.max() < 0.12 && h.position.z > 0.15 && h.position.z < 0.3);
            if let Some(hit) = hit {
                let mut target = hit.position;
                // Convert the top face measured in RGB-D to the declared grasp center.
                target.z = hit.position.z + hit.extent.z / 2. - self.item_height / 2.;
                self.target = Some([target.x, target.y, target.z]);
                let local = base.inverse_transform_point(&target.into()).coords;
                let link = format!("{}_wrist_housing", self.config.arm);
                let offset = Vector3::new(0.012, 0., -0.12);
                // Align the jaw with a cube face; a five-DoF arm needs approach
                // tilt freedom instead of an arbitrary full tool orientation.
                let dir = base.rotation.inverse()
                    * Vector3::y()
                    * if self.config.arm == "left" { 1. } else { -1. };
                let arm = self.robot.arm(&self.config.arm)?;
                let mut initial = q.to_vec();
                for (p, grip) in [
                    (local + Vector3::z() * 0.10, -0.2),
                    (local, -0.2),
                    (local, 0.75),
                    (local + Vector3::z() * 0.10, 0.75),
                ] {
                    let mut solved = self.robot.ik_axis(
                        &self.config.arm,
                        &link,
                        offset,
                        p,
                        Vector3::x(),
                        Some(dir),
                        &initial,
                    )?;
                    solved[arm[5]] = grip;
                    self.poses.push(solved.clone());
                    initial = solved;
                }
                if !self.external_skill {
                    let drop = base
                        .inverse_transform_point(&Vector3::from(self.config.drop).into())
                        .coords;
                    for (p, grip) in [
                        (drop + Vector3::z() * 0.10, 0.75),
                        (drop + Vector3::z() * 0.015, 0.75),
                        (drop + Vector3::z() * 0.015, -0.2),
                        (drop + Vector3::z() * 0.10, -0.2),
                    ] {
                        let mut solved = self.robot.ik_axis(
                            &self.config.arm,
                            &link,
                            offset,
                            p,
                            Vector3::x(),
                            Some(dir),
                            &initial,
                        )?;
                        solved[arm[5]] = grip;
                        self.poses.push(solved.clone());
                        initial = solved;
                    }
                }
                let mut previous = q;
                for pose in &self.poses {
                    crate::safety::validate_arm_transition(previous, pose, &arm[..5])?;
                    previous = pose;
                }
                write(
                    &self.config.output.join("ik-plan.json"),
                    &json!({"rgb_detection":self.target,"poses":self.poses,"offset":offset.as_slice(),"link":link}),
                )?;
                self.stage = "manipulate".into();
                self.stage_frame = seq;
            } else if seq - self.stage_frame > 30 {
                bail!("RGB target not found; no oracle fallback");
            }
        } else if self.stage == "manipulate" {
            let goal = &self.poses[self.pose_index];
            let arm = self.robot.arm(&self.config.arm)?;
            let mut error = 0f64;
            // Carrying a friction-held object needs lower arm acceleration than
            // the unloaded approach. The gripper keeps its independent closure.
            let carrying = matches!(self.pose_index + self.pose_offset, 3..=5);
            for &i in &arm {
                let e = goal[i] - q[i];
                let gripper = i == arm[5];
                let (gain, cap) = if carrying && !gripper {
                    (1.0, 0.20)
                } else {
                    (2.0, 0.45)
                };
                v[i] = (e * gain).clamp(-cap, cap);
                if !gripper {
                    error = error.max(e.abs());
                }
            }
            let minimum = if matches!(self.pose_index + self.pose_offset, 2 | 6) {
                15
            } else {
                5
            };
            if error < 0.025 && seq - self.stage_frame > minimum || seq - self.stage_frame > 150 {
                ensure!(
                    error < 0.08,
                    "Arm stalled at pose {}: {error:.3} rad",
                    self.pose_index
                );
                self.pose_index += 1;
                self.stage_frame = seq;
                if self.pose_index == self.poses.len() {
                    self.stage = "settle".into();
                }
            }
        } else if self.stage == "settle" && seq - self.stage_frame > 15 {
            return Ok((v, true, "trajectory finished; judge decides".into()));
        }
        Ok((v, false, "".into()))
    }
}
pub fn wheel_rates(forward: f64, lateral: f64, turn: f64) -> [f64; 3] {
    let mut r = [
        lateral / 0.034 - 3.4 * turn,
        ((3f64).sqrt() / 2. * forward - 0.5 * lateral) / 0.034 - 3.4 * turn,
        (-(3f64).sqrt() / 2. * forward - 0.5 * lateral) / 0.034 - 3.4 * turn,
    ];
    let scale = (5. / r.iter().copied().map(f64::abs).fold(1e-9, f64::max)).min(1.);
    for v in &mut r {
        *v *= scale;
    }
    r
}

#[cfg(test)]
mod household_tests {
    use super::*;
    /// TC-8 / MOB-3. Regression for the measured hall grasp that formerly produced an elbow flip.
    #[test]
    fn plan_four_placements_without_a_joint_branch_jump() {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/robots/adora_nano/model.urdf");
        let output =
            std::env::temp_dir().join(format!("household-place-regression-{}", std::process::id()));
        std::fs::create_dir_all(&output).unwrap();
        for station in crate::household::stations().unwrap() {
            let robot = Robot::read(&model).unwrap();
            let config = Config {
                model: model.clone(),
                output: output.clone(),
                map: Map {
                    bounds: [-6., 6., -5., 5.],
                    clearance: 0.55,
                    obstacles: vec![],
                },
                goal: station.goal,
                drop: station.drop,
                task: "household".into(),
                arm: "right".into(),
                seed: 0,
                max_frames: 1000,
            };
            let mut policy = Policy::new(config, robot).unwrap();
            let state = vec![
                station.goal[0] - 0.031,
                station.goal[1] + 0.0044,
                -0.0001,
                0.,
                0.,
                -0.0143,
                0.999898,
                0.,
                0.,
                0.,
                -0.603,
                0.30,
                0.2,
                0.,
                0.358,
                -0.998,
                0.,
                -0.054,
                1.446,
                -1.012,
                0.280,
                -0.257,
                0.,
                0.341,
            ];
            for (y_offset, yaw) in [(0.0044, -0.0286), (-0.018, 0.022), (0.018, -0.022)] {
                let mut measured = state.clone();
                measured[1] = station.goal[1] + y_offset;
                measured[5] = (0.5_f64 * yaw).sin();
                measured[6] = (0.5_f64 * yaw).cos();
                policy
                    .prepare_place(&measured, 100)
                    .unwrap_or_else(|e| panic!("{} ({y_offset}, {yaw}): {e}", station.station));
                assert_eq!(policy.poses.len(), 4);
            }
        }
        std::fs::remove_dir_all(output).unwrap();
    }
}
