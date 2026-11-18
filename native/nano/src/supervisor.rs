//! Asynchronous outer-loop decisions; only the native skills own joint commands.
use crate::{
    kinematics::Robot,
    runtime::{Config, stamp, transform, write},
    skills::{Controller, Skill},
};
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use dora_node_api::{
    DoraNode, Event, MetadataParameters, Parameter,
    futures::{StreamExt, executor::block_on},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn emit(node: &mut DoraNode, id: &str, m: MetadataParameters, v: Value) -> Result<()> {
    node.send_output(id.into(), m, v.to_string())
        .map_err(|e| anyhow::anyhow!("{e}"))
}
fn log(config: &Config, event: Value) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.output.join("supervisor.jsonl"))?;
    writeln!(f, "{event}")?;
    Ok(())
}
fn integer(m: &MetadataParameters, k: &str) -> Result<u64> {
    match m.get(k) {
        Some(Parameter::Integer(v)) if *v >= 0 => Ok(*v as u64),
        _ => bail!("invalid {k}"),
    }
}
fn goal(m: &MetadataParameters) -> Option<&str> {
    match m.get("goal_id") {
        Some(Parameter::String(s)) => Some(s),
        _ => None,
    }
}
fn png(rgb: &[u8], w: u32, h: u32) -> Result<String> {
    ensure!(
        rgb.len() == w as usize * h as usize * 3,
        "RGB size mismatch"
    );
    let mut b = Vec::new();
    {
        let mut e = png::Encoder::new(&mut b, w, h);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header()?.write_image_data(rgb)?;
    }
    Ok(format!("data:image/png;base64,{}", STANDARD.encode(b)))
}
pub use crate::admission::Admission;
pub fn run() -> Result<()> {
    let config: Config = serde_json::from_slice(&fs::read(
        std::env::var_os("AISLE_NANO_CONFIG").context("config missing")?,
    )?)?;
    let mission = config
        .output
        .file_name()
        .context("mission ID")?
        .to_string_lossy()
        .to_string();
    ensure!(mission.len() < 80, "mission ID too long");
    let mut c = Controller::new(config.clone(), Robot::read(&config.model)?)?;
    let (mut node, mut events) = DoraNode::init_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut admission = Admission::default();
    let mut feedback = json!({"status":"ready"});
    let mut history: Vec<Value> = Vec::new();
    let household = config.task == "household";
    let mut instruction = if household {
        "Tidy the furnished home in one pass. Deliver both folded garments inside their laundry baskets and both construction toys onto their coffee tables. Choose a nearby unfinished station using base_pose and goal_xy. Then pick and place the corresponding item."
    } else if config.task == "navigate" {
        "Navigate to station."
    } else {
        "Navigate to station, pick the red cube, then place it in placement_zone."
    }
    .to_string();
    let mut paused = false;
    let mut queued_control: Option<Value> = None;
    let mut control_id = 0;
    let mut proposal: Option<Value> = None;
    let mut skill_id = 0;
    let mut nav_done = false;
    let mut failures = 0;
    let mut packets: BTreeMap<String, dora_node_api::DoraArray> = BTreeMap::new();
    let mut frame: Option<(u64, u64, u64)> = None;
    let mut cached: Option<(u64, u64, Vec<f32>, bool, u64)> = None;
    let mut fatal: Option<String> = None;
    while let Some(event) = block_on(events.next()) {
        match event {
            Event::Stop { .. } => break,
            Event::Error(e) => bail!("Dora error: {e}"),
            Event::Input { id, metadata, data } => {
                let mut meta = metadata.parameters.clone();
                match id.as_str() {
                    "finished" => break,
                    "control_msg" => {
                        let text = String::try_from(&data).map_err(|e| anyhow::anyhow!("{e}"))?;
                        ensure!(text.len() <= 4096, "control too large");
                        let v: Value = serde_json::from_str(&text)?;
                        let n = v["request_id"].as_u64().unwrap_or(0);
                        if v["mission_id"] == mission
                            && n > control_id
                            && matches!(v["op"].as_str(), Some("cancel" | "replan"))
                        {
                            if v["op"] == "replan"
                                && v["instruction"]
                                    .as_str()
                                    .is_none_or(|s| s.is_empty() || s.len() > 2048)
                            {
                                continue;
                            }
                            control_id = n;
                            queued_control = Some(v);
                        }
                        continue;
                    }
                    "subgoal_msg" | "status_result" => {
                        let id = goal(&meta).unwrap_or("");
                        let t = integer(&meta, "sim_time_ns").unwrap_or(u64::MAX);
                        if !admission.accepts(id, t, now()) {
                            log(
                                &config,
                                json!({"event":"ignored_retired_reply","goal_id":id}),
                            )?;
                            continue;
                        }
                        let v = String::try_from(&data)
                            .map_err(|e| e.to_string())
                            .and_then(|text| crate::admission::reply(&text))
                            .unwrap_or_else(|reason| json!({"refused":reason}));
                        admission.cancel();
                        if v.pointer("/aisle/proposal").is_some() {
                            proposal = Some(v["aisle"]["proposal"].clone());
                        } else {
                            feedback = json!({"status":"planner_failed","detail":v});
                            failures += 1;
                        }
                        log(
                            &config,
                            json!({"event":"planner_result","reply":v,"goal_id":id}),
                        )?;
                        continue;
                    }
                    _ => {}
                }
                let (seq, t) = stamp(&metadata)?;
                let delivery = integer(&meta, "seq")?;
                if let Some((old, clock, command, done, sent)) = &mut cached {
                    if seq < *old {
                        continue;
                    }
                    if seq == *old {
                        ensure!(t == *clock, "changed retry timestamp");
                        if delivery > *sent {
                            if *done {
                                node.send_output("done".into(), meta, vec![seq])
                            } else {
                                node.send_output("joint_cmd".into(), meta, command.clone())
                            }
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                            *sent = delivery;
                        }
                        continue;
                    }
                }
                if let Some((n, clock, _)) = frame {
                    ensure!(n == seq && clock == t, "mismatched/incomplete sensor frame");
                } else {
                    frame = Some((seq, t, now()));
                }
                ensure!(
                    [
                        "state",
                        "rgb_head",
                        "rgb_left_wrist",
                        "rgb_right_wrist",
                        "depth",
                        "camera"
                    ]
                    .contains(&id.as_str()),
                    "unknown sensor topic"
                );
                packets.insert(id.to_string(), data);
                if packets.len() != 6 {
                    continue;
                }
                let captured = frame.unwrap().2;
                let w = integer(&meta, "w")? as usize;
                let h = integer(&meta, "h")? as usize;
                ensure!(
                    w > 0 && h > 0 && w <= 4096 && h <= 4096,
                    "invalid dimensions"
                );
                let state: &[f32] = (&packets["state"])
                    .try_into()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let state: Vec<f64> = state.iter().map(|x| *x as f64).collect();
                let rgb: &[u8] = (&packets["rgb_head"])
                    .try_into()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let depth: &[f32] = (&packets["depth"])
                    .try_into()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let camera: &[f64] = (&packets["camera"])
                    .try_into()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                if let Some(v) = queued_control.take() {
                    if let Some(old) = admission.cancel() {
                        let mut m = meta.clone();
                        m.insert("goal_id".into(), Parameter::String(old));
                        emit(
                            &mut node,
                            "cancel_msg",
                            m,
                            json!({"reason":"operator control"}),
                        )?;
                    }
                    proposal = None;
                    let cancelled = c.cancel(&state, "operator control at new observation")?;
                    paused = v["op"] == "cancel";
                    if !paused {
                        instruction = v["instruction"].as_str().unwrap().to_string();
                    }
                    feedback = json!({"status":if paused{"paused"}else{"replanning"},"cancelled":cancelled,"control":v,"observation_id":seq});
                    emit(&mut node, "skill_result", meta.clone(), feedback.clone())?;
                    log(&config, feedback.clone())?;
                }
                if let Some(v) = proposal.take() {
                    skill_id += 1;
                    history.push(json!({"role":"assistant","content":v.to_string()}));
                    match serde_json::from_value::<Skill>(v.clone()).and_then(|skill| {
                        c.start(skill_id, skill, seq, &state)
                            .map_err(serde::de::Error::custom)
                    }) {
                        Ok(()) => {
                            feedback = json!({"status":"accepted","skill_id":skill_id,"skill":v});
                        }
                        Err(e) => {
                            failures += 1;
                            feedback = json!({"status":"rejected","skill_id":skill_id,"skill":v,"reason":e.to_string()});
                        }
                    }
                    log(&config, feedback.clone())?;
                    emit(&mut node, "skill_result", meta.clone(), feedback.clone())?;
                }
                let (command, outcome) =
                    c.tick(seq, t, &state, rgb, depth, &transform(camera)?, w, h)?;
                if let Some(outcome) = outcome {
                    if matches!(outcome.skill,Skill::Navigate{ref target} if target=="station")
                        && outcome.status == "completed"
                    {
                        nav_done = true;
                    }
                    if outcome.status == "failed" {
                        failures += 1;
                    } else if household
                        && (outcome.status == "already_placed"
                            || outcome.status == "completed"
                                && matches!(
                                    outcome.skill,
                                    Skill::Navigate { .. } | Skill::Place { .. }
                                ))
                    {
                        // A new station gets its own retry budget. Frame and inference budgets bound the full mission.
                        failures = 0;
                    }
                    feedback = serde_json::to_value(outcome)?;
                    log(&config, feedback.clone())?;
                    emit(&mut node, "skill_result", meta.clone(), feedback.clone())?;
                }
                if admission
                    .pending
                    .as_ref()
                    .is_some_and(|p| now().saturating_sub(p.captured) > 30000)
                {
                    let old = admission.cancel().unwrap();
                    let mut m = meta.clone();
                    m.insert("goal_id".into(), Parameter::String(old));
                    emit(
                        &mut node,
                        "cancel_msg",
                        m,
                        json!({"reason":"planner deadline"}),
                    )?;
                    failures += 1;
                    feedback = json!({"status":"planner_timeout"});
                }
                let done = c.placed()
                    || config.task == "navigate" && nav_done
                    || seq >= config.max_frames
                    || failures >= 5
                    || admission.generation >= 32;
                if done && !c.placed() && !(config.task == "navigate" && nav_done) {
                    fatal = Some("mission budget exhausted".into());
                }
                if !done && !paused && !c.active() && admission.pending.is_none() {
                    let id = admission.request(&mission, t, captured);
                    let rules = if household {
                        "Choose exactly one available_subgoals entry. On a navigation decision prefer the closest station by map coordinates. The basket/table is beside the source shelf. Return pick before place. Completed trajectories require independent physics scoring."
                    } else {
                        "Only registered labels/arm. Navigate station first. After pick_trajectory_complete issue place, not navigate placement_zone. Judge alone verifies grasp and final success."
                    };
                    let context = json!({"mission":instruction,"capabilities":c.capabilities(),"feedback":feedback,"base_pose":&state[..7],"joints":&state[7..],"navigation_at_station":c.at_station([state[0],state[1]]),"rules":rules});
                    let mut parts = vec![json!({"type":"text","text":context.to_string()})];
                    for camera in ["head", "left_wrist", "right_wrist"] {
                        let a: &[u8] = (&packets[&format!("rgb_{camera}")])
                            .try_into()
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        parts.push(json!({"type":"text","text":format!("Camera: {camera}")}));
                        parts.push(json!({"type":"image_url","image_url":{"url":png(a,w as u32,h as u32)?}}));
                    }
                    // Retain compact decision/feedback history; only the current sensor snapshot is media.
                    let user = json!({"role":"user","content":context.to_string()});
                    let mut messages = vec![
                        json!({"role":"system","content":"Control Nano via one typed subgoal per response. Use measured feedback and three cameras. Never invent capabilities or claim physical success."}),
                    ];
                    if history.len() > 6 {
                        history.drain(..history.len() - 6);
                    }
                    messages.extend(history.clone());
                    messages.push(json!({"role":"user","content":parts}));
                    history.push(user);
                    let req = json!({"model":"deepseek-flash","max_tokens":512,"messages":messages,"observation":{"goal_id":id,"observation_ns":t,"env_id":0,"captured_at_ms":captured}});
                    meta.insert("goal_id".into(), Parameter::String(id.clone()));
                    emit(&mut node, "observation_msg", meta.clone(), req)?;
                    log(
                        &config,
                        json!({"event":"planner_request","goal_id":id,"observation_id":seq,"sim_time_ns":t,"context":context}),
                    )?;
                }
                write(
                    &config.output.join("supervisor-status.json"),
                    &json!({"mission_id":mission,"observation_id":seq,"phase":c.phase(),"pose_index":c.pose_index(),"paused":paused,"pending":admission.pending.as_ref().map(|p|&p.goal),"feedback":feedback,"done":done,"failure":fatal}),
                )?;
                // Real-time holding avoids spending the simulation budget while cloud inference waits.
                if !c.active() && !done {
                    std::thread::sleep(Duration::from_millis(200));
                }
                let command: Vec<f32> = command.into_iter().map(|x| x as f32).collect();
                cached = Some((seq, t, command.clone(), done, delivery));
                if done {
                    node.send_output("done".into(), meta, vec![seq])
                } else {
                    node.send_output("joint_cmd".into(), meta, command)
                }
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                packets.clear();
                frame = None;
            }
            _ => {}
        }
    }
    Ok(())
}
