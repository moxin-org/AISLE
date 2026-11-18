//! Explicit test fixture: emits prerecorded RGB once, never drives a robot.
use anyhow::{Context, Result, ensure};
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};

fn run() -> Result<Value> {
    let input = PathBuf::from(std::env::var("FIXTURE_INPUT")?);
    let output = PathBuf::from(std::env::var("FIXTURE_OUTPUT")?);
    let inference_started = PathBuf::from(std::env::var("FIXTURE_INFERENCE_STARTED")?);
    let cancel = std::env::var("FIXTURE_CANCEL").as_deref() == Ok("1");
    let replace = std::env::var("FIXTURE_REPLACE").as_deref() == Ok("1");
    let mut req: Value = serde_json::from_slice(&std::fs::read(input)?)?;
    req["observation"] = json!({"goal_id":"fixture-goal","observation_ns":123,"env_id":0,"captured_at_ms":aisle_octos::service::now_ms()});
    let (mut node, mut events) = DoraNode::init_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut sent = None;
    let mut cancelled = false;
    let mut replaced = false;
    let mut superseded_results = 0;
    let mut ticks = 0;
    let start = Instant::now();
    while let Some(event) = events.recv() {
        ensure!(start.elapsed().as_secs() < 20, "fixture deadline exceeded");
        if let Event::Input { id, data, metadata } = event {
            if id.as_str() == "tick" {
                ticks += 1;
                if sent.is_none() {
                    let meta = MetadataParameters::from([
                        ("goal_id".into(), Parameter::String("fixture-goal".into())),
                        ("sim_time_ns".into(), Parameter::Integer(123)),
                        ("env_id".into(), Parameter::Integer(0)),
                        ("seq".into(), Parameter::Integer(0)),
                        ("turn_epoch".into(), Parameter::Integer(7)),
                        ("turn_id".into(), Parameter::Integer(8)),
                    ]);
                    req["observation"]["captured_at_ms"] = json!(aisle_octos::service::now_ms());
                    node.send_output("observation_msg".into(), meta, req.to_string())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    sent = Some(Instant::now());
                } else if cancel
                    && !cancelled
                    && inference_started.exists()
                    && sent.is_some_and(|when| when.elapsed().as_millis() >= 200)
                {
                    node.send_output(
                        "cancel_msg".into(),
                        MetadataParameters::default(),
                        "cancel".to_owned(),
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    cancelled = true;
                } else if replace
                    && !replaced
                    && inference_started.exists()
                    && sent.is_some_and(|when| when.elapsed().as_millis() >= 200)
                {
                    req["observation"]["goal_id"] = json!("next-goal");
                    req["observation"]["observation_ns"] = json!(124);
                    req["observation"]["captured_at_ms"] = json!(aisle_octos::service::now_ms());
                    let meta = MetadataParameters::from([
                        ("goal_id".into(), Parameter::String("next-goal".into())),
                        ("sim_time_ns".into(), Parameter::Integer(124)),
                        ("env_id".into(), Parameter::Integer(0)),
                        ("seq".into(), Parameter::Integer(1)),
                        ("turn_epoch".into(), Parameter::Integer(7)),
                        ("turn_id".into(), Parameter::Integer(9)),
                    ]);
                    node.send_output("observation_msg".into(), meta, req.to_string())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    replaced = true;
                }
            } else if matches!(id.as_str(), "proposal" | "status") {
                let text =
                    String::try_from(&data).map_err(|_| anyhow::anyhow!("bad reply Arrow type"))?;
                let reply: Value = serde_json::from_str(&text)?;
                let next = replace && id.as_str() == "proposal";
                ensure!(
                    metadata.parameters.get("goal_id")
                        == Some(&Parameter::String(
                            if next { "next-goal" } else { "fixture-goal" }.into()
                        )),
                    "lost goal correlation"
                );
                ensure!(
                    metadata.parameters.get("turn_epoch") == Some(&Parameter::Integer(7)),
                    "lost epoch"
                );
                ensure!(
                    metadata.parameters.get("turn_id")
                        == Some(&Parameter::Integer(if next { 9 } else { 8 })),
                    "lost turn"
                );
                ensure!(
                    metadata.parameters.get("sim_time_ns")
                        == Some(&Parameter::Integer(if next { 124 } else { 123 })),
                    "lost observation stamp"
                );
                if replace && id.as_str() == "status" {
                    ensure!(
                        reply["cancelled"] == true && reply["reason"] == "superseded",
                        "missing supersession result"
                    );
                    superseded_results += 1;
                    continue;
                } else if cancel {
                    ensure!(
                        id.as_str() == "status" && reply["cancelled"] == true,
                        "cancelled turn emitted a proposal"
                    );
                } else {
                    ensure!(
                        id.as_str() == "proposal" && reply["ok"] == true,
                        "proposal failed: {reply}"
                    );
                }
                if replace {
                    ensure!(
                        superseded_results == 1,
                        "replacement did not terminate the previous goal"
                    );
                }
                let report = json!({"ok":true,"cancelled":cancel,"replaced":replace,"superseded_results":superseded_results,"timer_ticks_during_inference":ticks,"reply":reply});
                std::fs::write(output, serde_json::to_vec_pretty(&report)?)
                    .context("write fixture result")?;
                return Ok(report);
            }
        }
    }
    anyhow::bail!("fixture ended without a reply")
}
fn main() {
    match run() {
        Ok(v) => println!("{v}"),
        Err(e) => {
            println!("{}", json!({"ok":false,"error":e.to_string()}));
            std::process::exit(1);
        }
    }
}
