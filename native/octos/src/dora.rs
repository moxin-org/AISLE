//! Low-rate dialogue node. Camera/motor hot paths remain independent Arrow topics.
use crate::{
    input::{CompletionRequest, MAX_WIRE, Observation},
    service::{Failure, Service},
};
use anyhow::{Context, Result, ensure};
use axum::http::StatusCode;
use dora_node_api::{DoraNode, Event, MetadataParameters, Parameter, futures::StreamExt};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};
use tokio::task::JoinHandle;

fn integer(meta: &MetadataParameters, name: &str) -> Result<i64> {
    match meta.get(name) {
        Some(Parameter::Integer(v)) => Ok(*v),
        _ => anyhow::bail!("missing integer metadata {name}"),
    }
}
pub fn check_metadata(req: &CompletionRequest, meta: &MetadataParameters) -> Result<()> {
    let obs = req
        .observation
        .as_ref()
        .context("Dora observations require a timestamped context")?;
    ensure!(
        u64::try_from(integer(meta, "sim_time_ns")?)? == obs.observation_ns,
        "observation timestamp mismatch"
    );
    ensure!(
        u32::try_from(integer(meta, "env_id")?)? == obs.env_id,
        "environment mismatch"
    );
    ensure!(integer(meta, "seq")? >= 0, "negative sequence");
    match meta.get("goal_id") {
        Some(Parameter::String(id)) => ensure!(*id == obs.goal_id, "goal mismatch"),
        _ => anyhow::bail!("missing goal_id"),
    }
    ensure!(
        meta.contains_key("turn_id") == meta.contains_key("turn_epoch"),
        "incomplete lockstep turn metadata"
    );
    if meta.contains_key("turn_epoch") {
        ensure!(integer(meta, "turn_epoch")? >= 0, "negative turn epoch");
        ensure!(integer(meta, "turn_id")? >= 0, "negative turn id");
    }
    Ok(())
}
fn emit(node: &mut DoraNode, topic: &str, meta: MetadataParameters, value: Value) -> Result<()> {
    let text = value.to_string();
    node.send_output(topic.to_owned().into(), meta, text)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
type Pending = (JoinHandle<Result<Value, Failure>>, MetadataParameters);

type Identity = (u32, i64, String);

/// Bounded run-local ledger. Never evict a tombstone within an active epoch.
#[derive(Default)]
pub struct ObservationGate {
    epochs: BTreeMap<u32, i64>,
    goals: BTreeMap<Identity, (u64, bool)>,
    active: Option<Identity>,
}
impl ObservationGate {
    pub fn admit(&mut self, obs: &Observation, epoch: i64, busy: bool) -> Result<()> {
        ensure!(
            !obs.goal_id.is_empty() && obs.goal_id.len() <= 128,
            "invalid goal_id"
        );
        ensure!(epoch >= 0, "negative epoch");
        ensure!(
            self.epochs
                .get(&obs.env_id)
                .is_none_or(|previous| epoch >= *previous),
            "epoch rollback"
        );
        ensure!(
            self.epochs.contains_key(&obs.env_id) || self.epochs.len() < 32,
            "environment ledger full"
        );
        let key = (obs.env_id, epoch, obs.goal_id.clone());
        if let Some((stamp, closed)) = self.goals.get(&key) {
            ensure!(!closed, "retired or cancelled goal");
            ensure!(obs.observation_ns > *stamp, "replayed observation");
        }
        let same = self.active.as_ref() == Some(&key);
        ensure!(!(same && busy), "inference busy");
        let retained = self
            .goals
            .keys()
            .filter(|(env, e, _)| *env != obs.env_id || *e >= epoch)
            .count();
        ensure!(
            self.goals.contains_key(&key) || retained < 1024,
            "goal ledger full; start a new epoch"
        );
        if !same {
            self.cancel();
        }
        self.goals
            .retain(|(env, e, _), _| *env != obs.env_id || *e >= epoch);
        self.epochs.insert(obs.env_id, epoch);
        self.goals.insert(key.clone(), (obs.observation_ns, false));
        self.active = Some(key);
        Ok(())
    }
    pub fn matches_active(&self, env: u32, epoch: i64, goal: &str) -> bool {
        self.active
            .as_ref()
            .is_some_and(|(e, t, id)| *e == env && *t == epoch && id == goal)
    }
    pub fn cancel(&mut self) {
        if let Some(key) = &self.active
            && let Some((_, closed)) = self.goals.get_mut(key)
        {
            *closed = true;
        }
    }
}

async fn abort_pending(pending: &mut Option<Pending>) -> Option<MetadataParameters> {
    if let Some((task, meta)) = pending.take() {
        task.abort();
        let _ = task.await;
        Some(meta)
    } else {
        None
    }
}

async fn cancel_pending(
    node: &mut DoraNode,
    pending: &mut Option<Pending>,
    seq: &mut i64,
    reason: &str,
) -> Result<()> {
    if let Some(mut meta) = abort_pending(pending).await {
        meta.insert("seq".into(), Parameter::Integer(*seq));
        *seq += 1;
        emit(
            node,
            "status_result",
            meta,
            json!({"ok":false,"cancelled":true,"reason":reason,"execute":false}),
        )?;
    }
    Ok(())
}

async fn finish_run(result: Result<()>, pending: &mut Option<Pending>) -> Result<()> {
    abort_pending(pending).await;
    result
}

pub async fn run(service: Arc<Service>) -> Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut pending: Option<Pending> = None;
    let mut gate = ObservationGate::default();
    let mut output_seq = 0_i64;
    let result=async {
    loop {
        tokio::select! {
            // Stop/cancel/goal replacement always wins over a simultaneous reply.
            biased;
            event=events.next()=> {
                match event {
                    None|Some(Event::Stop{..})=>break,
                    Some(Event::Input{id,..}) if id.as_str()=="finished"=>break,
                    Some(Event::Input{id,metadata,..}) if id.as_str()=="cancel_msg"=> {
                        if let Some(Parameter::String(id))=metadata.parameters.get("goal_id") {
                            let env=integer(&metadata.parameters,"env_id").ok().and_then(|n|u32::try_from(n).ok());
                            let epoch=match metadata.parameters.get("turn_epoch"){Some(Parameter::Integer(n))=>*n,_=>0};
                            if !env.is_some_and(|env|gate.matches_active(env,epoch,id)) {continue;}
                        }
                        gate.cancel();
                        cancel_pending(&mut node,&mut pending,&mut output_seq,"cancelled").await?;
                    }
                    Some(Event::Input{id,metadata,data}) if id.as_str()=="observation_msg"=> {
                        let mut meta=metadata.parameters;
                        let decoded=(||->Result<CompletionRequest>{
                            let text=String::try_from(&data).map_err(|_|anyhow::anyhow!("expected one non-null Arrow UTF8 observation"))?;
                            ensure!(text.len()<=MAX_WIRE,"invalid observation payload");
                            let req:CompletionRequest=serde_json::from_str(&text)?;
                            check_metadata(&req,&meta)?;Ok(req)
                        })();
                        let req=match decoded {Ok(r)=>r,Err(e)=>{
                            meta.insert("seq".into(),Parameter::Integer(output_seq));output_seq+=1;
                            emit(&mut node,"status_result",meta,json!({"ok":false,"error":e.to_string(),"execute":false}))?;continue;
                        }};
                        let obs=req.observation.as_ref().unwrap();
                        let epoch=match meta.get("turn_epoch") {Some(Parameter::Integer(v))=>*v,_=>0};
                        if let Err(e)=gate.admit(obs,epoch,pending.is_some()) {
                            meta.insert("seq".into(),Parameter::Integer(output_seq));output_seq+=1;
                            emit(&mut node,"status_result",meta,json!({"ok":false,"error":e.to_string(),"execute":false}))?;
                            continue;
                        }
                        cancel_pending(&mut node,&mut pending,&mut output_seq,"superseded").await?;
                        let s=service.clone();pending=Some((tokio::spawn(async move{s.infer(req).await}),meta));
                    }
                    Some(Event::Error(error))=>anyhow::bail!("Dora event error: {error}"),
                    _=>{}
                }
            }
            finished=async { (&mut pending.as_mut().unwrap().0).await }, if pending.is_some()=> {
                let (_,mut meta)=pending.take().unwrap();
                meta.insert("seq".into(),Parameter::Integer(output_seq));output_seq+=1;
                match finished {
                    Ok(Ok(reply))=>emit(&mut node,service.mode.topic(),meta,reply)?,
                    Ok(Err(e))=>emit(&mut node,"status_result",meta,e.json())?,
                    Err(_)=>emit(&mut node,"status_result",meta,Failure::new(StatusCode::INTERNAL_SERVER_ERROR,"inference task failed").json())?,
                }
            }
        }
    }
    Ok(())
    }.await;
    finish_run(result, &mut pending).await
}

#[cfg(test)]
mod tests {
    use super::*;
    /// TC-3: even event/output errors await cancellation and delete camera files.
    #[tokio::test]
    async fn error_exit_waits_for_camera_cleanup() {
        let (ready, started) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let directory = tempfile::tempdir().unwrap();
            let camera = directory.path().join("head.png");
            std::fs::write(&camera, b"camera bytes").unwrap();
            ready.send(camera).unwrap();
            std::future::pending::<()>().await;
            drop(directory);
            Ok(json!({}))
        });
        let camera = started.await.unwrap();
        assert!(camera.exists());
        let mut pending = Some((task, MetadataParameters::default()));
        assert!(
            finish_run(Err(anyhow::anyhow!("event/output failure")), &mut pending)
                .await
                .is_err()
        );
        assert!(pending.is_none());
        assert!(!camera.exists());
    }
}
