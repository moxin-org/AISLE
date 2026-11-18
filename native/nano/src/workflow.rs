//! Offline Dora topic-reference validation for isolated workflow repair.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};
pub fn check(path: &Path) -> Result<Value> {
    let b = std::fs::read(path)?;
    ensure!(b.len() <= 1024 * 1024, "workflow too large");
    let v: Value = serde_yaml::from_slice(&b)?;
    let nodes = v["nodes"].as_array().context("nodes array required")?;
    ensure!(!nodes.is_empty() && nodes.len() <= 64, "invalid node count");
    let mut outputs = BTreeMap::new();
    for node in nodes {
        let id = node["id"].as_str().context("node ID required")?;
        ensure!(!id.is_empty() && !id.contains('/'), "invalid node ID");
        ensure!(
            outputs
                .insert(id, node["outputs"].as_array().cloned().unwrap_or_default())
                .is_none(),
            "duplicate node ID {id}"
        );
    }
    let mut edges = 0;
    for node in nodes {
        if let Some(inputs) = node["inputs"].as_object() {
            for (name, source) in inputs {
                let source = source
                    .as_str()
                    .or_else(|| source["source"].as_str())
                    .context("input source required")?;
                if let Some(period) = source.strip_prefix("dora/timer/millis/") {
                    ensure!(period.parse::<u64>()? > 0, "invalid timer");
                    continue;
                }
                let (producer, topic) = source
                    .split_once('/')
                    .context("source must be node/output")?;
                let declared = outputs.get(producer).with_context(|| {
                    format!("{}.{name}: unknown producer {producer}", node["id"])
                })?;
                ensure!(
                    declared.iter().any(|x| x.as_str() == Some(topic)),
                    "{}.{name}: {producer} does not declare output {topic}; declared: {declared:?}",
                    node["id"]
                );
                edges += 1;
            }
        }
    }
    Ok(
        json!({"ok":true,"nodes":nodes.len(),"edges":edges,"validation":"static Dora topic references; runtime/physics acceptance remains separate"}),
    )
}
#[cfg(test)]
mod tests {
    /// CON-4: a repaired workflow must reference actual declared output topics.
    #[test]
    fn missing_output_is_not_a_valid_workflow() {
        let p = std::env::temp_dir().join(format!("aisle-flow-{}.json", std::process::id()));
        let graph = serde_json::json!({"nodes":[{"id":"sensor","outputs":["rgb"]},{"id":"policy","inputs":{"rgb":"sensor/missing"}}]});
        std::fs::write(&p, graph.to_string()).unwrap();
        assert!(super::check(&p).is_err());
        let mut graph = graph;
        graph["nodes"][1]["inputs"]["rgb"] = serde_json::json!("sensor/rgb");
        std::fs::write(&p, graph.to_string()).unwrap();
        assert_eq!(super::check(&p).unwrap()["edges"], 1);
        std::fs::remove_file(p).unwrap();
    }
}
