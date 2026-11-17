//! Acceptance over an actual native Metal/Dora episode, never synthetic success.
use aisle_nano::{
    judge::{Judge, Sample},
    runtime::Config,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn sha(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}
fn same_verdict(mut actual: Value, mut expected: Value) {
    // JSON decimal parsing can differ by one f64 ULP. This tolerance is far
    // below a sensor/physics unit; booleans and counts must match exactly.
    let a = actual
        .as_object_mut()
        .unwrap()
        .remove("max_lift_m")
        .unwrap()
        .as_f64()
        .unwrap();
    let b = expected
        .as_object_mut()
        .unwrap()
        .remove("max_lift_m")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((a - b).abs() < 1e-12);
    assert_eq!(actual, expected);
}

/// CON-4, CON-5, TC-2, TC-3, TC-8, TC-9, BG-1, BG-2: retained physical evidence must support success.
#[test]
#[ignore = "Set AISLE_NANO_EVIDENCE to a completed native rollout directory"]
fn physical_episode_has_complete_guarded_evidence() {
    let root =
        std::path::PathBuf::from(std::env::var_os("AISLE_NANO_EVIDENCE").expect("episode path"));
    let root = if root.is_absolute() {
        root
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(root)
    };
    let config: Config =
        serde_json::from_slice(&fs::read(root.join("policy.json")).unwrap()).unwrap();
    let result = json(&root.join("rollout.json"));
    assert_eq!(result["ok"], true);
    assert_eq!(result["success"], true, "Physical task did not pass");
    let capture = json(&root.join("capture-complete.json"));
    assert_eq!(capture["terminal_acknowledged"], true);
    assert_eq!(
        capture["cameras"],
        serde_json::json!(["head", "left_wrist", "right_wrist"])
    );
    let samples: Vec<Sample> = fs::read_to_string(root.join("judge-trace.jsonl"))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(samples.len() as u64, capture["frames"].as_u64().unwrap());
    let mut judge = Judge::new(config.goal, samples[0].cube[2], config.drop);
    for (i, sample) in samples.iter().enumerate() {
        assert_eq!(sample.stamp, i as u64 * 200_000_000);
        judge.observe(*sample).unwrap();
    }
    let recomputed = serde_json::to_value(judge.result()).unwrap();
    same_verdict(recomputed.clone(), result["verdict"].clone());
    assert_eq!(recomputed["furniture_contacts"], 0);
    let guard: Vec<Value> = fs::read_to_string(root.join("guard.jsonl"))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(guard.len() + 1, samples.len());
    for (i, command) in guard.iter().enumerate() {
        assert_eq!(command["seq"], i);
        assert_eq!(command["stamp"], i as u64 * 200_000_000);
        let applied: Vec<f64> = serde_json::from_value(command["applied"].clone()).unwrap();
        assert_eq!(applied.len(), 17);
        assert!(
            applied
                .iter()
                .enumerate()
                .all(|(j, v)| v.is_finite() && v.abs() <= if j < 3 { 5. } else { 0.5 })
        );
        assert!(
            applied[..3].iter().all(|v| v.abs() < 1e-5)
                || applied[5..].iter().all(|v| v.abs() < 1e-5)
        );
    }
    let provenance = json(&root.join("provenance.json"));
    for (key, path) in [
        ("policy_sha256", "bin/aisle-nano"),
        ("simulator_sha256", "bin/dorobot-nano-runtime"),
        ("scene_sha256", "scene.json"),
        ("graph_sha256", "dataflow.yaml"),
        ("assets_sha256", "asset-manifest.json"),
    ] {
        assert_eq!(provenance[key], sha(&root.join(path)));
    }
    // Runs recorded since the model snapshot keep the exact URDF and meshes.
    if root.join("model").is_dir() {
        let manifest = json(&root.join("asset-manifest.json"));
        assert_eq!(provenance["model_sha256"], manifest["model.urdf"]);
        for (name, hash) in manifest.as_object().unwrap() {
            assert_eq!(hash, &sha(&root.join("model").join(name)), "{name}");
        }
    }
    let graph: Value =
        serde_yaml::from_slice(&fs::read(root.join("dataflow.yaml")).unwrap()).unwrap();
    let nodes = graph["nodes"].as_array().unwrap();
    let sim = nodes.iter().find(|n| n["id"] == "sim").unwrap();
    assert_eq!(sim["inputs"]["command"]["source"], "guard/joint_cmd_safe");
    let policy = nodes.iter().find(|n| n["id"] == "policy").unwrap();
    for edge in policy["inputs"].as_object().unwrap().values() {
        assert!(matches!(
            edge["source"].as_str().unwrap(),
            "sim/state" | "sim/rgb_head" | "sim/depth_head" | "sim/pose_head" | "sim/finished"
        ));
    }
    let info = json(&root.join("dataset/meta/info.json"));
    assert_eq!(info["total_frames"], samples.len());
    assert_eq!(info["total_videos"], 3);
    same_verdict(
        json(&root.join("dataset/meta/aisle-result.json"))["verdict"].clone(),
        recomputed,
    );
}
