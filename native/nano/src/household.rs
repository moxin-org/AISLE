//! Four independent grasp and placement stations within one continuous Nexus world.
use crate::{
    perception::Color,
    runtime::{self, Config},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Station {
    pub station: String,
    pub object: String,
    pub destination: String,
    pub kind: String,
    pub goal: [f64; 2],
    pub color: Color,
    pub size: [f64; 3],
    pub drop: [f64; 3],
    pub surface_z: f64,
    pub region: [f64; 4],
}

pub fn stations() -> Result<Vec<Station>> {
    Ok(serde_json::from_str(include_str!(
        "../../../examples/nano-household/stations.json"
    ))?)
}

pub fn prepare(
    out: &Path,
    model: &Path,
    sim: &Path,
    helper: Option<&Path>,
    launcher: &Path,
    start_at: Option<&str>,
) -> Result<Value> {
    let model = model.canonicalize()?;
    let assets = crate::assets::manifest(&model)?;
    let sim = sim.canonicalize()?;
    let helper = helper.map(Path::canonicalize).transpose()?;
    let launcher = launcher.canonicalize()?;
    let mut entries = stations()?;
    let mut objects = runtime::room(0, entries[0].goal, "right");
    objects.retain(|o| o["fixed"] == true && o["name"] != "pickup_table");
    // The installed Nexus bridge requires one legacy diagnostic body, excluded from the mission.
    objects.push(json!({"name":"cube_6","size":[0.04,0.04,0.04],"position":[-4.,-3.,0.0],"color":[0.38,0.38,0.38,1.0],"fixed":false}));
    let base_count = objects.len();
    objects.extend(serde_json::from_str::<Vec<Value>>(include_str!(
        "../../../examples/nano-household/furnishings.json"
    ))?);
    // A new episode may spawn at another station. Move the whole room into that
    // start frame before spawning physics, preserving every relative transform.
    let mut spawn = [0., 0.];
    if let Some(name) = start_at {
        let s = entries
            .iter()
            .find(|s| s.station == name)
            .context("unknown start station")?;
        spawn = [s.goal[0] - 0.17, s.goal[1]];
        for entry in &mut entries {
            for (i, offset) in spawn.iter().enumerate() {
                entry.goal[i] -= offset;
                entry.region[i] -= offset;
                entry.drop[i] -= offset;
            }
        }
        for object in &mut objects {
            for (i, offset) in spawn.iter().enumerate() {
                object["position"][i] =
                    json!(object["position"][i].as_f64().context("position")? - offset);
            }
        }
    }
    let mut map = runtime::map(&objects[..base_count]);
    for (i, offset) in spawn.iter().enumerate() {
        map.bounds[i * 2] -= offset;
        map.bounds[i * 2 + 1] -= offset;
    }
    let mut fixtures = runtime::map(&objects[base_count..]);
    for b in &mut fixtures.obstacles {
        b.clearance = Some(0.225);
    }
    map.obstacles.extend(fixtures.obstacles);
    let transit = map.transit();
    for a in &entries {
        let pre = [a.goal[0] - 0.7, a.goal[1]];
        for other in entries.iter().filter(|other| other.station != a.station) {
            let exit = [other.goal[0] - 0.8, other.goal[1]];
            transit
                .plan(exit, pre)
                .with_context(|| format!("{} to {}", other.station, a.station))?;
        }
        ensure!(
            map.clear_segment(pre, a.goal),
            "blocked docking lane at {}",
            a.station
        );
    }
    fs::create_dir(out)?;
    let out = out.canonicalize()?;
    let file = |s: &str| out.join(s);
    let config = Config {
        model: model.clone(),
        output: out.clone(),
        map,
        goal: entries[0].goal,
        drop: entries[0].drop,
        task: "household".into(),
        arm: "right".into(),
        seed: 0,
        max_frames: 6000,
    };
    runtime::write(&file("asset-manifest.json"), &assets)?;
    runtime::write(&file("policy.json"), &config)?;
    runtime::write(&file("scene.json"), &objects)?;
    runtime::write(&file("stations.json"), &entries)?;
    let mut graph: Value = serde_yaml::from_str(include_str!(
        "../../../examples/nano-household/dataflow.yaml"
    ))?;
    for n in graph["nodes"].as_array_mut().context("nodes")? {
        let mut config_path = file("policy.json");
        let id = n["id"].as_str().context("node id")?.to_owned();
        if id == "receipt" {
            let mut diagnostics = config.clone();
            diagnostics.output = file("single-body-diagnostics");
            fs::create_dir(&diagnostics.output)?;
            config_path = diagnostics.output.join("policy.json");
            runtime::write(&config_path, &diagnostics)?;
        }
        if id == "sim" {
            n["path"] = json!(sim);
        } else if matches!(id.as_str(), "guard" | "control" | "receipt") && helper.is_some() {
            n["path"] = json!(&helper);
        } else {
            n["path"] = json!(launcher);
            if matches!(id.as_str(), "guard" | "control" | "receipt") {
                n["args"] = json!(format!(
                    "nano dora-run {}",
                    n["args"].as_str().context("helper args")?
                ));
            }
        }
        n["env"] = json!({"AISLE_NANO_CONFIG":config_path,"AISLE_NANO_OUTPUT":out,"AISLE_NANO_SCENE":file("scene.json"),"AISLE_NANO_MODEL":model,"AISLE_NANO_ARM":"right"});
        for src in n["inputs"].as_object_mut().context("inputs")?.values_mut() {
            *src = json!({"source":src.clone(),"queue_size":8});
        }
    }
    fs::write(file("dataflow.yaml"), serde_yaml::to_string(&graph)?)?;
    crate::workflow::check(&file("dataflow.yaml"))?;
    Ok(json!({"graph":file("dataflow.yaml"),"stream":file("preview.sock"),"objects":entries.len()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// TC-7, MOB-2/3: the stand-alone example prepares all four stations and
    /// connects the planner, current guard and measured receipt without any helper.
    #[test]
    fn prepares_guarded_four_station_graph() -> Result<()> {
        let temp = std::env::temp_dir().join(format!("aisle-example-{}", std::process::id()));
        fs::create_dir(&temp)?;
        // The graph is generated offline. Use a small mesh-free model and the
        // test executable for path resolution; no runtime or network is started.
        let model = temp.join("test.urdf");
        fs::write(
            &model,
            r#"<robot name="fixture"><link name="base"/></robot>"#,
        )?;
        let executable = std::env::current_exe()?.canonicalize()?;
        let launcher = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/aisle-jit")
            .canonicalize()?;
        for (name, helper, start) in [
            ("default", None, None),
            ("helper", Some(executable.as_path()), Some("bedroom bench")),
        ] {
            let out = temp.join(name);
            let plan = prepare(&out, &model, &executable, helper, &launcher, start)?;
            assert_eq!(plan["objects"], 4);
            let graph: Value = serde_yaml::from_slice(&fs::read(out.join("dataflow.yaml"))?)?;
            let node = |id| {
                graph["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|n| n["id"] == id)
                    .unwrap()
            };
            for id in ["guard", "control", "receipt"] {
                assert_eq!(node(id)["path"], json!(helper.unwrap_or(&launcher)));
                assert_eq!(
                    node(id)["args"]
                        .as_str()
                        .unwrap()
                        .starts_with("nano dora-run node "),
                    helper.is_none()
                );
            }
            assert_eq!(
                node("sim")["inputs"]["command"]["source"],
                "guard/joint_cmd_safe"
            );
            assert_eq!(node("receipt")["inputs"]["truth"]["source"], "sim/truth");
            assert_eq!(node("receipt")["inputs"]["state"]["source"], "sim/state");
            assert_eq!(
                node("policy")["inputs"]["subgoal_msg"]["source"],
                "planner/subgoal_msg"
            );
            for view in ["head", "left_wrist", "right_wrist"] {
                let rgb = format!("rgb_{view}");
                assert_eq!(
                    node("policy")["inputs"][&rgb]["source"],
                    format!("sim/{rgb}")
                );
            }
            assert!(prepare(&out, &model, &executable, helper, &launcher, start).is_err());
        }
        fs::remove_dir_all(&temp)?;
        Ok(())
    }
}
