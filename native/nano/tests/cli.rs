/// CON-8: invalid CLI requests must fail with a structured JSON error.
#[test]
fn malformed_request_is_json_and_nonzero() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_aisle-nano"))
        .arg("not-a-command")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["ok"], false);
    assert!(response["error"].is_string());
}

/// CON-4, CON-5: serialized measured metrics survive supervisor parse/copy exactly.
#[test]
fn evidence_metric_survives_json_roundtrip() {
    let measured = 0.09749113023281097_f64;
    let bytes = serde_json::to_vec(&serde_json::json!({"max_lift_m":measured})).unwrap();
    let copied: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(copied["max_lift_m"].as_f64().unwrap(), measured);
}

/// CON-8: requesting help is a successful structured CLI operation.
#[test]
fn help_is_successful_json() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_aisle-nano"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["ok"], true);
    assert!(response["help"].as_str().unwrap().contains("rollout"));
}

const URDF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/robots/adora_nano/model.urdf"
);

/// A scratch directory holding a complete model (empty mesh files) and a
/// stand-in for both the Dora CLI and the simulator. `FAKE_DORA` selects what
/// the stand-in rollout leaves behind.
fn fixture(name: &str) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("aisle-nano-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("model")).unwrap();
    let urdf = std::fs::read_to_string(URDF).unwrap();
    for mesh in urdf.split("filename=\"").skip(1) {
        let mesh = dir.join("model").join(mesh.split('"').next().unwrap());
        std::fs::create_dir_all(mesh.parent().unwrap()).unwrap();
        std::fs::write(mesh, b"mesh").unwrap();
    }
    std::fs::write(dir.join("model/model.urdf"), urdf).unwrap();
    // In "hang" mode Dora exits on SIGINT while its worker ignores SIGINT.
    let script = format!(
        r#"#!/bin/sh
if [ "$1" = --version ]; then echo "${{FAKE_DORA_VERSION:-dora-cli 1.0.1}}"; exit 0; fi
out=$(dirname "$2")
if [ "$FAKE_DORA" = hang ]; then
  (trap '' INT; sh -c 'echo $$ > {0}/worker.pid; exec sleep 300') &
  trap 'exit 130' INT
  wait
fi
printf '%s' '{{"ok":true,"success":true,"verdict":{{}}}}' > "$out/result.json"
printf '%s' '{{"simulation_settings":{{"physics_hz":100}}}}' > "$out/scene-info.json"
if [ "$FAKE_DORA" != no-capture ]; then
  printf '%s' '{{"ok":true,"terminal_acknowledged":true}}' > "$out/capture-complete.json"
fi
if [ "$FAKE_DORA" = fault ]; then
  printf '%s' '{{"done":true,"fault":true,"reason":"IK unreachable"}}' > "$out/policy-status.json"
fi
if [ "$FAKE_DORA" = exit1 ]; then exit 1; fi
exit 0
"#,
        dir.display()
    );
    for name in ["dora", "simulator"] {
        std::fs::write(dir.join(name), &script).unwrap();
        std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    dir
}
/// A navigation rollout of the fixture; `args` override the defaults.
fn rollout(dir: &std::path::Path, mode: &str, args: &[&str]) -> std::process::Command {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_aisle-nano"));
    command
        .arg("rollout")
        .arg("--simulator")
        .arg(dir.join("simulator"))
        .arg("--dora")
        .arg(dir.join("dora"))
        .args(["--grace-seconds", "1"])
        .args(args)
        .env("FAKE_DORA", mode)
        .env_remove("FAKE_DORA_VERSION");
    let defaults = [
        ("--model", dir.join("model/model.urdf")),
        ("--output", dir.join("run")),
        ("--task", "navigate".into()),
    ];
    for (flag, value) in defaults {
        if !args.contains(&flag) {
            command.arg(flag).arg(value);
        }
    }
    command
}
fn report(output: &std::process::Output) -> serde_json::Value {
    serde_json::from_slice(&output.stdout).unwrap()
}
fn sha(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

/// CON-8: an interrupted rollout stops every process it started and still reports JSON.
#[test]
fn interrupted_rollout_reaps_its_process_group() {
    let dir = fixture("interrupt");
    let child = rollout(&dir, "hang", &[])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let pid_file = dir.join("worker.pid");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let worker: i32 = loop {
        if let Some(pid) = std::fs::read_to_string(&pid_file)
            .ok()
            .and_then(|s| s.trim().parse().ok())
        {
            break pid;
        }
        assert!(std::time::Instant::now() < deadline, "worker never started");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(report(&output)["ok"], false);
    assert_eq!(unsafe { libc::kill(worker, 0) }, -1, "worker survived");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// CON-4, CON-5, CON-8: a rollout runs from, and hashes, its own frozen copy of the model.
#[test]
fn rollout_executes_a_snapshot_of_the_model() {
    let dir = fixture("snapshot");
    let output = rollout(&dir, "ok", &[]).output().unwrap();
    let result = report(&output);
    assert_eq!(output.status.code(), Some(0), "{result}");
    assert_eq!(
        (&result["ok"], &result["success"]),
        (&true.into(), &true.into())
    );
    assert!(result["policy_fault"].is_null());
    let run = dir.join("run").canonicalize().unwrap();
    let frozen = run.join("model/model.urdf");
    assert_eq!(sha(&frozen), sha(&dir.join("model/model.urdf")));
    let json = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(run.join(name)).unwrap()).unwrap()
    };
    // Policy, guard, judge and simulator are all pointed at the frozen copy.
    assert_eq!(json("policy.json")["model"], frozen.to_str().unwrap());
    let graph: serde_json::Value =
        serde_yaml::from_slice(&std::fs::read(run.join("dataflow.yaml")).unwrap()).unwrap();
    let nodes = graph["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 4);
    for node in nodes {
        assert_eq!(node["env"]["AISLE_NANO_MODEL"], frozen.to_str().unwrap());
    }
    // The independent judge reads simulator outputs only, including base heading.
    let judge = nodes.iter().find(|n| n["id"] == "judge").unwrap();
    assert_eq!(judge["inputs"]["state"]["source"], "sim/state");
    assert_eq!(judge["inputs"]["truth"]["source"], "sim/truth");
    // The manifest lists the URDF and every mesh, each matching the frozen bytes.
    let manifest = json("asset-manifest.json");
    let manifest = manifest.as_object().unwrap();
    assert_eq!(manifest.len(), 138);
    for (name, hash) in manifest {
        assert_eq!(hash, &sha(&run.join("model").join(name)), "{name}");
    }
    let provenance = json("provenance.json");
    assert_eq!(provenance["model_sha256"], manifest["model.urdf"]);
    assert_eq!(
        provenance["assets_sha256"],
        sha(&run.join("asset-manifest.json"))
    );
    assert_eq!(provenance["simulator_settings"]["physics_hz"], 100);
    // Editing the source model afterwards cannot change what the run recorded.
    std::fs::write(dir.join("model/model.urdf"), "<robot name=\"edited\"/>").unwrap();
    assert_eq!(sha(&frozen), manifest["model.urdf"].as_str().unwrap());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// CON-8: `ok` needs complete capture, a clean Dora exit and no policy fault; otherwise exit 1.
#[test]
fn rollout_is_not_ok_without_capture_clean_exit_and_a_healthy_policy() {
    for (mode, fault) in [
        ("no-capture", None),
        ("exit1", None),
        ("fault", Some("IK unreachable")),
    ] {
        let dir = fixture(mode);
        let output = rollout(&dir, mode, &[]).output().unwrap();
        let result = report(&output);
        assert_eq!(output.status.code(), Some(1), "{mode}: {result}");
        assert_eq!(result["ok"], false, "{mode}");
        assert_eq!(result["success"], false, "{mode}");
        assert_eq!(result["capture_complete"], mode != "no-capture", "{mode}");
        assert_eq!(
            result["dora_exited_successfully"],
            mode != "exit1",
            "{mode}"
        );
        assert_eq!(result["policy_fault"].as_str(), fault, "{mode}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// CON-8: a refused request reports why and leaves the output directory unused.
#[test]
fn refused_rollout_does_not_consume_the_output_directory() {
    let dir = fixture("refused");
    let refused = |command: &mut std::process::Command, why: &str| {
        let output = command.output().unwrap();
        let result = report(&output);
        assert_eq!(output.status.code(), Some(1), "{result}");
        assert_eq!(result["ok"], false);
        let error = result["error"].as_str().unwrap();
        assert!(error.contains(why), "{error}");
        assert!(
            !dir.join("run").exists(),
            "{why}: output directory was created"
        );
    };
    // The committed kinematic-only URDF has no mesh files beside it.
    refused(
        &mut rollout(&dir, "ok", &["--model", URDF]),
        "Model mesh missing",
    );
    refused(
        rollout(&dir, "ok", &[]).env("FAKE_DORA_VERSION", "dora-cli 0.9.0"),
        "Dora CLI 1.0.1 required",
    );
    let blocked = ["--task", "pickup", "--goal-x", "1.9", "--goal-y", "-2.5"];
    refused(&mut rollout(&dir, "ok", &blocked), "Pickup docking point");
    refused(
        &mut rollout(&dir, "ok", &["--frames", "0"]),
        "Invalid rollout budget",
    );
    // An existing output directory is never reused.
    let used = dir.join("used");
    std::fs::create_dir(&used).unwrap();
    let output = rollout(&dir, "ok", &["--output", used.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let error = report(&output)["error"].as_str().unwrap().to_owned();
    assert!(error.contains("new output directory"), "{error}");
    assert_eq!(std::fs::read_dir(&used).unwrap().count(), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// CON-8: inspection is not refused for a goal the base could not drive to.
#[test]
fn inspection_does_not_need_a_drivable_goal() {
    let dir = fixture("inspect");
    let on_the_coffee_table = ["--goal-x", "0", "--goal-y", "1.1"];
    let inspect = [&["--task", "inspect"][..], &on_the_coffee_table].concat();
    let output = rollout(&dir, "ok", &inspect).output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", report(&output));
    let blocked = dir.join("blocked");
    let navigate = [
        &["--output", blocked.to_str().unwrap()][..],
        &on_the_coffee_table,
    ]
    .concat();
    let output = rollout(&dir, "ok", &navigate).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(!blocked.exists());
    std::fs::remove_dir_all(&dir).unwrap();
}
