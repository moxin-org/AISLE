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

/// CON-4/5: serialized measured metrics survive supervisor parse/copy exactly.
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

/// CON-8: an interrupted rollout stops every process it started and still reports JSON.
#[test]
fn interrupted_rollout_reaps_its_process_group() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("aisle-nano-interrupt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("model")).unwrap();
    let urdf = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/robots/adora_nano/model.urdf"
    ))
    .unwrap();
    for mesh in urdf.split("filename=\"").skip(1) {
        let mesh = dir.join("model").join(mesh.split('"').next().unwrap());
        std::fs::create_dir_all(mesh.parent().unwrap()).unwrap();
        std::fs::write(mesh, b"").unwrap();
    }
    std::fs::write(dir.join("model/model.urdf"), urdf).unwrap();
    // The stand-in Dora exits on SIGINT; its worker ignores SIGINT and must be killed.
    let script = format!(
        "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'dora-cli 1.0.1'; exit 0; fi\n\
         (trap '' INT; sh -c 'echo $$ > {0}/worker.pid; exec sleep 300') &\n\
         trap 'exit 130' INT\nwait\n",
        dir.display()
    );
    for name in ["dora", "simulator"] {
        std::fs::write(dir.join(name), &script).unwrap();
        std::fs::set_permissions(dir.join(name), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_aisle-nano"))
        .arg("rollout")
        .arg("--model")
        .arg(dir.join("model/model.urdf"))
        .arg("--simulator")
        .arg(dir.join("simulator"))
        .arg("--dora")
        .arg(dir.join("dora"))
        .arg("--output")
        .arg(dir.join("run"))
        .args(["--task", "navigate", "--grace-seconds", "1"])
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
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["ok"], false);
    assert_eq!(unsafe { libc::kill(worker, 0) }, -1, "worker survived");
    std::fs::remove_dir_all(&dir).unwrap();
}
