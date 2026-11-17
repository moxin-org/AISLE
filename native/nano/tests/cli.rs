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
