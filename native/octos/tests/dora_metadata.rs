use aisle_octos::{dora::check_metadata, input::CompletionRequest};
use dora_node_api::{MetadataParameters, Parameter};
use serde_json::json;

/// TC-2/7: delayed epochs and retired goals cannot regain action authority.
#[test]
fn admission_rejects_epoch_rollback_and_retired_goal_replay() {
    use aisle_octos::{dora::ObservationGate, input::Observation};
    let mut gate = ObservationGate::default();
    let mut obs = Observation {
        goal_id: "a".into(),
        observation_ns: 1,
        env_id: 0,
        captured_at_ms: 1000,
    };
    gate.admit(&obs, 7, false).unwrap();
    obs.observation_ns = 2;
    assert!(gate.admit(&obs, 7, true).is_err());
    gate.admit(&obs, 7, false).unwrap();
    gate.cancel();
    obs.observation_ns = 3;
    assert!(gate.admit(&obs, 7, false).is_err());
    obs.goal_id = "b".into();
    gate.admit(&obs, 7, false).unwrap();
    obs.goal_id = "a".into();
    obs.observation_ns = 4;
    assert!(gate.admit(&obs, 7, false).is_err());
    gate.admit(&obs, 8, false).unwrap();
    obs.goal_id = "c".into();
    obs.observation_ns = 5;
    assert!(gate.admit(&obs, 7, false).is_err());
    assert!(gate.admit(&obs, 0, false).is_err());
    gate.admit(&obs, 8, false).unwrap();
    obs.goal_id = "a".into();
    assert!(gate.admit(&obs, 8, false).is_err());
    // An invalid identity must not advance epoch state or occupy a ledger slot.
    obs.goal_id = "x".repeat(129);
    assert!(gate.admit(&obs, 99, false).is_err());
    obs.goal_id.clear();
    assert!(gate.admit(&obs, 99, false).is_err());
    obs.goal_id = "valid-next".into();
    gate.admit(&obs, 8, false).unwrap();
}

/// TC-2/7: only same-observation, same-environment, same-goal metadata is accepted.
#[test]
fn metadata_cannot_relabel_a_different_observation() {
    let request: CompletionRequest =
        serde_json::from_value(json!({"model":"deepseek-flash","messages":[],
        "observation":{"goal_id":"g","observation_ns":123,"env_id":0,"captured_at_ms":1000}}))
        .unwrap();
    let meta = MetadataParameters::from([
        ("sim_time_ns".into(), Parameter::Integer(123)),
        ("env_id".into(), Parameter::Integer(0)),
        ("seq".into(), Parameter::Integer(0)),
        ("goal_id".into(), Parameter::String("g".into())),
    ]);
    assert!(check_metadata(&request, &meta).is_ok());
    for (key, bad) in [
        ("sim_time_ns", Parameter::Integer(124)),
        ("env_id", Parameter::Integer(1)),
        ("seq", Parameter::Integer(-1)),
        ("goal_id", Parameter::String("other".into())),
        ("turn_id", Parameter::Integer(1)),
    ] {
        let mut changed = meta.clone();
        changed.insert(key.into(), bad);
        assert!(check_metadata(&request, &changed).is_err(), "{key}");
    }
}

/// CON-8: CLI output is JSON, including usage errors and help.
#[test]
fn cli_help_and_errors_are_structured() {
    for (arg, success) in [("--help", true), ("invalid-command", false)] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_aisle-octos"))
            .arg(arg)
            .output()
            .unwrap();
        assert_eq!(out.status.success(), success);
        let body: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(body["ok"], success);
    }
}

/// TC-7: a scoped late cancellation cannot affect another environment or epoch.
#[test]
fn cancellation_identity_includes_environment_and_epoch() {
    use aisle_octos::{dora::ObservationGate, input::Observation};
    let mut g = ObservationGate::default();
    let obs = Observation {
        goal_id: "same".into(),
        observation_ns: 1,
        env_id: 0,
        captured_at_ms: 1,
    };
    g.admit(&obs, 1, false).unwrap();
    assert!(g.matches_active(0, 1, "same"));
    g.admit(&obs, 2, false).unwrap();
    assert!(!g.matches_active(0, 1, "same"));
    assert!(g.matches_active(0, 2, "same"));
    assert!(!g.matches_active(1, 2, "same"));
}
