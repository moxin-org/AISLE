use aisle_octos::contract::{Mode, validate_answer};
use aisle_octos::input::{CompletionRequest, prepare};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

fn action_value(x: f64, y: f64) -> Value {
    json!({
        "left_arm":{"x":x,"y":y,"z":0,"pitch":0,"roll":0},
        "right_arm":{"x":0,"y":0,"z":0,"pitch":0,"roll":0},
        "left_gripper":{"angle":0},"right_gripper":{"angle":0}})
}
fn action(x: f64, y: f64) -> String {
    format!("```json\n{}\n```", action_value(x, y))
}

pub fn request() -> Value {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[255, 0, 0, 0, 0, 255])
            .unwrap();
    }
    let url = format!("data:image/png;base64,{}", STANDARD.encode(bytes));
    let mut content =
        vec![json!({"type":"text","text":"Measured joint state: [0,0,0]. Inspect the red cube."})];
    for camera in ["head", "left_wrist", "right_wrist"] {
        content.push(json!({"type":"text","text":format!("Camera: {camera}")}));
        content.push(json!({"type":"image_url","image_url":{"url":url}}));
    }
    json!({"model":"deepseek-flash","messages":[
        {"role":"system","content":"Control Nano using measured state and RGB only."},
        {"role":"user","content":content}],"max_tokens":1024,"stream":false,
        "observation":{"goal_id":"test-goal","observation_ns":123,"env_id":0,"captured_at_ms":1000}})
}

/// CON-3/4: action proposals are bounded before reaching any Dora actuator.
#[test]
fn vector_norm_is_not_a_per_axis_limit() {
    assert!(validate_answer(&action(0.3, 0.), Mode::Action).is_ok());
    assert!(validate_answer(&action(0.3, 0.2), Mode::Action).is_err());
}

/// CON-3: one action authority, held second arm and grippers in probe mode.
#[test]
fn rejects_dual_arm_gripper_and_unknown_fields() {
    // JSON key order changes with Cargo feature unification (preserve_order).
    // Mutate values, so every variant still exercises its intended violation.
    let valid = action_value(0.2, 0.);
    let mut dual = valid.clone();
    dual["right_arm"]["x"] = json!(0.1);
    let mut gripper = valid.clone();
    gripper["left_gripper"]["angle"] = json!(0.1);
    let mut unknown = valid.clone();
    unknown["unexpected"] = json!(1);
    for value in [dual, gripper, unknown] {
        assert_ne!(value, valid);
        let bad = format!("```json\n{value}\n```");
        assert!(validate_answer(&bad, Mode::Action).is_err(), "{bad}");
    }
}

/// CON-3: ambiguous, truncated and duplicate-key actions never execute.
#[test]
fn rejects_ambiguous_or_incomplete_answers() {
    for bad in [
        format!("{} trailing", action(0., 0.)),
        format!("{}\n{}", action(0., 0.), action(0., 0.)),
        action(0., 0.).replace("\"x\":0.0", "\"x\":0.0,\"x\":1.0"),
        action(0., 0.).replace("\"x\":0.0", "\"x\":true"),
        "```json\n{}".into(),
    ] {
        assert!(validate_answer(&bad, Mode::Action).is_err(), "{bad}");
    }
}

/// TC-7/8: subgoals delegate bounded intent; they cannot declare task success.
#[test]
fn subgoal_contract_is_separate_from_motor_actions() {
    for goal in [
        json!({"kind":"navigate","target":"coffee table"}),
        json!({"kind":"pick","object":"red cube","arm":"right"}),
        json!({"kind":"place","target":"tray","arm":"right"}),
        json!({"kind":"hold","reason":"Need a fresh observation"}),
    ] {
        assert!(validate_answer(&goal.to_string(), Mode::Subgoal).is_ok());
        assert!(validate_answer(&goal.to_string(), Mode::Action).is_err());
    }
    for goal in [
        json!({"kind":"success"}),
        json!({"kind":"navigate","target":"table","x":99}),
        json!({"kind":"pick","object":"cube","arm":"both"}),
        json!({"kind":"hold","reason":""}),
    ] {
        assert!(validate_answer(&goal.to_string(), Mode::Subgoal).is_err());
    }
}

/// TC-2/3/9: preserve camera role/order and decoded bytes; never silently drop RGB.
#[test]
fn images_survive_octos_conversion_and_are_request_scoped() {
    let req: CompletionRequest = serde_json::from_value(request()).unwrap();
    let prepared = prepare(&req, Mode::Action, 1000, 30000).unwrap();
    assert_eq!(prepared.image_count, 3);
    let msg = prepared.messages.last().unwrap();
    assert_eq!(msg.media.len(), 3);
    assert!(msg.content.contains("head"));
    assert!(msg.content.contains("left_wrist"));
    let path = std::path::PathBuf::from(&msg.media[0]);
    assert!(
        std::fs::read(&path)
            .unwrap()
            .starts_with(b"\x89PNG\r\n\x1a\n")
    );
    drop(prepared);
    assert!(!path.exists());
}

/// TC-2/3: missing, duplicate, remote or corrupt cameras fail closed.
#[test]
fn refuses_invalid_camera_envelopes() {
    let original = request();
    let mut variants = vec![];
    let mut v = original.clone();
    v["messages"][1]["content"].as_array_mut().unwrap().pop();
    variants.push(v);
    let mut v = original.clone();
    v["messages"][1]["content"][3]["text"] = json!("Camera: head");
    variants.push(v);
    let mut v = original.clone();
    v["messages"][1]["content"][2]["image_url"]["url"] = json!("https://example.com/a.png");
    variants.push(v);
    let mut v = original.clone();
    v["messages"][1]["content"][2]["image_url"]["url"] = json!("data:image/png;base64,aGVsbG8=");
    variants.push(v);
    for v in variants {
        let req: CompletionRequest = serde_json::from_value(v).unwrap();
        assert!(prepare(&req, Mode::Action, 1000, 30000).is_err());
    }
}

/// TC-2/CON-5: age checks use injected time, both before and after inference.
#[test]
fn rejects_stale_or_future_observations() {
    let req: CompletionRequest = serde_json::from_value(request()).unwrap();
    assert!(prepare(&req, Mode::Action, 31001, 30000).is_err());
    assert!(prepare(&req, Mode::Action, 0, 30000).is_err());
    assert!(prepare(&req, Mode::Action, 31000, 30000).is_ok());
}
