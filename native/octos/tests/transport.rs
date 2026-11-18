use aisle_octos::{
    contract::Mode,
    input::CompletionRequest,
    service::{Service, now_ms, router},
};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use base64::{Engine, engine::general_purpose::STANDARD};
use octos_llm::openai::{ModelHints, OpenAIProvider, ReasoningStyle};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{Mutex, Notify},
    task::JoinHandle,
};

// HTTP contract tests can cold-compile the client and server under Relax.
const HTTP_DEADLINE: Duration = Duration::from_secs(25);

struct Upstream {
    requests: Mutex<Vec<Value>>,
    status: StatusCode,
    reply: Value,
    delay: Duration,
    hold_next: AtomicBool,
    entered: Notify,
    release: Notify,
}
async fn respond(
    State(s): State<Arc<Upstream>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    s.requests.lock().await.push(body);
    if s.hold_next.swap(false, Ordering::SeqCst) {
        s.entered.notify_one();
        s.release.notified().await;
    }
    tokio::time::sleep(s.delay).await;
    (s.status, Json(s.reply.clone()))
}
struct Fixture {
    service: Arc<Service>,
    upstream: Arc<Upstream>,
    server: JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}
async fn fixture(
    mode: Mode,
    status: StatusCode,
    reply: Value,
    delay: Duration,
    timeout: Duration,
    max_age: u64,
) -> Fixture {
    let upstream = Arc::new(Upstream {
        requests: Mutex::new(vec![]),
        hold_next: AtomicBool::new(false),
        entered: Notify::new(),
        release: Notify::new(),
        status,
        reply,
        delay,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/v1/chat/completions", post(respond))
        .with_state(upstream.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let provider = OpenAIProvider::new("fixture-key", "deepseek-flash")
        .with_base_url(url)
        .with_hints(ModelHints {
            reasoning_style: ReasoningStyle::EffortAndThinkingToggle,
            ..Default::default()
        });
    let service = Arc::new(Service::new(Arc::new(provider), mode, timeout, max_age).unwrap());
    Fixture {
        service,
        upstream,
        server,
    }
}
fn reply(content: &str) -> Value {
    json!({"model":"deepseek-flash","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":content}}],"usage":{"prompt_tokens":20,"completion_tokens":10}})
}
fn held_action() -> Value {
    json!({"left_arm":{"x":0,"y":0,"z":0,"pitch":0,"roll":0},"right_arm":{"x":0,"y":0,"z":0,"pitch":0,"roll":0},"left_gripper":{"angle":0},"right_gripper":{"angle":0}})
}
fn hold() -> String {
    format!("```json\n{}\n```", held_action())
}
fn request() -> CompletionRequest {
    let mut png = vec![];
    {
        let mut encoder = png::Encoder::new(&mut png, 1, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[255, 0, 0])
            .unwrap();
    }
    let url = format!("data:image/png;base64,{}", STANDARD.encode(png));
    let mut parts =
        vec![json!({"type":"text","text":"Measured robot state. Inspect without contact."})];
    for role in ["head", "left_wrist", "right_wrist"] {
        parts.push(json!({"type":"text","text":format!("Camera: {role}")}));
        parts.push(json!({"type":"image_url","image_url":{"url":url}}));
    }
    serde_json::from_value(json!({"model":"deepseek-flash","messages":[{"role":"system","content":"Robot policy"},{"role":"user","content":parts}],
        "observation":{"goal_id":"test","observation_ns":0,"env_id":0,"captured_at_ms":now_ms()}})).unwrap()
}

/// TC-3/9, CON-1: real Octos provider sends three inline images and no tools.
#[tokio::test]
async fn real_octos_http_path_preserves_rgb_and_disables_thinking_and_tools() {
    let f = fixture(
        Mode::Action,
        StatusCode::OK,
        reply(&hold()),
        Duration::ZERO,
        HTTP_DEADLINE,
        30000,
    )
    .await;
    let result = f.service.infer(request()).await.unwrap();
    assert_eq!(result["aisle"]["image_count"], 3);
    assert_eq!(result["aisle"]["execute"], false);
    let bodies = f.upstream.requests.lock().await;
    assert_eq!(bodies.len(), 1);
    let body = &bodies[0];
    assert!(body.get("tools").is_none());
    assert_eq!(body["thinking"]["type"], "disabled");
    let parts = body["messages"][1]["content"].as_array().unwrap();
    assert_eq!(parts.iter().filter(|p| p["type"] == "image_url").count(), 3);
    assert!(
        body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("EUCLIDEAN")
    );
}

/// TC-3/9: Octos FailFast must never silently retry a vision error as text-only.
#[tokio::test]
async fn vision_refusal_has_exactly_one_upstream_request() {
    let f = fixture(
        Mode::Action,
        StatusCode::BAD_REQUEST,
        json!({"error":{"message":"image_url is not supported"}}),
        Duration::ZERO,
        HTTP_DEADLINE,
        30000,
    )
    .await;
    assert_eq!(
        f.service.infer(request()).await.unwrap_err().status,
        StatusCode::BAD_GATEWAY
    );
    assert_eq!(f.upstream.requests.lock().await.len(), 1);
}

/// TC-7/8: subgoal mode returns intent without motor output or success claims.
#[tokio::test]
async fn subgoal_uses_same_multimodal_provider_but_separate_contract() {
    let f = fixture(
        Mode::Subgoal,
        StatusCode::OK,
        reply(r#"{"kind":"pick","object":"red cube","arm":"right"}"#),
        Duration::ZERO,
        HTTP_DEADLINE,
        30000,
    )
    .await;
    let result = f.service.infer(request()).await.unwrap();
    assert_eq!(result["aisle"]["proposal"]["kind"], "pick");
    assert_eq!(result["aisle"]["execute"], false);
}

/// CON-3: incomplete/tool replies and the observed 3.61 mm failure are refused.
#[tokio::test]
async fn refuses_incomplete_tool_and_oversize_actions() {
    let mut truncated = reply(&hold());
    truncated["choices"][0]["finish_reason"] = json!("length");
    let mut tool = reply(&hold());
    tool["choices"][0]["message"]["tool_calls"] =
        json!([{"id":"bad","type":"function","function":{"name":"shell","arguments":"{}"}}]);
    let mut large = held_action();
    large["left_arm"]["x"] = json!(0.3);
    large["left_arm"]["y"] = json!(0.2);
    let oversized = reply(&format!("```json\n{large}\n```"));
    for body in [truncated, tool, oversized] {
        let f = fixture(
            Mode::Action,
            StatusCode::OK,
            body,
            Duration::ZERO,
            HTTP_DEADLINE,
            30000,
        )
        .await;
        assert!(f.service.infer(request()).await.is_err());
    }
}

/// TC-2: a frame valid before inference can expire while inference is running.
#[tokio::test]
async fn stale_after_inference_and_deadline_both_fail_closed() {
    let f = fixture(
        Mode::Action,
        StatusCode::OK,
        reply(&hold()),
        Duration::from_millis(2500),
        HTTP_DEADLINE,
        2000,
    )
    .await;
    let mut req = request();
    req.observation.as_mut().unwrap().captured_at_ms = now_ms();
    assert_eq!(
        f.service.infer(req).await.unwrap_err().status,
        StatusCode::CONFLICT
    );
    let f = fixture(
        Mode::Action,
        StatusCode::OK,
        reply(&hold()),
        Duration::ZERO,
        Duration::from_millis(20),
        30000,
    )
    .await;
    f.upstream.hold_next.store(true, Ordering::SeqCst);
    assert_eq!(
        f.service.infer(request()).await.unwrap_err().status,
        StatusCode::GATEWAY_TIMEOUT
    );
    f.upstream.release.notify_one();
}

/// CON-3/8: no queued commands; aborting an owner releases its inference slot.
#[tokio::test]
async fn concurrent_request_is_rejected_and_cancellation_releases_slot() {
    let f = fixture(
        Mode::Action,
        StatusCode::OK,
        reply(&hold()),
        Duration::ZERO,
        HTTP_DEADLINE,
        30000,
    )
    .await;
    f.upstream.hold_next.store(true, Ordering::SeqCst);
    let s = f.service.clone();
    let task = tokio::spawn(async move { s.infer(request()).await });
    tokio::time::timeout(HTTP_DEADLINE, f.upstream.entered.notified())
        .await
        .unwrap();
    assert_eq!(
        f.service.infer(request()).await.unwrap_err().status,
        StatusCode::CONFLICT
    );
    task.abort();
    let _ = task.await;
    f.upstream.release.notify_one();
    assert!(f.service.infer(request()).await.is_ok());
}

/// CON-8: AISLE's HTTP client receives normal completions and structured errors.
#[tokio::test]
async fn loopback_service_accepts_existing_chat_client_wire_shape() {
    let f = fixture(
        Mode::Action,
        StatusCode::OK,
        reply(&hold()),
        Duration::ZERO,
        HTTP_DEADLINE,
        30000,
    )
    .await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let app = router(f.service.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let mut req = serde_json::to_value(request()).unwrap();
    req.as_object_mut().unwrap().remove("observation");
    req["temperature"] = json!(0);
    req["seed"] = json!(42);
    req["chat_template_kwargs"] = json!({"enable_thinking":true});
    let client = reqwest::Client::new();
    let response = client.post(&url).json(&req).send().await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let result: Value = response.json().await.unwrap();
    assert_eq!(result["choices"][0]["finish_reason"], "stop");
    req["model"] = json!("unapproved-model");
    let response = client.post(&url).json(&req).send().await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.json::<Value>().await.unwrap()["execute"], false);
    task.abort();
}
