use crate::{
    contract::{Mode, validate_answer},
    input::{CompletionRequest, MAX_WIRE, prepare},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use octos_llm::{
    ChatConfig, LlmCallPolicy, LlmProvider, ReasoningEffort, StopReason, ToolChoice,
    with_llm_call_policy,
};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Semaphore;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[derive(Debug)]
pub struct Failure {
    pub status: StatusCode,
    pub message: String,
}
impl Failure {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
    pub fn json(&self) -> Value {
        json!({"ok":false,"error":{"message":self.message,"status":self.status.as_u16()},"execute":false})
    }
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.status, Json(self.json())).into_response()
    }
}

pub struct Service {
    provider: Arc<dyn LlmProvider>,
    pub mode: Mode,
    pub model: String,
    timeout: Duration,
    max_age_ms: u64,
    slot: Semaphore,
}
impl Service {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        mode: Mode,
        timeout: Duration,
        max_age_ms: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !timeout.is_zero() && timeout <= Duration::from_secs(120),
            "timeout must be 1..120000 ms"
        );
        anyhow::ensure!(
            (1..=120000).contains(&max_age_ms),
            "observation age limit must be 1..120000 ms"
        );
        Ok(Self {
            model: provider.model_id().to_owned(),
            provider,
            mode,
            timeout,
            max_age_ms,
            slot: Semaphore::new(1),
        })
    }
    pub async fn infer(&self, req: CompletionRequest) -> Result<Value, Failure> {
        let _slot = self
            .slot
            .try_acquire()
            .map_err(|_| Failure::new(StatusCode::CONFLICT, "inference busy; no command queued"))?;
        if req.model != self.model {
            return Err(Failure::new(
                StatusCode::BAD_REQUEST,
                "model must match configured provider",
            ));
        }
        let max_tokens = req.max_tokens.unwrap_or(4096);
        if !(1..=4096).contains(&max_tokens) {
            return Err(Failure::new(
                StatusCode::BAD_REQUEST,
                "max_tokens must be 1..4096",
            ));
        }
        let prepared = prepare(&req, self.mode, now_ms(), self.max_age_ms)
            .map_err(|e| Failure::new(StatusCode::BAD_REQUEST, e.to_string()))?;
        let config = ChatConfig {
            max_tokens: Some(max_tokens),
            temperature: Some(0.),
            tool_choice: ToolChoice::None,
            reasoning_effort: Some(ReasoningEffort::Disabled),
            media_scope_root: Some(prepared.directory.path().canonicalize().map_err(|_| {
                Failure::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "temporary camera directory unavailable",
                )
            })?),
            ..Default::default()
        };
        let started = Instant::now();
        let reply = tokio::time::timeout(
            self.timeout,
            with_llm_call_policy(
                LlmCallPolicy::FailFast,
                self.provider.chat(&prepared.messages, &[], &config),
            ),
        )
        .await
        .map_err(|_| {
            Failure::new(
                StatusCode::GATEWAY_TIMEOUT,
                "inference deadline exceeded; no command emitted",
            )
        })?
        // Upstream error bodies may contain credentials or echoed inputs.
        .map_err(|_| {
            Failure::new(
                StatusCode::BAD_GATEWAY,
                "Octos inference failed; no text-only retry or command emitted",
            )
        })?;
        if reply.stop_reason != StopReason::EndTurn || !reply.tool_calls.is_empty() {
            return Err(Failure::new(
                StatusCode::BAD_GATEWAY,
                "incomplete completion or unexpected tool call",
            ));
        }
        if let Some(obs) = &req.observation {
            obs.check_age(now_ms(), self.max_age_ms)
                .map_err(|e| Failure::new(StatusCode::CONFLICT, e.to_string()))?;
        }
        let answer = reply
            .content
            .as_deref()
            .ok_or_else(|| Failure::new(StatusCode::BAD_GATEWAY, "empty final answer"))?;
        let proposal = validate_answer(answer, self.mode)
            .map_err(|e| Failure::new(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
        let answer = if self.mode == Mode::Subgoal {
            format!("```json\n{}\n```", proposal)
        } else {
            answer.to_owned()
        };
        let usage = reply.usage;
        Ok(json!({
            "ok":true,"object":"chat.completion","model":self.model,
            "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":answer}}],
            "usage":{"prompt_tokens":u64::from(usage.input_tokens)+u64::from(usage.cache_read_tokens)+u64::from(usage.cache_write_tokens),
                "completion_tokens":usage.output_tokens,"octos":usage,"adapter_wall_time_s":started.elapsed().as_secs_f64()},
            "aisle":{"mode":self.mode,"proposal":proposal,"observation":req.observation,
                "image_count":prepared.image_count,"image_sha256":prepared.image_sha256,
                "provider":"octos-llm","configured_model":self.model,"upstream_model_attested":false,
                "tool_calls":0,"execute":false}
        }))
    }
}

pub fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(completion))
        .layer(DefaultBodyLimit::max(MAX_WIRE))
        .with_state(service)
}
async fn health(State(s): State<Arc<Service>>) -> Json<Value> {
    Json(json!({"ok":true,"mode":s.mode,"model":s.model,"provider":"octos-llm"}))
}
async fn completion(
    State(s): State<Arc<Service>>,
    request: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<Value>, Failure> {
    let Json(req) =
        request.map_err(|e| Failure::new(e.status(), "invalid or oversized completion request"))?;
    s.infer(req).await.map(Json)
}
