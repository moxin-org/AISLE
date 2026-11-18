use crate::contract::Mode;
use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use octos_core::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, io::Cursor};
use tempfile::TempDir;

pub const MAX_WIRE: usize = 32 * 1024 * 1024;
const CAMERAS: [&str; 3] = ["head", "left_wrist", "right_wrist"];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub goal_id: String,
    pub observation_ns: u64,
    pub env_id: u32,
    pub captured_at_ms: u64,
}
impl Observation {
    pub fn check_age(&self, now_ms: u64, max_age_ms: u64) -> Result<()> {
        ensure!(
            !self.goal_id.is_empty() && self.goal_id.len() <= 128,
            "invalid goal_id"
        );
        let age = now_ms
            .checked_sub(self.captured_at_ms)
            .context("observation is in the future")?;
        ensure!(age <= max_age_ms, "stale observation");
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub observation: Option<Observation>,
    // Accepted for compatibility with the existing AISLE ChatClient. Octos does
    // not promise seeded cloud sampling; server-owned settings take precedence.
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WireMessage {
    pub role: String,
    pub content: Content,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Part {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
}

pub struct Prepared {
    pub messages: Vec<Message>,
    pub image_count: usize,
    pub image_sha256: Vec<String>,
    pub directory: TempDir,
}

pub fn prepare(
    req: &CompletionRequest,
    mode: Mode,
    now_ms: u64,
    max_age_ms: u64,
) -> Result<Prepared> {
    ensure!(
        !req.stream,
        "streaming completions are not accepted for robot actions"
    );
    ensure!(
        serde_json::to_vec(req)?.len() <= MAX_WIRE,
        "observation transcript too large"
    );
    ensure!(
        (2..=10).contains(&req.messages.len()),
        "expected bounded system/history/observation"
    );
    ensure!(
        req.messages[0].role == "system" && req.messages.last().unwrap().role == "user",
        "invalid message roles"
    );
    if let Some(obs) = &req.observation {
        obs.check_age(now_ms, max_age_ms)?;
    }
    let directory = tempfile::Builder::new().prefix("aisle-octos-").tempdir()?;
    let mut prepared = Prepared {
        messages: vec![],
        image_count: 0,
        image_sha256: vec![],
        directory,
    };
    for (index, wire) in req.messages.iter().enumerate() {
        ensure!(
            index == 0 || matches!(wire.role.as_str(), "user" | "assistant"),
            "unexpected history role"
        );
        let mut msg = match wire.role.as_str() {
            "system" => Message::system(""),
            "user" => Message::user(""),
            "assistant" => Message::assistant(""),
            _ => anyhow::bail!("unsupported message role"),
        };
        let parts = match &wire.content {
            Content::Text(text) => vec![Part::Text { text: text.clone() }],
            Content::Parts(p) => p.clone(),
        };
        ensure!(parts.len() <= 32, "too many content parts");
        let mut camera = None;
        let mut seen = BTreeSet::new();
        for part in parts {
            match part {
                Part::Text { text } => {
                    ensure!(text.len() <= 32768, "text part too large");
                    if let Some(name) = text.strip_prefix("Camera: ") {
                        ensure!(
                            camera.is_none() && CAMERAS.contains(&name),
                            "invalid camera role"
                        );
                        camera = Some(name.to_owned());
                    }
                    msg.content.push_str(&text);
                    msg.content.push('\n');
                }
                Part::ImageUrl { image_url } => {
                    ensure!(
                        wire.role == "user",
                        "images are only allowed in observation messages"
                    );
                    let name = camera
                        .take()
                        .context("camera image needs an explicit role label")?;
                    ensure!(seen.insert(name.clone()), "duplicate camera role");
                    let encoded = image_url
                        .url
                        .strip_prefix("data:image/png;base64,")
                        .context("only inline PNG images accepted")?;
                    ensure!(encoded.len() <= 8 * 1024 * 1024, "camera image too large");
                    let bytes = STANDARD.decode(encoded)?;
                    let mut decoder = png::Decoder::new(Cursor::new(&bytes));
                    decoder.set_limits(png::Limits {
                        bytes: 32 * 1024 * 1024,
                    });
                    let mut reader = decoder.read_info().context("invalid PNG")?;
                    ensure!(
                        reader.info().width > 0
                            && reader.info().height > 0
                            && u64::from(reader.info().width) * u64::from(reader.info().height)
                                <= 4_194_304,
                        "PNG dimensions too large"
                    );
                    let mut decoded = vec![0; reader.output_buffer_size()];
                    reader.next_frame(&mut decoded).context("corrupt PNG")?;
                    let path = prepared
                        .directory
                        .path()
                        .join(format!("{index}-{name}.png"));
                    std::fs::write(&path, &bytes)?;
                    msg.content.push_str(&format!(
                        "[Attached image {}: camera {name}]\n",
                        msg.media.len() + 1
                    ));
                    msg.media
                        .push(path.to_str().context("non-UTF8 temporary path")?.to_owned());
                    prepared.image_count += 1;
                    prepared
                        .image_sha256
                        .push(format!("{:x}", Sha256::digest(&bytes)));
                }
            }
        }
        ensure!(camera.is_none(), "camera role has no image");
        if !seen.is_empty() || index == req.messages.len() - 1 {
            ensure!(
                seen.len() == 3 && CAMERAS.iter().all(|name| seen.contains(*name)),
                "all three Nano cameras required"
            );
        }
        if index == 0 {
            msg.content.push_str(mode.instruction());
        }
        prepared.messages.push(msg);
    }
    ensure!(
        (3..=15).contains(&prepared.image_count),
        "expected at most five three-camera observations"
    );
    Ok(prepared)
}
