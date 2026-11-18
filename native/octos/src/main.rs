use aisle_octos::{
    contract::Mode,
    service::{Service, router},
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use octos_llm::openai::{ModelHints, OpenAIProvider, ReasoningStyle};
use serde_json::{Value, json};
use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

#[derive(Parser)]
#[command(about = "AISLE Octos multimodal inference; emits proposals, never motor commands")]
struct Args {
    #[command(subcommand)]
    command: Command,
    #[arg(long, global = true, default_value = "subgoal")]
    mode: Mode,
    #[arg(long, global = true, default_value = "deepseek-flash")]
    model: String,
    #[arg(long, global = true, default_value = "https://api.deepseek.com/v1")]
    upstream: String,
    #[arg(long, global = true, default_value = "DEEPSEEK_API_KEY")]
    api_key_env: String,
    /// Read only the selected profile's DeepSeek credential; no hooks/tools/memory.
    #[arg(long, global = true)]
    octos_profile: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 12000)]
    timeout_ms: u64,
    #[arg(long, global = true, default_value_t = 30000)]
    max_observation_age_ms: u64,
}
#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8099")]
        listen: SocketAddr,
    },
    #[cfg(feature = "dora")]
    Node,
    Infer {
        #[arg(long)]
        request: PathBuf,
    },
}
fn credential(args: &Args) -> Result<String> {
    if let Ok(key) = std::env::var(&args.api_key_env) {
        ensure!(!key.is_empty(), "empty API key");
        return Ok(key);
    }
    if let Some(path) = &args.octos_profile {
        let bytes = std::fs::read(path).context("cannot read selected Octos profile")?;
        ensure!(bytes.len() <= 1024 * 1024, "Octos profile too large");
        let profile: Value =
            serde_json::from_slice(&bytes).context("invalid Octos profile JSON")?;
        let key = profile
            .pointer("/config/env_vars/DEEPSEEK_API_KEY")
            .and_then(Value::as_str)
            .context("selected profile has no DeepSeek credential")?;
        ensure!(
            !key.is_empty() && !key.starts_with("keychain:"),
            "use API-key environment for keychain-backed profiles"
        );
        return Ok(key.to_owned());
    }
    anyhow::bail!("set {} or select --octos-profile", args.api_key_env)
}
#[tokio::main]
async fn main() {
    let result = run().await;
    match result {
        Ok(v) => {
            println!("{v}");
            if v["ok"] == false {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!(
                "{}",
                json!({"ok":false,"error":e.to_string(),"execute":false})
            );
            std::process::exit(1);
        }
    }
}
async fn run() -> Result<Value> {
    let mut args = match Args::try_parse() {
        Ok(a) => a,
        Err(e)
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            return Ok(json!({"ok":true,"help":e.to_string()}));
        }
        Err(e) => anyhow::bail!("{e}"),
    };
    if args.octos_profile.is_none() {
        args.octos_profile = std::env::var_os("AISLE_OCTOS_PROFILE").map(PathBuf::from);
    }
    ensure!(
        args.model == "deepseek-flash",
        "this adapter pins the deepseek-flash model contract"
    );
    // Explicit alternate endpoints are for local replay/mock validation only.
    let allowed = args.upstream == "https://api.deepseek.com/v1"
        || args
            .upstream
            .strip_prefix("http://")
            .and_then(|s| s.strip_suffix("/v1"))
            .and_then(|s| s.parse::<SocketAddr>().ok())
            .is_some_and(|a| a.ip().is_loopback());
    ensure!(
        allowed,
        "upstream must be official DeepSeek or an explicit loopback /v1 endpoint"
    );
    let provider = OpenAIProvider::new(credential(&args)?, &args.model)
        .with_provider_label("deepseek")
        .with_base_url(&args.upstream)
        .with_hints(ModelHints {
            reasoning_style: ReasoningStyle::EffortAndThinkingToggle,
            ..ModelHints::detect(&args.model)
        })
        .with_http_timeout(args.timeout_ms.div_ceil(1000).max(1), 5);
    let provider = Arc::new(provider);
    let service = Arc::new(Service::new(
        provider,
        args.mode,
        Duration::from_millis(args.timeout_ms),
        args.max_observation_age_ms,
    )?);
    match args.command {
        Command::Infer { request } => {
            let bytes = std::fs::read(request)?;
            ensure!(
                bytes.len() <= aisle_octos::input::MAX_WIRE,
                "request too large"
            );
            service
                .infer(serde_json::from_slice(&bytes)?)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e.message))
        }
        Command::Serve { listen } => {
            ensure!(
                matches!(listen.ip(),IpAddr::V4(a) if a.is_loopback()),
                "listen must be IPv4 loopback"
            );
            let socket = tokio::net::TcpListener::bind(listen).await?;
            eprintln!(
                "{}",
                json!({"event":"listening","address":socket.local_addr()?.to_string(),"mode":args.mode})
            );
            axum::serve(socket, router(service))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
            Ok(json!({"ok":true,"stopped":true}))
        }
        #[cfg(feature = "dora")]
        Command::Node => {
            aisle_octos::dora::run(service).await?;
            Ok(json!({"ok":true,"stopped":true}))
        }
    }
}
