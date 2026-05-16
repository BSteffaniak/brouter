#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Command-line interface for brouter.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::OnceLock;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent};
use brouter_catalog_models::ResolvedModelMetadata;
use brouter_config::{
    apply_default_config, llm_judge_config_or_default, load_config, resolve_config_path,
    routeable_models, routing_rules, scoring_weights, validate_config_warnings,
};
use brouter_config_models::{BrouterConfig, ProviderConfig, ProviderKind};
use brouter_provider::{ProviderClient, ProviderRegistry};
use brouter_provider_models::{ModelId, ProviderId, RouteableModel};
use brouter_router::Router;
use brouter_router::{
    DEFAULT_JUDGE_SYSTEM_PROMPT, build_judge_prompt, judge_request, parse_judge_response,
    should_fire_trigger, top_2_score_gap,
};
use brouter_router_models::{
    JudgeConfig, ReasoningLevel, RoutingDecision, RoutingObjective, ScoredCandidate,
    SelectedRequestControls,
};
use brouter_telemetry::TelemetryStore;
use brouter_telemetry_models::{RoutingEvent, RoutingEventKind, SessionSummary};
use clap::{Parser, Subcommand};
use rand::TryRng as _;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use tracing::{debug, info, warn as tracing_warn};
use zeroize::Zeroizing;

static LOG_GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

/// Runs the brouter command-line interface.
///
/// # Errors
///
/// Returns an error when configuration loading, route explanation, JSON output,
/// or server startup fails.
pub async fn run_cli() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { config, port } => serve(config.as_deref(), port).await,
        Command::CheckConfig {
            config,
            strict,
            json,
        } => check_config(&resolve_and_load(config.as_deref())?, strict, json),
        Command::PrintExampleConfig => {
            print_example_config();
            Ok(())
        }
        Command::Auth { command } => auth(command).await,
        Command::Doctor { config, json } => {
            doctor(&resolve_and_load(config.as_deref())?, json).await
        }
        Command::Sessions { config, command } => {
            sessions(&resolve_and_load(config.as_deref())?, command).await
        }
        Command::Route {
            config,
            prompt,
            first_message,
            objective,
            judge,
        } => {
            explain_route(
                &resolve_and_load(config.as_deref())?,
                &prompt,
                first_message,
                objective.as_deref(),
                judge,
            )
            .await
        }
    }
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let json = std::env::var("BROUTER_LOG_FORMAT").is_ok_and(|value| value == "json");
    if let Some(path) = log_file_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let (writer, guard) = tracing_appender::non_blocking(file);
            let _ = LOG_GUARD.set(guard);
            if json {
                let _ = tracing_subscriber::fmt()
                    .json()
                    .with_writer(writer)
                    .with_env_filter(filter)
                    .try_init();
            } else {
                let _ = tracing_subscriber::fmt()
                    .with_writer(writer)
                    .with_env_filter(filter)
                    .try_init();
            }
            return;
        }
    }
    if json {
        let _ = tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    }
}

fn log_file_path() -> Option<PathBuf> {
    match std::env::var("BROUTER_LOG_FILE") {
        Ok(value) if value == "off" || value == "false" || value.is_empty() => None,
        Ok(value) => Some(expand_user_path(&value)),
        Err(_) => Some(default_state_file("brouter.log")),
    }
}

fn default_state_file(file_name: &str) -> PathBuf {
    if let Ok(state_home) = std::env::var("XDG_STATE_HOME") {
        return PathBuf::from(state_home).join("brouter").join(file_name);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("brouter")
            .join(file_name);
    }
    PathBuf::from(".brouter-state").join(file_name)
}

fn expand_user_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(stripped);
    }
    PathBuf::from(path)
}

fn resolve_and_load(config: Option<&Path>) -> Result<BrouterConfig> {
    let path = resolve_config_path(config).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(load_config(&path)?)
}

/// Runs brouter with the configuration at the resolved path, or with a
/// default configuration if no config file is found.
async fn serve(config: Option<&Path>, port_override: Option<u16>) -> Result<()> {
    let mut config = match resolve_config_path(config) {
        Ok(path) => load_config(&path)?,
        Err(brouter_config::ConfigError::ConfigPathNotFound { .. }) => {
            let mut config = BrouterConfig::default();
            apply_default_config(&mut config);
            config
        }
        Err(e) => return Err(e.into()),
    };
    if let Some(port) = port_override {
        config.server.port = port;
    }
    brouter_server::serve(config).await?;
    Ok(())
}

fn check_config(config: &BrouterConfig, strict: bool, json_output: bool) -> Result<()> {
    let warnings = validate_config_warnings(config);
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": warnings.is_empty(),
                "providers": config.providers.len(),
                "models": config.models.len(),
                "warnings": warnings.iter().map(ToString::to_string).collect::<Vec<_>>(),
            }))?
        );
    } else {
        println!(
            "config ok: {} providers, {} models, {} warnings",
            config.providers.len(),
            config.models.len(),
            warnings.len()
        );
        for warning in &warnings {
            eprintln!("warning: {warning}");
        }
    }
    if strict && !warnings.is_empty() {
        bail!(
            "strict config check failed with {} warnings",
            warnings.len()
        );
    }
    Ok(())
}

async fn doctor(config: &BrouterConfig, json_output: bool) -> Result<()> {
    let warnings = validate_config_warnings(config);
    let models = routeable_models(config);
    let judge = llm_judge_config_or_default(config, &models);
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": warnings.is_empty(),
                "providers": config.providers.len(),
                "models": config.models.len(),
                "warnings": warnings.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "introspection": {
                    "refresh_on_startup": config.router.metadata.refresh_on_startup,
                    "allow_stale_on_provider_error": config.router.metadata.allow_stale_on_provider_error,
                    "cache_path": &config.router.metadata.cache_path,
                    "enabled_providers": config.providers.iter().filter(|(_, provider)| provider.introspection.enabled && !provider.introspection.disabled).map(|(id, _)| id).collect::<Vec<_>>(),
                },
                "telemetry": {
                    "disabled": config.telemetry.disabled,
                    "database_path": &config.telemetry.database_path,
                },
                "logs": {
                    "file": log_file_path().map(|path| path.display().to_string()),
                },
                "judge": judge.as_ref().map(|judge| serde_json::json!({
                    "model": judge.model.to_string(),
                    "provider": judge.provider.as_ref().map(ToString::to_string),
                })),
            }))?
        );
        return Ok(());
    }
    println!(
        "config ok: {} providers, {} models, {} warnings",
        config.providers.len(),
        config.models.len(),
        warnings.len()
    );
    println!(
        "introspection: refresh_on_startup={}, cache_path={}",
        config.router.metadata.refresh_on_startup,
        config
            .router
            .metadata
            .cache_path
            .as_deref()
            .unwrap_or("<disabled>")
    );
    println!(
        "telemetry: backend={}, database_path={}",
        if config.telemetry.disabled {
            "memory"
        } else {
            "sqlite"
        },
        config
            .telemetry
            .database_path
            .as_deref()
            .unwrap_or("<memory>")
    );
    if let Some(judge) = &judge {
        println!("judge: default/explicit model {}", judge.model);
    } else {
        println!("judge: disabled or unavailable");
    }
    if let Some(path) = log_file_path() {
        println!("logs: {}", path.display());
    }
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let mut failures = 0_u32;
    for (provider_id, provider) in &config.providers {
        match check_provider(&client, config, provider_id, provider).await {
            Ok(message) => println!("provider {provider_id}: ok ({message})"),
            Err(error) => {
                failures = failures.saturating_add(1);
                eprintln!("provider {provider_id}: failed ({error})");
            }
        }
    }

    if failures > 0 {
        bail!("doctor failed: {failures} provider checks failed");
    }
    Ok(())
}

async fn check_provider(
    client: &reqwest::Client,
    config: &BrouterConfig,
    provider_id: &str,
    provider: &ProviderConfig,
) -> Result<String> {
    if provider.kind == ProviderKind::OpenaiCodex {
        let configured_models = config
            .models
            .values()
            .filter(|model| model.provider == provider_id)
            .count();
        return Ok(format!(
            "sshenv-backed ChatGPT/Codex provider configured, {configured_models} configured models"
        ));
    }
    let base_url = provider.base_url.as_deref().unwrap_or(match provider.kind {
        ProviderKind::OpenAiCompatible => bail!("missing base_url"),
        ProviderKind::Anthropic => "https://api.anthropic.com/v1",
        ProviderKind::OpenaiCodex => unreachable!("openai-codex handled above"),
    });
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut request = client.get(url);
    if let Some(api_key_env) = &provider.api_key_env {
        let api_key = std::env::var(api_key_env)?;
        request = match provider.kind {
            ProviderKind::OpenAiCompatible => request.bearer_auth(api_key),
            ProviderKind::Anthropic => request
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01"),
            ProviderKind::OpenaiCodex => unreachable!("openai-codex handled above"),
        };
    }
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        bail!("GET /models returned {status}");
    }
    let configured_models = config
        .models
        .values()
        .filter(|model| model.provider == provider_id)
        .count();
    Ok(format!("reachable, {configured_models} configured models"))
}

async fn sessions(config: &BrouterConfig, command: SessionsCommand) -> Result<()> {
    let store = telemetry_store_from_config(config).await?;
    match command {
        SessionsCommand::List { json } => {
            let sessions = store.sessions().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&sessions)?);
            } else if sessions.is_empty() {
                println!("no brouter sessions recorded");
            } else {
                for summary in sessions {
                    print_session_summary(&summary);
                }
            }
        }
        SessionsCommand::Show { session_id, json } => {
            let sessions = store.sessions().await?;
            let Some(summary) = sessions
                .into_iter()
                .find(|summary| summary.session_id == session_id)
            else {
                bail!("unknown brouter session {session_id}");
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&summary)?);
            } else {
                print_session_summary(&summary);
            }
        }
        SessionsCommand::Timeline {
            session_id,
            last,
            json,
        } => {
            let mut events = store.routing_events_for_session(&session_id).await?;
            if let Some(last) = last
                && events.len() > last
            {
                events = events.split_off(events.len() - last);
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&events)?);
            } else if events.is_empty() {
                println!("no events recorded for session {session_id}");
            } else {
                for event in &events {
                    print_routing_event(event);
                }
            }
        }
        SessionsCommand::Event { event_id, json } => {
            let events = store.routing_events().await?;
            let Some(event) = events.into_iter().find(|event| event.event_id == event_id) else {
                bail!("unknown brouter routing event {event_id}");
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&event)?);
            } else {
                print_routing_event(&event);
                println!("{}", serde_json::to_string_pretty(&event.payload)?);
            }
        }
    }
    Ok(())
}

async fn telemetry_store_from_config(config: &BrouterConfig) -> Result<TelemetryStore> {
    if config.telemetry.disabled {
        bail!("telemetry is disabled; no session event log is available");
    }
    let Some(path) = &config.telemetry.database_path else {
        bail!("telemetry.database_path is not configured; no session event log is available");
    };
    Ok(TelemetryStore::sqlite(&expand_user_path(path)).await?)
}

fn print_session_summary(summary: &SessionSummary) {
    println!(
        "{}  events={} requests={} first={} last={}",
        summary.session_id,
        summary.event_count,
        summary.request_count,
        summary.first_timestamp_ms,
        summary.last_timestamp_ms
    );
}

fn print_routing_event(event: &RoutingEvent) {
    match event.kind {
        RoutingEventKind::RouteDecision => print_route_decision_event(event),
        RoutingEventKind::ProviderAttempt => print_provider_attempt_event(event),
        _ => println!(
            "{}  {}  {}  {}",
            event.timestamp_ms,
            event.kind.as_str(),
            event.request_id,
            event.event_id
        ),
    }
}

fn print_route_decision_event(event: &RoutingEvent) {
    let payload = &event.payload;
    let selected = payload
        .get("selected_model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<unknown>");
    let provider = payload
        .get("provider")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<unknown>");
    let tier = payload
        .get("request_controls")
        .and_then(|controls| controls.get("service_tier"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default");
    let reasoning = payload
        .get("request_controls")
        .and_then(|controls| controls.get("reasoning_effort"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default");
    let intent = payload
        .get("features")
        .and_then(|features| features.get("intent"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let detected_reasoning = payload
        .get("features")
        .and_then(|features| features.get("reasoning"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    println!(
        "{}  route  {} via {} tier={} reasoning={} intent={}/{} request={} event={}",
        event.timestamp_ms,
        selected,
        provider,
        tier,
        reasoning,
        intent,
        detected_reasoning,
        event.request_id,
        event.event_id
    );
    if let Some(reasons) = payload.get("reasons").and_then(serde_json::Value::as_array)
        && !reasons.is_empty()
    {
        let reasons = reasons
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>()
            .join("; ");
        println!("          why: {reasons}");
    }
    if let Some(sources) = payload
        .get("control_sources")
        .and_then(serde_json::Value::as_object)
    {
        println!(
            "          controls: service_tier from {}, reasoning_effort from {}",
            sources
                .get("service_tier")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            sources
                .get("reasoning_effort")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
        );
    }
    print_judge_summary(payload);
    print_candidate_summary(payload, selected);
}

fn print_judge_summary(payload: &serde_json::Value) {
    if let Some(trigger) = payload.get("judge_trigger") {
        let configured = trigger
            .get("configured")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let fired = trigger
            .get("fired")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let reason = trigger
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let gap = trigger
            .get("score_gap")
            .and_then(serde_json::Value::as_f64)
            .map_or_else(|| "unknown".to_string(), |gap| format!("{gap:.2}"));
        let state = if fired { "fired" } else { "skipped" };
        println!("          judge: configured={configured} {state} gap={gap} ({reason})");
    }
    if let Some(judge) = payload.get("judge")
        && !judge.is_null()
    {
        let model = judge
            .get("model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let chosen = judge
            .get("chosen_model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let overridden = judge
            .get("overridden")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let rationale = judge
            .get("rationale")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        println!("          judge result: {model} chose {chosen} overridden={overridden}");
        if !rationale.is_empty() {
            println!("          judge why: {rationale}");
        }
    }
}

fn print_candidate_summary(payload: &serde_json::Value, selected: &str) {
    let Some(candidates) = payload
        .get("candidates")
        .and_then(serde_json::Value::as_array)
    else {
        return;
    };
    if candidates.is_empty() {
        return;
    }
    println!("          top candidates:");
    let top_score = candidates
        .first()
        .and_then(|candidate| candidate.get("score"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    for candidate in candidates.iter().take(3) {
        let model = candidate
            .get("model_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let marker = if model == selected { "*" } else { " " };
        let score = candidate
            .get("score")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let delta = score - top_score;
        let cost = candidate
            .get("estimated_cost")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let reasons = candidate
            .get("reasons")
            .and_then(serde_json::Value::as_array)
            .map(|reasons| {
                reasons
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_default();
        println!(
            "          {marker} {model}: score={score:.2} Δ={delta:.2} cost=${cost:.6} {reasons}"
        );
    }
}

fn print_provider_attempt_event(event: &RoutingEvent) {
    let payload = &event.payload;
    let model = payload
        .get("model_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<unknown>");
    let status = payload
        .get("status_code")
        .and_then(serde_json::Value::as_u64)
        .map_or_else(|| "error".to_string(), |status| status.to_string());
    let fallback = payload
        .get("fallback_used")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    println!(
        "{}  attempt  {} status={} fallback={} request={} event={}",
        event.timestamp_ms, model, status, fallback, event.request_id, event.event_id
    );
    if let Some(error) = payload
        .get("provider_error")
        .and_then(serde_json::Value::as_str)
    {
        println!("          error: {error}");
    }
}

fn print_example_config() {
    print!("{}", include_str!("../../../brouter.example.toml"));
}

async fn auth(command: AuthCommand) -> Result<()> {
    match command {
        AuthCommand::OpenaiCodex { command } => match command {
            OpenaiCodexAuthCommand::Login {
                profile,
                vault,
                recipient_key,
                headless,
            } => login_openai_codex(profile, vault, recipient_key, headless).await,
        },
        AuthCommand::Openrouter { command } => match command {
            OpenrouterAuthCommand::Login {
                profile,
                vault,
                recipient_key,
                api_key,
            } => login_openrouter(profile, vault, recipient_key, api_key),
        },
        AuthCommand::OpencodeZen { command } => match command {
            OpencodeZenAuthCommand::Login {
                profile,
                vault,
                recipient_key,
                api_key,
            } => login_opencode_zen(profile, vault, recipient_key, api_key),
        },
        AuthCommand::Openaicom { command } => match command {
            OpenaicomAuthCommand::Login {
                profile,
                vault,
                recipient_key,
                api_key,
            } => login_openaicom(profile, vault, recipient_key, api_key),
        },
    }
}

async fn login_openai_codex(
    profile: String,
    vault: PathBuf,
    recipient_key: Option<String>,
    headless: bool,
) -> Result<()> {
    let store = open_auth_store(&vault, recipient_key)?;
    let oauth = run_openai_codex_oauth(headless).await?;
    let expires_at =
        unix_timestamp().saturating_add(oauth.expires_in.unwrap_or(3_600).saturating_sub(60));
    let account_id = oauth
        .id_token
        .as_deref()
        .and_then(chatgpt_account_id_from_access_token)
        .or_else(|| chatgpt_account_id_from_access_token(&oauth.access_token));

    set_auth_secret(&store, &profile, "BROUTER_OPENAI_AUTH_MODE", "chatgpt")?;
    set_auth_secret(
        &store,
        &profile,
        "BROUTER_OPENAI_CODEX_ACCESS_TOKEN",
        oauth.access_token,
    )?;
    if let Some(id_token) = oauth.id_token {
        set_auth_secret(&store, &profile, "BROUTER_OPENAI_CODEX_ID_TOKEN", id_token)?;
    }
    if let Some(refresh_token) = oauth.refresh_token {
        set_auth_secret(
            &store,
            &profile,
            "BROUTER_OPENAI_CODEX_REFRESH_TOKEN",
            refresh_token,
        )?;
    }
    set_auth_secret(
        &store,
        &profile,
        "BROUTER_OPENAI_CODEX_EXPIRES_AT",
        expires_at.to_string(),
    )?;
    if let Some(account_id) = account_id {
        set_auth_secret(
            &store,
            &profile,
            "BROUTER_OPENAI_CODEX_ACCOUNT_ID",
            account_id,
        )?;
    }

    println!(
        "OpenAI ChatGPT/Codex subscription login saved to profile {profile} in {}",
        vault.display()
    );
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn login_openrouter(
    profile: String,
    vault: PathBuf,
    recipient_key: Option<String>,
    api_key: Option<String>,
) -> Result<()> {
    let api_key = resolve_api_key(api_key, "OPENROUTER_API_KEY", "OpenRouter")?;
    let store = open_auth_store(&vault, recipient_key)?;
    set_auth_secret(&store, &profile, "OPENROUTER_API_KEY", api_key)?;
    println!(
        "OpenRouter API key saved to profile {profile} in {}",
        vault.display()
    );
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn login_opencode_zen(
    profile: String,
    vault: PathBuf,
    recipient_key: Option<String>,
    api_key: Option<String>,
) -> Result<()> {
    let api_key = resolve_api_key(api_key, "OPENCODE_API_KEY", "OpenCode Zen")?;
    let store = open_auth_store(&vault, recipient_key)?;
    set_auth_secret(&store, &profile, "OPENCODE_API_KEY", api_key)?;
    println!(
        "OpenCode Zen API key saved to profile {profile} in {}",
        vault.display()
    );
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn login_openaicom(
    profile: String,
    vault: PathBuf,
    recipient_key: Option<String>,
    api_key: Option<String>,
) -> Result<()> {
    let api_key = resolve_api_key(api_key, "OPENAI_API_KEY", "OpenAI")?;
    let store = open_auth_store(&vault, recipient_key)?;
    set_auth_secret(&store, &profile, "OPENAI_API_KEY", api_key)?;
    println!(
        "OpenAI API key saved to profile {profile} in {}",
        vault.display()
    );
    Ok(())
}

fn resolve_api_key(provided: Option<String>, env_var: &str, name: &str) -> Result<String> {
    if let Some(key) = provided {
        return Ok(key);
    }
    if let Ok(key) = std::env::var(env_var) {
        println!("Read {name} API key from {env_var} environment variable.");
        return Ok(key);
    }
    print!(
        "Enter your {name} API key (or paste the full https://api.openai.com/dashboard/API/... URL): "
    );
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let key = input.trim().to_string();
    if key.is_empty() {
        bail!("No API key provided");
    }
    Ok(key)
}

fn open_auth_store(
    vault_path: &Path,
    recipient_key: Option<String>,
) -> Result<sshenv_vault::SshenvStore> {
    let store = sshenv_vault::SshenvStore::new(sshenv_vault::SshenvStoreConfig::new(
        vault_path.to_path_buf(),
    ));
    if !vault_path.exists() {
        let recipient_key = resolve_recipient_key(recipient_key)?;
        store.init(&recipient_key)?;
    }
    Ok(store)
}

fn set_auth_secret(
    store: &sshenv_vault::SshenvStore,
    profile: &str,
    key: &str,
    value: impl Into<String>,
) -> Result<()> {
    store.set_secret(profile, key, Zeroizing::new(value.into()))?;
    Ok(())
}

async fn run_openai_codex_oauth(headless: bool) -> Result<OpenAiOauthTokenResponse> {
    if headless {
        run_openai_codex_device_oauth().await
    } else {
        run_openai_codex_browser_oauth().await
    }
}

async fn run_openai_codex_browser_oauth() -> Result<OpenAiOauthTokenResponse> {
    let listeners = open_oauth_listeners()?;
    let redirect_uri = format!("http://localhost:{OPENAI_CODEX_OAUTH_PORT}/auth/callback");
    let state = random_urlsafe(32)?;
    let verifier = random_pkce_verifier(43)?;
    let challenge = pkce_challenge(&verifier);
    let authorize_url = openai_codex_authorize_url(&redirect_uri, &state, &challenge);
    println!("OpenAI ChatGPT/Codex subscription browser login");
    println!("Open this URL if your browser does not open automatically:\n{authorize_url}\n");
    println!(
        "If your browser says localhost refused to connect, copy the full redirected localhost URL, paste it here, and press Enter."
    );
    open_browser(&authorize_url);
    let code = wait_for_oauth_code(&listeners, &state)?;
    exchange_openai_codex_code_async(&redirect_uri, &verifier, &code).await
}

async fn run_openai_codex_device_oauth() -> Result<OpenAiOauthTokenResponse> {
    let device = start_openai_codex_device_auth().await?;
    println!("OpenAI ChatGPT/Codex subscription device login");
    println!("Open this URL:\nhttps://auth.openai.com/codex/device\n");
    println!("Enter this code: {}", device.user_code);
    let interval = device.interval.parse::<u64>().unwrap_or(5).max(1);
    let token = poll_openai_codex_device_auth(&device, interval).await?;
    exchange_openai_codex_code_async(
        "https://auth.openai.com/deviceauth/callback",
        &token.code_verifier,
        &token.authorization_code,
    )
    .await
}

async fn start_openai_codex_device_auth() -> Result<OpenAiDeviceUserCodeResponse> {
    let response = reqwest::Client::new()
        .post("https://auth.openai.com/api/accounts/deviceauth/usercode")
        .header(
            "User-Agent",
            format!("brouter/{}", env!("CARGO_PKG_VERSION")),
        )
        .json(&serde_json::json!({ "client_id": OPENAI_CODEX_CLIENT_ID }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        bail!("OpenAI device authorization failed with HTTP {status}: {body}");
    }
    Ok(serde_json::from_str(&body)?)
}

async fn poll_openai_codex_device_auth(
    device: &OpenAiDeviceUserCodeResponse,
    interval_seconds: u64,
) -> Result<OpenAiDeviceTokenResponse> {
    loop {
        tokio::time::sleep(Duration::from_secs(interval_seconds.saturating_add(3))).await;
        let response = reqwest::Client::new()
            .post("https://auth.openai.com/api/accounts/deviceauth/token")
            .header(
                "User-Agent",
                format!("brouter/{}", env!("CARGO_PKG_VERSION")),
            )
            .json(&serde_json::json!({
                "device_auth_id": device.device_auth_id,
                "user_code": device.user_code,
            }))
            .send()
            .await?;
        if response.status().is_success() {
            let body = response.text().await?;
            return Ok(serde_json::from_str(&body)?);
        }
        if !matches!(response.status().as_u16(), 403 | 404) {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("OpenAI device authorization polling failed with HTTP {status}: {body}");
        }
    }
}

async fn exchange_openai_codex_code_async(
    redirect_uri: &str,
    verifier: &str,
    code: &str,
) -> Result<OpenAiOauthTokenResponse> {
    let params = [
        ("grant_type", "authorization_code"),
        ("client_id", OPENAI_CODEX_CLIENT_ID),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("code_verifier", verifier),
    ];
    let response = reqwest::Client::new()
        .post(OPENAI_CODEX_TOKEN_URL)
        .form(&params)
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        bail!("OpenAI OAuth token exchange failed with HTTP {status}: {body}");
    }
    Ok(serde_json::from_str(&body)?)
}

fn openai_codex_authorize_url(redirect_uri: &str, state: &str, challenge: &str) -> String {
    let params = [
        ("response_type", "code"),
        ("client_id", OPENAI_CODEX_CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("scope", OPENAI_CODEX_SCOPE),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("state", state),
        ("originator", "brouter"),
    ];
    let query = params
        .into_iter()
        .map(|(key, value)| format!("{}={}", pct_encode(key), pct_encode(value)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{OPENAI_CODEX_AUTHORIZE_URL}?{query}")
}

fn open_oauth_listeners() -> Result<Vec<TcpListener>> {
    let mut listeners = Vec::new();
    let mut errors = Vec::new();
    for address in ["127.0.0.1", "::1"] {
        match TcpListener::bind((address, OPENAI_CODEX_OAUTH_PORT)) {
            Ok(listener) => {
                listener.set_nonblocking(true)?;
                listeners.push(listener);
            }
            Err(error) => errors.push(format!("{address}: {error}")),
        }
    }
    if listeners.is_empty() {
        bail!(
            "failed to bind OpenAI OAuth callback server on localhost:{OPENAI_CODEX_OAUTH_PORT}: {}",
            errors.join("; ")
        );
    }
    Ok(listeners)
}

fn wait_for_oauth_code(listeners: &[TcpListener], expected_state: &str) -> Result<String> {
    let manual_callback = spawn_manual_oauth_callback_reader();
    let deadline = Instant::now() + Duration::from_mins(5);
    loop {
        if let Some(code) = poll_manual_oauth_callback(&manual_callback, expected_state)? {
            return Ok(code);
        }
        if Instant::now() >= deadline {
            bail!("OpenAI OAuth callback timed out");
        }
        if let Some(code) = poll_oauth_listeners(listeners, expected_state)? {
            return Ok(code);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn poll_oauth_listeners(listeners: &[TcpListener], expected_state: &str) -> Result<Option<String>> {
    for listener in listeners {
        match listener.accept() {
            Ok((mut stream, _)) => match handle_oauth_callback_stream(&mut stream, expected_state)?
            {
                OAuthCallback::Code(code) => return Ok(Some(code)),
                OAuthCallback::Ignored => {}
            },
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(None)
}

fn spawn_manual_oauth_callback_reader() -> Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        loop {
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                break;
            }
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    receiver
}

fn poll_manual_oauth_callback(
    receiver: &Receiver<String>,
    expected_state: &str,
) -> Result<Option<String>> {
    match receiver.try_recv() {
        Ok(input) => manual_oauth_callback_code(&input, expected_state),
        Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => Ok(None),
    }
}

fn manual_oauth_callback_code(input: &str, expected_state: &str) -> Result<Option<String>> {
    if input.trim().is_empty() {
        return Ok(None);
    }
    match parse_oauth_callback(input.trim()) {
        OAuthCallbackParse::Code { code, state } if state == expected_state => Ok(Some(code)),
        OAuthCallbackParse::Code { .. } => {
            eprintln!(
                "Pasted OpenAI OAuth callback state did not match; paste the newest redirected URL from this login attempt."
            );
            Ok(None)
        }
        OAuthCallbackParse::Error(error) => bail!("OpenAI OAuth failed: {error}"),
        OAuthCallbackParse::Ignored => {
            eprintln!(
                "Pasted text was not an OpenAI OAuth callback URL; paste the full localhost callback URL."
            );
            Ok(None)
        }
    }
}

fn handle_oauth_callback_stream(
    stream: &mut std::net::TcpStream,
    expected_state: &str,
) -> Result<OAuthCallback> {
    let mut request = [0_u8; 8192];
    let size = stream.read(&mut request)?;
    let request = String::from_utf8_lossy(&request[..size]);
    let first_line = request.lines().next().unwrap_or_default();
    match parse_oauth_callback(first_line) {
        OAuthCallbackParse::Code { code, state } if state == expected_state => {
            write_oauth_response(stream, true)?;
            Ok(OAuthCallback::Code(code))
        }
        OAuthCallbackParse::Code { .. } => {
            write_oauth_response(stream, false)?;
            bail!("OpenAI OAuth callback state did not match");
        }
        OAuthCallbackParse::Error(error) => {
            write_oauth_response(stream, false)?;
            bail!("OpenAI OAuth failed: {error}");
        }
        OAuthCallbackParse::Ignored => {
            write_oauth_response(stream, false)?;
            Ok(OAuthCallback::Ignored)
        }
    }
}

fn write_oauth_response(stream: &mut std::net::TcpStream, success: bool) -> Result<()> {
    let response_body = if success {
        "brouter OpenAI login complete. You can close this tab."
    } else {
        "brouter OpenAI login did not complete. Return to your terminal."
    };
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        response_body.len(),
        response_body
    )?;
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum OAuthCallback {
    Code(String),
    Ignored,
}

#[derive(Debug, PartialEq, Eq)]
enum OAuthCallbackParse {
    Code { code: String, state: String },
    Error(String),
    Ignored,
}

fn parse_oauth_callback(input: &str) -> OAuthCallbackParse {
    let Some(path) = oauth_callback_path(input) else {
        return OAuthCallbackParse::Ignored;
    };
    if !path.starts_with("/auth/callback") {
        return OAuthCallbackParse::Ignored;
    }
    let Some(query) = path.split_once('?').map(|(_, query)| query) else {
        return OAuthCallbackParse::Ignored;
    };
    let mut code = None;
    let mut state = None;
    let mut error = None;
    let mut error_description = None;
    for pair in query.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        match pct_decode(key).as_deref() {
            Some("code") => code = pct_decode(value),
            Some("state") => state = pct_decode(value),
            Some("error") => error = pct_decode(value),
            Some("error_description") => error_description = pct_decode(value),
            _ => {}
        }
    }
    if let Some(error) = error_description.or(error) {
        return OAuthCallbackParse::Error(error);
    }
    match (code, state) {
        (Some(code), Some(state)) => OAuthCallbackParse::Code { code, state },
        _ => OAuthCallbackParse::Ignored,
    }
}

fn oauth_callback_path(input: &str) -> Option<&str> {
    let candidate = if input.starts_with("GET ") || input.starts_with("POST ") {
        input.split_whitespace().nth(1)?
    } else {
        oauth_callback_url_from_text(input.trim())?
    };
    if candidate.starts_with("/auth/callback") {
        return Some(candidate);
    }
    let (_, without_scheme) = candidate.split_once("://")?;
    let path_start = without_scheme.find('/')?;
    Some(&without_scheme[path_start..])
}

fn oauth_callback_url_from_text(input: &str) -> Option<&str> {
    if input.starts_with("/auth/callback") {
        return Some(input);
    }
    let start = input
        .find("http://localhost:")
        .or_else(|| input.find("http://127.0.0.1:"))
        .or_else(|| input.find("http://[::1]:"))?;
    let rest = &input[start..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    Some(&rest[..end])
}

fn random_urlsafe(bytes: usize) -> Result<String> {
    let mut data = vec![0_u8; bytes];
    rand::rngs::SysRng.try_fill_bytes(&mut data)?;
    Ok(URL_SAFE_NO_PAD.encode(data))
}

fn random_pkce_verifier(length: usize) -> Result<String> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut data = vec![0_u8; length];
    rand::rngs::SysRng.try_fill_bytes(&mut data)?;
    Ok(data
        .into_iter()
        .map(|byte| char::from(CHARS[usize::from(byte) % CHARS.len()]))
        .collect())
}

fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

fn pct_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

fn pct_decode(value: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut iter = value.as_bytes().iter().copied();
    while let Some(byte) = iter.next() {
        if byte == b'%' {
            let high = iter.next()?;
            let low = iter.next()?;
            bytes.push(hex_byte(high, low)?);
        } else if byte == b'+' {
            bytes.push(b' ');
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).ok()
}

fn hex_byte(high: u8, low: u8) -> Option<u8> {
    const fn digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }
    Some(digit(high)? << 4 | digit(low)?)
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let command = ("open", vec![url]);
    #[cfg(target_os = "windows")]
    let command = ("cmd", vec!["/C", "start", url]);
    #[cfg(all(unix, not(target_os = "macos")))]
    let command = ("xdg-open", vec![url]);
    let _ = ProcessCommand::new(command.0)
        .args(command.1)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

fn resolve_recipient_key(recipient_key: Option<String>) -> Result<String> {
    if let Some(recipient_key) = recipient_key {
        return public_key_line_from_path_or_literal(&recipient_key);
    }
    let Some(path) = sshenv_vault::identity::discover_public_key_paths()
        .into_iter()
        .next()
    else {
        bail!("no SSH public key found; pass --recipient-key <path-or-public-key>");
    };
    public_key_line_from_path_or_literal(&path.display().to_string())
}

fn public_key_line_from_path_or_literal(value: &str) -> Result<String> {
    if value.starts_with("ssh-") {
        return Ok(value.to_string());
    }
    let contents = std::fs::read_to_string(value)?;
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToString::to_string)
        .ok_or_else(|| anyhow::anyhow!("no public key line found in {value}"))
}

fn chatgpt_account_id_from_access_token(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let claims = serde_json::from_slice::<serde_json::Value>(&bytes).ok()?;
    claims
        .get("chatgpt_account_id")
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(|auth| auth.get("chatgpt_account_id"))
        })
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

fn default_auth_vault_path() -> PathBuf {
    if let Ok(path) = std::env::var("BROUTER_AUTH_VAULT") {
        return PathBuf::from(path);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("brouter")
            .join("auth")
            .join("vault");
    }
    PathBuf::from("brouter-auth-vault")
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const OPENAI_CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OPENAI_CODEX_SCOPE: &str = "openid profile email offline_access";
const OPENAI_CODEX_OAUTH_PORT: u16 = 1455;

#[derive(Debug, Deserialize)]
struct OpenAiOauthTokenResponse {
    access_token: String,
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDeviceUserCodeResponse {
    device_auth_id: String,
    user_code: String,
    interval: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiDeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

async fn explain_route(
    config: &BrouterConfig,
    prompt: &str,
    first_message: bool,
    objective: Option<&str>,
    force_judge: bool,
) -> Result<()> {
    let models = routeable_models(config);
    let judge_config = llm_judge_config_or_default(config, &models);
    let objective = objective.map_or_else(
        || RoutingObjective::from_name(&config.router.default_objective),
        RoutingObjective::from_name,
    );
    let router = Router::new_with_rules(
        models,
        objective,
        scoring_weights(config),
        routing_rules(config),
    );
    let decision = router.route_chat(&prompt_request(prompt), first_message)?;
    let mut decision =
        maybe_invoke_judge(config, judge_config.as_ref(), &decision, force_judge).await?;
    apply_explain_controls(&mut decision);
    println!("{}", serde_json::to_string_pretty(&decision)?);
    Ok(())
}

fn apply_explain_controls(decision: &mut RoutingDecision) {
    let selected = decision
        .candidates
        .iter()
        .find(|candidate| candidate.model_id == decision.selected_model);
    let mut controls = SelectedRequestControls {
        service_tier: selected
            .and_then(|candidate| candidate.attributes.get("latency_class"))
            .cloned(),
        reasoning_effort: Some(match decision.features.reasoning {
            ReasoningLevel::Low => "low".to_string(),
            ReasoningLevel::Medium => "medium".to_string(),
            ReasoningLevel::High => "high".to_string(),
        }),
        resource_pools: Vec::new(),
    };
    if let Some(reasoning) = &decision.reasoning {
        if reasoning.service_tier.is_some() {
            controls.service_tier.clone_from(&reasoning.service_tier);
        }
        if reasoning.reasoning_effort.is_some() {
            controls
                .reasoning_effort
                .clone_from(&reasoning.reasoning_effort);
        }
    }
    decision.request_controls = controls;
}

/// Optionally invokes the LLM judge and returns the updated decision.
#[allow(clippy::too_many_lines)]
async fn maybe_invoke_judge(
    config: &BrouterConfig,
    judge_config: Option<&JudgeConfig>,
    decision: &RoutingDecision,
    force_judge: bool,
) -> Result<RoutingDecision> {
    let Some(judge_config) = judge_config else {
        return Ok(decision.clone());
    };

    // Determine if the judge should fire.
    let rule_matched = decision.features.matched_rules.iter().any(|name| {
        routing_rules(config)
            .iter()
            .any(|r| &r.name == name && r.llm_judge)
    });
    let gap = top_2_score_gap(&decision.candidates);
    let should_fire = force_judge || should_fire_trigger(&judge_config.trigger, gap, rule_matched);
    if !should_fire {
        return Ok(decision.clone());
    }

    // Build judge request components.
    let judge_model_id = &judge_config.model;
    let system_prompt = judge_config
        .system_prompt
        .as_deref()
        .unwrap_or(DEFAULT_JUDGE_SYSTEM_PROMPT);
    // Build the list of candidate judge models from ALL configured models, grouped by
    // provider (judge model first, then others from same provider, then other providers).
    // Within each provider group, sort by quality ascending so cheaper/faster models are tried first.
    let models = routeable_models(config);
    let primary_provider = judge_config.provider.clone();
    let mut judge_model_ids: Vec<ModelId> = vec![judge_model_id.clone()];
    // Collect remaining models grouped by provider, sorted by quality ascending (cheapest first).
    let mut by_provider: std::collections::BTreeMap<String, Vec<ModelId>> =
        std::collections::BTreeMap::new();
    for m in models.iter().filter(|m| &m.id != judge_model_id) {
        by_provider
            .entry(m.provider.to_string())
            .or_default()
            .push(m.id.clone());
    }
    // Sort each provider group by quality ascending (lower quality = cheaper/faster).
    for ids in by_provider.values_mut() {
        ids.sort_by(|a_id, b_id| {
            let a_quality = models
                .iter()
                .find(|m| &m.id == a_id)
                .map_or(0.0, |m| f64::from(m.quality));
            let b_quality = models
                .iter()
                .find(|m| &m.id == b_id)
                .map_or(0.0, |m| f64::from(m.quality));
            a_quality
                .partial_cmp(&b_quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    // Append same-provider models first (to avoid propagating provider failures), then other providers.
    if let Some(pp) = &primary_provider
        && let Some(same) = by_provider.remove(&pp.to_string())
    {
        judge_model_ids.extend(same);
    }
    for (_, ids) in by_provider {
        judge_model_ids.extend(ids);
    }

    // Try each judge model in fallback order until one succeeds.
    // Track failed judge providers so we can filter them from candidate shortlist.
    let registry = ProviderRegistry::from_config(config);
    let client = ProviderClient::new();
    let mut raw_text = String::new();
    let mut used_judge_id = judge_model_id;
    let mut failed_judge_providers: BTreeSet<ProviderId> = BTreeSet::new();

    // Start with all candidates - will be filtered as judge providers fail.
    let mut active_candidates: Vec<_> = decision.candidates.clone();

    for candidate_id in &judge_model_ids {
        debug!(
            model_id = %candidate_id,
            "trying judge model"
        );

        // Look up the actual model config to get the correct provider.
        let configured_judge_model = models.iter().find(|m| &m.id == candidate_id).cloned();
        let provider_id = configured_judge_model.as_ref().map_or_else(
            || {
                judge_config
                    .provider
                    .clone()
                    .unwrap_or_else(|| ProviderId::new("local"))
            },
            |model| model.provider.clone(),
        );

        // Skip judge providers that have already failed (don't retry).
        if failed_judge_providers.contains(&provider_id) {
            debug!(
                model_id = %candidate_id,
                provider = %provider_id,
                "skipping judge model (provider already failed)"
            );
            continue;
        }

        // Build fresh judge request with current filtered candidates.
        let session_context = brouter_router_models::JudgeSessionContext::default();
        let judge_user_prompt = build_judge_prompt(
            &decision.features.original_prompt,
            &format!("{:?}", decision.features.intent),
            decision.objective,
            &active_candidates,
            &session_context,
        );

        // Structured logging: judge invocation
        tracing::info!(
            component = "judge",
            judge_model = %candidate_id,
            provider = %provider_id,
            candidate_count = active_candidates.len(),
            candidate_ids = ?active_candidates.iter().map(|c| c.model_id.as_str()).collect::<Vec<_>>(),
            prompt_length = judge_user_prompt.len(),
            "judge invocation started"
        );

        // Trace level: full judge prompt for debugging
        tracing::trace!(
            component = "judge",
            judge_model = %candidate_id,
            judge_prompt = %judge_user_prompt,
            "full judge prompt"
        );

        let judge_req = judge_request(judge_config, system_prompt, &judge_user_prompt);

        let judge_model = configured_judge_model.unwrap_or_else(|| RouteableModel {
            id: candidate_id.clone(),
            provider: provider_id.clone(),
            upstream_model: candidate_id.to_string(),
            context_window: 128_000,
            input_cost_per_million: 0.0,
            output_cost_per_million: 0.0,
            quality: 50,
            capabilities: vec![],
            attributes: BTreeMap::new(),
            display_badges: vec![],
            metadata: ResolvedModelMetadata::default(),
        });

        match client
            .chat_completions(&registry, &judge_model, &judge_req)
            .await
        {
            Ok(response) => {
                let body = response.body.to_string();
                // Skip non-JSON responses (e.g. HTML error pages)
                if !body.trim_start().starts_with('{') {
                    tracing_warn!(
                        model_id = %candidate_id,
                        provider = %provider_id,
                        bytes = body.len(),
                        sample = %body.trim().chars().take(300).collect::<String>(),
                        "judge returned non-JSON"
                    );
                    failed_judge_providers.insert(provider_id.clone());
                    filter_candidates_by_failed_providers(
                        &mut active_candidates,
                        &failed_judge_providers,
                    );
                    continue;
                }
                // Skip JSON error responses (e.g. usage limit reached)
                if let Some(err_msg) = extract_api_error(&body) {
                    tracing_warn!(
                        model_id = %candidate_id,
                        provider = %provider_id,
                        error = %err_msg,
                        response = %body.trim().chars().take(500).collect::<String>(),
                        "judge API error"
                    );
                    failed_judge_providers.insert(provider_id.clone());
                    filter_candidates_by_failed_providers(
                        &mut active_candidates,
                        &failed_judge_providers,
                    );
                    continue;
                }
                // Validate response is actually parseable as judge JSON.
                // Use active candidates (filtered by healthy providers).
                let parsed_result = parse_judge_response(&body, &active_candidates, candidate_id);
                if let Some(err) = &parsed_result.error {
                    tracing_warn!(
                        model_id = %candidate_id,
                        provider = %provider_id,
                        parse_error = %err,
                        sample = %body.trim().chars().take(500).collect::<String>(),
                        "judge parse failure"
                    );
                    failed_judge_providers.insert(provider_id.clone());
                    filter_candidates_by_failed_providers(
                        &mut active_candidates,
                        &failed_judge_providers,
                    );
                    continue;
                }
                // Accept this valid response
                info!(
                    model_id = %candidate_id,
                    provider = %provider_id,
                    response_length = body.len(),
                    "judge succeeded"
                );

                // Trace level: full judge response for debugging
                tracing::trace!(
                    component = "judge",
                    judge_model = %candidate_id,
                    judge_response = %body,
                    "full judge response"
                );

                raw_text = body;
                used_judge_id = candidate_id;
                break;
            }
            Err(e) => {
                tracing_warn!(
                    model_id = %candidate_id,
                    provider = %provider_id,
                    error = %e,
                    "judge unavailable"
                );
                failed_judge_providers.insert(provider_id.clone());
                filter_candidates_by_failed_providers(
                    &mut active_candidates,
                    &failed_judge_providers,
                );
            }
        }
    }

    // Final validation using filtered candidates (judge was prompted with active models).
    let reasoning = if raw_text.is_empty() {
        tracing_warn!(
            component = "judge",
            failed_providers = ?failed_judge_providers.iter().map(ProviderId::as_str).collect::<Vec<_>>(),
            "all judge models failed, using original decision"
        );
        return Ok(decision.clone());
    } else {
        parse_judge_response(&raw_text, &active_candidates, used_judge_id)
    };

    let mut updated = decision.clone();
    updated.reasoning = Some(reasoning);
    if updated.reasoning.as_ref().is_some_and(|r| r.overridden) {
        updated.selected_model = updated.reasoning.as_ref().unwrap().chosen_model.clone();
    }

    // Structured logging: final routing decision
    tracing::info!(
        component = "router",
        objective = ?decision.objective,
        intent = ?decision.features.intent,
        candidates_count = decision.candidates.len(),
        selected_model = %updated.selected_model,
        judge_overridden = %updated.reasoning.as_ref().is_some_and(|r| r.overridden),
        "routing decision made"
    );

    Ok(updated)
}

/// Filters candidates by removing any whose provider is in the failed set.
fn filter_candidates_by_failed_providers(
    candidates: &mut Vec<ScoredCandidate>,
    failed_providers: &BTreeSet<ProviderId>,
) {
    candidates.retain(|c| {
        let provider_id = ProviderId::new(&c.provider);
        !failed_providers.contains(&provider_id)
    });
}

/// Returns the API error message from a judge response if it looks like an API error.
fn extract_api_error(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<ApiErrorResponse>(trimmed)
        && !parsed.error.message.is_empty()
    {
        return Some(parsed.error.message);
    }
    None
}

/// API error response shape for detecting provider failures.
#[derive(serde::Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(serde::Deserialize)]
struct ApiErrorDetail {
    message: String,
}

fn prompt_request(prompt: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "brouter/auto".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(prompt.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            extra: BTreeMap::new(),
        }],
        temperature: None,
        top_p: None,
        max_tokens: None,
        reasoning_effort: None,
        stream: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        metadata: None,
        extra: BTreeMap::new(),
    }
}

#[derive(Debug, Parser)]
#[command(name = "brouter", version)]
#[command(about = "Local OpenAI-compatible LLM router")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum AuthCommand {
    /// Manages ChatGPT/Codex Max subscription credentials.
    OpenaiCodex {
        #[command(subcommand)]
        command: OpenaiCodexAuthCommand,
    },
    /// Manages `OpenRouter` credentials.
    Openrouter {
        #[command(subcommand)]
        command: OpenrouterAuthCommand,
    },
    /// Manages `OpenCode` Zen credentials.
    OpencodeZen {
        #[command(subcommand)]
        command: OpencodeZenAuthCommand,
    },
    /// Manages `OpenAI` direct API credentials.
    Openaicom {
        #[command(subcommand)]
        command: OpenaicomAuthCommand,
    },
}

#[derive(Debug, Subcommand)]
enum OpenaiCodexAuthCommand {
    /// Runs browser login and stores tokens in brouter's sshenv vault.
    Login {
        /// sshenv profile to write.
        #[arg(long, default_value = "openai-max")]
        profile: String,
        /// sshenv vault path to use.
        #[arg(long, default_value_os_t = default_auth_vault_path())]
        vault: PathBuf,
        /// SSH public key or public-key file used when initializing a new vault.
        #[arg(long)]
        recipient_key: Option<String>,
        /// Use Codex device-code auth instead of browser OAuth.
        #[arg(long)]
        headless: bool,
    },
}

#[derive(Debug, Subcommand)]
enum OpenrouterAuthCommand {
    /// Stores an `OpenRouter` API key in the sshenv vault.
    Login {
        /// sshenv profile to write.
        #[arg(long, default_value = "openrouter")]
        profile: String,
        /// sshenv vault path to use.
        #[arg(long, default_value_os_t = default_auth_vault_path())]
        vault: PathBuf,
        /// SSH public key or public-key file used when initializing a new vault.
        #[arg(long)]
        recipient_key: Option<String>,
        /// `OpenRouter` API key (leave blank to interactively paste or read from env).
        #[arg(long)]
        api_key: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum OpencodeZenAuthCommand {
    /// Stores an `OpenCode` Zen API key in the sshenv vault.
    Login {
        /// sshenv profile to write.
        #[arg(long, default_value = "opencode-zen")]
        profile: String,
        /// sshenv vault path to use.
        #[arg(long, default_value_os_t = default_auth_vault_path())]
        vault: PathBuf,
        /// SSH public key or public-key file used when initializing a new vault.
        #[arg(long)]
        recipient_key: Option<String>,
        /// `OpenCode` Zen API key (leave blank to interactively paste or read from env).
        #[arg(long)]
        api_key: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum OpenaicomAuthCommand {
    /// Stores an `OpenAI` API key in the sshenv vault.
    Login {
        /// sshenv profile to write.
        #[arg(long, default_value = "openai")]
        profile: String,
        /// sshenv vault path to use.
        #[arg(long, default_value_os_t = default_auth_vault_path())]
        vault: PathBuf,
        /// SSH public key or public-key file used when initializing a new vault.
        #[arg(long)]
        recipient_key: Option<String>,
        /// `OpenAI` API key (leave blank to interactively paste or read from env).
        #[arg(long)]
        api_key: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum SessionsCommand {
    /// Lists sessions that have brouter routing events.
    List {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Shows a single session summary.
    Show {
        /// Session identifier.
        session_id: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Shows the routing timeline for a session.
    Timeline {
        /// Session identifier.
        session_id: String,
        /// Maximum number of events to show.
        #[arg(long)]
        last: Option<usize>,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Shows a single routing event.
    Event {
        /// Event identifier.
        event_id: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Starts the local HTTP router service.
    Serve {
        /// Path to the brouter TOML config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Override the server port (overrides config file setting).
        #[arg(long, short = 'p')]
        port: Option<u16>,
    },
    /// Validates the brouter TOML config.
    CheckConfig {
        /// Path to the brouter TOML config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Treat validation warnings as errors.
        #[arg(long)]
        strict: bool,
        /// Emit machine-readable JSON diagnostics.
        #[arg(long)]
        json: bool,
    },
    /// Prints the example TOML config to stdout.
    PrintExampleConfig,
    /// Manages brouter-owned auth credentials.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// Validates config and checks provider reachability.
    Doctor {
        /// Path to the brouter TOML config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Emit machine-readable JSON diagnostics.
        #[arg(long)]
        json: bool,
    },
    /// Inspects session-scoped routing event logs.
    Sessions {
        /// Path to the brouter TOML config.
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(subcommand)]
        command: SessionsCommand,
    },
    /// Explains how a prompt would be routed.
    Route {
        /// Path to the brouter TOML config.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Prompt text to classify and route.
        prompt: String,
        /// Treat the prompt as the first message in a session.
        #[arg(long, default_value_t = true)]
        first_message: bool,
        /// Override the configured routing objective.
        #[arg(long)]
        objective: Option<String>,
        /// Force LLM judge re-evaluation regardless of trigger settings.
        #[arg(long)]
        judge: bool,
    },
}
