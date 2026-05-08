#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Command-line interface for brouter.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, bail};
use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent};
use brouter_config::{
    load_config, routeable_models, routing_rules, scoring_weights, validate_config_warnings,
};
use brouter_config_models::{BrouterConfig, ProviderConfig, ProviderKind};
use brouter_router::Router;
use brouter_router_models::RoutingObjective;
use clap::{Parser, Subcommand};

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
        Command::Serve { config } => serve(&config).await,
        Command::CheckConfig { config, strict } => check_config(&config, strict),
        Command::Doctor { config } => doctor(&config).await,
        Command::Route {
            config,
            prompt,
            first_message,
            objective,
        } => explain_route(&config, &prompt, first_message, objective.as_deref()),
    }
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

async fn serve(config: &Path) -> Result<()> {
    let config = load_config(config)?;
    brouter_server::serve(config).await?;
    Ok(())
}

fn check_config(config: &Path, strict: bool) -> Result<()> {
    let config = load_config(config)?;
    let warnings = validate_config_warnings(&config);
    println!(
        "config ok: {} providers, {} models, {} warnings",
        config.providers.len(),
        config.models.len(),
        warnings.len()
    );
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    if strict && !warnings.is_empty() {
        bail!(
            "strict config check failed with {} warnings",
            warnings.len()
        );
    }
    Ok(())
}

async fn doctor(config: &Path) -> Result<()> {
    let config = load_config(config)?;
    let warnings = validate_config_warnings(&config);
    println!(
        "config ok: {} providers, {} models, {} warnings",
        config.providers.len(),
        config.models.len(),
        warnings.len()
    );
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let mut failures = 0_u32;
    for (provider_id, provider) in &config.providers {
        match check_provider(&client, &config, provider_id, provider).await {
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
    let base_url = provider.base_url.as_deref().unwrap_or(match provider.kind {
        ProviderKind::OpenAiCompatible => bail!("missing base_url"),
        ProviderKind::Anthropic => "https://api.anthropic.com/v1",
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

fn explain_route(
    config: &Path,
    prompt: &str,
    first_message: bool,
    objective: Option<&str>,
) -> Result<()> {
    let config = load_config(config)?;
    let objective = objective.map_or_else(
        || RoutingObjective::from_name(&config.router.default_objective),
        RoutingObjective::from_name,
    );
    let router = Router::new_with_rules(
        routeable_models(&config),
        objective,
        scoring_weights(&config),
        routing_rules(&config),
    );
    let decision = router.route_chat(&prompt_request(prompt), first_message)?;
    println!("{}", serde_json::to_string_pretty(&decision)?);
    Ok(())
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
        }],
        temperature: None,
        top_p: None,
        max_tokens: None,
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
enum Command {
    /// Starts the local HTTP router service.
    Serve {
        /// Path to the brouter TOML config.
        #[arg(long, default_value = "brouter.toml")]
        config: PathBuf,
    },
    /// Validates the brouter TOML config.
    CheckConfig {
        /// Path to the brouter TOML config.
        #[arg(long, default_value = "brouter.toml")]
        config: PathBuf,
        /// Treat validation warnings as errors.
        #[arg(long)]
        strict: bool,
    },
    /// Validates config and checks provider reachability.
    Doctor {
        /// Path to the brouter TOML config.
        #[arg(long, default_value = "brouter.toml")]
        config: PathBuf,
    },
    /// Explains how a prompt would be routed.
    Route {
        /// Path to the brouter TOML config.
        #[arg(long, default_value = "brouter.toml")]
        config: PathBuf,
        /// Prompt text to classify and route.
        prompt: String,
        /// Treat the prompt as the first message in a session.
        #[arg(long, default_value_t = true)]
        first_message: bool,
        /// Override the configured routing objective.
        #[arg(long)]
        objective: Option<String>,
    },
}
