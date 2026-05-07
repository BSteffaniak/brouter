#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Command-line interface for brouter.

use std::path::{Path, PathBuf};

use anyhow::Result;
use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent};
use brouter_config::{load_config, routeable_models, scoring_weights};
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
        Command::CheckConfig { config } => check_config(&config),
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

fn check_config(config: &Path) -> Result<()> {
    let config = load_config(config)?;
    println!(
        "config ok: {} providers, {} models",
        config.providers.len(),
        config.models.len()
    );
    Ok(())
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
    let router = Router::new_with_scoring(
        routeable_models(&config),
        objective,
        scoring_weights(&config),
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
