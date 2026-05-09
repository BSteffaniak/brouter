#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Command-line interface for brouter.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent};
use brouter_config::{
    apply_default_config, load_config, routeable_models, routing_rules, scoring_weights,
    validate_config_warnings,
};
use brouter_config_models::{BrouterConfig, ProviderConfig, ProviderKind};
use brouter_router::Router;
use brouter_router_models::RoutingObjective;
use clap::{Parser, Subcommand};
use rand::TryRngCore as _;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use zeroize::Zeroizing;

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
        Command::CheckConfig {
            config,
            strict,
            json,
        } => check_config(&config, strict, json),
        Command::PrintExampleConfig => {
            print_example_config();
            Ok(())
        }
        Command::Auth { command } => auth(command).await,
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
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if std::env::var("BROUTER_LOG_FORMAT").is_ok_and(|value| value == "json") {
        let _ = tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    }
}

async fn serve(config: &Path) -> Result<()> {
    let config = if config.exists() {
        load_config(config)?
    } else {
        let mut config = BrouterConfig::default();
        apply_default_config(&mut config);
        config
    };
    brouter_server::serve(config).await?;
    Ok(())
}

fn check_config(config: &Path, strict: bool, json_output: bool) -> Result<()> {
    let config = load_config(config)?;
    let warnings = validate_config_warnings(&config);
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
    rand::rngs::OsRng.try_fill_bytes(&mut data)?;
    Ok(URL_SAFE_NO_PAD.encode(data))
}

fn random_pkce_verifier(length: usize) -> Result<String> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut data = vec![0_u8; length];
    rand::rngs::OsRng.try_fill_bytes(&mut data)?;
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
