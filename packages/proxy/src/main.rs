//! Brouter Proxy - Lightweight forwarding proxy with observability
//!
//! Accepts requests from clients (pi, opencode, etc.) and forwards to any
//! LLM backend (brouter, `OpenAI`, etc.) while adding trace IDs and structured logging.
//!
//! Configuration via environment variables:
//! - `PROXY_PORT` - Port to listen on (default: 8581)
//! - `PROXY_BACKEND` - Backend URL to forward to (e.g., `http://127.0.0.1:8582`)

#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

use axum::{
    Router, body::Body, extract::State, http::Request as HttpRequest, response::Response,
    routing::any,
};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::oneshot;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use uuid::Uuid;

/// Default port if not specified via `PROXY_PORT`
const DEFAULT_PORT: u16 = 8581;

/// Default backend if not specified via `PROXY_BACKEND`
const DEFAULT_BACKEND: &str = "http://127.0.0.1:8582";

/// Default brouter backend port used by `up`.
const DEFAULT_BACKEND_PORT: u16 = 8582;

#[derive(Debug, Parser)]
#[command(name = "brouter-proxy", version)]
#[command(about = "Local forwarding proxy for brouter-compatible LLM APIs")]
struct Cli {
    #[command(subcommand)]
    command: Option<ProxyCommand>,
}

#[derive(Debug, Subcommand)]
enum ProxyCommand {
    /// Starts only the forwarding proxy.
    Serve {
        /// Port for the proxy to listen on.
        #[arg(long, short = 'p')]
        port: Option<u16>,
        /// Backend URL to forward requests to.
        #[arg(long)]
        backend: Option<String>,
    },
    /// Builds and starts brouter plus the proxy in one multiplexed process.
    Up {
        /// Path to the brouter TOML config passed to the backend.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Port for the brouter backend to listen on.
        #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
        backend_port: u16,
        /// Port for the proxy to listen on.
        #[arg(long, default_value_t = DEFAULT_PORT)]
        proxy_port: u16,
        /// Skip the initial backend binary build step.
        #[arg(long)]
        no_build: bool,
    },
}

#[derive(Debug, Clone)]
struct ProxyOptions {
    port: u16,
    backend_url: String,
}

impl ProxyOptions {
    fn from_env() -> Self {
        let port = std::env::var("PROXY_PORT")
            .unwrap_or_else(|_| DEFAULT_PORT.to_string())
            .parse()
            .unwrap_or(DEFAULT_PORT);
        let backend_url =
            std::env::var("PROXY_BACKEND").unwrap_or_else(|_| DEFAULT_BACKEND.to_string());
        Self { port, backend_url }
    }

    fn with_overrides(port: Option<u16>, backend_url: Option<String>) -> Self {
        let mut options = Self::from_env();
        if let Some(port) = port {
            options.port = port;
        }
        if let Some(backend_url) = backend_url {
            options.backend_url = backend_url;
        }
        options
    }
}

#[derive(Clone)]
struct ProxyState {
    backend_url: String,
    client: reqwest::Client,
}

impl ProxyState {
    fn new(backend_url: String) -> Self {
        info!(backend_url = %backend_url, "proxy configuration");

        Self {
            backend_url,
            client: reqwest::Client::builder()
                .build()
                .expect("failed to create HTTP client"),
        }
    }
}

async fn proxy_handler(State(state): State<ProxyState>, request: HttpRequest<Body>) -> Response {
    #![allow(clippy::too_many_lines)]
    let request_id = Uuid::new_v4().to_string();

    // Get the full path including any path segments
    let path = request.uri().path().to_string();
    let query = request.uri().query().map(String::from);

    // Get method and headers before consuming request
    let method = request.method().clone();
    let headers: Vec<_> = request
        .headers()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Build the backend path
    let backend_path = format!("{}{}", state.backend_url.trim_end_matches('/'), path);
    let backend_url = if let Some(q) = &query {
        format!("{backend_path}?{q}")
    } else {
        backend_path
    };

    // Extract headers for logging
    let content_type = request
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");

    let content_length = request
        .headers()
        .get(http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let request_session_headers = diagnostic_headers(
        request.headers(),
        &[
            "x-brouter-session",
            "session_id",
            "x-session-affinity",
            "x-session-id",
            "x-request-session-id",
            "x-client-request-id",
        ],
    );

    info!(
        request_id = %request_id,
        method = %method,
        path = %path,
        content_type = %content_type,
        content_length = ?content_length,
        backend_url = %backend_url,
        session_headers = ?request_session_headers,
        "forwarding request to backend"
    );

    // Convert the request body to bytes
    let body_bytes = match axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!(request_id = %request_id, error = %e, "failed to read request body");
            return Response::builder()
                .status(http::StatusCode::BAD_REQUEST)
                .body(Body::from("Failed to read request body"))
                .unwrap();
        }
    };

    // Build the backend request
    let mut req_builder = match method.as_str() {
        "GET" => state.client.get(&backend_url),
        "POST" => state.client.post(&backend_url),
        "PUT" => state.client.put(&backend_url),
        "DELETE" => state.client.delete(&backend_url),
        "PATCH" => state.client.patch(&backend_url),
        _ => {
            warn!(request_id = %request_id, method = %method, "unsupported HTTP method");
            return Response::builder()
                .status(http::StatusCode::METHOD_NOT_ALLOWED)
                .body(Body::from("Method not supported"))
                .unwrap();
        }
    };

    // Forward all headers except hop-by-hop ones
    let hop_by_hop = [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    ];

    for (key, value) in &headers {
        let key_str = key.as_str().to_lowercase();
        if hop_by_hop.contains(&key_str.as_str()) {
            continue;
        }
        req_builder = req_builder.header(key.as_str(), value.as_bytes());
    }

    // Add X-Request-ID for correlation
    req_builder = req_builder.header("X-Request-ID", request_id.as_str());

    // Send the request
    let backend_response = req_builder.body(body_bytes).send().await;

    match backend_response {
        Ok(response) => {
            let status = response.status();
            let response_headers: Vec<_> = response
                .headers()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            // Check if streaming response
            let is_streaming = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.contains("event-stream") || v.contains("stream"));

            let response_context_headers = diagnostic_headers(
                response.headers(),
                &[
                    "x-brouter-session",
                    "x-brouter-session-source",
                    "x-brouter-context-window",
                    "x-brouter-context-tokens",
                    "x-brouter-context-percent",
                    "x-brouter-context-source",
                ],
            );

            info!(
                request_id = %request_id,
                status = %status,
                streaming = %is_streaming,
                context_headers = ?response_context_headers,
                "forwarding response from backend"
            );

            // Build the response
            let mut builder = Response::builder().status(status);

            // Forward relevant response headers
            for (key, value) in &response_headers {
                let key_str = key.as_str().to_lowercase();
                if hop_by_hop.contains(&key_str.as_str()) {
                    continue;
                }
                builder = builder.header(key.as_str(), value.as_bytes());
            }

            if is_streaming {
                // Stream the response body directly
                let stream = response.bytes_stream();
                let body = Body::from_stream(stream);
                builder.body(body).unwrap()
            } else {
                // Buffer non-streaming responses
                let body_bytes = response.bytes().await.unwrap_or_default();
                builder.body(Body::from(body_bytes)).unwrap()
            }
        }
        Err(e) => {
            warn!(request_id = %request_id, error = %e, "backend request failed");
            Response::builder()
                .status(http::StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Backend error: {e}")))
                .unwrap()
        }
    }
}

fn diagnostic_headers(headers: &http::HeaderMap, names: &[&str]) -> Vec<String> {
    names
        .iter()
        .filter_map(|name| {
            headers
                .get(*name)
                .and_then(|value| value.to_str().ok())
                .map(|value| format!("{name}={value}"))
        })
        .collect()
}

async fn run_proxy(
    options: ProxyOptions,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let state = ProxyState::new(options.backend_url);

    info!(
        port = options.port,
        backend = %state.backend_url,
        "starting brouter-proxy"
    );

    let app = Router::new()
        .route("/v1/{*path}", any(proxy_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], options.port));

    info!(addr = %addr, "brouter-proxy listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn run_up(
    config: Option<PathBuf>,
    backend_port: u16,
    proxy_port: u16,
    no_build: bool,
) -> anyhow::Result<()> {
    if !no_build {
        build_backend().await?;
    }

    let mut backend = spawn_backend(config, backend_port, no_build)?;
    if let Some(stdout) = backend.stdout.take() {
        spawn_output_task(stdout, "backend stdout");
    }
    if let Some(stderr) = backend.stderr.take() {
        spawn_output_task(stderr, "backend stderr");
    }

    let backend_task = tokio::spawn(async move { backend.wait().await });
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let proxy_options = ProxyOptions {
        port: proxy_port,
        backend_url: format!("http://127.0.0.1:{backend_port}"),
    };

    info!(
        backend_port = backend_port,
        proxy_port = proxy_port,
        "brouter backend and proxy are starting"
    );

    tokio::select! {
        status = backend_task => {
            let _ = shutdown_tx.send(());
            let status = status??;
            if status.success() {
                info!(%status, "brouter backend exited");
                Ok(())
            } else {
                anyhow::bail!("brouter backend exited with {status}");
            }
        }
        result = run_proxy(proxy_options, async move {
            let _ = shutdown_rx.await;
        }) => result,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            info!("received shutdown signal; stopping brouter backend and proxy");
            let _ = shutdown_tx.send(());
            Ok(())
        }
    }
}

async fn build_backend() -> anyhow::Result<()> {
    info!("building brouter backend");
    let status = Command::new("cargo")
        .args(["build", "--package", "brouter_cli", "--bin", "brouter"])
        .status()
        .await?;
    if !status.success() {
        anyhow::bail!("backend build failed with {status}");
    }
    Ok(())
}

fn spawn_backend(
    config: Option<PathBuf>,
    backend_port: u16,
    use_cargo_run: bool,
) -> anyhow::Result<Child> {
    let mut command = if use_cargo_run {
        let mut command = Command::new("cargo");
        command.args(["run", "--package", "brouter_cli", "--bin", "brouter", "--"]);
        command
    } else {
        Command::new(resolve_backend_binary())
    };

    command.args(["serve", "--port", &backend_port.to_string()]);
    if let Some(config) = config {
        command.arg("--config").arg(config);
    }
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(Into::into)
}

fn resolve_backend_binary() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("brouter")))
        .filter(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("brouter"))
}

fn spawn_output_task<R>(reader: R, label: &'static str)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => info!(target: "brouter_proxy::child", %label, "{line}"),
                Ok(None) => break,
                Err(error) => {
                    warn!(%label, %error, "failed to read child process output");
                    break;
                }
            }
        }
    });
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command.unwrap_or(ProxyCommand::Serve {
        port: None,
        backend: None,
    }) {
        ProxyCommand::Serve { port, backend } => {
            let options = ProxyOptions::with_overrides(port, backend);
            run_proxy(options, async {
                if let Err(error) = tokio::signal::ctrl_c().await {
                    warn!(%error, "failed to wait for shutdown signal");
                }
            })
            .await
        }
        ProxyCommand::Up {
            config,
            backend_port,
            proxy_port,
            no_build,
        } => run_up(config, backend_port, proxy_port, no_build).await,
    }
}
