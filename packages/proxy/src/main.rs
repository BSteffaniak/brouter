//! Brouter Proxy - Lightweight forwarding proxy with observability
//!
//! Accepts requests from clients (pi, opencode, etc.) and forwards to any
//! LLM backend (brouter, `OpenAI`, etc.) while adding trace IDs and structured logging.
//!
//! Configuration via environment variables:
//! - `PROXY_PORT` - Port to listen on (default: 8081)
//! - `PROXY_BACKEND` - Backend URL to forward to (e.g., `http://127.0.0.1:8080`)

#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

use axum::{
    Router, body::Body, extract::State, http::Request as HttpRequest, response::Response,
    routing::any,
};
use std::net::SocketAddr;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use uuid::Uuid;

/// Default port if not specified via `PROXY_PORT`
const DEFAULT_PORT: u16 = 8081;

/// Default backend if not specified via `PROXY_BACKEND`
const DEFAULT_BACKEND: &str = "http://127.0.0.1:8080";

#[derive(Clone)]
struct ProxyState {
    backend_url: String,
    client: reqwest::Client,
}

impl ProxyState {
    fn new() -> Self {
        let backend_url =
            std::env::var("PROXY_BACKEND").unwrap_or_else(|_| DEFAULT_BACKEND.to_string());

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

    info!(
        request_id = %request_id,
        method = %method,
        path = %path,
        content_type = %content_type,
        content_length = ?content_length,
        backend_url = %backend_url,
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

            info!(
                request_id = %request_id,
                status = %status,
                streaming = %is_streaming,
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

#[tokio::main]
async fn main() {
    // Initialize tracing
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(filter)
        .init();

    let state = ProxyState::new();

    let port: u16 = std::env::var("PROXY_PORT")
        .unwrap_or_else(|_| DEFAULT_PORT.to_string())
        .parse()
        .unwrap_or(DEFAULT_PORT);

    info!(
        port = port,
        backend = %state.backend_url,
        "starting brouter-proxy"
    );

    let app = Router::new()
        .route("/v1/{*path}", any(proxy_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    info!(addr = %addr, "brouter-proxy listening");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
