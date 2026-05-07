#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! HTTP server for brouter.

use std::net::SocketAddr;
use std::path::Path;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use brouter_api_models::{ChatCompletionRequest, ErrorResponse, ModelListResponse, ModelObject};
use brouter_config::{ConfigError, routeable_models, scoring_weights};
use brouter_config_models::BrouterConfig;
use brouter_provider::{ProviderClient, ProviderRegistry};
use brouter_provider_models::{ModelId, RouteableModel};
use brouter_router::{Router, RouterError};
use brouter_router_models::{RoutingDecision, RoutingObjective};
use brouter_telemetry::{TelemetryError, TelemetryStore, now_millis};
use brouter_telemetry_models::UsageEvent;
use serde::Serialize;
use thiserror::Error;

/// HTTP server startup error.
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("invalid bind address {address}: {source}")]
    AddressParse {
        address: String,
        source: std::net::AddrParseError,
    },
    #[error("invalid configuration: {0}")]
    Config(#[from] ConfigError),
    #[error("failed to bind HTTP listener at {address}: {source}")]
    Bind {
        address: SocketAddr,
        source: std::io::Error,
    },
    #[error("telemetry error: {0}")]
    Telemetry(#[from] TelemetryError),
    #[error("HTTP server error: {0}")]
    Serve(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
struct AppState {
    router: Router,
    providers: ProviderRegistry,
    provider_client: ProviderClient,
    telemetry: TelemetryStore,
}

#[derive(Debug, Clone)]
struct AttemptTelemetry {
    model_id: ModelId,
    estimated_cost: f64,
    latency_ms: Option<u64>,
    status_code: Option<u16>,
    provider_error: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    success: bool,
}

/// Starts the brouter HTTP server.
///
/// # Errors
///
/// Returns an error when the configuration is invalid, the bind address cannot
/// be parsed, the TCP listener cannot be created, or the HTTP server exits with
/// an I/O error.
pub async fn serve(config: BrouterConfig) -> Result<(), ServerError> {
    brouter_config::validate_config(&config)?;
    let bind_address = config.server.bind_address();
    let address = parse_bind_address(&bind_address)?;
    let telemetry = telemetry_store(&config).await?;
    let app = build_app(&config, telemetry);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;

    tracing::info!(%address, "starting brouter server");
    axum::serve(listener, app).await.map_err(ServerError::Serve)
}

fn build_app(config: &BrouterConfig, telemetry: TelemetryStore) -> AxumRouter {
    let objective = RoutingObjective::from_name(&config.router.default_objective);
    let router =
        Router::new_with_scoring(routeable_models(config), objective, scoring_weights(config));
    let state = AppState {
        router,
        providers: ProviderRegistry::from_config(config),
        provider_client: ProviderClient::new(),
        telemetry,
    };

    AxumRouter::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/brouter/route/explain", post(route_explain))
        .route("/v1/brouter/usage", get(usage))
        .with_state(state)
}

async fn telemetry_store(config: &BrouterConfig) -> Result<TelemetryStore, TelemetryError> {
    if let Some(path) = &config.telemetry.database_path {
        TelemetryStore::sqlite(Path::new(path)).await
    } else {
        Ok(TelemetryStore::memory())
    }
}

fn parse_bind_address(address: &str) -> Result<SocketAddr, ServerError> {
    address.parse().map_err(|source| ServerError::AddressParse {
        address: address.to_string(),
        source,
    })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { ok: true })
}

async fn models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let mut models = vec![ModelObject::new("brouter/auto", "brouter")];
    models.extend(
        state
            .router
            .models()
            .iter()
            .map(|model| ModelObject::new(model.id.as_str(), model.provider.as_str())),
    );
    Json(ModelListResponse::model_list(models))
}

async fn route_explain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<RoutingDecision>, (StatusCode, Json<ErrorResponse>)> {
    let decision = route_request(&state, &headers, &request).await?;
    Ok(Json(decision))
}

async fn usage(
    State(state): State<AppState>,
) -> Result<Json<Vec<UsageEvent>>, (StatusCode, Json<ErrorResponse>)> {
    state
        .telemetry
        .events()
        .await
        .map(Json)
        .map_err(|error| telemetry_error_response(&error))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let decision = match route_request(&state, &headers, &request).await {
        Ok(decision) => decision,
        Err(error) => return error.into_response(),
    };

    if request.is_streaming() {
        forward_streaming_with_fallbacks(&state, &headers, &request, &decision).await
    } else {
        forward_with_fallbacks(&state, &headers, &request, &decision).await
    }
}

async fn forward_streaming_with_fallbacks(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    decision: &RoutingDecision,
) -> Response {
    let mut last_error = None;
    let mut last_status = None;

    for candidate in &decision.candidates {
        let model = match model_by_id(state, &candidate.model_id) {
            Ok(model) => model,
            Err(error) => return error.into_response(),
        };

        let started_at = Instant::now();
        match state
            .provider_client
            .chat_completions_response(&state.providers, &model, request)
            .await
        {
            Ok(response) => {
                let status = provider_status(response.status().as_u16());
                if let Err(error) = record_model_attempt(
                    state,
                    headers,
                    request,
                    AttemptTelemetry {
                        model_id: candidate.model_id.clone(),
                        estimated_cost: candidate.estimated_cost,
                        latency_ms: Some(elapsed_millis(started_at)),
                        status_code: Some(status.as_u16()),
                        provider_error: None,
                        prompt_tokens: None,
                        completion_tokens: None,
                        success: status.is_success(),
                    },
                )
                .await
                {
                    return telemetry_error_response(&error).into_response();
                }

                if status.is_success() || !should_try_fallback(status) {
                    let mut response_headers = route_headers(&model, &candidate.model_id, decision);
                    response_headers.insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    return (
                        status,
                        response_headers,
                        Body::from_stream(response.bytes_stream()),
                    )
                        .into_response();
                }
                last_status = Some(status);
            }
            Err(error) => {
                let error = error.to_string();
                if let Err(telemetry_error) = record_model_attempt(
                    state,
                    headers,
                    request,
                    AttemptTelemetry {
                        model_id: candidate.model_id.clone(),
                        estimated_cost: candidate.estimated_cost,
                        latency_ms: Some(elapsed_millis(started_at)),
                        status_code: None,
                        provider_error: Some(error.clone()),
                        prompt_tokens: None,
                        completion_tokens: None,
                        success: false,
                    },
                )
                .await
                {
                    return telemetry_error_response(&telemetry_error).into_response();
                }
                last_error = Some(error);
            }
        }
    }

    error_response(
        last_status.unwrap_or(StatusCode::BAD_GATEWAY),
        last_error.unwrap_or_else(|| "all streaming provider attempts failed".to_string()),
        "provider_error",
    )
    .into_response()
}

async fn forward_with_fallbacks(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    decision: &RoutingDecision,
) -> Response {
    let mut last_error = None;
    let mut last_response = None;

    for candidate in &decision.candidates {
        let model = match model_by_id(state, &candidate.model_id) {
            Ok(model) => model,
            Err(error) => return error.into_response(),
        };

        let started_at = Instant::now();
        match state
            .provider_client
            .chat_completions(&state.providers, &model, request)
            .await
        {
            Ok(provider_response) => {
                let status = provider_status(provider_response.status);
                if let Err(error) = record_model_attempt(
                    state,
                    headers,
                    request,
                    AttemptTelemetry {
                        model_id: candidate.model_id.clone(),
                        estimated_cost: candidate.estimated_cost,
                        latency_ms: Some(elapsed_millis(started_at)),
                        status_code: Some(status.as_u16()),
                        provider_error: None,
                        prompt_tokens: prompt_tokens(&provider_response.body),
                        completion_tokens: completion_tokens(&provider_response.body),
                        success: status.is_success(),
                    },
                )
                .await
                {
                    return telemetry_error_response(&error).into_response();
                }

                if status.is_success() || !should_try_fallback(status) {
                    let response_headers = route_headers(&model, &candidate.model_id, decision);
                    return (status, response_headers, Json(provider_response.body))
                        .into_response();
                }
                last_response = Some((status, provider_response.body));
            }
            Err(error) => {
                let error = error.to_string();
                if let Err(telemetry_error) = record_model_attempt(
                    state,
                    headers,
                    request,
                    AttemptTelemetry {
                        model_id: candidate.model_id.clone(),
                        estimated_cost: candidate.estimated_cost,
                        latency_ms: Some(elapsed_millis(started_at)),
                        status_code: None,
                        provider_error: Some(error.clone()),
                        prompt_tokens: None,
                        completion_tokens: None,
                        success: false,
                    },
                )
                .await
                {
                    return telemetry_error_response(&telemetry_error).into_response();
                }
                last_error = Some(error);
            }
        }
    }

    if let Some((status, body)) = last_response {
        return (status, Json(body)).into_response();
    }
    error_response(
        StatusCode::BAD_GATEWAY,
        last_error.unwrap_or_else(|| "all provider attempts failed".to_string()),
        "provider_error",
    )
    .into_response()
}

fn model_by_id(
    state: &AppState,
    model_id: &ModelId,
) -> Result<RouteableModel, (StatusCode, Json<ErrorResponse>)> {
    state
        .router
        .models()
        .iter()
        .find(|model| &model.id == model_id)
        .cloned()
        .ok_or_else(|| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "selected model is no longer configured",
                "internal_error",
            )
        })
}

fn should_try_fallback(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn route_headers(
    model: &RouteableModel,
    attempted_model_id: &ModelId,
    decision: &RoutingDecision,
) -> HeaderMap {
    let mut headers = HeaderMap::new();
    insert_header(&mut headers, "x-brouter-selected-model", model.id.as_str());
    insert_header(&mut headers, "x-brouter-provider", model.provider.as_str());
    insert_header(
        &mut headers,
        "x-brouter-upstream-model",
        &model.upstream_model,
    );
    insert_header(
        &mut headers,
        "x-brouter-fallback-used",
        fallback_used(attempted_model_id, decision),
    );
    headers
}

fn fallback_used(attempted_model_id: &ModelId, decision: &RoutingDecision) -> &'static str {
    if attempted_model_id == &decision.selected_model {
        "false"
    } else {
        "true"
    }
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

async fn route_request(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
) -> Result<RoutingDecision, (StatusCode, Json<ErrorResponse>)> {
    let session_id = session_id(headers, request);
    let has_session = match session_id.as_deref() {
        Some(id) => state
            .telemetry
            .has_session(id)
            .await
            .map_err(|error| telemetry_error_response(&error))?,
        None => false,
    };
    let is_first_message = !has_session;
    state
        .router
        .route_chat(request, is_first_message)
        .map_err(|error| router_error_response(&error))
}

fn router_error_response(error: &RouterError) -> (StatusCode, Json<ErrorResponse>) {
    error_response(StatusCode::BAD_REQUEST, error.to_string(), "routing_error")
}

fn telemetry_error_response(error: &TelemetryError) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        error.to_string(),
        "telemetry_error",
    )
}

fn error_response(
    status: StatusCode,
    message: impl Into<String>,
    error_type: impl Into<String>,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse::new(message, error_type, status.as_u16())),
    )
}

fn provider_status(status: u16) -> StatusCode {
    StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY)
}

fn elapsed_millis(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn prompt_tokens(body: &serde_json::Value) -> Option<u64> {
    body.get("usage")
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
}

fn completion_tokens(body: &serde_json::Value) -> Option<u64> {
    body.get("usage")
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
}

async fn record_model_attempt(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    attempt: AttemptTelemetry,
) -> Result<(), TelemetryError> {
    state
        .telemetry
        .record(&UsageEvent {
            timestamp_ms: now_millis(),
            session_id: session_id(headers, request),
            selected_model: ModelId::new(attempt.model_id.as_str()),
            estimated_cost: attempt.estimated_cost,
            latency_ms: attempt.latency_ms,
            status_code: attempt.status_code,
            provider_error: attempt.provider_error,
            prompt_tokens: attempt.prompt_tokens,
            completion_tokens: attempt.completion_tokens,
            success: attempt.success,
        })
        .await
}

fn session_id(headers: &HeaderMap, request: &ChatCompletionRequest) -> Option<String> {
    headers
        .get("x-brouter-session")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
        .or_else(|| metadata_session_id(request))
}

fn metadata_session_id(request: &ChatCompletionRequest) -> Option<String> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("session_id"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
}

#[derive(Debug, Clone, Serialize)]
struct HealthResponse {
    ok: bool,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::net::SocketAddr;

    use axum::body::{Body, to_bytes};
    use axum::http::{Method, Request};
    use axum::routing::post;
    use axum::{Json, Router as AxumRouter};
    use brouter_config_models::{ModelConfig, ProviderConfig, ProviderKind};
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn chat_completion_falls_back_and_sets_route_headers() {
        let failing_upstream = spawn_status_upstream(StatusCode::INTERNAL_SERVER_ERROR).await;
        let healthy_upstream = spawn_echo_upstream().await;
        let config = fallback_config(failing_upstream, healthy_upstream);
        let app = build_app(&config, TelemetryStore::memory());

        let response = app
            .clone()
            .oneshot(chat_request("debug this Rust error", false))
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("x-brouter-selected-model"),
            Some(&HeaderValue::from_static("cheap_cloud"))
        );
        assert_eq!(
            response.headers().get("x-brouter-fallback-used"),
            Some(&HeaderValue::from_static("true"))
        );
        let body = response_json(response).await;
        assert_eq!(body["model"], "cheap-upstream");

        let usage_response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/brouter/usage")
                    .body(Body::empty())
                    .expect("usage request should build"),
            )
            .await
            .expect("usage request should complete");
        let usage = response_json(usage_response).await;
        assert_eq!(usage.as_array().map_or(0, Vec::len), 2);
    }

    #[tokio::test]
    async fn streaming_completion_sets_route_headers() {
        let upstream = spawn_streaming_upstream().await;
        let config = single_provider_config(upstream);
        let app = build_app(&config, TelemetryStore::memory());

        let response = app
            .oneshot(chat_request("hello", true))
            .await
            .expect("streaming request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/event-stream"))
        );
        assert_eq!(
            response.headers().get("x-brouter-selected-model"),
            Some(&HeaderValue::from_static("cheap_cloud"))
        );
        let bytes = to_bytes(response.into_body(), 1024)
            .await
            .expect("stream body should read");
        let body = String::from_utf8(bytes.to_vec()).expect("stream should be utf8");
        assert!(body.contains("[DONE]"));
    }

    async fn spawn_echo_upstream() -> String {
        async fn echo(Json(body): Json<Value>) -> Json<Value> {
            Json(json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": body["model"].clone(),
                "choices": []
            }))
        }
        spawn_upstream(AxumRouter::new().route("/v1/chat/completions", post(echo))).await
    }

    async fn spawn_streaming_upstream() -> String {
        async fn stream() -> (HeaderMap, &'static str) {
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            (headers, "data: {\"choices\":[]}\n\ndata: [DONE]\n\n")
        }
        spawn_upstream(AxumRouter::new().route("/v1/chat/completions", post(stream))).await
    }

    async fn spawn_status_upstream(status: StatusCode) -> String {
        async fn fail() -> StatusCode {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        spawn_upstream(AxumRouter::new().route("/v1/chat/completions", post(fail))).await
    }

    async fn spawn_upstream(app: AxumRouter) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let address = listener.local_addr().expect("address should be available");
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("fake upstream should serve");
        });
        base_url(address)
    }

    fn base_url(address: SocketAddr) -> String {
        format!("http://{address}/v1")
    }

    async fn response_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), 4096)
            .await
            .expect("response body should read");
        serde_json::from_slice(&bytes).expect("response body should be json")
    }

    fn chat_request(prompt: &str, stream: bool) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "brouter/auto",
                    "stream": stream,
                    "messages": [{"role": "user", "content": prompt}],
                    "metadata": {"session_id": "test-session"}
                })
                .to_string(),
            ))
            .expect("chat request should build")
    }

    fn single_provider_config(base_url: String) -> BrouterConfig {
        let mut config = BrouterConfig::default();
        config.providers.insert(
            "healthy".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some(base_url),
                api_key_env: None,
            },
        );
        config.models.insert(
            "cheap_cloud".to_string(),
            ModelConfig {
                provider: "healthy".to_string(),
                model: "cheap-upstream".to_string(),
                context_window: 128_000,
                input_cost_per_million: 0.15,
                output_cost_per_million: 0.60,
                quality: Some(70),
                capabilities: vec!["chat".to_string()],
            },
        );
        config
    }

    fn fallback_config(failing_base_url: String, healthy_base_url: String) -> BrouterConfig {
        let mut providers = BTreeMap::new();
        providers.insert(
            "failing".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some(failing_base_url),
                api_key_env: None,
            },
        );
        providers.insert(
            "healthy".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some(healthy_base_url),
                api_key_env: None,
            },
        );

        let mut models = BTreeMap::new();
        models.insert(
            "strong_cloud".to_string(),
            ModelConfig {
                provider: "failing".to_string(),
                model: "strong-upstream".to_string(),
                context_window: 128_000,
                input_cost_per_million: 2.0,
                output_cost_per_million: 8.0,
                quality: Some(90),
                capabilities: vec!["chat".to_string(), "code".to_string()],
            },
        );
        models.insert(
            "cheap_cloud".to_string(),
            ModelConfig {
                provider: "healthy".to_string(),
                model: "cheap-upstream".to_string(),
                context_window: 128_000,
                input_cost_per_million: 0.15,
                output_cost_per_million: 0.60,
                quality: Some(70),
                capabilities: vec!["chat".to_string(), "code".to_string()],
            },
        );

        BrouterConfig {
            providers,
            models,
            ..BrouterConfig::default()
        }
    }
}
