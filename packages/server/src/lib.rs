#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! HTTP server for brouter.

use std::net::SocketAddr;
use std::path::Path;

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
                    &candidate.model_id,
                    candidate.estimated_cost,
                    status.is_success(),
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
                if let Err(telemetry_error) = record_model_attempt(
                    state,
                    headers,
                    request,
                    &candidate.model_id,
                    candidate.estimated_cost,
                    false,
                )
                .await
                {
                    return telemetry_error_response(&telemetry_error).into_response();
                }
                last_error = Some(error.to_string());
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
                    &candidate.model_id,
                    candidate.estimated_cost,
                    status.is_success(),
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
                if let Err(telemetry_error) = record_model_attempt(
                    state,
                    headers,
                    request,
                    &candidate.model_id,
                    candidate.estimated_cost,
                    false,
                )
                .await
                {
                    return telemetry_error_response(&telemetry_error).into_response();
                }
                last_error = Some(error.to_string());
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

async fn record_model_attempt(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    model_id: &ModelId,
    estimated_cost: f64,
    success: bool,
) -> Result<(), TelemetryError> {
    state
        .telemetry
        .record(&UsageEvent {
            timestamp_ms: now_millis(),
            session_id: session_id(headers, request),
            selected_model: ModelId::new(model_id.as_str()),
            estimated_cost,
            latency_ms: None,
            success,
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
