#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! HTTP server for brouter.

use std::net::SocketAddr;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use brouter_api_models::{ChatCompletionRequest, ErrorResponse, ModelListResponse, ModelObject};
use brouter_config::{ConfigError, routeable_models};
use brouter_config_models::BrouterConfig;
use brouter_provider::{ProviderClient, ProviderError, ProviderRegistry};
use brouter_provider_models::{ModelId, RouteableModel};
use brouter_router::{Router, RouterError};
use brouter_router_models::{RoutingDecision, RoutingObjective};
use brouter_telemetry::TelemetryStore;
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
    let app = build_app(&config);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;

    tracing::info!(%address, "starting brouter server");
    axum::serve(listener, app).await.map_err(ServerError::Serve)
}

fn build_app(config: &BrouterConfig) -> AxumRouter {
    let objective = RoutingObjective::from_name(&config.router.default_objective);
    let router = Router::new(routeable_models(config), objective);
    let state = AppState {
        router,
        providers: ProviderRegistry::from_config(config),
        provider_client: ProviderClient::new(),
        telemetry: TelemetryStore::default(),
    };

    AxumRouter::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/brouter/route/explain", post(route_explain))
        .route("/v1/brouter/usage", get(usage))
        .with_state(state)
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
    let decision = route_request(&state, &headers, &request)?;
    Ok(Json(decision))
}

async fn usage(State(state): State<AppState>) -> Json<Vec<UsageEvent>> {
    Json(state.telemetry.events())
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.is_streaming() {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "streaming provider forwarding is not implemented yet",
            "not_implemented",
        )
        .into_response();
    }

    let decision = match route_request(&state, &headers, &request) {
        Ok(decision) => decision,
        Err(error) => return error.into_response(),
    };
    let model = match selected_model(&state, &decision) {
        Ok(model) => model,
        Err(error) => return error.into_response(),
    };

    match state
        .provider_client
        .chat_completions(&state.providers, &model, &request)
        .await
    {
        Ok(provider_response) => {
            let status = provider_status(provider_response.status);
            record_decision(&state, &headers, &request, &decision, status.is_success());
            (status, Json(provider_response.body)).into_response()
        }
        Err(error) => {
            record_decision(&state, &headers, &request, &decision, false);
            provider_error_response(&error).into_response()
        }
    }
}

fn selected_model(
    state: &AppState,
    decision: &RoutingDecision,
) -> Result<RouteableModel, (StatusCode, Json<ErrorResponse>)> {
    state
        .router
        .models()
        .iter()
        .find(|model| model.id == decision.selected_model)
        .cloned()
        .ok_or_else(|| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "selected model is no longer configured",
                "internal_error",
            )
        })
}

fn route_request(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
) -> Result<RoutingDecision, (StatusCode, Json<ErrorResponse>)> {
    let session_id = session_id(headers, request);
    let is_first_message = session_id
        .as_deref()
        .is_none_or(|id| !state.telemetry.has_session(id));
    state
        .router
        .route_chat(request, is_first_message)
        .map_err(|error| router_error_response(&error))
}

fn router_error_response(error: &RouterError) -> (StatusCode, Json<ErrorResponse>) {
    error_response(StatusCode::BAD_REQUEST, error.to_string(), "routing_error")
}

fn provider_error_response(error: &ProviderError) -> (StatusCode, Json<ErrorResponse>) {
    error_response(StatusCode::BAD_GATEWAY, error.to_string(), "provider_error")
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

fn record_decision(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    decision: &RoutingDecision,
    success: bool,
) {
    let estimated_cost = decision
        .candidates
        .iter()
        .find(|candidate| candidate.model_id == decision.selected_model)
        .map_or(0.0, |candidate| candidate.estimated_cost);
    state.telemetry.record(UsageEvent {
        session_id: session_id(headers, request),
        selected_model: ModelId::new(decision.selected_model.as_str()),
        estimated_cost,
        latency_ms: None,
        success,
    });
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
