#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! HTTP server for brouter.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use brouter_api_models::{
    ChatCompletionRequest, EmbeddingsRequest, ErrorResponse, ModelListResponse, ModelObject,
};
use brouter_catalog_models::{MetadataProvenance, MetadataSource};
use brouter_config::{
    ConfigError, context_policy, routeable_models_with_introspection, routing_profiles,
    routing_rules, scoring_weights,
};
use brouter_config_models::BrouterConfig;
use brouter_introspection::{DynamicPolicyConfig, dynamic_policy_effects};
use brouter_introspection_models::{
    AccountSnapshot, AccountStatus, DynamicPolicyEffect, IntrospectionRequest,
    IntrospectionSnapshot, ResourceKind, ResourcePool, ResourceScope, ResourceSelector,
    ResourceUnit, SnapshotSource, SnapshotSourceKind,
};
use brouter_provider::{ProviderClient, ProviderRegistry};
use brouter_provider_models::{ModelCapability, ModelId, ProviderId, RouteableModel};
use brouter_router::{Router, RouterError};
use brouter_router_models::{RoutingDecision, RoutingObjective, RoutingOptions};
use brouter_telemetry::{TelemetryError, TelemetryStore, now_millis};
use brouter_telemetry_models::UsageEvent;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tower_http::cors::{Any, CorsLayer};

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
    #[error("server API key environment variable {env_var} is not set")]
    MissingServerApiKey { env_var: String },
    #[error("HTTP server error: {0}")]
    Serve(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
struct AppState {
    router: Router,
    providers: ProviderRegistry,
    provider_client: ProviderClient,
    telemetry: TelemetryStore,
    api_key: Option<String>,
    provider_health: Arc<Mutex<BTreeMap<ProviderId, ProviderHealth>>>,
    provider_failure_threshold: u32,
    provider_cooldown_ms: u64,
    max_estimated_cost: Option<f64>,
    max_session_estimated_cost: Option<f64>,
    model_budgets: BTreeMap<ModelId, f64>,
    provider_budgets: BTreeMap<ProviderId, f64>,
    groups: BTreeMap<String, Vec<ModelId>>,
    default_profile: Option<String>,
    session_context_tokens: Arc<Mutex<BTreeMap<String, u32>>>,
    introspection_snapshots: Vec<IntrospectionSnapshot>,
    dynamic_policy_effects: Vec<DynamicPolicyEffect>,
}

#[derive(Debug, Clone, Default)]
struct ProviderHealth {
    consecutive_failures: u32,
    cooldown_until: Option<Instant>,
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
    let api_key = server_api_key(&config)?;
    let snapshots = introspection_snapshots(&config).await;
    let app = build_app_with_api_key_and_introspection(&config, telemetry, api_key, snapshots);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;

    tracing::info!(%address, "starting brouter server");
    axum::serve(listener, app).await.map_err(ServerError::Serve)
}

#[cfg(test)]
fn build_app(config: &BrouterConfig, telemetry: TelemetryStore) -> AxumRouter {
    build_app_with_api_key(config, telemetry, None)
}

#[cfg(test)]
fn build_app_with_api_key(
    config: &BrouterConfig,
    telemetry: TelemetryStore,
    api_key: Option<String>,
) -> AxumRouter {
    build_app_with_api_key_and_introspection(config, telemetry, api_key, Vec::new())
}

fn build_app_with_api_key_and_introspection(
    config: &BrouterConfig,
    telemetry: TelemetryStore,
    api_key: Option<String>,
    introspection_snapshots: Vec<IntrospectionSnapshot>,
) -> AxumRouter {
    let objective = RoutingObjective::from_name(&config.router.default_objective);
    let dynamic_policy_effects = dynamic_policy_effects(
        introspection_snapshots.clone(),
        DynamicPolicyConfig {
            low_remaining_ratio: config.router.dynamic_policy.low_remaining_ratio,
            critical_remaining_ratio: config.router.dynamic_policy.critical_remaining_ratio,
            low_remaining_penalty: config.router.dynamic_policy.low_remaining_penalty,
            exclude_when_exhausted: config.router.dynamic_policy.exclude_when_exhausted,
        },
        &config.router.dynamic_policy.disable_attributes_when_low,
    );
    let router = Router::new_with_policy(
        routeable_models_with_introspection(config, &introspection_snapshots),
        objective,
        scoring_weights(config),
        routing_rules(config),
        routing_profiles(config),
        context_policy(config),
    );
    let state = AppState {
        router,
        providers: ProviderRegistry::from_config(config),
        provider_client: ProviderClient::new(),
        telemetry,
        api_key,
        provider_health: Arc::new(Mutex::new(BTreeMap::new())),
        provider_failure_threshold: config.router.provider_failure_threshold,
        provider_cooldown_ms: config.router.provider_cooldown_ms,
        max_estimated_cost: config.router.max_estimated_cost,
        max_session_estimated_cost: config.router.max_session_estimated_cost,
        model_budgets: config
            .models
            .iter()
            .filter_map(|(id, model)| {
                model
                    .max_estimated_cost
                    .map(|budget| (ModelId::new(id.clone()), budget))
            })
            .collect(),
        provider_budgets: config
            .providers
            .iter()
            .filter_map(|(id, provider)| {
                provider
                    .max_estimated_cost
                    .map(|budget| (ProviderId::new(id.clone()), budget))
            })
            .collect(),
        groups: config
            .router
            .groups
            .iter()
            .map(|(group, models)| {
                (
                    group.clone(),
                    models.iter().cloned().map(ModelId::new).collect(),
                )
            })
            .collect(),
        default_profile: config.router.default_profile.clone(),
        session_context_tokens: Arc::new(Mutex::new(BTreeMap::new())),
        introspection_snapshots,
        dynamic_policy_effects,
    };

    let app = AxumRouter::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/brouter/route/explain", post(route_explain))
        .route("/v1/brouter/usage", get(usage))
        .route("/v1/brouter/usage/summary", get(usage_summary))
        .route("/v1/brouter/introspection", get(introspection))
        .layer(DefaultBodyLimit::max(config.server.max_request_body_bytes))
        .with_state(state);
    apply_cors(app, &config.server.cors_allowed_origins)
}

fn apply_cors(app: AxumRouter, allowed_origins: &[String]) -> AxumRouter {
    if allowed_origins.is_empty() {
        return app;
    }
    let layer = if allowed_origins.iter().any(|origin| origin == "*") {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any)
    } else {
        let mut layer = CorsLayer::new()
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any);
        for origin in allowed_origins {
            if let Ok(origin) = HeaderValue::from_str(origin) {
                layer = layer.allow_origin(origin);
            }
        }
        layer
    };
    app.layer(layer)
}

async fn introspection_snapshots(config: &BrouterConfig) -> Vec<IntrospectionSnapshot> {
    let live = refresh_introspection_snapshots(config).await;
    merge_configured_resource_pools(config, live)
}

async fn refresh_introspection_snapshots(config: &BrouterConfig) -> Vec<IntrospectionSnapshot> {
    if !config.router.metadata.refresh_on_startup {
        return Vec::new();
    }
    let registry = ProviderRegistry::from_config(config);
    let client = ProviderClient::new();
    let mut snapshots = Vec::new();
    for provider_id in registry.provider_ids() {
        let Some(provider) = config.providers.get(provider_id.as_str()) else {
            continue;
        };
        if !provider.introspection.enabled {
            continue;
        }
        let request = IntrospectionRequest {
            include_catalog: provider.introspection.catalog,
            include_account: provider.introspection.account,
            include_limits: provider.introspection.limits,
        };
        match client.introspect(&registry, &provider_id, request).await {
            Ok(snapshot) => snapshots.push(snapshot),
            Err(error) => tracing::warn!(%provider_id, %error, "provider introspection failed"),
        }
    }
    snapshots
}

fn merge_configured_resource_pools(
    config: &BrouterConfig,
    mut snapshots: Vec<IntrospectionSnapshot>,
) -> Vec<IntrospectionSnapshot> {
    for (provider_id, provider) in &config.providers {
        if provider.resource_pools.is_empty() {
            continue;
        }
        let provider_id = ProviderId::new(provider_id.clone());
        let snapshot_index = snapshots
            .iter()
            .position(|snapshot| snapshot.provider == provider_id);
        let configured_pools = provider
            .resource_pools
            .iter()
            .map(|pool| configured_resource_pool(&provider_id, pool))
            .collect::<Vec<_>>();
        if let Some(index) = snapshot_index {
            let account = snapshots[index]
                .account
                .get_or_insert_with(|| AccountSnapshot {
                    account_id: None,
                    status: AccountStatus::Unknown,
                    pools: Vec::new(),
                });
            for pool in configured_pools {
                merge_resource_pool(&mut account.pools, pool);
            }
        } else {
            snapshots.push(IntrospectionSnapshot {
                provider: provider_id,
                fetched_at_ms: now_millis(),
                source: SnapshotSource {
                    kind: SnapshotSourceKind::Runtime,
                    endpoint: None,
                    label: Some("configured_resource_pools".to_string()),
                },
                catalog: None,
                account: Some(AccountSnapshot {
                    account_id: None,
                    status: AccountStatus::Unknown,
                    pools: configured_pools,
                }),
                warnings: Vec::new(),
            });
        }
    }
    snapshots
}

fn merge_resource_pool(pools: &mut Vec<ResourcePool>, configured: ResourcePool) {
    if let Some(existing) = pools.iter_mut().find(|pool| pool.id == configured.id) {
        existing.total = existing.total.or(configured.total);
        existing.remaining = existing.remaining.or_else(|| {
            configured.remaining.or_else(|| {
                existing
                    .total
                    .zip(existing.used)
                    .map(|(total, used)| (total - used).max(0.0))
            })
        });
        existing.used = existing.used.or(configured.used);
        existing.refill_at_ms = existing.refill_at_ms.or(configured.refill_at_ms);
        existing.reset_at_ms = existing.reset_at_ms.or(configured.reset_at_ms);
        existing.expires_at_ms = existing.expires_at_ms.or(configured.expires_at_ms);
    } else {
        pools.push(configured);
    }
}

fn configured_resource_pool(
    provider_id: &ProviderId,
    pool: &brouter_config_models::ResourcePoolConfig,
) -> ResourcePool {
    let mut selector = configured_resource_selector(&pool.applies_to);
    if selector.providers.is_empty() {
        selector.providers.push(provider_id.clone());
    }
    ResourcePool {
        id: pool.id.clone(),
        scope: resource_scope(&pool.scope),
        kind: resource_kind(&pool.kind),
        unit: resource_unit(&pool.unit),
        remaining: pool.remaining,
        total: pool.total,
        used: pool.used,
        refill_at_ms: pool.refill_at_ms,
        reset_at_ms: pool.reset_at_ms,
        expires_at_ms: pool.expires_at_ms,
        applies_to: selector,
        provenance: MetadataProvenance::new(MetadataSource::UserConfig),
    }
}

fn configured_resource_selector(
    selector: &brouter_config_models::ResourceSelectorConfig,
) -> ResourceSelector {
    ResourceSelector {
        providers: selector
            .providers
            .iter()
            .cloned()
            .map(ProviderId::new)
            .collect(),
        upstream_models: selector.upstream_models.clone(),
        configured_models: selector.models.iter().cloned().map(ModelId::new).collect(),
        attributes: selector.attributes.clone(),
        capabilities: selector
            .capabilities
            .iter()
            .filter_map(|capability| capability.parse().ok())
            .collect(),
    }
}

fn resource_scope(value: &str) -> ResourceScope {
    match value {
        "provider" => ResourceScope::Provider,
        "account" => ResourceScope::Account,
        "model" => ResourceScope::Model,
        "attribute" => ResourceScope::Attribute,
        _ => ResourceScope::Unknown,
    }
}

fn resource_kind(value: &str) -> ResourceKind {
    match value {
        "monetary_credit" => ResourceKind::MonetaryCredit,
        "subscription_allowance" => ResourceKind::SubscriptionAllowance,
        "token_budget" => ResourceKind::TokenBudget,
        "request_budget" => ResourceKind::RequestBudget,
        "rate_limit" => ResourceKind::RateLimit,
        "priority_allowance" => ResourceKind::PriorityAllowance,
        _ => ResourceKind::Unknown,
    }
}

fn resource_unit(value: &str) -> ResourceUnit {
    match value {
        "usd" => ResourceUnit::Usd,
        "tokens" => ResourceUnit::Tokens,
        "requests" => ResourceUnit::Requests,
        "requests_per_minute" => ResourceUnit::RequestsPerMinute,
        "tokens_per_minute" => ResourceUnit::TokensPerMinute,
        "percent" => ResourceUnit::Percent,
        _ => ResourceUnit::Unknown,
    }
}

async fn telemetry_store(config: &BrouterConfig) -> Result<TelemetryStore, TelemetryError> {
    if let Some(path) = &config.telemetry.database_path {
        TelemetryStore::sqlite(Path::new(path)).await
    } else {
        Ok(TelemetryStore::memory())
    }
}

fn server_api_key(config: &BrouterConfig) -> Result<Option<String>, ServerError> {
    config
        .server
        .api_key_env
        .as_ref()
        .map(|env_var| {
            std::env::var(env_var).map_err(|_| ServerError::MissingServerApiKey {
                env_var: env_var.clone(),
            })
        })
        .transpose()
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

async fn models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<ModelListResponse>, (StatusCode, Json<ErrorResponse>)> {
    authorize(&state, &headers)?;
    let mut models = vec![ModelObject::new("brouter/auto", "brouter")];
    models.extend(
        state
            .router
            .models()
            .iter()
            .map(|model| ModelObject::new(model.id.as_str(), model.provider.as_str())),
    );
    Ok(Json(ModelListResponse::model_list(models)))
}

async fn route_explain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<RoutingDecision>, (StatusCode, Json<ErrorResponse>)> {
    authorize(&state, &headers)?;
    let decision = route_request(&state, &headers, &request).await?;
    Ok(Json(decision))
}

async fn usage(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<UsageQuery>,
) -> Result<Json<Vec<UsageEvent>>, (StatusCode, Json<ErrorResponse>)> {
    authorize(&state, &headers)?;
    state
        .telemetry
        .events()
        .await
        .map(|events| Json(filter_usage_events(events, &query)))
        .map_err(|error| telemetry_error_response(&error))
}

async fn introspection(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<IntrospectionSnapshot>>, (StatusCode, Json<ErrorResponse>)> {
    authorize(&state, &headers)?;
    Ok(Json(state.introspection_snapshots))
}

async fn usage_summary(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<UsageQuery>,
) -> Result<Json<UsageSummary>, (StatusCode, Json<ErrorResponse>)> {
    authorize(&state, &headers)?;
    state
        .telemetry
        .events()
        .await
        .map(|events| Json(summarize_usage(&filter_usage_events(events, &query))))
        .map_err(|error| telemetry_error_response(&error))
}

async fn metrics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    authorize(&state, &headers)?;
    let events = state
        .telemetry
        .events()
        .await
        .map_err(|error| telemetry_error_response(&error))?;
    let summary = summarize_usage(&events);
    Ok(format!(
        "# TYPE brouter_requests_total counter\nbrouter_requests_total {}\n# TYPE brouter_request_success_total counter\nbrouter_request_success_total {}\n# TYPE brouter_estimated_cost_total counter\nbrouter_estimated_cost_total {}\n# TYPE brouter_latency_ms_sum counter\nbrouter_latency_ms_sum {}\n",
        summary.requests, summary.successes, summary.estimated_cost, summary.latency_ms_sum,
    ))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }

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

#[allow(clippy::too_many_lines)]
async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<EmbeddingsRequest>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }

    let model = match embedding_model(&state, &request) {
        Ok(model) => model,
        Err(error) => return error.into_response(),
    };
    match state
        .provider_client
        .embeddings(&state.providers, &model, &request)
        .await
    {
        Ok(provider_response) => {
            let status = provider_status(provider_response.status);
            let mut headers = HeaderMap::new();
            insert_header(&mut headers, "x-brouter-selected-model", model.id.as_str());
            insert_header(&mut headers, "x-brouter-provider", model.provider.as_str());
            insert_header(
                &mut headers,
                "x-brouter-upstream-model",
                &model.upstream_model,
            );
            insert_route_attribute_headers(&mut headers, &model);
            (status, headers, Json(provider_response.body)).into_response()
        }
        Err(error) => error_response(StatusCode::BAD_GATEWAY, error.to_string(), "provider_error")
            .into_response(),
    }
}

fn embedding_model(
    state: &AppState,
    request: &EmbeddingsRequest,
) -> Result<RouteableModel, (StatusCode, Json<ErrorResponse>)> {
    if matches!(request.model.as_str(), "brouter/auto" | "auto") {
        return state
            .router
            .models()
            .iter()
            .find(|model| model.has_capability(ModelCapability::Embeddings))
            .cloned()
            .ok_or_else(|| {
                error_response(
                    StatusCode::BAD_REQUEST,
                    "no configured embedding model",
                    "routing_error",
                )
            });
    }
    let model = state
        .router
        .models()
        .iter()
        .find(|model| model.id.as_str() == request.model)
        .cloned()
        .ok_or_else(|| {
            error_response(
                StatusCode::BAD_REQUEST,
                format!("unknown embedding model {}", request.model),
                "routing_error",
            )
        })?;
    if !model.has_capability(ModelCapability::Embeddings) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            format!("model {} does not support embeddings", request.model),
            "routing_error",
        ));
    }
    Ok(model)
}

#[allow(clippy::too_many_lines)]
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

        match within_budget(state, headers, request, &model, candidate.estimated_cost).await {
            Ok(true) => {}
            Ok(false) => {
                last_status = Some(StatusCode::PAYMENT_REQUIRED);
                last_error = Some(format!(
                    "model {} exceeds configured max estimated cost",
                    candidate.model_id
                ));
                continue;
            }
            Err(error) => return telemetry_error_response(&error).into_response(),
        }
        if provider_in_cooldown(state, &model.provider) {
            last_status = Some(StatusCode::SERVICE_UNAVAILABLE);
            last_error = Some(format!("provider {} is cooling down", model.provider));
            continue;
        }

        let started_at = Instant::now();
        match state
            .provider_client
            .chat_completions_response(&state.providers, &model, request)
            .await
        {
            Ok(response) => {
                let status = provider_status(response.status);
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

                if status.is_success() {
                    mark_provider_success(state, &model.provider);
                } else if should_try_fallback(status) {
                    mark_provider_failure(state, &model.provider);
                }

                if status.is_success() || !should_try_fallback(status) {
                    let mut response_headers = route_headers(&model, &candidate.model_id, decision);
                    response_headers.insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    return (status, response_headers, Body::from_stream(response.stream))
                        .into_response();
                }
                last_status = Some(status);
            }
            Err(error) => {
                mark_provider_failure(state, &model.provider);
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

        match within_budget(state, headers, request, &model, candidate.estimated_cost).await {
            Ok(true) => {}
            Ok(false) => {
                last_error = Some(format!(
                    "model {} exceeds configured max estimated cost",
                    candidate.model_id
                ));
                continue;
            }
            Err(error) => return telemetry_error_response(&error).into_response(),
        }
        if provider_in_cooldown(state, &model.provider) {
            last_error = Some(format!("provider {} is cooling down", model.provider));
            continue;
        }

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

                if status.is_success() {
                    mark_provider_success(state, &model.provider);
                } else if should_try_fallback(status) {
                    mark_provider_failure(state, &model.provider);
                }

                if status.is_success() || !should_try_fallback(status) {
                    let response_headers = route_headers(&model, &candidate.model_id, decision);
                    return (status, response_headers, Json(provider_response.body))
                        .into_response();
                }
                last_response = Some((status, provider_response.body));
            }
            Err(error) => {
                mark_provider_failure(state, &model.provider);
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

async fn within_budget(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
    model: &RouteableModel,
    estimated_cost: f64,
) -> Result<bool, TelemetryError> {
    let request_budget_allows = state
        .max_estimated_cost
        .is_none_or(|max_estimated_cost| estimated_cost <= max_estimated_cost)
        && state
            .provider_budgets
            .get(&model.provider)
            .is_none_or(|max_estimated_cost| estimated_cost <= *max_estimated_cost)
        && state
            .model_budgets
            .get(&model.id)
            .is_none_or(|max_estimated_cost| estimated_cost <= *max_estimated_cost);
    if !request_budget_allows {
        return Ok(false);
    }
    let Some(max_session_cost) = state.max_session_estimated_cost else {
        return Ok(true);
    };
    let Some(session_id) = session_id(headers, request) else {
        return Ok(true);
    };
    let session_cost = state
        .telemetry
        .events()
        .await?
        .iter()
        .filter(|event| event.session_id.as_ref() == Some(&session_id))
        .map(|event| event.estimated_cost)
        .sum::<f64>();
    Ok(session_cost + estimated_cost <= max_session_cost)
}

fn provider_in_cooldown(state: &AppState, provider_id: &ProviderId) -> bool {
    let Ok(mut health) = state.provider_health.lock() else {
        return false;
    };
    let Some(provider_health) = health.get_mut(provider_id) else {
        return false;
    };
    match provider_health.cooldown_until {
        Some(until) if Instant::now() < until => true,
        Some(_) => {
            provider_health.cooldown_until = None;
            provider_health.consecutive_failures = 0;
            false
        }
        None => false,
    }
}

fn mark_provider_success(state: &AppState, provider_id: &ProviderId) {
    if let Ok(mut health) = state.provider_health.lock() {
        health.remove(provider_id);
    }
}

fn mark_provider_failure(state: &AppState, provider_id: &ProviderId) {
    let Ok(mut health) = state.provider_health.lock() else {
        return;
    };
    let provider_health = health.entry(provider_id.clone()).or_default();
    provider_health.consecutive_failures = provider_health.consecutive_failures.saturating_add(1);
    if provider_health.consecutive_failures >= state.provider_failure_threshold {
        provider_health.cooldown_until =
            Some(Instant::now() + std::time::Duration::from_millis(state.provider_cooldown_ms));
    }
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
    insert_route_attribute_headers(&mut headers, model);
    headers
}

fn insert_route_attribute_headers(headers: &mut HeaderMap, model: &RouteableModel) {
    let attributes = route_attributes_header(model);
    if !attributes.is_empty() {
        insert_header(headers, "x-brouter-attributes", &attributes);
    }
    let badges = route_badges_header(model);
    if !badges.is_empty() {
        insert_header(headers, "x-brouter-display-badges", &badges);
    }
}

fn route_attributes_header(model: &RouteableModel) -> String {
    model
        .attributes
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn route_badges_header(model: &RouteableModel) -> String {
    if !model.display_badges.is_empty() {
        return model.display_badges.join(",");
    }
    model
        .attributes
        .iter()
        .filter_map(|(key, value)| display_badge_for_attribute(key, value))
        .collect::<Vec<_>>()
        .join(",")
}

fn display_badge_for_attribute(key: &str, value: &str) -> Option<String> {
    matches!(
        key,
        "latency_class" | "service_class" | "quality_lane" | "billing_class"
    )
    .then(|| value.to_string())
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
    let allowed_models = group_models(state, &request.model);
    let session_context_tokens = session_id
        .as_deref()
        .and_then(|id| session_context_tokens(state, id));
    let profile = requested_profile(state, headers, request);
    let decision = state
        .router
        .route_chat_with_options(
            request,
            is_first_message,
            RoutingOptions {
                allowed_models,
                profile,
                session_context_tokens,
                dynamic_policy_effects: state.dynamic_policy_effects.clone(),
            },
        )
        .map_err(|error| router_error_response(&error))?;
    if let Some(session_id) = session_id {
        update_session_context_tokens(
            state,
            &session_id,
            decision.features.required_context_tokens,
        );
    }
    Ok(decision)
}

fn group_models(state: &AppState, model: &str) -> Option<Vec<ModelId>> {
    let group_name = model.strip_prefix("group:").unwrap_or(model);
    state.groups.get(group_name).cloned()
}

fn requested_profile(
    state: &AppState,
    headers: &HeaderMap,
    request: &ChatCompletionRequest,
) -> Option<String> {
    header_value(headers, "x-brouter-profile")
        .map(ToOwned::to_owned)
        .or_else(|| metadata_string(request, "brouter_profile"))
        .or_else(|| {
            request
                .model
                .strip_prefix("profile:")
                .map(ToOwned::to_owned)
        })
        .or_else(|| state.default_profile.clone())
}

fn metadata_string(request: &ChatCompletionRequest, key: &str) -> Option<String> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(key))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
}

fn session_context_tokens(state: &AppState, session_id: &str) -> Option<u32> {
    state
        .session_context_tokens
        .lock()
        .ok()
        .and_then(|context| context.get(session_id).copied())
}

fn update_session_context_tokens(state: &AppState, session_id: &str, required_context_tokens: u32) {
    if let Ok(mut context) = state.session_context_tokens.lock() {
        let entry = context.entry(session_id.to_string()).or_default();
        *entry = (*entry).max(required_context_tokens);
    }
}

fn authorize(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let Some(expected_key) = state.api_key.as_deref() else {
        return Ok(());
    };
    if bearer_token(headers).is_some_and(|token| token == expected_key)
        || header_value(headers, "x-api-key").is_some_and(|token| token == expected_key)
    {
        return Ok(());
    }
    Err(error_response(
        StatusCode::UNAUTHORIZED,
        "missing or invalid brouter API key",
        "authentication_error",
    ))
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = header_value(headers, header::AUTHORIZATION.as_str())?;
    value.strip_prefix("Bearer ")
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
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

#[derive(Debug, Clone, Default, Serialize)]
struct UsageSummary {
    requests: u64,
    successes: u64,
    failures: u64,
    estimated_cost: f64,
    latency_ms_sum: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct UsageQuery {
    session_id: Option<String>,
    model: Option<String>,
    success: Option<bool>,
    since_ms: Option<u64>,
    until_ms: Option<u64>,
}

fn summarize_usage(events: &[UsageEvent]) -> UsageSummary {
    UsageSummary {
        requests: u64::try_from(events.len()).unwrap_or(u64::MAX),
        successes: u64::try_from(events.iter().filter(|event| event.success).count())
            .unwrap_or(u64::MAX),
        failures: u64::try_from(events.iter().filter(|event| !event.success).count())
            .unwrap_or(u64::MAX),
        estimated_cost: events.iter().map(|event| event.estimated_cost).sum(),
        latency_ms_sum: events.iter().filter_map(|event| event.latency_ms).sum(),
        prompt_tokens: events.iter().filter_map(|event| event.prompt_tokens).sum(),
        completion_tokens: events
            .iter()
            .filter_map(|event| event.completion_tokens)
            .sum(),
    }
}

fn filter_usage_events(events: Vec<UsageEvent>, query: &UsageQuery) -> Vec<UsageEvent> {
    events
        .into_iter()
        .filter(|event| {
            query
                .session_id
                .as_ref()
                .is_none_or(|session_id| event.session_id.as_ref() == Some(session_id))
        })
        .filter(|event| {
            query
                .model
                .as_ref()
                .is_none_or(|model| event.selected_model.as_str() == model)
        })
        .filter(|event| query.success.is_none_or(|success| event.success == success))
        .filter(|event| {
            query
                .since_ms
                .is_none_or(|since| event.timestamp_ms >= since)
        })
        .filter(|event| {
            query
                .until_ms
                .is_none_or(|until| event.timestamp_ms <= until)
        })
        .collect()
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
            .clone()
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

        let second_response = app
            .clone()
            .oneshot(chat_request("debug this Rust error", false))
            .await
            .expect("second request should complete");
        assert_eq!(second_response.status(), StatusCode::OK);

        let usage_response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/brouter/usage?success=false")
                    .body(Body::empty())
                    .expect("usage request should build"),
            )
            .await
            .expect("usage request should complete");
        let usage = response_json(usage_response).await;
        assert_eq!(usage.as_array().map_or(0, Vec::len), 1);
    }

    #[tokio::test]
    async fn embeddings_forward_to_embedding_model() {
        let upstream = spawn_embeddings_upstream().await;
        let config = single_provider_config(upstream);
        let app = build_app(&config, TelemetryStore::memory());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"model": "brouter/auto", "input": "hello"}).to_string(),
                    ))
                    .expect("embeddings request should build"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("x-brouter-selected-model"),
            Some(&HeaderValue::from_static("cheap_cloud"))
        );
        let body = response_json(response).await;
        assert_eq!(body["model"], "cheap-upstream");
    }

    #[tokio::test]
    async fn authenticated_routes_require_api_key() {
        let upstream = spawn_echo_upstream().await;
        let config = single_provider_config(upstream);
        let app = build_app_with_api_key(
            &config,
            TelemetryStore::memory(),
            Some("secret".to_string()),
        );

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("models request should build"),
            )
            .await
            .expect("request should complete");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/models")
                    .header(header::AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .expect("models request should build"),
            )
            .await
            .expect("request should complete");
        assert_eq!(response.status(), StatusCode::OK);
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

    async fn spawn_embeddings_upstream() -> String {
        async fn embeddings(Json(body): Json<Value>) -> Json<Value> {
            Json(json!({
                "object": "list",
                "model": body["model"].clone(),
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]
            }))
        }
        spawn_upstream(AxumRouter::new().route("/v1/embeddings", post(embeddings))).await
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
                timeout_ms: None,
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: BTreeMap::new(),
            },
        );
        config.models.insert(
            "cheap_cloud".to_string(),
            ModelConfig {
                provider: "healthy".to_string(),
                model: "cheap-upstream".to_string(),
                context_window: Some(128_000),
                input_cost_per_million: 0.15,
                output_cost_per_million: 0.60,
                quality: Some(70),
                capabilities: vec!["chat".to_string(), "embeddings".to_string()],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
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
                timeout_ms: None,
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: BTreeMap::new(),
            },
        );
        providers.insert(
            "healthy".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some(healthy_base_url),
                api_key_env: None,
                timeout_ms: None,
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: BTreeMap::new(),
            },
        );

        let mut models = BTreeMap::new();
        models.insert(
            "strong_cloud".to_string(),
            ModelConfig {
                provider: "failing".to_string(),
                model: "strong-upstream".to_string(),
                context_window: Some(128_000),
                input_cost_per_million: 2.0,
                output_cost_per_million: 8.0,
                quality: Some(90),
                capabilities: vec!["chat".to_string(), "code".to_string()],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
            },
        );
        models.insert(
            "cheap_cloud".to_string(),
            ModelConfig {
                provider: "healthy".to_string(),
                model: "cheap-upstream".to_string(),
                context_window: Some(128_000),
                input_cost_per_million: 0.15,
                output_cost_per_million: 0.60,
                quality: Some(70),
                capabilities: vec!["chat".to_string(), "code".to_string()],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
            },
        );

        let mut config = BrouterConfig {
            providers,
            models,
            ..BrouterConfig::default()
        };
        config.router.provider_failure_threshold = 1;
        config
    }
}
