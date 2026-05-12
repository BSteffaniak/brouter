use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent, ReasoningEffort};
use brouter_catalog_models::{MetadataProvenance, MetadataSource};
use brouter_config_models::ProviderConfig;
use brouter_introspection_models::{
    AccountSnapshot, AccountStatus, IntrospectionRequest, IntrospectionSnapshot,
    IntrospectionWarning, ResourceKind, ResourcePool, ResourceScope, ResourceSelector,
    ResourceUnit, SnapshotSource, SnapshotSourceKind,
};
use brouter_provider_models::{ModelCapability, ProviderId, RouteableModel};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zeroize::Zeroizing;

use crate::{
    ProviderError, ProviderResponse, ProviderStreamResponse, RequestTimeoutScope,
    apply_attribute_mappings, send_provider_request,
};

const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const OPENAI_CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OPENAI_CODEX_API_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
const BROUTER_PREFIX: &str = "BROUTER_OPENAI";

#[derive(Debug, Clone)]
struct CodexAuth {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: Option<u64>,
    account_id: Option<String>,
    profile: String,
    vault: PathBuf,
}

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

pub async fn chat_completions(
    http: &reqwest::Client,
    provider: &ProviderConfig,
    model: &RouteableModel,
    request: &ChatCompletionRequest,
) -> Result<ProviderResponse, ProviderError> {
    let mut auth = codex_auth(provider, model)?;
    refresh_auth_if_needed(http, &mut auth, model).await?;
    let response = send_codex_request(
        http,
        provider,
        model,
        request,
        &auth,
        RequestTimeoutScope::FullResponse,
    )
    .await?;
    let status = response.status().as_u16();
    let text = response.text().await?;
    if !(200..300).contains(&status) {
        return Ok(ProviderResponse {
            status,
            body: codex_error_body(status, &text),
        });
    }
    Ok(ProviderResponse {
        status,
        body: codex_sse_to_chat_completion(&text, &model.upstream_model),
    })
}

pub async fn chat_completions_response(
    http: &reqwest::Client,
    provider: &ProviderConfig,
    model: &RouteableModel,
    request: &ChatCompletionRequest,
) -> Result<ProviderStreamResponse, ProviderError> {
    let mut auth = codex_auth(provider, model)?;
    refresh_auth_if_needed(http, &mut auth, model).await?;
    let response = send_codex_request(
        http,
        provider,
        model,
        request,
        &auth,
        RequestTimeoutScope::InitialResponse,
    )
    .await?;
    let status = response.status().as_u16();
    if !response.status().is_success() {
        return Ok(codex_error_stream_response(response).await);
    }

    let upstream = response.bytes_stream();
    let converter = CodexSseConverter::new(model.upstream_model.clone());
    let stream = stream::unfold(
        (upstream, converter, String::new(), false),
        |(mut upstream, mut converter, mut buffer, mut done)| async move {
            loop {
                if done {
                    return None;
                }
                match upstream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        let output = drain_complete_sse_lines(&mut buffer, &mut converter);
                        if !output.is_empty() {
                            return Some((
                                Ok(Bytes::from(output)),
                                (upstream, converter, buffer, done),
                            ));
                        }
                    }
                    Some(Err(error)) => {
                        done = true;
                        return Some((
                            Err(ProviderError::Http(error)),
                            (upstream, converter, buffer, done),
                        ));
                    }
                    None => {
                        done = true;
                        let output = converter.finish();
                        if output.is_empty() {
                            return None;
                        }
                        return Some((
                            Ok(Bytes::from(output)),
                            (upstream, converter, buffer, done),
                        ));
                    }
                }
            }
        },
    );

    Ok(ProviderStreamResponse {
        status,
        stream: Box::pin(stream),
    })
}

pub async fn introspect(
    http: &reqwest::Client,
    provider_id: &ProviderId,
    provider: &ProviderConfig,
    request: IntrospectionRequest,
) -> Result<IntrospectionSnapshot, ProviderError> {
    let model = introspection_model(provider_id);
    let mut auth = codex_auth(provider, &model)?;
    refresh_auth_if_needed(http, &mut auth, &model).await?;

    let mut warnings = Vec::new();
    let mut account = None;
    if request.include_account || request.include_limits {
        match fetch_usage_snapshot(http, provider_id, provider, &auth).await {
            Some(snapshot) => account = Some(snapshot),
            None => warnings.push(IntrospectionWarning::new(
                "codex_usage_unavailable",
                "OpenAI Codex usage endpoints did not return a recognized usage payload",
            )),
        }
    }

    Ok(IntrospectionSnapshot {
        provider: provider_id.clone(),
        fetched_at_ms: unix_millis(),
        source: SnapshotSource {
            kind: SnapshotSourceKind::ProviderApi,
            endpoint: Some("chatgpt.com/backend-api".to_string()),
            label: Some("openai_codex_usage".to_string()),
        },
        catalog: None,
        account,
        warnings,
    })
}

fn introspection_model(provider_id: &ProviderId) -> RouteableModel {
    RouteableModel {
        id: brouter_provider_models::ModelId::new("openai_codex_introspection"),
        provider: provider_id.clone(),
        upstream_model: "gpt-5.5".to_string(),
        context_window: 1_000_000,
        input_cost_per_million: 0.0,
        output_cost_per_million: 0.0,
        quality: 0,
        capabilities: vec![ModelCapability::Chat],
        attributes: BTreeMap::new(),
        display_badges: Vec::new(),
        metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
    }
}

async fn fetch_usage_snapshot(
    http: &reqwest::Client,
    provider_id: &ProviderId,
    provider: &ProviderConfig,
    auth: &CodexAuth,
) -> Option<AccountSnapshot> {
    for endpoint in codex_usage_endpoints(provider) {
        let Ok(response) = codex_usage_request(http, provider, auth, &endpoint).await else {
            continue;
        };
        if !response.status().is_success() {
            continue;
        }
        let Ok(value) = response.json::<Value>().await else {
            continue;
        };
        let pools = usage_resource_pools(provider_id, &endpoint, &value);
        if !pools.is_empty() {
            return Some(AccountSnapshot {
                account_id: auth.account_id.clone(),
                status: AccountStatus::Available,
                pools,
            });
        }
    }
    None
}

fn codex_usage_endpoints(provider: &ProviderConfig) -> Vec<String> {
    if !provider.introspection.usage_endpoints.is_empty() {
        return provider.introspection.usage_endpoints.clone();
    }
    vec![
        "https://chatgpt.com/backend-api/codex/usage".to_string(),
        "https://chatgpt.com/backend-api/codex/usage_limits".to_string(),
        "https://chatgpt.com/backend-api/codex/limits".to_string(),
        "https://chatgpt.com/backend-api/accounts/check/v4".to_string(),
    ]
}

async fn codex_usage_request(
    http: &reqwest::Client,
    provider: &ProviderConfig,
    auth: &CodexAuth,
    endpoint: &str,
) -> Result<reqwest::Response, ProviderError> {
    let url = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("https://chatgpt.com{endpoint}")
    };
    let mut builder = http
        .get(url)
        .bearer_auth(&auth.access_token)
        .header("OpenAI-Beta", "responses=experimental")
        .header("originator", "brouter")
        .header("User-Agent", "brouter/0.1.0")
        .header("accept", "application/json");
    if let Some(account_id) = &auth.account_id {
        builder = builder.header("ChatGPT-Account-Id", account_id);
    }
    send_provider_request(builder, provider, RequestTimeoutScope::FullResponse).await
}

fn usage_resource_pools(
    provider_id: &ProviderId,
    endpoint: &str,
    value: &Value,
) -> Vec<ResourcePool> {
    let mut pools = Vec::new();
    collect_usage_pools(provider_id, endpoint, value, "usage", &mut pools);
    pools
}

fn collect_usage_pools(
    provider_id: &ProviderId,
    endpoint: &str,
    value: &Value,
    path: &str,
    pools: &mut Vec<ResourcePool>,
) {
    match value {
        Value::Object(object) => {
            if let Some(pool) = resource_pool_from_object(provider_id, endpoint, path, object) {
                pools.push(pool);
            }
            for (key, child) in object {
                collect_usage_pools(
                    provider_id,
                    endpoint,
                    child,
                    &format!("{path}.{key}"),
                    pools,
                );
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                collect_usage_pools(
                    provider_id,
                    endpoint,
                    child,
                    &format!("{path}[{index}]"),
                    pools,
                );
            }
        }
        _ => {}
    }
}

fn resource_pool_from_object(
    provider_id: &ProviderId,
    endpoint: &str,
    path: &str,
    object: &serde_json::Map<String, Value>,
) -> Option<ResourcePool> {
    let total = number_field(object, &["total", "limit", "quota", "max", "maximum"]);
    let used = number_field(object, &["used", "consumed", "usage", "current"]);
    let remaining = number_field(
        object,
        &["remaining", "available", "left", "remaining_count"],
    )
    .or_else(|| total.zip(used).map(|(total, used)| (total - used).max(0.0)));
    if total.is_none() && used.is_none() && remaining.is_none() {
        return None;
    }

    let label = string_field(object, &["id", "name", "type", "bucket", "window"])
        .unwrap_or_else(|| path.to_string());
    Some(ResourcePool {
        id: sanitize_pool_id(&label),
        scope: resource_scope_for_label(&label),
        kind: resource_kind_for_label(&label),
        unit: resource_unit_for_label(&label),
        remaining,
        total,
        used,
        refill_at_ms: timestamp_ms_field(object, &["refill_at", "refill_at_ms"]),
        reset_at_ms: timestamp_ms_field(object, &["reset_at", "reset_at_ms", "resets_at"]),
        expires_at_ms: timestamp_ms_field(object, &["expires_at", "expires_at_ms"]),
        applies_to: usage_selector(provider_id, &label),
        provenance: MetadataProvenance {
            source: MetadataSource::ProviderApi,
            source_label: Some("openai_codex_usage".to_string()),
            source_url: Some(endpoint.to_string()),
            reason: Some(path.to_string()),
            verified_at_ms: Some(unix_millis()),
        },
    })
}

fn number_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<f64> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(value_as_f64))
}

fn string_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        object
            .get(*name)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

fn timestamp_ms_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<u64> {
    number_field(object, names).map(|value| {
        if !value.is_finite() || value.is_sign_negative() {
            return 0;
        }
        let seconds = if value < 10_000_000_000.0 {
            value
        } else {
            value / 1_000.0
        };
        let millis = std::time::Duration::from_secs_f64(seconds).as_millis();
        u64::try_from(millis).unwrap_or(u64::MAX)
    })
}

fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn sanitize_pool_id(label: &str) -> String {
    let id = label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    id.trim_matches('_').to_string()
}

fn resource_scope_for_label(label: &str) -> ResourceScope {
    let label = label.to_lowercase();
    if label.contains("priority") || label.contains("tier") || label.contains("thinking") {
        ResourceScope::Attribute
    } else if label.contains("model") || label.contains("gpt") || label.contains("codex") {
        ResourceScope::Model
    } else {
        ResourceScope::Account
    }
}

fn resource_kind_for_label(label: &str) -> ResourceKind {
    let label = label.to_lowercase();
    if label.contains("priority") {
        ResourceKind::PriorityAllowance
    } else if label.contains("token") {
        ResourceKind::TokenBudget
    } else if label.contains("rate") {
        ResourceKind::RateLimit
    } else if label.contains("request") || label.contains("message") || label.contains("5h") {
        ResourceKind::RequestBudget
    } else {
        ResourceKind::SubscriptionAllowance
    }
}

fn resource_unit_for_label(label: &str) -> ResourceUnit {
    let label = label.to_lowercase();
    if label.contains("token") {
        ResourceUnit::Tokens
    } else if label.contains("rpm") || label.contains("request_per_minute") {
        ResourceUnit::RequestsPerMinute
    } else if label.contains("percent") || label.contains("ratio") {
        ResourceUnit::Percent
    } else {
        ResourceUnit::Requests
    }
}

fn usage_selector(provider_id: &ProviderId, label: &str) -> ResourceSelector {
    let mut attributes = BTreeMap::new();
    let lower = label.to_lowercase();
    if lower.contains("priority") {
        attributes.insert("latency_class".to_string(), "priority".to_string());
    }
    if lower.contains("standard") {
        attributes.insert("latency_class".to_string(), "standard".to_string());
    }
    if lower.contains("high") || lower.contains("reasoning") || lower.contains("thinking") {
        attributes.insert("reasoning_effort".to_string(), "high".to_string());
    }
    ResourceSelector {
        providers: vec![provider_id.clone()],
        upstream_models: Vec::new(),
        configured_models: Vec::new(),
        attributes,
        capabilities: Vec::new(),
    }
}

fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}

fn codex_auth(
    provider: &ProviderConfig,
    model: &RouteableModel,
) -> Result<CodexAuth, ProviderError> {
    if provider.auth_backend.as_deref().unwrap_or("sshenv") != "sshenv" {
        return Err(ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: "openai-codex providers currently require auth_backend = \"sshenv\""
                .to_string(),
        });
    }
    let profile = provider
        .auth_profile
        .clone()
        .ok_or_else(|| ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: "openai-codex provider requires auth_profile".to_string(),
        })?;
    let vault = provider
        .auth_vault_path
        .as_ref()
        .map_or_else(default_auth_vault_path, |path| expand_home(path));
    let store = sshenv_vault::SshenvStore::new(sshenv_vault::SshenvStoreConfig::new(vault.clone()));
    let values = store
        .get_profile(&profile)
        .map_err(|error| ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: format!("failed to read sshenv auth profile {profile}: {error}"),
        })?
        .ok_or_else(|| ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: format!(
                "sshenv auth profile {profile} was not found in {}",
                vault.display()
            ),
        })?
        .into_iter()
        .map(|(key, value)| (key, value.to_string()))
        .collect::<BTreeMap<_, _>>();

    let access_token = values
        .get(&format!("{BROUTER_PREFIX}_CODEX_ACCESS_TOKEN"))
        .cloned()
        .ok_or_else(|| ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: format!(
                "sshenv auth profile {profile} does not contain ChatGPT/Codex credentials"
            ),
        })?;
    let refresh_token = values
        .get(&format!("{BROUTER_PREFIX}_CODEX_REFRESH_TOKEN"))
        .cloned();
    let expires_at = values
        .get(&format!("{BROUTER_PREFIX}_CODEX_EXPIRES_AT"))
        .and_then(|value| value.parse().ok());
    let account_id = values
        .get(&format!("{BROUTER_PREFIX}_CODEX_ACCOUNT_ID"))
        .cloned()
        .or_else(|| {
            values
                .get(&format!("{BROUTER_PREFIX}_CODEX_ID_TOKEN"))
                .and_then(|token| chatgpt_account_id_from_access_token(token))
        })
        .or_else(|| chatgpt_account_id_from_access_token(&access_token));

    Ok(CodexAuth {
        access_token,
        refresh_token,
        expires_at,
        account_id,
        profile,
        vault,
    })
}

async fn refresh_auth_if_needed(
    http: &reqwest::Client,
    auth: &mut CodexAuth,
    model: &RouteableModel,
) -> Result<(), ProviderError> {
    let Some(expires_at) = auth.expires_at else {
        return Ok(());
    };
    if expires_at > unix_timestamp().saturating_add(60) {
        return Ok(());
    }
    let refresh_token = auth
        .refresh_token
        .clone()
        .ok_or_else(|| ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: "ChatGPT/Codex token is expired and no refresh token is available".to_string(),
        })?;
    let refreshed = refresh_openai_codex_token(http, &refresh_token, model).await?;
    let next_refresh_token = refreshed.refresh_token.clone().unwrap_or(refresh_token);
    let next_expires_at =
        unix_timestamp().saturating_add(refreshed.expires_in.unwrap_or(3_600).saturating_sub(60));
    let account_id = refreshed
        .id_token
        .as_deref()
        .and_then(chatgpt_account_id_from_access_token)
        .or_else(|| chatgpt_account_id_from_access_token(&refreshed.access_token));
    store_refreshed_auth(
        auth,
        &refreshed,
        &next_refresh_token,
        next_expires_at,
        account_id.as_deref(),
        model,
    )?;
    auth.access_token = refreshed.access_token;
    auth.refresh_token = Some(next_refresh_token);
    auth.expires_at = Some(next_expires_at);
    auth.account_id = account_id;
    Ok(())
}

async fn refresh_openai_codex_token(
    http: &reqwest::Client,
    refresh_token: &str,
    model: &RouteableModel,
) -> Result<OpenAiOauthTokenResponse, ProviderError> {
    let params = [
        ("grant_type", "refresh_token"),
        ("client_id", OPENAI_CODEX_CLIENT_ID),
        ("refresh_token", refresh_token),
    ];
    let response = http
        .post(OPENAI_CODEX_TOKEN_URL)
        .form(&params)
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: format!("ChatGPT/Codex token refresh failed with {status}: {body}"),
        });
    }
    serde_json::from_str(&body).map_err(|error| ProviderError::Auth {
        provider_id: model.provider.to_string(),
        message: format!("failed to decode ChatGPT/Codex token refresh response: {error}"),
    })
}

fn store_refreshed_auth(
    auth: &CodexAuth,
    refreshed: &OpenAiOauthTokenResponse,
    next_refresh_token: &str,
    next_expires_at: u64,
    account_id: Option<&str>,
    model: &RouteableModel,
) -> Result<(), ProviderError> {
    let store =
        sshenv_vault::SshenvStore::new(sshenv_vault::SshenvStoreConfig::new(auth.vault.clone()));
    set_secret(
        &store,
        auth,
        "CODEX_ACCESS_TOKEN",
        refreshed.access_token.clone(),
        model,
    )?;
    if let Some(id_token) = &refreshed.id_token {
        set_secret(&store, auth, "CODEX_ID_TOKEN", id_token.clone(), model)?;
    }
    set_secret(
        &store,
        auth,
        "CODEX_REFRESH_TOKEN",
        next_refresh_token.to_string(),
        model,
    )?;
    set_secret(
        &store,
        auth,
        "CODEX_EXPIRES_AT",
        next_expires_at.to_string(),
        model,
    )?;
    if let Some(account_id) = account_id {
        set_secret(
            &store,
            auth,
            "CODEX_ACCOUNT_ID",
            account_id.to_string(),
            model,
        )?;
    }
    Ok(())
}

fn set_secret(
    store: &sshenv_vault::SshenvStore,
    auth: &CodexAuth,
    key: &str,
    value: String,
    model: &RouteableModel,
) -> Result<(), ProviderError> {
    store
        .set_secret(
            &auth.profile,
            &format!("{BROUTER_PREFIX}_{key}"),
            Zeroizing::new(value),
        )
        .map_err(|error| ProviderError::Auth {
            provider_id: model.provider.to_string(),
            message: format!("failed to store refreshed ChatGPT/Codex token: {error}"),
        })
}

async fn send_codex_request(
    http: &reqwest::Client,
    provider: &ProviderConfig,
    model: &RouteableModel,
    request: &ChatCompletionRequest,
    auth: &CodexAuth,
    timeout_scope: RequestTimeoutScope,
) -> Result<reqwest::Response, ProviderError> {
    let mut body = codex_request_body(model, request);
    apply_attribute_mappings(provider, model, &mut body);
    let mut builder = http
        .post(OPENAI_CODEX_API_ENDPOINT)
        .bearer_auth(&auth.access_token)
        .header("OpenAI-Beta", "responses=experimental")
        .header("originator", "brouter")
        .header("User-Agent", "brouter/0.1.0")
        .header("accept", "text/event-stream")
        .json(&body);
    if let Some(account_id) = &auth.account_id {
        builder = builder.header("ChatGPT-Account-Id", account_id);
    }
    if let Some(session_id) = codex_session_id(request) {
        builder = builder.header("session_id", session_id);
    }
    send_provider_request(builder, provider, timeout_scope).await
}

#[derive(Debug, Serialize)]
struct CodexResponsesRequest {
    model: String,
    instructions: String,
    input: Vec<Value>,
    stream: bool,
    store: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    text: CodexTextOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<CodexReasoningOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    include: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CodexTextOptions {
    verbosity: &'static str,
}

#[derive(Debug, Serialize)]
struct CodexReasoningOptions {
    effort: String,
}

fn codex_request_body(model: &RouteableModel, request: &ChatCompletionRequest) -> Value {
    let input = request
        .messages
        .iter()
        .flat_map(message_to_responses_input)
        .collect::<Vec<_>>();
    let tools = request.tools.as_ref().map_or_else(Vec::new, |tools| {
        tools
            .iter()
            .filter_map(openai_tool_to_responses_tool)
            .collect()
    });
    let body = CodexResponsesRequest {
        model: model.upstream_model.clone(),
        instructions: codex_instructions(request),
        input,
        stream: true,
        store: false,
        tools,
        tool_choice: "auto",
        parallel_tool_calls: true,
        text: CodexTextOptions { verbosity: "low" },
        reasoning: codex_reasoning(request),
        service_tier: request
            .extra
            .get("service_tier")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        include: vec!["reasoning.encrypted_content"],
        prompt_cache_key: codex_session_id(request),
        temperature: request.temperature,
        top_p: request.top_p,
    };
    serde_json::to_value(body).unwrap_or_else(|_| json!({}))
}

fn codex_reasoning(request: &ChatCompletionRequest) -> Option<CodexReasoningOptions> {
    let effort = match request.reasoning_effort? {
        ReasoningEffort::None => return None,
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High | ReasoningEffort::Max => "high",
    };
    Some(CodexReasoningOptions {
        effort: effort.to_string(),
    })
}

fn codex_session_id(request: &ChatCompletionRequest) -> Option<String> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("session_id"))
        .and_then(Value::as_str)
        .or_else(|| request.extra.get("session_id").and_then(Value::as_str))
        .filter(|session_id| !session_id.trim().is_empty())
        .map(ToOwned::to_owned)
}

async fn codex_error_stream_response(response: reqwest::Response) -> ProviderStreamResponse {
    let status = response.status().as_u16();
    let text = response.text().await.unwrap_or_default();
    let body =
        serde_json::to_vec(&codex_error_body(status, &text)).unwrap_or_else(|_| text.into_bytes());
    ProviderStreamResponse {
        status,
        stream: Box::pin(stream::once(async move { Ok(Bytes::from(body)) })),
    }
}

fn codex_error_body(status: u16, text: &str) -> Value {
    if text.trim().is_empty() {
        return json!({
            "error": {
                "message": format!("ChatGPT/Codex upstream returned {status} with an empty body"),
                "type": "upstream_error",
                "code": status,
            }
        });
    }
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

fn codex_instructions(request: &ChatCompletionRequest) -> String {
    system_instructions(request).unwrap_or_else(|| "You are a helpful assistant.".to_string())
}

fn system_instructions(request: &ChatCompletionRequest) -> Option<String> {
    let parts = request
        .messages
        .iter()
        .filter(|message| message.role == "system" || message.role == "developer")
        .map(|message| message.content.as_text())
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn message_to_responses_input(message: &ChatMessage) -> Vec<Value> {
    match message.role.as_str() {
        "system" | "developer" => Vec::new(),
        "assistant" => assistant_items(message),
        "tool" => message
            .tool_call_id
            .as_ref()
            .map(|call_id| {
                vec![json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.content.as_text(),
                })]
            })
            .unwrap_or_default(),
        _ => text_message("user", &message.content, true),
    }
}

fn text_message(role: &str, content: &MessageContent, input_text: bool) -> Vec<Value> {
    let text = content.as_text();
    if text.trim().is_empty() {
        return Vec::new();
    }
    let content_type = if input_text {
        "input_text"
    } else {
        "output_text"
    };
    vec![json!({
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    })]
}

fn assistant_items(message: &ChatMessage) -> Vec<Value> {
    let mut items = text_message("assistant", &message.content, false);
    if let Some(tool_calls) = &message.tool_calls {
        items.extend(tool_calls.iter().filter_map(|tool_call| {
            let function = tool_call.get("function")?;
            Some(json!({
                "type": "function_call",
                "call_id": tool_call.get("id").and_then(Value::as_str).unwrap_or("call"),
                "name": function.get("name").and_then(Value::as_str).unwrap_or("tool"),
                "arguments": function.get("arguments").and_then(Value::as_str).unwrap_or("{}"),
            }))
        }));
    }
    items
}

fn openai_tool_to_responses_tool(tool: &Value) -> Option<Value> {
    let function = tool.get("function")?;
    Some(json!({
        "type": "function",
        "name": function.get("name")?.clone(),
        "description": function.get("description").cloned().unwrap_or_else(|| Value::String(String::new())),
        "parameters": function.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    }))
}

fn drain_complete_sse_lines(buffer: &mut String, converter: &mut CodexSseConverter) -> String {
    let mut output = String::new();
    while let Some(position) = buffer.find('\n') {
        let mut line = buffer[..position].to_string();
        if line.ends_with('\r') {
            line.pop();
        }
        buffer.drain(..=position);
        output.push_str(&converter.process_line(line.trim()));
    }
    output
}

#[derive(Debug, Clone)]
struct CodexSseConverter {
    model: String,
    emitted_role: bool,
    done: bool,
}

impl CodexSseConverter {
    const fn new(model: String) -> Self {
        Self {
            model,
            emitted_role: false,
            done: false,
        }
    }

    fn process_line(&mut self, line: &str) -> String {
        let Some(data) = line.strip_prefix("data: ") else {
            return String::new();
        };
        if data == "[DONE]" {
            self.done = true;
            return self.done_chunk("stop");
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            return String::new();
        };
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => event
                .get("delta")
                .and_then(Value::as_str)
                .map_or_else(String::new, |delta| self.text_delta(delta)),
            Some("response.output_item.added" | "response.output_item.done") => {
                self.output_item(&event)
            }
            Some("response.function_call_arguments.delta") => self.tool_arguments_delta(&event),
            Some("response.completed" | "response.done") => {
                self.done = true;
                self.done_chunk("stop")
            }
            Some("response.incomplete") => {
                self.done = true;
                self.done_chunk("length")
            }
            _ => String::new(),
        }
    }

    fn text_delta(&mut self, delta: &str) -> String {
        let role = (!self.emitted_role).then_some("assistant");
        self.emitted_role = true;
        let mut delta_obj = json!({"content": delta});
        if let Some(role) = role {
            delta_obj["role"] = Value::String(role.to_string());
        }
        self.chunk(&delta_obj, None)
    }

    fn output_item(&self, event: &Value) -> String {
        let Some(item) = event.get("item") else {
            return String::new();
        };
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return String::new();
        }
        let index = event
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let id = item
            .get("call_id")
            .and_then(Value::as_str)
            .unwrap_or("call");
        let name = item.get("name").and_then(Value::as_str).unwrap_or("tool");
        self.chunk(
            &json!({
                "tool_calls": [{
                    "index": index,
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }]
            }),
            None,
        )
    }

    fn tool_arguments_delta(&self, event: &Value) -> String {
        let index = event
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let delta = event
            .get("delta")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.chunk(
            &json!({
                "tool_calls": [{
                    "index": index,
                    "function": {"arguments": delta},
                }]
            }),
            None,
        )
    }

    fn done_chunk(&self, reason: &str) -> String {
        self.chunk(&json!({}), Some(reason)) + "data: [DONE]\n\n"
    }

    fn finish(&mut self) -> String {
        if self.done {
            String::new()
        } else {
            self.done = true;
            self.done_chunk("stop")
        }
    }

    fn chunk(&self, delta: &Value, finish_reason: Option<&str>) -> String {
        format!(
            "data: {}\n\n",
            json!({
                "id": "chatcmpl-brouter-codex",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }]
            })
        )
    }
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

fn codex_sse_to_chat_completion(text: &str, model: &str) -> Value {
    let mut content = String::new();
    let mut tool_calls = BTreeMap::<u64, ToolCallAccumulator>::new();
    for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if line == "[DONE]" {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    content.push_str(delta);
                }
            }
            Some("response.output_item.added" | "response.output_item.done") => {
                if let Some(item) = event.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                {
                    let index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let entry = tool_calls.entry(index).or_default();
                    entry.id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    entry.name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                }
            }
            Some("response.function_call_arguments.delta") => {
                let index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    tool_calls
                        .entry(index)
                        .or_default()
                        .arguments
                        .push_str(delta);
                }
            }
            _ => {}
        }
    }
    let tool_calls = tool_calls
        .into_values()
        .filter_map(|tool_call| {
            Some(json!({
                "id": tool_call.id?,
                "type": "function",
                "function": {
                    "name": tool_call.name?,
                    "arguments": tool_call.arguments,
                }
            }))
        })
        .collect::<Vec<_>>();
    let finish_reason = if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    json!({
        "id": "chatcmpl-brouter-codex",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    })
}

fn chatgpt_account_id_from_access_token(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let claims = serde_json::from_slice::<Value>(&bytes).ok()?;
    claims
        .get("chatgpt_account_id")
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(|auth| auth.get("chatgpt_account_id"))
        })
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn default_auth_vault_path() -> PathBuf {
    if let Ok(path) = std::env::var("BROUTER_AUTH_VAULT") {
        return expand_home(&path);
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

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use brouter_api_models::{ChatMessage, MessageContent};
    use brouter_provider_models::{ModelCapability, ModelId, ProviderId};

    use super::*;

    #[test]
    fn codex_request_uses_responses_shape() {
        let model = routeable_model();
        let request = ChatCompletionRequest {
            model: "auto".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text("be helpful".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    extra: BTreeMap::new(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text("hello".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    extra: BTreeMap::new(),
                },
            ],
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            response_format: None,
            metadata: None,
            extra: BTreeMap::new(),
        };

        let body = codex_request_body(&model, &request);

        assert_eq!(body["model"], "gpt-5.5");
        assert_eq!(body["instructions"], "be helpful");
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn codex_request_maps_reasoning_and_service_tier() {
        let model = routeable_model();
        let mut request = test_request();
        request.reasoning_effort = Some(ReasoningEffort::High);
        request.extra.insert(
            "service_tier".to_string(),
            Value::String("priority".to_string()),
        );

        let body = codex_request_body(&model, &request);

        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["service_tier"], "priority");
    }

    #[test]
    fn codex_usage_parser_builds_priority_pool() {
        let provider_id = ProviderId::new("openai_max");
        let value = json!({
            "usage_limits": {
                "priority": {
                    "used": 8,
                    "total": 10,
                    "reset_at_ms": 12345
                }
            }
        });

        let pools = usage_resource_pools(&provider_id, "https://example.test/usage", &value);

        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].kind, ResourceKind::PriorityAllowance);
        assert_eq!(pools[0].remaining, Some(2.0));
        assert_eq!(
            pools[0].applies_to.attributes.get("latency_class"),
            Some(&"priority".to_string())
        );
    }

    #[test]
    fn codex_sse_converts_text_to_openai_completion() {
        let body = codex_sse_to_chat_completion(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hel\"}\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\ndata: {\"type\":\"response.completed\"}\n\n",
            "gpt-5.5",
        );

        assert_eq!(body["choices"][0]["message"]["content"], "hello");
    }

    fn test_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "auto".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                extra: BTreeMap::new(),
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            response_format: None,
            metadata: None,
            extra: BTreeMap::new(),
        }
    }

    fn routeable_model() -> RouteableModel {
        RouteableModel {
            id: ModelId::new("codex"),
            provider: ProviderId::new("openai_max"),
            upstream_model: "gpt-5.5".to_string(),
            context_window: 1_000_000,
            input_cost_per_million: 0.0,
            output_cost_per_million: 0.0,
            quality: 95,
            capabilities: vec![ModelCapability::Chat],
            attributes: std::collections::BTreeMap::new(),
            display_badges: Vec::new(),
            metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
        }
    }
}
