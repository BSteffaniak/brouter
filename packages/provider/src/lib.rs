#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Provider registry and forwarding primitives for brouter.

mod openai_codex;

use std::collections::BTreeMap;
use std::pin::Pin;
use std::time::Duration;

use brouter_api_models::{
    ChatCompletionRequest, EmbeddingsRequest, MessageContent, ReasoningEffort,
};
use brouter_catalog_models::{MetadataProvenance, MetadataSource};
use brouter_config_models::{BrouterConfig, ProviderConfig, ProviderKind};
use brouter_introspection_models::{
    CatalogModel, IntrospectionRequest, IntrospectionSnapshot, IntrospectionWarning, MetadataField,
    ModelCatalogSnapshot, ModelMetadataFields, SnapshotSource,
};
use brouter_provider_models::{ModelCapability, ProviderId, RouteableModel};
use bytes::Bytes;
use futures_util::{Stream, StreamExt, TryStreamExt};
use serde_json::{Map, Value, json};
use thiserror::Error;

/// Registry of configured upstream providers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderRegistry {
    providers: BTreeMap<ProviderId, ProviderConfig>,
}

impl ProviderRegistry {
    /// Builds a registry from loaded configuration.
    #[must_use]
    pub fn from_config(config: &BrouterConfig) -> Self {
        let providers = config
            .providers
            .iter()
            .map(|(id, provider)| (ProviderId::new(id.clone()), provider.clone()))
            .collect();
        Self { providers }
    }

    /// Returns the number of configured providers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Returns true when no providers are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    fn provider(&self, id: &ProviderId) -> Option<&ProviderConfig> {
        self.providers.get(id)
    }

    /// Returns configured provider IDs.
    #[must_use]
    pub fn provider_ids(&self) -> Vec<ProviderId> {
        self.providers.keys().cloned().collect()
    }
}

/// Client used to forward requests to upstream providers.
#[derive(Debug, Clone)]
pub struct ProviderClient {
    http: reqwest::Client,
}

impl Default for ProviderClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderClient {
    /// Creates a provider client.
    #[must_use]
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    /// Fetches generic live introspection metadata for a provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is missing, required configuration is
    /// missing, authentication is unavailable, or the provider HTTP request
    /// fails.
    pub async fn introspect(
        &self,
        registry: &ProviderRegistry,
        provider_id: &ProviderId,
        request: IntrospectionRequest,
    ) -> Result<IntrospectionSnapshot, ProviderError> {
        let provider =
            registry
                .provider(provider_id)
                .ok_or_else(|| ProviderError::UnknownProvider {
                    provider_id: provider_id.to_string(),
                })?;
        match provider.kind {
            ProviderKind::OpenAiCompatible => {
                self.openai_compatible_introspection(provider_id, provider, request)
                    .await
            }
            ProviderKind::Anthropic => {
                self.anthropic_introspection(provider_id, provider, request)
                    .await
            }
            ProviderKind::OpenaiCodex => Ok(unsupported_introspection_snapshot(
                provider_id,
                "openai-codex live introspection adapter is not implemented yet",
            )),
        }
    }

    /// Forwards a chat completion request to the selected upstream model.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected provider is missing, unsupported,
    /// lacks required configuration, has a missing API key environment variable,
    /// or the HTTP request fails.
    pub async fn chat_completions(
        &self,
        registry: &ProviderRegistry,
        model: &RouteableModel,
        request: &ChatCompletionRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let provider = provider_for(registry, model)?;
        match provider.kind {
            ProviderKind::OpenAiCompatible => {
                let response = self
                    .openai_compatible_chat_completions(provider, model, request)
                    .await?;
                buffered_provider_response(response).await
            }
            ProviderKind::Anthropic => {
                self.anthropic_chat_completions(provider, model, request)
                    .await
            }
            ProviderKind::OpenaiCodex => {
                openai_codex::chat_completions(&self.http, provider, model, request).await
            }
        }
    }

    /// Forwards a chat completion request and returns the raw upstream response.
    ///
    /// This is used for streaming responses where the server must proxy the
    /// upstream byte stream without buffering it first.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected provider is missing, unsupported,
    /// lacks required configuration, has a missing API key environment variable,
    /// or the HTTP request fails.
    pub async fn chat_completions_response(
        &self,
        registry: &ProviderRegistry,
        model: &RouteableModel,
        request: &ChatCompletionRequest,
    ) -> Result<ProviderStreamResponse, ProviderError> {
        let provider = provider_for(registry, model)?;
        match provider.kind {
            ProviderKind::OpenAiCompatible => {
                let response = self
                    .openai_compatible_chat_completions(provider, model, request)
                    .await?;
                Ok(raw_stream_response(response))
            }
            ProviderKind::Anthropic => {
                let response = self
                    .anthropic_chat_completions_response(provider, model, request)
                    .await?;
                Ok(anthropic_stream_response(response, &model.upstream_model))
            }
            ProviderKind::OpenaiCodex => {
                openai_codex::chat_completions_response(&self.http, provider, model, request).await
            }
        }
    }

    /// Forwards an embeddings request to an OpenAI-compatible upstream.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected provider is missing, unsupported,
    /// lacks required configuration, has a missing API key environment variable,
    /// or the HTTP request fails.
    pub async fn embeddings(
        &self,
        registry: &ProviderRegistry,
        model: &RouteableModel,
        request: &EmbeddingsRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let provider = provider_for(registry, model)?;
        match provider.kind {
            ProviderKind::OpenAiCompatible => {
                let response = self
                    .openai_compatible_embeddings(provider, model, request)
                    .await?;
                buffered_provider_response(response).await
            }
            ProviderKind::Anthropic => Err(ProviderError::UnsupportedProviderKind {
                provider_id: model.provider.to_string(),
                kind: "anthropic embeddings".to_string(),
            }),
            ProviderKind::OpenaiCodex => Err(ProviderError::UnsupportedProviderKind {
                provider_id: model.provider.to_string(),
                kind: "openai-codex embeddings".to_string(),
            }),
        }
    }

    async fn openai_compatible_introspection(
        &self,
        provider_id: &ProviderId,
        provider: &ProviderConfig,
        request: IntrospectionRequest,
    ) -> Result<IntrospectionSnapshot, ProviderError> {
        let base_url =
            provider
                .base_url
                .as_deref()
                .ok_or_else(|| ProviderError::MissingBaseUrl {
                    provider_id: provider_id.to_string(),
                })?;
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        if !request.include_catalog {
            return Ok(empty_introspection_snapshot(provider_id, &url));
        }
        let mut builder = self.http.get(&url);
        if let Some(timeout) = provider_timeout(provider) {
            builder = builder.timeout(timeout);
        }
        if let Some(api_key_env) = &provider.api_key_env {
            let api_key = std::env::var(api_key_env).map_err(|_| ProviderError::MissingApiKey {
                env_var: api_key_env.clone(),
                provider_id: provider_id.to_string(),
            })?;
            builder = builder.bearer_auth(api_key);
        }
        let response = builder.send().await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(ProviderError::Introspection {
                provider_id: provider_id.to_string(),
                message: format!("GET /models returned {status}: {body}"),
            });
        }
        let value = serde_json::from_str::<Value>(&body).unwrap_or(Value::String(body));
        Ok(IntrospectionSnapshot {
            provider: provider_id.clone(),
            fetched_at_ms: unix_millis(),
            source: SnapshotSource::provider_api(url),
            catalog: Some(openai_compatible_catalog(&value)),
            account: None,
            warnings: Vec::new(),
        })
    }

    async fn anthropic_introspection(
        &self,
        provider_id: &ProviderId,
        provider: &ProviderConfig,
        request: IntrospectionRequest,
    ) -> Result<IntrospectionSnapshot, ProviderError> {
        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        if !request.include_catalog {
            return Ok(empty_introspection_snapshot(provider_id, &url));
        }
        let mut builder = self
            .http
            .get(&url)
            .header("anthropic-version", "2023-06-01");
        if let Some(timeout) = provider_timeout(provider) {
            builder = builder.timeout(timeout);
        }
        if let Some(api_key_env) = &provider.api_key_env {
            let api_key = std::env::var(api_key_env).map_err(|_| ProviderError::MissingApiKey {
                env_var: api_key_env.clone(),
                provider_id: provider_id.to_string(),
            })?;
            builder = builder.header("x-api-key", api_key);
        }
        let response = builder.send().await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(ProviderError::Introspection {
                provider_id: provider_id.to_string(),
                message: format!("GET /models returned {status}: {body}"),
            });
        }
        let value = serde_json::from_str::<Value>(&body).unwrap_or(Value::String(body));
        Ok(IntrospectionSnapshot {
            provider: provider_id.clone(),
            fetched_at_ms: unix_millis(),
            source: SnapshotSource::provider_api(url),
            catalog: Some(partial_model_catalog(&value)),
            account: None,
            warnings: vec![IntrospectionWarning::new(
                "partial_catalog",
                "Anthropic model list does not expose context/pricing in the generic adapter",
            )],
        })
    }

    async fn anthropic_chat_completions(
        &self,
        provider: &ProviderConfig,
        model: &RouteableModel,
        request: &ChatCompletionRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let url = format!("{}/messages", base_url.trim_end_matches('/'));
        let anthropic_request = anthropic_request(model, request);
        let mut request_builder = self
            .http
            .post(url)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_request);
        if let Some(timeout) = provider_timeout(provider) {
            request_builder = request_builder.timeout(timeout);
        }
        if let Some(api_key_env) = &provider.api_key_env {
            let api_key = std::env::var(api_key_env).map_err(|_| ProviderError::MissingApiKey {
                env_var: api_key_env.clone(),
                provider_id: model.provider.to_string(),
            })?;
            request_builder = request_builder.header("x-api-key", api_key);
        }

        let response = request_builder.send().await?;
        let status = response.status();
        let response = buffered_provider_response(response).await?;
        if status.is_success() {
            Ok(ProviderResponse {
                status: response.status,
                body: anthropic_response_to_openai(&response.body, model),
            })
        } else {
            Ok(response)
        }
    }

    async fn anthropic_chat_completions_response(
        &self,
        provider: &ProviderConfig,
        model: &RouteableModel,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let base_url = provider
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let url = format!("{}/messages", base_url.trim_end_matches('/'));
        let anthropic_request = anthropic_request(model, request);
        let mut request_builder = self
            .http
            .post(url)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_request);
        if let Some(timeout) = provider_timeout(provider) {
            request_builder = request_builder.timeout(timeout);
        }
        if let Some(api_key_env) = &provider.api_key_env {
            let api_key = std::env::var(api_key_env).map_err(|_| ProviderError::MissingApiKey {
                env_var: api_key_env.clone(),
                provider_id: model.provider.to_string(),
            })?;
            request_builder = request_builder.header("x-api-key", api_key);
        }
        Ok(request_builder.send().await?)
    }

    async fn openai_compatible_chat_completions(
        &self,
        provider: &ProviderConfig,
        model: &RouteableModel,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let base_url =
            provider
                .base_url
                .as_deref()
                .ok_or_else(|| ProviderError::MissingBaseUrl {
                    provider_id: model.provider.to_string(),
                })?;
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let upstream_request = openai_compatible_request_body(provider, model, request);

        let mut request_builder = self.http.post(url).json(&upstream_request);
        if let Some(timeout) = provider_timeout(provider) {
            request_builder = request_builder.timeout(timeout);
        }
        if let Some(api_key_env) = &provider.api_key_env {
            let api_key = std::env::var(api_key_env).map_err(|_| ProviderError::MissingApiKey {
                env_var: api_key_env.clone(),
                provider_id: model.provider.to_string(),
            })?;
            request_builder = request_builder.bearer_auth(api_key);
        }

        Ok(request_builder.send().await?)
    }

    async fn openai_compatible_embeddings(
        &self,
        provider: &ProviderConfig,
        model: &RouteableModel,
        request: &EmbeddingsRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let base_url =
            provider
                .base_url
                .as_deref()
                .ok_or_else(|| ProviderError::MissingBaseUrl {
                    provider_id: model.provider.to_string(),
                })?;
        let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
        let mut upstream_request = request.clone();
        upstream_request.model.clone_from(&model.upstream_model);

        let mut request_builder = self.http.post(url).json(&upstream_request);
        if let Some(timeout) = provider_timeout(provider) {
            request_builder = request_builder.timeout(timeout);
        }
        if let Some(api_key_env) = &provider.api_key_env {
            let api_key = std::env::var(api_key_env).map_err(|_| ProviderError::MissingApiKey {
                env_var: api_key_env.clone(),
                provider_id: model.provider.to_string(),
            })?;
            request_builder = request_builder.bearer_auth(api_key);
        }

        Ok(request_builder.send().await?)
    }
}

fn openai_compatible_request_body(
    provider: &ProviderConfig,
    model: &RouteableModel,
    request: &ChatCompletionRequest,
) -> Value {
    let mut body = serde_json::to_value(request).unwrap_or_else(|_| json!({}));
    let Some(object) = body.as_object_mut() else {
        return body;
    };
    object.insert(
        "model".to_string(),
        Value::String(model.upstream_model.clone()),
    );
    match request.reasoning_effort {
        Some(ReasoningEffort::None) => {
            object.remove("reasoning_effort");
        }
        Some(ReasoningEffort::Max) => {
            object.insert(
                "reasoning_effort".to_string(),
                Value::String("high".to_string()),
            );
        }
        Some(
            ReasoningEffort::Minimal
            | ReasoningEffort::Low
            | ReasoningEffort::Medium
            | ReasoningEffort::High,
        )
        | None => {}
    }
    apply_attribute_mappings(provider, model, &mut body);
    body
}

pub(crate) fn apply_attribute_mappings(
    provider: &ProviderConfig,
    model: &RouteableModel,
    body: &mut Value,
) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    for (key, value) in &model.attributes {
        let Some(mapping) = provider
            .attribute_mappings
            .get(key)
            .and_then(|values| values.get(value))
        else {
            continue;
        };
        for field in &mapping.omit_request_fields {
            object.remove(field);
        }
        for (field, field_value) in &mapping.request_fields {
            object.insert(field.clone(), field_value.clone());
        }
    }
}

fn openai_compatible_catalog(value: &Value) -> ModelCatalogSnapshot {
    let mut catalog = ModelCatalogSnapshot::default();
    for model in model_array(value) {
        let Some(id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        let provenance = MetadataProvenance::new(MetadataSource::ProviderApi);
        let supported_parameters = string_array(model.get("supported_parameters"));
        let mut capabilities = vec![ModelCapability::Chat];
        if supported_parameters.iter().any(|value| value == "tools") {
            capabilities.push(ModelCapability::Tools);
        }
        if supported_parameters
            .iter()
            .any(|value| value == "response_format")
        {
            capabilities.push(ModelCapability::Json);
        }
        let prompt_cost = model
            .get("pricing")
            .and_then(|pricing| pricing.get("prompt"))
            .and_then(value_as_f64)
            .map(|cost| cost * 1_000_000.0);
        let completion_cost = model
            .get("pricing")
            .and_then(|pricing| pricing.get("completion"))
            .and_then(value_as_f64)
            .map(|cost| cost * 1_000_000.0);
        catalog.models.insert(
            id.to_string(),
            CatalogModel {
                upstream_model: id.to_string(),
                aliases: Vec::new(),
                fields: ModelMetadataFields {
                    context_window: optional_field(
                        model
                            .get("context_length")
                            .or_else(|| model.get("context_window"))
                            .and_then(value_as_u32),
                        provenance.clone(),
                    ),
                    max_output_tokens: optional_field(
                        model
                            .get("top_provider")
                            .and_then(|provider| provider.get("max_completion_tokens"))
                            .or_else(|| model.get("max_output_tokens"))
                            .and_then(value_as_u32),
                        provenance.clone(),
                    ),
                    input_cost_per_million: optional_field(prompt_cost, provenance.clone()),
                    output_cost_per_million: optional_field(completion_cost, provenance.clone()),
                    capabilities: MetadataField::new(capabilities, provenance.clone()),
                    supported_parameters: MetadataField::new(supported_parameters, provenance),
                },
                raw_capabilities: BTreeMap::new(),
                raw_parameters: model
                    .as_object()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
            },
        );
    }
    catalog
}

fn partial_model_catalog(value: &Value) -> ModelCatalogSnapshot {
    let mut catalog = ModelCatalogSnapshot::default();
    let provenance = MetadataProvenance::new(MetadataSource::ProviderApi);
    for model in model_array(value) {
        let Some(id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        catalog.models.insert(
            id.to_string(),
            CatalogModel {
                upstream_model: id.to_string(),
                aliases: Vec::new(),
                fields: ModelMetadataFields {
                    capabilities: MetadataField::new(
                        vec![ModelCapability::Chat],
                        provenance.clone(),
                    ),
                    ..ModelMetadataFields::default()
                },
                raw_capabilities: BTreeMap::new(),
                raw_parameters: model
                    .as_object()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
            },
        );
    }
    catalog
}

fn model_array(value: &Value) -> Vec<&Value> {
    value
        .get("data")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |models| models.iter().collect())
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
}

const fn optional_field<T>(value: Option<T>, provenance: MetadataProvenance) -> MetadataField<T> {
    MetadataField { value, provenance }
}

fn value_as_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn value_as_u32(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn empty_introspection_snapshot(provider_id: &ProviderId, endpoint: &str) -> IntrospectionSnapshot {
    IntrospectionSnapshot {
        provider: provider_id.clone(),
        fetched_at_ms: unix_millis(),
        source: SnapshotSource::provider_api(endpoint),
        catalog: None,
        account: None,
        warnings: Vec::new(),
    }
}

fn unsupported_introspection_snapshot(
    provider_id: &ProviderId,
    message: &str,
) -> IntrospectionSnapshot {
    IntrospectionSnapshot {
        provider: provider_id.clone(),
        fetched_at_ms: unix_millis(),
        source: SnapshotSource {
            kind: brouter_introspection_models::SnapshotSourceKind::Unknown,
            endpoint: None,
            label: None,
        },
        catalog: None,
        account: None,
        warnings: vec![IntrospectionWarning::new("unsupported", message)],
    }
}

fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}

fn provider_timeout(provider: &ProviderConfig) -> Option<Duration> {
    provider.timeout_ms.map(Duration::from_millis)
}

/// Provider response payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponse {
    pub status: u16,
    pub body: Value,
}

/// Provider streaming response payload.
pub struct ProviderStreamResponse {
    pub status: u16,
    pub stream: Pin<Box<dyn Stream<Item = Result<Bytes, ProviderError>> + Send>>,
}

/// Provider forwarding error.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("unknown provider {provider_id}")]
    UnknownProvider { provider_id: String },
    #[error("provider {provider_id} requires base_url")]
    MissingBaseUrl { provider_id: String },
    #[error("provider {provider_id} requires environment variable {env_var}")]
    MissingApiKey {
        provider_id: String,
        env_var: String,
    },
    #[error("provider {provider_id} kind {kind} is not implemented yet")]
    UnsupportedProviderKind { provider_id: String, kind: String },
    #[error("provider {provider_id} introspection failed: {message}")]
    Introspection {
        provider_id: String,
        message: String,
    },
    #[error("provider {provider_id} authentication failed: {message}")]
    Auth {
        provider_id: String,
        message: String,
    },
    #[error("upstream HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
}

fn provider_for<'a>(
    registry: &'a ProviderRegistry,
    model: &RouteableModel,
) -> Result<&'a ProviderConfig, ProviderError> {
    registry
        .provider(&model.provider)
        .ok_or_else(|| ProviderError::UnknownProvider {
            provider_id: model.provider.to_string(),
        })
}

async fn buffered_provider_response(
    response: reqwest::Response,
) -> Result<ProviderResponse, ProviderError> {
    let status = response.status().as_u16();
    let text = response.text().await?;
    let body = parse_provider_body(&text);
    Ok(ProviderResponse { status, body })
}

fn parse_provider_body(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

pub(crate) fn raw_stream_response(response: reqwest::Response) -> ProviderStreamResponse {
    let status = response.status().as_u16();
    let stream = response.bytes_stream().map_err(ProviderError::Http);
    ProviderStreamResponse {
        status,
        stream: Box::pin(stream),
    }
}

fn anthropic_stream_response(response: reqwest::Response, model: &str) -> ProviderStreamResponse {
    let status = response.status().as_u16();
    if !response.status().is_success() {
        return raw_stream_response(response);
    }
    let model = model.to_string();
    let stream = response.bytes_stream().map(move |chunk| {
        chunk
            .map(|bytes| anthropic_sse_chunk_to_openai(&String::from_utf8_lossy(&bytes), &model))
            .map(Bytes::from)
            .map_err(ProviderError::Http)
    });
    ProviderStreamResponse {
        status,
        stream: Box::pin(stream),
    }
}

fn anthropic_sse_chunk_to_openai(chunk: &str, model: &str) -> String {
    let mut output = String::new();
    for line in chunk.lines() {
        if let Some(event) = line.strip_prefix("data: ")
            && let Some(openai_event) = anthropic_event_to_openai(event, model)
        {
            output.push_str("data: ");
            output.push_str(&openai_event);
            output.push_str("\n\n");
        }
    }
    output
}

fn anthropic_event_to_openai(event: &str, model: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(event).ok()?;
    match value.get("type").and_then(Value::as_str) {
        Some("message_start") => Some(
            json!({
                "id": value.get("message").and_then(|message| message.get("id")).cloned().unwrap_or_else(|| Value::String("anthropic".to_string())),
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            })
            .to_string(),
        ),
        Some("content_block_delta") => value
            .get("delta")
            .and_then(|delta| delta.get("text"))
            .and_then(Value::as_str)
            .map(|text| {
                json!({
                    "id": "anthropic",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                })
                .to_string()
            }),
        Some("message_stop") => Some("[DONE]".to_string()),
        _ => None,
    }
}

fn anthropic_request(model: &RouteableModel, request: &ChatCompletionRequest) -> Value {
    let mut body = Map::new();
    body.insert(
        "model".to_string(),
        Value::String(model.upstream_model.clone()),
    );
    let max_tokens = anthropic_max_tokens(request);
    body.insert("max_tokens".to_string(), Value::from(max_tokens));
    if let Some(temperature) = request.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".to_string(), Value::from(top_p));
    }
    if request.is_streaming() {
        body.insert("stream".to_string(), Value::Bool(true));
    }
    if let Some(thinking) = anthropic_thinking(request.reasoning_effort, max_tokens) {
        body.insert("thinking".to_string(), thinking);
        body.remove("temperature");
    }

    let system = anthropic_system_prompt(request);
    if !system.is_empty() {
        body.insert("system".to_string(), Value::String(system));
    }
    body.insert(
        "messages".to_string(),
        Value::Array(anthropic_messages(request)),
    );
    Value::Object(body)
}

fn anthropic_max_tokens(request: &ChatCompletionRequest) -> u32 {
    let requested = request.max_tokens.unwrap_or(1_024);
    request
        .reasoning_effort
        .and_then(anthropic_desired_thinking_budget)
        .map_or(requested, |budget| {
            requested.max(budget.saturating_add(1_024))
        })
}

fn anthropic_thinking(effort: Option<ReasoningEffort>, max_tokens: u32) -> Option<Value> {
    let desired_budget = effort.and_then(anthropic_desired_thinking_budget)?;
    let budget_tokens = desired_budget.min(max_tokens.saturating_sub(1)).max(1_024);
    Some(json!({"type": "enabled", "budget_tokens": budget_tokens}))
}

const fn anthropic_desired_thinking_budget(effort: ReasoningEffort) -> Option<u32> {
    match effort {
        ReasoningEffort::None => None,
        ReasoningEffort::Minimal => Some(1_024),
        ReasoningEffort::Low => Some(2_048),
        ReasoningEffort::Medium => Some(4_096),
        ReasoningEffort::High => Some(8_192),
        ReasoningEffort::Max => Some(16_384),
    }
}

fn anthropic_system_prompt(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .iter()
        .filter(|message| message.role == "system")
        .map(|message| message.content.as_text())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn anthropic_messages(request: &ChatCompletionRequest) -> Vec<Value> {
    request
        .messages
        .iter()
        .filter(|message| message.role == "user" || message.role == "assistant")
        .map(|message| {
            json!({
                "role": message.role.clone(),
                "content": anthropic_message_content(&message.content),
            })
        })
        .collect()
}

fn anthropic_message_content(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => Value::String(text.clone()),
        MessageContent::Parts(parts) => Value::Array(parts.clone()),
        MessageContent::Null => Value::String(String::new()),
    }
}

fn anthropic_response_to_openai(response: &Value, model: &RouteableModel) -> Value {
    let content = response
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let finish_reason = response
        .get("stop_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    json!({
        "id": response.get("id").cloned().unwrap_or_else(|| Value::String("anthropic".to_string())),
        "object": "chat.completion",
        "created": 0,
        "model": model.upstream_model.clone(),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
            "completion_tokens": usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
            "total_tokens": usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0)
                + usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::net::SocketAddr;

    use axum::routing::post;
    use axum::{Json, Router as AxumRouter};
    use brouter_api_models::{ChatMessage, MessageContent, ReasoningEffort};
    use brouter_config_models::{BrouterConfig, ModelConfig, ProviderConfig, ProviderKind};
    use brouter_introspection_models::IntrospectionRequest;
    use brouter_provider_models::{ModelCapability, ModelId, ProviderId};
    use serde_json::{Value, json};

    use super::*;

    #[tokio::test]
    async fn openai_compatible_forwarding_rewrites_model() {
        let upstream = spawn_echo_upstream().await;
        let config = test_config(upstream);
        let registry = ProviderRegistry::from_config(&config);
        let model = RouteableModel {
            id: ModelId::new("auto_selected"),
            provider: ProviderId::new("fake"),
            upstream_model: "real-upstream-model".to_string(),
            context_window: 8_192,
            input_cost_per_million: 0.0,
            output_cost_per_million: 0.0,
            quality: 80,
            capabilities: vec![ModelCapability::Chat],
            attributes: BTreeMap::new(),
            display_badges: Vec::new(),
            metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
        };

        let response = ProviderClient::new()
            .chat_completions(&registry, &model, &chat_request(false))
            .await
            .expect("provider forwarding should succeed");

        assert_eq!(response.status, 200);
        assert_eq!(response.body["model"], "real-upstream-model");
    }

    #[test]
    fn openai_compatible_request_maps_reasoning_effort() {
        let model = RouteableModel {
            id: ModelId::new("auto_selected"),
            provider: ProviderId::new("fake"),
            upstream_model: "real-upstream-model".to_string(),
            context_window: 8_192,
            input_cost_per_million: 0.0,
            output_cost_per_million: 0.0,
            quality: 80,
            capabilities: vec![ModelCapability::Chat],
            attributes: BTreeMap::new(),
            display_badges: Vec::new(),
            metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
        };
        let provider = ProviderConfig {
            kind: ProviderKind::OpenAiCompatible,
            base_url: Some("http://localhost/v1".to_string()),
            api_key_env: None,
            timeout_ms: None,
            max_estimated_cost: None,
            auth_backend: None,
            auth_profile: None,
            auth_vault_path: None,
            introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
            attribute_mappings: BTreeMap::new(),
        };
        let mut request = chat_request(false);

        request.reasoning_effort = Some(ReasoningEffort::None);
        let body = openai_compatible_request_body(&provider, &model, &request);
        assert!(body.get("reasoning_effort").is_none());

        request.reasoning_effort = Some(ReasoningEffort::Max);
        let body = openai_compatible_request_body(&provider, &model, &request);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[tokio::test]
    async fn openai_compatible_introspection_maps_rich_model_catalog() {
        let upstream = spawn_models_upstream().await;
        let config = test_config(upstream);
        let registry = ProviderRegistry::from_config(&config);
        let snapshot = ProviderClient::new()
            .introspect(
                &registry,
                &ProviderId::new("fake"),
                IntrospectionRequest::default(),
            )
            .await
            .expect("introspection should succeed");

        let model = snapshot
            .catalog
            .expect("catalog should exist")
            .models
            .get("real-upstream-model")
            .expect("model should be mapped")
            .clone();
        assert_eq!(model.fields.context_window.value, Some(128_000));
        assert_eq!(model.fields.input_cost_per_million.value, Some(1.0));
        assert!(
            model
                .fields
                .capabilities
                .value
                .expect("capabilities should exist")
                .contains(&ModelCapability::Tools)
        );
    }

    #[tokio::test]
    async fn anthropic_forwarding_converts_response_to_openai_shape() {
        let upstream = spawn_anthropic_upstream().await;
        let config = anthropic_config(upstream);
        let registry = ProviderRegistry::from_config(&config);
        let model = RouteableModel {
            id: ModelId::new("anthropic_selected"),
            provider: ProviderId::new("anthropic"),
            upstream_model: "claude-test".to_string(),
            context_window: 200_000,
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
            quality: 90,
            capabilities: vec![ModelCapability::Chat, ModelCapability::Reasoning],
            attributes: BTreeMap::new(),
            display_badges: Vec::new(),
            metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
        };

        let response = ProviderClient::new()
            .chat_completions(&registry, &model, &chat_request(false))
            .await
            .expect("anthropic forwarding should succeed");

        assert_eq!(response.status, 200);
        assert_eq!(response.body["object"], "chat.completion");
        assert_eq!(response.body["model"], "claude-test");
        assert_eq!(
            response.body["choices"][0]["message"]["content"],
            "anthropic hello"
        );
    }

    #[tokio::test]
    async fn anthropic_streaming_converts_to_openai_sse() {
        let upstream = spawn_anthropic_streaming_upstream().await;
        let config = anthropic_config(upstream);
        let registry = ProviderRegistry::from_config(&config);
        let model = RouteableModel {
            id: ModelId::new("anthropic_selected"),
            provider: ProviderId::new("anthropic"),
            upstream_model: "claude-test".to_string(),
            context_window: 200_000,
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
            quality: 90,
            capabilities: vec![ModelCapability::Chat],
            attributes: BTreeMap::new(),
            display_badges: Vec::new(),
            metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
        };

        let mut response = ProviderClient::new()
            .chat_completions_response(&registry, &model, &chat_request(true))
            .await
            .expect("anthropic stream should succeed");
        let body = response
            .stream
            .next()
            .await
            .expect("stream should yield")
            .expect("body should be readable");
        let body = String::from_utf8(body.to_vec()).expect("body should be utf8");

        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("anthropic hello"));
        assert!(body.contains("[DONE]"));
    }

    #[tokio::test]
    async fn openai_compatible_streaming_returns_raw_response() {
        let upstream = spawn_echo_upstream().await;
        let config = test_config(upstream);
        let registry = ProviderRegistry::from_config(&config);
        let model = RouteableModel {
            id: ModelId::new("auto_selected"),
            provider: ProviderId::new("fake"),
            upstream_model: "real-streaming-model".to_string(),
            context_window: 8_192,
            input_cost_per_million: 0.0,
            output_cost_per_million: 0.0,
            quality: 80,
            capabilities: vec![ModelCapability::Chat],
            attributes: BTreeMap::new(),
            display_badges: Vec::new(),
            metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
        };

        let mut response = ProviderClient::new()
            .chat_completions_response(&registry, &model, &chat_request(true))
            .await
            .expect("provider forwarding should succeed");
        let body = response
            .stream
            .next()
            .await
            .expect("stream should yield")
            .expect("body should be readable");
        let body = String::from_utf8(body.to_vec()).expect("body should be utf8");

        assert!(body.contains("real-streaming-model"));
    }

    async fn spawn_anthropic_streaming_upstream() -> String {
        async fn messages() -> &'static str {
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-test\"}}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"anthropic hello\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n"
        }
        let app = AxumRouter::new().route("/v1/messages", post(messages));
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

    async fn spawn_anthropic_upstream() -> String {
        async fn messages(Json(body): Json<Value>) -> Json<Value> {
            assert_eq!(body["model"], "claude-test");
            Json(json!({
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": body["model"].clone(),
                "content": [{"type": "text", "text": "anthropic hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2}
            }))
        }

        let app = AxumRouter::new().route("/v1/messages", post(messages));
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

    async fn spawn_models_upstream() -> String {
        async fn models() -> Json<Value> {
            Json(json!({
                "data": [{
                    "id": "real-upstream-model",
                    "context_length": 128_000,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["tools", "response_format"]
                }]
            }))
        }
        let app = AxumRouter::new().route("/v1/models", axum::routing::get(models));
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

    async fn spawn_echo_upstream() -> String {
        async fn echo(Json(body): Json<Value>) -> Json<Value> {
            Json(json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": body["model"].clone(),
                "choices": []
            }))
        }

        let app = AxumRouter::new().route("/v1/chat/completions", post(echo));
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

    fn anthropic_config(base_url: String) -> BrouterConfig {
        let mut providers = BTreeMap::new();
        providers.insert(
            "anthropic".to_string(),
            ProviderConfig {
                kind: ProviderKind::Anthropic,
                base_url: Some(base_url),
                api_key_env: None,
                timeout_ms: None,
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                attribute_mappings: BTreeMap::new(),
            },
        );
        let mut models = BTreeMap::new();
        models.insert(
            "anthropic_selected".to_string(),
            ModelConfig {
                provider: "anthropic".to_string(),
                model: "claude-test".to_string(),
                context_window: Some(200_000),
                input_cost_per_million: 3.0,
                output_cost_per_million: 15.0,
                quality: Some(90),
                capabilities: vec!["chat".to_string(), "reasoning".to_string()],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
            },
        );
        BrouterConfig {
            providers,
            models,
            ..BrouterConfig::default()
        }
    }

    fn test_config(base_url: String) -> BrouterConfig {
        let mut providers = BTreeMap::new();
        providers.insert(
            "fake".to_string(),
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
                attribute_mappings: BTreeMap::new(),
            },
        );
        let mut models = BTreeMap::new();
        models.insert(
            "auto_selected".to_string(),
            ModelConfig {
                provider: "fake".to_string(),
                model: "real-upstream-model".to_string(),
                context_window: Some(8_192),
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: Some(80),
                capabilities: vec!["chat".to_string()],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
            },
        );
        BrouterConfig {
            providers,
            models,
            ..BrouterConfig::default()
        }
    }

    fn chat_request(stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "brouter/auto".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            stream: Some(stream),
            tools: None,
            tool_choice: None,
            response_format: None,
            metadata: None,
            extra: BTreeMap::new(),
        }
    }
}
