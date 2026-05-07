#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Provider registry and forwarding primitives for brouter.

use std::collections::BTreeMap;

use brouter_api_models::{ChatCompletionRequest, MessageContent};
use brouter_config_models::{BrouterConfig, ProviderConfig, ProviderKind};
use brouter_provider_models::{ProviderId, RouteableModel};
use serde_json::{Map, Value, json};
use thiserror::Error;

/// Registry of configured upstream providers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    ) -> Result<reqwest::Response, ProviderError> {
        let provider = provider_for(registry, model)?;
        match provider.kind {
            ProviderKind::OpenAiCompatible => {
                self.openai_compatible_chat_completions(provider, model, request)
                    .await
            }
            ProviderKind::Anthropic => Err(ProviderError::UnsupportedProviderKind {
                provider_id: model.provider.to_string(),
                kind: "anthropic streaming".to_string(),
            }),
        }
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
        let mut upstream_request = request.clone();
        upstream_request.model.clone_from(&model.upstream_model);

        let mut request_builder = self.http.post(url).json(&upstream_request);
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

/// Provider response payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponse {
    pub status: u16,
    pub body: Value,
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

fn anthropic_request(model: &RouteableModel, request: &ChatCompletionRequest) -> Value {
    let mut body = Map::new();
    body.insert(
        "model".to_string(),
        Value::String(model.upstream_model.clone()),
    );
    body.insert(
        "max_tokens".to_string(),
        Value::from(request.max_tokens.unwrap_or(1_024)),
    );
    if let Some(temperature) = request.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }
    if let Some(top_p) = request.top_p {
        body.insert("top_p".to_string(), Value::from(top_p));
    }
    if request.is_streaming() {
        body.insert("stream".to_string(), Value::Bool(true));
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
    use brouter_api_models::{ChatMessage, MessageContent};
    use brouter_config_models::{BrouterConfig, ModelConfig, ProviderConfig, ProviderKind};
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
        };

        let response = ProviderClient::new()
            .chat_completions(&registry, &model, &chat_request(false))
            .await
            .expect("provider forwarding should succeed");

        assert_eq!(response.status, 200);
        assert_eq!(response.body["model"], "real-upstream-model");
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
        };

        let response = ProviderClient::new()
            .chat_completions_response(&registry, &model, &chat_request(true))
            .await
            .expect("provider forwarding should succeed");
        let body = response.text().await.expect("body should be readable");

        assert!(body.contains("real-streaming-model"));
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
            },
        );
        let mut models = BTreeMap::new();
        models.insert(
            "anthropic_selected".to_string(),
            ModelConfig {
                provider: "anthropic".to_string(),
                model: "claude-test".to_string(),
                context_window: 200_000,
                input_cost_per_million: 3.0,
                output_cost_per_million: 15.0,
                quality: Some(90),
                capabilities: vec!["chat".to_string(), "reasoning".to_string()],
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
            },
        );
        let mut models = BTreeMap::new();
        models.insert(
            "auto_selected".to_string(),
            ModelConfig {
                provider: "fake".to_string(),
                model: "real-upstream-model".to_string(),
                context_window: 8_192,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: Some(80),
                capabilities: vec!["chat".to_string()],
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
            stream: Some(stream),
            tools: None,
            tool_choice: None,
            response_format: None,
            metadata: None,
        }
    }
}
