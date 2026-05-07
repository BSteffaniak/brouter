#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Provider registry and forwarding primitives for brouter.

use std::collections::BTreeMap;

use brouter_api_models::ChatCompletionRequest;
use brouter_config_models::{BrouterConfig, ProviderConfig, ProviderKind};
use brouter_provider_models::{ProviderId, RouteableModel};
use serde_json::Value;
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
        let response = self
            .chat_completions_response(registry, model, request)
            .await?;
        let status = response.status().as_u16();
        let text = response.text().await?;
        let body = parse_provider_body(&text);
        Ok(ProviderResponse { status, body })
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
        let provider =
            registry
                .provider(&model.provider)
                .ok_or_else(|| ProviderError::UnknownProvider {
                    provider_id: model.provider.to_string(),
                })?;
        match provider.kind {
            ProviderKind::OpenAiCompatible => {
                self.openai_compatible_chat_completions(provider, model, request)
                    .await
            }
            ProviderKind::Anthropic => Err(ProviderError::UnsupportedProviderKind {
                provider_id: model.provider.to_string(),
                kind: "anthropic".to_string(),
            }),
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

fn parse_provider_body(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}
