#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Configuration loading, validation, and conversion helpers for brouter.

use std::path::Path;

use brouter_config_models::BrouterConfig;
use brouter_provider_models::{ModelId, ProviderId, RouteableModel};
use brouter_router_models::ScoringWeights;
use thiserror::Error;

/// Configuration loading and validation error.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read configuration file {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse configuration file {path}: {source}")]
    Parse {
        path: String,
        source: toml::de::Error,
    },
    #[error("model {model_id} references unknown provider {provider_id}")]
    UnknownProvider {
        model_id: String,
        provider_id: String,
    },
    #[error("model {model_id} has context_window = 0")]
    EmptyContextWindow { model_id: String },
}

/// Loads and validates a brouter TOML configuration file.
///
/// # Errors
///
/// Returns an error when:
///
/// * The file cannot be read.
/// * The TOML cannot be parsed.
/// * The parsed configuration is invalid.
pub fn load_config(path: &Path) -> Result<BrouterConfig, ConfigError> {
    let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let config =
        toml::from_str::<BrouterConfig>(&contents).map_err(|source| ConfigError::Parse {
            path: path.display().to_string(),
            source,
        })?;
    validate_config(&config)?;
    Ok(config)
}

/// Validates a parsed brouter configuration.
///
/// # Errors
///
/// Returns an error when a model references an unknown provider or declares an
/// invalid context window.
pub fn validate_config(config: &BrouterConfig) -> Result<(), ConfigError> {
    for (model_id, model) in &config.models {
        if !config.providers.contains_key(&model.provider) {
            return Err(ConfigError::UnknownProvider {
                model_id: model_id.clone(),
                provider_id: model.provider.clone(),
            });
        }
        if model.context_window == 0 {
            return Err(ConfigError::EmptyContextWindow {
                model_id: model_id.clone(),
            });
        }
    }
    Ok(())
}

/// Converts router scoring configuration into concrete scoring weights.
#[must_use]
pub fn scoring_weights(config: &BrouterConfig) -> ScoringWeights {
    let defaults = ScoringWeights::default();
    ScoringWeights {
        quality_weight: config
            .router
            .scoring
            .quality_weight
            .unwrap_or(defaults.quality_weight),
        balanced_cost_weight: config
            .router
            .scoring
            .balanced_cost_weight
            .unwrap_or(defaults.balanced_cost_weight),
        cheapest_cost_weight: config
            .router
            .scoring
            .cheapest_cost_weight
            .unwrap_or(defaults.cheapest_cost_weight),
        local_bonus: config
            .router
            .scoring
            .local_bonus
            .unwrap_or(defaults.local_bonus),
        strongest_quality_weight: config
            .router
            .scoring
            .strongest_quality_weight
            .unwrap_or(defaults.strongest_quality_weight),
        first_message_reasoning_bonus: config
            .router
            .scoring
            .first_message_reasoning_bonus
            .unwrap_or(defaults.first_message_reasoning_bonus),
        code_bonus: config
            .router
            .scoring
            .code_bonus
            .unwrap_or(defaults.code_bonus),
        reasoning_bonus: config
            .router
            .scoring
            .reasoning_bonus
            .unwrap_or(defaults.reasoning_bonus),
    }
}

/// Converts configured models into router candidates.
#[must_use]
pub fn routeable_models(config: &BrouterConfig) -> Vec<RouteableModel> {
    config
        .models
        .iter()
        .map(|(id, model)| RouteableModel {
            id: ModelId::new(id.clone()),
            provider: ProviderId::new(model.provider.clone()),
            upstream_model: model.model.clone(),
            context_window: model.context_window,
            input_cost_per_million: model.input_cost_per_million,
            output_cost_per_million: model.output_cost_per_million,
            quality: model
                .quality
                .unwrap_or_else(|| inferred_quality(&model.capabilities)),
            capabilities: model
                .capabilities
                .iter()
                .filter_map(|capability| capability.parse().ok())
                .collect(),
        })
        .collect()
}

fn inferred_quality(capabilities: &[String]) -> u8 {
    if capabilities
        .iter()
        .any(|capability| capability == "reasoning")
    {
        90
    } else if capabilities.iter().any(|capability| capability == "code") {
        75
    } else {
        60
    }
}

#[cfg(test)]
mod tests {
    use brouter_config_models::{ModelConfig, ProviderConfig, ProviderKind};

    use super::*;

    #[test]
    fn validates_known_provider() {
        let mut config = BrouterConfig::default();
        config.providers.insert(
            "local".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some("http://localhost:11434/v1".to_string()),
                api_key_env: None,
            },
        );
        config.models.insert(
            "fast".to_string(),
            ModelConfig {
                provider: "local".to_string(),
                model: "llama".to_string(),
                context_window: 8_192,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: None,
                capabilities: vec!["chat".to_string()],
            },
        );

        validate_config(&config).expect("config should be valid");
    }
}
