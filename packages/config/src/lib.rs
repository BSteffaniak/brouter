#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Configuration loading, validation, and conversion helpers for brouter.

use std::path::Path;

use brouter_config_models::{BrouterConfig, ProviderKind};
use brouter_provider_models::{ModelId, ProviderId, RouteableModel};
use brouter_router_models::{RoutingObjective, RoutingRule, ScoringWeights};
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

/// Configuration validation warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigWarning {
    ServerApiKeyEnvMissing {
        env_var: String,
    },
    UnknownModelCapability {
        model_id: String,
        capability: String,
    },
    ModelMissingChatCapability {
        model_id: String,
    },
    OpenAiCompatibleProviderMissingBaseUrl {
        provider_id: String,
    },
    ProviderApiKeyEnvMissing {
        provider_id: String,
        env_var: String,
    },
    UnknownRuleIntent {
        rule_name: String,
        intent: String,
    },
    UnknownDefaultObjective {
        objective: String,
    },
    DefaultLocalOnlyWithoutLocalModel,
    UnknownRuleObjective {
        rule_name: String,
        objective: String,
    },
    UnknownRuleCapability {
        rule_name: String,
        capability: String,
    },
    LocalOnlyRuleWithoutLocalModel {
        rule_name: String,
    },
}

impl std::fmt::Display for ConfigWarning {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ServerApiKeyEnvMissing { env_var } => write!(
                formatter,
                "server references missing API key environment variable {env_var}"
            ),
            Self::UnknownModelCapability {
                model_id,
                capability,
            } => {
                write!(
                    formatter,
                    "model {model_id} declares unknown capability {capability}"
                )
            }
            Self::ModelMissingChatCapability { model_id } => {
                write!(
                    formatter,
                    "model {model_id} does not declare chat capability"
                )
            }
            Self::OpenAiCompatibleProviderMissingBaseUrl { provider_id } => write!(
                formatter,
                "open-ai-compatible provider {provider_id} is missing base_url"
            ),
            Self::ProviderApiKeyEnvMissing {
                provider_id,
                env_var,
            } => write!(
                formatter,
                "provider {provider_id} references missing environment variable {env_var}"
            ),
            Self::UnknownRuleIntent { rule_name, intent } => {
                write!(
                    formatter,
                    "rule {rule_name} declares unknown intent {intent}"
                )
            }
            Self::UnknownDefaultObjective { objective } => {
                write!(
                    formatter,
                    "router declares unknown default objective {objective}"
                )
            }
            Self::DefaultLocalOnlyWithoutLocalModel => write!(
                formatter,
                "router default objective can select local_only but no model declares local capability"
            ),
            Self::UnknownRuleObjective {
                rule_name,
                objective,
            } => write!(
                formatter,
                "rule {rule_name} declares unknown objective {objective}"
            ),
            Self::UnknownRuleCapability {
                rule_name,
                capability,
            } => write!(
                formatter,
                "rule {rule_name} declares unknown capability {capability}"
            ),
            Self::LocalOnlyRuleWithoutLocalModel { rule_name } => write!(
                formatter,
                "rule {rule_name} can select local_only but no model declares local capability"
            ),
        }
    }
}

/// Returns non-fatal configuration validation warnings.
#[must_use]
pub fn validate_config_warnings(config: &BrouterConfig) -> Vec<ConfigWarning> {
    let mut warnings = Vec::new();
    collect_server_warnings(config, &mut warnings);
    collect_provider_warnings(config, &mut warnings);
    collect_model_warnings(config, &mut warnings);
    collect_rule_warnings(config, &mut warnings);
    warnings
}

fn collect_server_warnings(config: &BrouterConfig, warnings: &mut Vec<ConfigWarning>) {
    if let Some(env_var) = &config.server.api_key_env
        && std::env::var_os(env_var).is_none()
    {
        warnings.push(ConfigWarning::ServerApiKeyEnvMissing {
            env_var: env_var.clone(),
        });
    }
}

fn collect_provider_warnings(config: &BrouterConfig, warnings: &mut Vec<ConfigWarning>) {
    for (provider_id, provider) in &config.providers {
        if matches!(provider.kind, ProviderKind::OpenAiCompatible) && provider.base_url.is_none() {
            warnings.push(ConfigWarning::OpenAiCompatibleProviderMissingBaseUrl {
                provider_id: provider_id.clone(),
            });
        }
        if let Some(env_var) = &provider.api_key_env
            && std::env::var_os(env_var).is_none()
        {
            warnings.push(ConfigWarning::ProviderApiKeyEnvMissing {
                provider_id: provider_id.clone(),
                env_var: env_var.clone(),
            });
        }
    }
}

fn collect_model_warnings(config: &BrouterConfig, warnings: &mut Vec<ConfigWarning>) {
    for (model_id, model) in &config.models {
        let mut has_chat = false;
        for capability in &model.capabilities {
            match capability.parse() {
                Ok(brouter_provider_models::ModelCapability::Chat) => has_chat = true,
                Ok(_) => {}
                Err(_) => warnings.push(ConfigWarning::UnknownModelCapability {
                    model_id: model_id.clone(),
                    capability: capability.clone(),
                }),
            }
        }
        if !has_chat {
            warnings.push(ConfigWarning::ModelMissingChatCapability {
                model_id: model_id.clone(),
            });
        }
    }
}

fn collect_rule_warnings(config: &BrouterConfig, warnings: &mut Vec<ConfigWarning>) {
    let has_local_model = config.models.values().any(|model| {
        model
            .capabilities
            .iter()
            .any(|capability| capability == "local")
    });
    if !is_known_objective(&config.router.default_objective) {
        warnings.push(ConfigWarning::UnknownDefaultObjective {
            objective: config.router.default_objective.clone(),
        });
    }
    if matches!(
        config.router.default_objective.as_str(),
        "local_only" | "local-only" | "local"
    ) && !has_local_model
    {
        warnings.push(ConfigWarning::DefaultLocalOnlyWithoutLocalModel);
    }
    for rule in &config.router.rules {
        if let Some(intent) = &rule.intent
            && intent
                .parse::<brouter_router_models::PromptIntent>()
                .is_err()
        {
            warnings.push(ConfigWarning::UnknownRuleIntent {
                rule_name: rule.name.clone(),
                intent: intent.clone(),
            });
        }
        if let Some(objective) = &rule.objective {
            if !is_known_objective(objective) {
                warnings.push(ConfigWarning::UnknownRuleObjective {
                    rule_name: rule.name.clone(),
                    objective: objective.clone(),
                });
            }
            if matches!(objective.as_str(), "local_only" | "local-only" | "local")
                && !has_local_model
            {
                warnings.push(ConfigWarning::LocalOnlyRuleWithoutLocalModel {
                    rule_name: rule.name.clone(),
                });
            }
        }
        for capability in rule
            .prefer_capabilities
            .iter()
            .chain(rule.require_capabilities.iter())
        {
            if capability
                .parse::<brouter_provider_models::ModelCapability>()
                .is_err()
            {
                warnings.push(ConfigWarning::UnknownRuleCapability {
                    rule_name: rule.name.clone(),
                    capability: capability.clone(),
                });
            }
        }
    }
}

fn is_known_objective(value: &str) -> bool {
    matches!(
        value,
        "cheapest" | "fastest" | "balanced" | "strongest" | "local_only" | "local-only" | "local"
    )
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

/// Converts configured router rules into runtime routing rules.
#[must_use]
pub fn routing_rules(config: &BrouterConfig) -> Vec<RoutingRule> {
    config
        .router
        .rules
        .iter()
        .map(|rule| RoutingRule {
            name: rule.name.clone(),
            when_contains: rule
                .when_contains
                .iter()
                .map(|value| value.to_lowercase())
                .collect(),
            intent: rule
                .intent
                .as_deref()
                .and_then(|intent| intent.parse().ok()),
            objective: rule.objective.as_deref().map(RoutingObjective::from_name),
            prefer_capabilities: rule
                .prefer_capabilities
                .iter()
                .filter_map(|capability| capability.parse().ok())
                .collect(),
            require_capabilities: rule
                .require_capabilities
                .iter()
                .filter_map(|capability| capability.parse().ok())
                .collect(),
        })
        .collect()
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
    use brouter_config_models::{ModelConfig, ProviderConfig, ProviderKind, RouterRuleConfig};

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
                timeout_ms: None,
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

    #[test]
    fn reports_non_fatal_config_warnings() {
        let mut config = BrouterConfig::default();
        config.providers.insert(
            "local".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: None,
                api_key_env: None,
                timeout_ms: None,
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
                capabilities: vec!["made_up".to_string()],
            },
        );
        config.router.rules.push(RouterRuleConfig {
            name: "private".to_string(),
            objective: Some("local_only".to_string()),
            require_capabilities: vec!["also_made_up".to_string()],
            ..RouterRuleConfig::default()
        });

        let warnings = validate_config_warnings(&config);

        assert!(
            warnings.contains(&ConfigWarning::OpenAiCompatibleProviderMissingBaseUrl {
                provider_id: "local".to_string(),
            })
        );
        assert!(warnings.contains(&ConfigWarning::UnknownModelCapability {
            model_id: "fast".to_string(),
            capability: "made_up".to_string(),
        }));
        assert!(
            warnings.contains(&ConfigWarning::ModelMissingChatCapability {
                model_id: "fast".to_string(),
            })
        );
        assert!(warnings.contains(&ConfigWarning::UnknownRuleCapability {
            rule_name: "private".to_string(),
            capability: "also_made_up".to_string(),
        }));
        assert!(
            warnings.contains(&ConfigWarning::LocalOnlyRuleWithoutLocalModel {
                rule_name: "private".to_string(),
            })
        );
    }
}
