#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Configuration loading, validation, and conversion helpers for brouter.

use std::path::Path;

use brouter_catalog::FallbackCatalog;
use brouter_catalog_models::{
    MetadataOverrideMode, MetadataProvenance, MetadataSource, ResolvedModelMetadata,
};
use brouter_config_models::{BrouterConfig, ProviderKind};
use brouter_introspection_models::{CatalogModel, IntrospectionSnapshot};
use brouter_provider_models::{ModelCapability, ModelId, ProviderId, RouteableModel};
use brouter_router_models::{
    CandidateDenyRule, CandidateSelector, ContextPolicy, RoutingObjective, RoutingProfile,
    RoutingRule, ScoringWeights,
};
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
        if model.context_window == Some(0) {
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
    UnknownAliasTarget {
        alias: String,
        target: String,
    },
    UnknownGroupModel {
        group: String,
        model_id: String,
    },
    UnknownDefaultProfile {
        profile: String,
    },
    UnknownProfileModel {
        profile: String,
        model_id: String,
    },
    UnknownProfileProvider {
        profile: String,
        provider_id: String,
    },
    UnknownProfileCapability {
        profile: String,
        capability: String,
    },
    UnknownProfileObjective {
        profile: String,
        objective: String,
    },
    MissingModelContextMetadata {
        model_id: String,
    },
    ForcedMetadataOverrideMissingProvenance {
        model_id: String,
    },
}

impl std::fmt::Display for ConfigWarning {
    #[allow(clippy::too_many_lines)]
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
                    "model {model_id} does not declare chat or embeddings capability"
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
            Self::UnknownAliasTarget { alias, target } => {
                write!(formatter, "alias {alias} references unknown model {target}")
            }
            Self::UnknownGroupModel { group, model_id } => {
                write!(
                    formatter,
                    "group {group} references unknown model {model_id}"
                )
            }
            Self::UnknownDefaultProfile { profile } => {
                write!(
                    formatter,
                    "router default profile {profile} is not configured"
                )
            }
            Self::UnknownProfileModel { profile, model_id } => write!(
                formatter,
                "profile {profile} references unknown model {model_id}"
            ),
            Self::UnknownProfileProvider {
                profile,
                provider_id,
            } => write!(
                formatter,
                "profile {profile} references unknown provider {provider_id}"
            ),
            Self::UnknownProfileCapability {
                profile,
                capability,
            } => write!(
                formatter,
                "profile {profile} declares unknown capability {capability}"
            ),
            Self::UnknownProfileObjective { profile, objective } => write!(
                formatter,
                "profile {profile} declares unknown objective {objective}"
            ),
            Self::MissingModelContextMetadata { model_id } => write!(
                formatter,
                "model {model_id} has no context metadata from config, override, or fallback catalog"
            ),
            Self::ForcedMetadataOverrideMissingProvenance { model_id } => write!(
                formatter,
                "model {model_id} uses forced metadata overrides without reason or source_url"
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
    collect_alias_warnings(config, &mut warnings);
    collect_profile_warnings(config, &mut warnings);
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
    let catalog = FallbackCatalog::default();
    for (model_id, model) in &config.models {
        let mut has_chat = false;
        let mut has_embeddings = false;
        for capability in &model.capabilities {
            match capability.parse() {
                Ok(brouter_provider_models::ModelCapability::Chat) => has_chat = true,
                Ok(brouter_provider_models::ModelCapability::Embeddings) => has_embeddings = true,
                Ok(_) => {}
                Err(_) => warnings.push(ConfigWarning::UnknownModelCapability {
                    model_id: model_id.clone(),
                    capability: capability.clone(),
                }),
            }
        }
        let catalog_model = config.providers.get(&model.provider).and_then(|provider| {
            catalog.find(
                provider.kind,
                &provider_family(&model.provider, provider.kind, provider.base_url.as_deref()),
                &model.model,
            )
        });
        if let Some(catalog_model) = catalog_model.as_ref() {
            for capability in &catalog_model.capabilities {
                match capability.parse() {
                    Ok(ModelCapability::Chat) => has_chat = true,
                    Ok(ModelCapability::Embeddings) => has_embeddings = true,
                    Ok(_) | Err(_) => {}
                }
            }
        }
        if !has_chat && !has_embeddings {
            warnings.push(ConfigWarning::ModelMissingChatCapability {
                model_id: model_id.clone(),
            });
        }
        let override_config = model.metadata_overrides.as_ref();
        if model.context_window.is_none()
            && override_config
                .and_then(|overrides| overrides.context_window)
                .is_none()
            && catalog_model
                .and_then(|metadata| metadata.context_window)
                .is_none()
        {
            warnings.push(ConfigWarning::MissingModelContextMetadata {
                model_id: model_id.clone(),
            });
        }
        if override_config.is_some_and(|overrides| {
            MetadataOverrideMode::from_name(&overrides.mode) == MetadataOverrideMode::Force
                && overrides.reason.is_none()
                && overrides.source_url.is_none()
        }) {
            warnings.push(ConfigWarning::ForcedMetadataOverrideMissingProvenance {
                model_id: model_id.clone(),
            });
        }
    }
}

fn collect_alias_warnings(config: &BrouterConfig, warnings: &mut Vec<ConfigWarning>) {
    for (alias, target) in &config.router.aliases {
        if !config.models.contains_key(target) {
            warnings.push(ConfigWarning::UnknownAliasTarget {
                alias: alias.clone(),
                target: target.clone(),
            });
        }
    }
    for (group, models) in &config.router.groups {
        for model_id in models {
            if !config.models.contains_key(model_id) {
                warnings.push(ConfigWarning::UnknownGroupModel {
                    group: group.clone(),
                    model_id: model_id.clone(),
                });
            }
        }
    }
}

fn collect_profile_warnings(config: &BrouterConfig, warnings: &mut Vec<ConfigWarning>) {
    if let Some(default_profile) = &config.router.default_profile
        && !config.router.profiles.contains_key(default_profile)
    {
        warnings.push(ConfigWarning::UnknownDefaultProfile {
            profile: default_profile.clone(),
        });
    }
    for (profile_name, profile) in &config.router.profiles {
        if let Some(objective) = &profile.objective
            && !is_known_objective(objective)
        {
            warnings.push(ConfigWarning::UnknownProfileObjective {
                profile: profile_name.clone(),
                objective: objective.clone(),
            });
        }
        for selector in &profile.allow {
            collect_selector_warnings(config, warnings, profile_name, selector);
        }
        for deny in &profile.deny {
            let selector = brouter_config_models::CandidateSelectorConfig {
                models: deny.models.clone(),
                upstream_models: deny.upstream_models.clone(),
                providers: deny.providers.clone(),
                capabilities: deny.capabilities.clone(),
                attributes: deny.attributes.clone(),
            };
            collect_selector_warnings(config, warnings, profile_name, &selector);
        }
    }
}

fn collect_selector_warnings(
    config: &BrouterConfig,
    warnings: &mut Vec<ConfigWarning>,
    profile_name: &str,
    selector: &brouter_config_models::CandidateSelectorConfig,
) {
    for model_id in &selector.models {
        if !config.models.contains_key(model_id) {
            warnings.push(ConfigWarning::UnknownProfileModel {
                profile: profile_name.to_string(),
                model_id: model_id.clone(),
            });
        }
    }
    for provider_id in &selector.providers {
        if !config.providers.contains_key(provider_id) {
            warnings.push(ConfigWarning::UnknownProfileProvider {
                profile: profile_name.to_string(),
                provider_id: provider_id.clone(),
            });
        }
    }
    for capability in &selector.capabilities {
        if capability
            .parse::<brouter_provider_models::ModelCapability>()
            .is_err()
        {
            warnings.push(ConfigWarning::UnknownProfileCapability {
                profile: profile_name.to_string(),
                capability: capability.clone(),
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
        policy_penalty: config
            .router
            .scoring
            .policy_penalty
            .unwrap_or(defaults.policy_penalty),
    }
}

/// Converts context configuration into a runtime context policy.
#[must_use]
pub const fn context_policy(config: &BrouterConfig) -> ContextPolicy {
    ContextPolicy {
        safety_margin_ratio: config.router.context.safety_margin_ratio,
        preserve_session_context_floor: config.router.context.preserve_session_context_floor,
        allow_context_downgrade: config.router.context.allow_context_downgrade,
    }
}

/// Converts configured router profiles into runtime routing profiles.
#[must_use]
pub fn routing_profiles(
    config: &BrouterConfig,
) -> std::collections::BTreeMap<String, RoutingProfile> {
    config
        .router
        .profiles
        .iter()
        .map(|(name, profile)| {
            (
                name.clone(),
                RoutingProfile {
                    objective: profile
                        .objective
                        .as_deref()
                        .map(RoutingObjective::from_name),
                    allow: profile.allow.iter().map(selector_config).collect(),
                    deny: profile
                        .deny
                        .iter()
                        .map(|deny| CandidateDenyRule {
                            selector: deny_selector_config(deny),
                            reason: deny.reason.clone(),
                            hard: deny.hard,
                            penalty: deny.penalty,
                        })
                        .collect(),
                    context_policy: profile.context.as_ref().map(|context| ContextPolicy {
                        safety_margin_ratio: context.safety_margin_ratio,
                        preserve_session_context_floor: context.preserve_session_context_floor,
                        allow_context_downgrade: context.allow_context_downgrade,
                    }),
                },
            )
        })
        .collect()
}

fn selector_config(selector: &brouter_config_models::CandidateSelectorConfig) -> CandidateSelector {
    CandidateSelector {
        models: selector.models.iter().cloned().map(ModelId::new).collect(),
        upstream_models: selector.upstream_models.clone(),
        providers: selector
            .providers
            .iter()
            .cloned()
            .map(ProviderId::new)
            .collect(),
        capabilities: selector
            .capabilities
            .iter()
            .filter_map(|capability| capability.parse().ok())
            .collect(),
        attributes: selector.attributes.clone(),
    }
}

fn deny_selector_config(deny: &brouter_config_models::DenyRuleConfig) -> CandidateSelector {
    CandidateSelector {
        models: deny.models.iter().cloned().map(ModelId::new).collect(),
        upstream_models: deny.upstream_models.clone(),
        providers: deny
            .providers
            .iter()
            .cloned()
            .map(ProviderId::new)
            .collect(),
        capabilities: deny
            .capabilities
            .iter()
            .filter_map(|capability| capability.parse().ok())
            .collect(),
        attributes: deny.attributes.clone(),
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
            prefer_attributes: rule.prefer_attributes.clone(),
            require_attributes: rule.require_attributes.clone(),
        })
        .collect()
}

/// Converts configured models into router candidates.
#[must_use]
pub fn routeable_models(config: &BrouterConfig) -> Vec<RouteableModel> {
    routeable_models_with_introspection(config, &[])
}

/// Converts configured models into router candidates using live provider snapshots.
#[must_use]
pub fn routeable_models_with_introspection(
    config: &BrouterConfig,
    snapshots: &[IntrospectionSnapshot],
) -> Vec<RouteableModel> {
    let catalog = FallbackCatalog::default();
    let mut models = config
        .models
        .iter()
        .map(|(id, model)| model_config_to_routeable(config, &catalog, snapshots, id, model))
        .collect::<Vec<_>>();
    for (alias, target) in &config.router.aliases {
        if let Some(model) = config.models.get(target) {
            models.push(model_config_to_routeable(
                config, &catalog, snapshots, alias, model,
            ));
        }
    }
    models
}

fn model_config_to_routeable(
    config: &BrouterConfig,
    catalog: &FallbackCatalog,
    snapshots: &[IntrospectionSnapshot],
    id: &str,
    model: &brouter_config_models::ModelConfig,
) -> RouteableModel {
    let provider = config.providers.get(&model.provider);
    let catalog_model = provider.and_then(|provider| {
        catalog.find(
            provider.kind,
            &provider_family(&model.provider, provider.kind, provider.base_url.as_deref()),
            &model.model,
        )
    });
    let live_model = live_catalog_model(snapshots, &model.provider, &model.model);
    let resolved = resolve_model_metadata(model, live_model, catalog_model);
    RouteableModel {
        id: ModelId::new(id.to_string()),
        provider: ProviderId::new(model.provider.clone()),
        upstream_model: model.model.clone(),
        context_window: resolved.context_window,
        input_cost_per_million: resolved.input_cost_per_million,
        output_cost_per_million: resolved.output_cost_per_million,
        quality: model
            .quality
            .unwrap_or_else(|| inferred_quality(&model.capabilities)),
        capabilities: resolved.capabilities,
        attributes: model.attributes.clone(),
        display_badges: model.display_badges.clone(),
        metadata: resolved.metadata,
    }
}

#[derive(Debug, Clone)]
struct ResolvedMetadataValues {
    context_window: u32,
    input_cost_per_million: f64,
    output_cost_per_million: f64,
    capabilities: Vec<ModelCapability>,
    metadata: ResolvedModelMetadata,
}

fn resolve_model_metadata(
    model: &brouter_config_models::ModelConfig,
    live_model: Option<&CatalogModel>,
    catalog_model: Option<&brouter_catalog_models::CatalogModelMetadata>,
) -> ResolvedMetadataValues {
    let override_config = model.metadata_overrides.as_ref();
    let override_mode = override_config
        .map(|overrides| MetadataOverrideMode::from_name(&overrides.mode))
        .unwrap_or_default();
    let context_resolution = resolve_context_window(
        model,
        live_model,
        catalog_model,
        override_config,
        override_mode,
    );
    let cost_resolution = resolve_cost(
        model,
        live_model,
        catalog_model,
        override_config,
        override_mode,
    );
    let capabilities_resolution = resolve_capabilities(
        model,
        live_model,
        catalog_model,
        override_config,
        override_mode,
    );
    let max_output_resolution =
        resolve_max_output_tokens(live_model, catalog_model, override_config, override_mode);

    ResolvedMetadataValues {
        context_window: context_resolution.0,
        input_cost_per_million: cost_resolution.0,
        output_cost_per_million: cost_resolution.1,
        capabilities: capabilities_resolution.0,
        metadata: ResolvedModelMetadata {
            context_window: context_resolution.1,
            max_output_tokens: max_output_resolution.0,
            max_output_tokens_source: max_output_resolution.1,
            cost: cost_resolution.2,
            capabilities: capabilities_resolution.1,
        },
    }
}

fn resolve_context_window(
    model: &brouter_config_models::ModelConfig,
    live_model: Option<&CatalogModel>,
    catalog_model: Option<&brouter_catalog_models::CatalogModelMetadata>,
    override_config: Option<&brouter_config_models::ModelMetadataOverridesConfig>,
    override_mode: MetadataOverrideMode,
) -> (u32, MetadataProvenance) {
    if override_mode == MetadataOverrideMode::Force
        && let Some(value) = override_config.and_then(|overrides| overrides.context_window)
    {
        return (
            value,
            override_provenance(override_config, MetadataSource::UserForcedOverride),
        );
    }
    if let Some(field) = live_model.map(|model| &model.fields.context_window)
        && let Some(value) = field.value
    {
        return (value, field.provenance.clone());
    }
    if let Some(value) = model.context_window {
        return (value, MetadataProvenance::new(MetadataSource::UserConfig));
    }
    if override_mode == MetadataOverrideMode::Fallback
        && let Some(value) = override_config.and_then(|overrides| overrides.context_window)
    {
        return (
            value,
            override_provenance(override_config, MetadataSource::UserVerifiedFallback),
        );
    }
    if let Some(catalog_model) = catalog_model
        && let Some(value) = catalog_model.context_window
    {
        return (value, catalog_provenance(catalog_model));
    }
    (0, MetadataProvenance::new(MetadataSource::Unknown))
}

fn resolve_cost(
    model: &brouter_config_models::ModelConfig,
    live_model: Option<&CatalogModel>,
    catalog_model: Option<&brouter_catalog_models::CatalogModelMetadata>,
    override_config: Option<&brouter_config_models::ModelMetadataOverridesConfig>,
    override_mode: MetadataOverrideMode,
) -> (f64, f64, MetadataProvenance) {
    if override_mode == MetadataOverrideMode::Force
        && let Some(input_cost) =
            override_config.and_then(|overrides| overrides.input_cost_per_million)
    {
        return (
            input_cost,
            override_config
                .and_then(|overrides| overrides.output_cost_per_million)
                .unwrap_or(model.output_cost_per_million),
            override_provenance(override_config, MetadataSource::UserForcedOverride),
        );
    }
    if let Some(live_model) = live_model
        && let Some(input_cost) = live_model.fields.input_cost_per_million.value
    {
        return (
            input_cost,
            live_model
                .fields
                .output_cost_per_million
                .value
                .unwrap_or_default(),
            live_model.fields.input_cost_per_million.provenance.clone(),
        );
    }
    if model.input_cost_per_million > 0.0 || model.output_cost_per_million > 0.0 {
        return (
            model.input_cost_per_million,
            model.output_cost_per_million,
            MetadataProvenance::new(MetadataSource::UserConfig),
        );
    }
    if override_mode == MetadataOverrideMode::Fallback
        && let Some(input_cost) =
            override_config.and_then(|overrides| overrides.input_cost_per_million)
    {
        return (
            input_cost,
            override_config
                .and_then(|overrides| overrides.output_cost_per_million)
                .unwrap_or_default(),
            override_provenance(override_config, MetadataSource::UserVerifiedFallback),
        );
    }
    if let Some(catalog_model) = catalog_model
        && let Some(input_cost) = catalog_model.input_cost_per_million
    {
        return (
            input_cost,
            catalog_model.output_cost_per_million.unwrap_or_default(),
            catalog_provenance(catalog_model),
        );
    }
    (0.0, 0.0, MetadataProvenance::new(MetadataSource::Unknown))
}

fn resolve_capabilities(
    model: &brouter_config_models::ModelConfig,
    live_model: Option<&CatalogModel>,
    catalog_model: Option<&brouter_catalog_models::CatalogModelMetadata>,
    override_config: Option<&brouter_config_models::ModelMetadataOverridesConfig>,
    override_mode: MetadataOverrideMode,
) -> (Vec<ModelCapability>, MetadataProvenance) {
    if override_mode == MetadataOverrideMode::Force
        && let Some(overrides) = override_config
        && !overrides.capabilities.is_empty()
    {
        return (
            parse_capabilities(&overrides.capabilities),
            override_provenance(override_config, MetadataSource::UserForcedOverride),
        );
    }
    if let Some(field) = live_model.map(|model| &model.fields.capabilities)
        && let Some(value) = field.value.clone()
        && !value.is_empty()
    {
        return (value, field.provenance.clone());
    }
    if !model.capabilities.is_empty() {
        return (
            parse_capabilities(&model.capabilities),
            MetadataProvenance::new(MetadataSource::UserConfig),
        );
    }
    if override_mode == MetadataOverrideMode::Fallback
        && let Some(overrides) = override_config
        && !overrides.capabilities.is_empty()
    {
        return (
            parse_capabilities(&overrides.capabilities),
            override_provenance(override_config, MetadataSource::UserVerifiedFallback),
        );
    }
    if let Some(catalog_model) = catalog_model
        && !catalog_model.capabilities.is_empty()
    {
        return (
            parse_capabilities(&catalog_model.capabilities),
            catalog_provenance(catalog_model),
        );
    }
    (Vec::new(), MetadataProvenance::new(MetadataSource::Unknown))
}

fn resolve_max_output_tokens(
    live_model: Option<&CatalogModel>,
    catalog_model: Option<&brouter_catalog_models::CatalogModelMetadata>,
    override_config: Option<&brouter_config_models::ModelMetadataOverridesConfig>,
    override_mode: MetadataOverrideMode,
) -> (Option<u32>, MetadataProvenance) {
    if override_mode == MetadataOverrideMode::Force
        && let Some(value) = override_config.and_then(|overrides| overrides.max_output_tokens)
    {
        return (
            Some(value),
            override_provenance(override_config, MetadataSource::UserForcedOverride),
        );
    }
    if let Some(field) = live_model.map(|model| &model.fields.max_output_tokens)
        && let Some(value) = field.value
    {
        return (Some(value), field.provenance.clone());
    }
    if override_mode == MetadataOverrideMode::Fallback
        && let Some(value) = override_config.and_then(|overrides| overrides.max_output_tokens)
    {
        return (
            Some(value),
            override_provenance(override_config, MetadataSource::UserVerifiedFallback),
        );
    }
    if let Some(catalog_model) = catalog_model
        && let Some(value) = catalog_model.max_output_tokens
    {
        return (Some(value), catalog_provenance(catalog_model));
    }
    (None, MetadataProvenance::new(MetadataSource::Unknown))
}

fn live_catalog_model<'a>(
    snapshots: &'a [IntrospectionSnapshot],
    provider_id: &str,
    upstream_model: &str,
) -> Option<&'a CatalogModel> {
    snapshots
        .iter()
        .find(|snapshot| snapshot.provider.as_str() == provider_id)
        .and_then(|snapshot| snapshot.catalog.as_ref())
        .and_then(|catalog| catalog.models.get(upstream_model))
}

fn parse_capabilities(capabilities: &[String]) -> Vec<ModelCapability> {
    capabilities
        .iter()
        .filter_map(|capability| capability.parse().ok())
        .collect()
}

fn override_provenance(
    override_config: Option<&brouter_config_models::ModelMetadataOverridesConfig>,
    source: MetadataSource,
) -> MetadataProvenance {
    let mut provenance = MetadataProvenance::new(source);
    if let Some(overrides) = override_config {
        provenance.reason.clone_from(&overrides.reason);
        provenance.source_url.clone_from(&overrides.source_url);
        provenance.verified_at_ms = overrides.verified_at_ms;
    }
    provenance
}

fn catalog_provenance(
    catalog_model: &brouter_catalog_models::CatalogModelMetadata,
) -> MetadataProvenance {
    MetadataProvenance {
        source: MetadataSource::BrouterFallbackCatalog,
        source_label: Some(catalog_model.source_label.clone()),
        source_url: catalog_model.source_url.clone(),
        reason: None,
        verified_at_ms: catalog_model.verified_at_ms,
    }
}

fn provider_family(
    provider_id: &str,
    provider_kind: ProviderKind,
    base_url: Option<&str>,
) -> String {
    match provider_kind {
        ProviderKind::Anthropic => "anthropic".to_string(),
        ProviderKind::OpenaiCodex => "openai".to_string(),
        ProviderKind::OpenAiCompatible
            if base_url.is_some_and(|url| url.contains("openai.com")) =>
        {
            "openai".to_string()
        }
        ProviderKind::OpenAiCompatible => provider_id.to_string(),
    }
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
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: std::collections::BTreeMap::new(),
            },
        );
        config.models.insert(
            "fast".to_string(),
            ModelConfig {
                provider: "local".to_string(),
                model: "llama".to_string(),
                context_window: Some(8_192),
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: None,
                capabilities: vec!["chat".to_string()],
                attributes: std::collections::BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
            },
        );

        validate_config(&config).expect("config should be valid");
    }

    #[test]
    fn fallback_catalog_fills_missing_model_metadata() {
        let mut config = BrouterConfig::default();
        config.providers.insert(
            "openai".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some("https://api.openai.com/v1".to_string()),
                api_key_env: None,
                timeout_ms: None,
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: std::collections::BTreeMap::new(),
            },
        );
        config.models.insert(
            "strong".to_string(),
            ModelConfig {
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                context_window: None,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: None,
                capabilities: Vec::new(),
                attributes: std::collections::BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
            },
        );

        let models = routeable_models(&config);

        assert_eq!(models[0].context_window, 1_047_576);
        assert_eq!(
            models[0].metadata.context_window.source,
            MetadataSource::BrouterFallbackCatalog
        );
        assert!(models[0].has_capability(ModelCapability::Reasoning));
    }

    #[test]
    fn forced_metadata_override_wins_over_fallback_catalog() {
        let mut config = BrouterConfig::default();
        config.providers.insert(
            "openai".to_string(),
            ProviderConfig {
                kind: ProviderKind::OpenAiCompatible,
                base_url: Some("https://api.openai.com/v1".to_string()),
                api_key_env: None,
                timeout_ms: None,
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: std::collections::BTreeMap::new(),
            },
        );
        config.models.insert(
            "strong".to_string(),
            ModelConfig {
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                context_window: None,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: None,
                capabilities: Vec::new(),
                attributes: std::collections::BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: Some(brouter_config_models::ModelMetadataOverridesConfig {
                    mode: "force".to_string(),
                    reason: Some("manual test override".to_string()),
                    source_url: None,
                    verified_at_ms: None,
                    context_window: Some(42_000),
                    max_output_tokens: None,
                    input_cost_per_million: None,
                    output_cost_per_million: None,
                    capabilities: Vec::new(),
                }),
            },
        );

        let models = routeable_models(&config);

        assert_eq!(models[0].context_window, 42_000);
        assert_eq!(
            models[0].metadata.context_window.source,
            MetadataSource::UserForcedOverride
        );
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
                max_estimated_cost: None,
                auth_backend: None,
                auth_profile: None,
                auth_vault_path: None,
                introspection: brouter_config_models::ProviderIntrospectionConfig::default(),
                resource_pools: Vec::new(),
                attribute_mappings: std::collections::BTreeMap::new(),
            },
        );
        config.models.insert(
            "fast".to_string(),
            ModelConfig {
                provider: "local".to_string(),
                model: "llama".to_string(),
                context_window: Some(8_192),
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: None,
                capabilities: vec!["made_up".to_string()],
                attributes: std::collections::BTreeMap::new(),
                display_badges: Vec::new(),
                max_estimated_cost: None,
                metadata_overrides: None,
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
