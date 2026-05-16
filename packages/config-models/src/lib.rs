#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Serde-compatible configuration models for brouter.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Top-level brouter configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BrouterConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelConfig>,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,
    #[serde(default)]
    pub cors_allowed_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key_env: None,
            max_request_body_bytes: default_max_request_body_bytes(),
            cors_allowed_origins: Vec::new(),
        }
    }
}

impl ServerConfig {
    /// Returns the TCP bind address for the configured server.
    #[must_use]
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

const fn default_port() -> u16 {
    8080
}

const fn default_max_request_body_bytes() -> usize {
    1_048_576
}

/// Router behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterConfig {
    #[serde(default = "default_objective")]
    pub default_objective: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub debug_headers: bool,
    #[serde(default)]
    pub scoring: ScoringConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub context_reporting: ContextReportingConfig,
    #[serde(default)]
    pub streaming: StreamingConfig,
    #[serde(default)]
    pub metadata: MetadataConfig,
    #[serde(default)]
    pub dynamic_policy: DynamicPolicyConfig,
    #[serde(default)]
    pub rules: Vec<RouterRuleConfig>,
    #[serde(default)]
    pub aliases: BTreeMap<String, String>,
    #[serde(default)]
    pub groups: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub profiles: BTreeMap<String, RouterProfileConfig>,
    #[serde(default)]
    pub classifier: Option<ClassifierConfig>,
    #[serde(default)]
    pub llm_judge: Option<LlmJudgeConfig>,
    #[serde(default = "default_default_judge")]
    pub default_judge: bool,
    #[serde(default = "default_provider_failure_threshold")]
    pub provider_failure_threshold: u32,
    #[serde(default = "default_provider_cooldown_ms")]
    pub provider_cooldown_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_estimated_cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_session_estimated_cost: Option<f64>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            default_objective: default_objective(),
            default_profile: None,
            debug_headers: false,
            scoring: ScoringConfig::default(),
            context: ContextConfig::default(),
            context_reporting: ContextReportingConfig::default(),
            streaming: StreamingConfig::default(),
            metadata: MetadataConfig::default(),
            dynamic_policy: DynamicPolicyConfig::default(),
            rules: Vec::new(),
            aliases: BTreeMap::new(),
            groups: BTreeMap::new(),
            profiles: BTreeMap::new(),
            classifier: None,
            llm_judge: None,
            default_judge: default_default_judge(),
            provider_failure_threshold: default_provider_failure_threshold(),
            provider_cooldown_ms: default_provider_cooldown_ms(),
            max_estimated_cost: None,
            max_session_estimated_cost: None,
        }
    }
}

fn default_objective() -> String {
    "balanced".to_string()
}

const fn default_default_judge() -> bool {
    true
}

const fn default_provider_failure_threshold() -> u32 {
    3
}

const fn default_provider_cooldown_ms() -> u64 {
    30_000
}

/// Context reporting behavior for client-visible responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextReportingConfig {
    #[serde(default)]
    pub mode: ContextReportingMode,
}

impl Default for ContextReportingConfig {
    fn default() -> Self {
        Self {
            mode: ContextReportingMode::Headers,
        }
    }
}

/// Client-visible context reporting mode.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextReportingMode {
    /// Preserve provider-compatible bodies/streams and omit brouter context headers.
    Compat,
    /// Preserve provider-compatible bodies/streams and emit brouter context headers.
    #[default]
    Headers,
    /// Preserve provider-compatible bodies/streams and emit brouter context headers.
    /// Reserved for future protocol-safe metadata envelopes.
    Envelope,
    /// Emit brouter context headers. Reserved for future opt-in usage rewriting.
    RewriteUsage,
}

impl ContextReportingMode {
    /// Returns true when brouter context headers should be emitted.
    #[must_use]
    pub const fn emits_headers(self) -> bool {
        matches!(self, Self::Headers | Self::Envelope | Self::RewriteUsage)
    }
}

/// Streaming response behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamingConfig {
    #[serde(default = "default_streaming_context_usage_observer")]
    pub context_usage_observer: bool,
    #[serde(default = "default_request_stream_usage")]
    pub request_usage: bool,
    #[serde(default = "default_stream_usage_buffer_limit_bytes")]
    pub usage_buffer_limit_bytes: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            context_usage_observer: default_streaming_context_usage_observer(),
            request_usage: default_request_stream_usage(),
            usage_buffer_limit_bytes: default_stream_usage_buffer_limit_bytes(),
        }
    }
}

const fn default_streaming_context_usage_observer() -> bool {
    true
}

const fn default_request_stream_usage() -> bool {
    true
}

const fn default_stream_usage_buffer_limit_bytes() -> usize {
    65_536
}

/// Router scoring overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ScoringConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub balanced_cost_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cheapest_cost_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_bonus: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strongest_quality_weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_message_reasoning_bonus: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_bonus: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_bonus: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_penalty: Option<f64>,
}

/// Metadata introspection/cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct MetadataConfig {
    #[serde(default)]
    pub strict: bool,
    #[serde(default = "default_metadata_max_age_ms")]
    pub max_age_ms: u64,
    #[serde(default = "default_metadata_refresh_on_startup")]
    pub refresh_on_startup: bool,
    #[serde(default = "default_metadata_refresh_interval_ms")]
    pub refresh_interval_ms: u64,
    #[serde(default = "default_refresh_before_expensive_route")]
    pub refresh_before_expensive_route: bool,
    #[serde(default = "default_allow_stale_on_provider_error")]
    pub allow_stale_on_provider_error: bool,
    #[serde(default = "default_allow_fallback_catalog")]
    pub allow_fallback_catalog: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<String>,
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            strict: false,
            max_age_ms: default_metadata_max_age_ms(),
            refresh_on_startup: default_metadata_refresh_on_startup(),
            refresh_interval_ms: default_metadata_refresh_interval_ms(),
            refresh_before_expensive_route: default_refresh_before_expensive_route(),
            allow_stale_on_provider_error: default_allow_stale_on_provider_error(),
            allow_fallback_catalog: default_allow_fallback_catalog(),
            cache_path: None,
        }
    }
}

const fn default_metadata_max_age_ms() -> u64 {
    86_400_000
}

const fn default_metadata_refresh_on_startup() -> bool {
    true
}

const fn default_metadata_refresh_interval_ms() -> u64 {
    300_000
}

const fn default_refresh_before_expensive_route() -> bool {
    true
}

const fn default_allow_stale_on_provider_error() -> bool {
    true
}

const fn default_allow_fallback_catalog() -> bool {
    true
}

/// Dynamic account policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DynamicPolicyConfig {
    #[serde(default = "default_low_remaining_ratio")]
    pub low_remaining_ratio: f64,
    #[serde(default = "default_critical_remaining_ratio")]
    pub critical_remaining_ratio: f64,
    #[serde(default = "default_low_remaining_penalty")]
    pub low_remaining_penalty: f64,
    #[serde(default = "default_exclude_when_exhausted")]
    pub exclude_when_exhausted: bool,
    #[serde(default)]
    pub disable_attributes_when_low: BTreeMap<String, String>,
}

impl Default for DynamicPolicyConfig {
    fn default() -> Self {
        Self {
            low_remaining_ratio: default_low_remaining_ratio(),
            critical_remaining_ratio: default_critical_remaining_ratio(),
            low_remaining_penalty: default_low_remaining_penalty(),
            exclude_when_exhausted: default_exclude_when_exhausted(),
            disable_attributes_when_low: BTreeMap::new(),
        }
    }
}

const fn default_low_remaining_ratio() -> f64 {
    0.25
}

const fn default_critical_remaining_ratio() -> f64 {
    0.05
}

const fn default_low_remaining_penalty() -> f64 {
    30.0
}

const fn default_exclude_when_exhausted() -> bool {
    true
}

/// Context safety configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextConfig {
    #[serde(default = "default_context_safety_margin_ratio")]
    pub safety_margin_ratio: f64,
    #[serde(default = "default_preserve_session_context_floor")]
    pub preserve_session_context_floor: bool,
    #[serde(default)]
    pub allow_context_downgrade: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            safety_margin_ratio: default_context_safety_margin_ratio(),
            preserve_session_context_floor: default_preserve_session_context_floor(),
            allow_context_downgrade: false,
        }
    }
}

const fn default_context_safety_margin_ratio() -> f64 {
    0.15
}

const fn default_preserve_session_context_floor() -> bool {
    true
}

/// Named router profile configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RouterProfileConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default)]
    pub allow: Vec<CandidateSelectorConfig>,
    #[serde(default)]
    pub deny: Vec<DenyRuleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextConfig>,
}

/// Candidate selector configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateSelectorConfig {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub upstream_models: Vec<String>,
    #[serde(default)]
    pub providers: Vec<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// Candidate deny rule configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenyRuleConfig {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub upstream_models: Vec<String>,
    #[serde(default)]
    pub providers: Vec<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default = "default_deny_reason")]
    pub reason: String,
    #[serde(default = "default_hard_deny")]
    pub hard: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub penalty: Option<f64>,
}

impl Default for DenyRuleConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            upstream_models: Vec::new(),
            providers: Vec::new(),
            capabilities: Vec::new(),
            attributes: BTreeMap::new(),
            reason: default_deny_reason(),
            hard: default_hard_deny(),
            penalty: None,
        }
    }
}

fn default_deny_reason() -> String {
    "denied by routing profile".to_string()
}

const fn default_hard_deny() -> bool {
    true
}

/// Configurable router rule.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouterRuleConfig {
    pub name: String,
    #[serde(default)]
    pub when_contains: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default)]
    pub prefer_capabilities: Vec<String>,
    #[serde(default)]
    pub require_capabilities: Vec<String>,
    #[serde(default)]
    pub prefer_attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub require_attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub llm_judge: bool,
}

/// LLM judge configuration for model selection reasoning.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmJudgeConfig {
    /// The model to use for LLM judge calls.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub trigger: LlmJudgeTriggerConfig,
    #[serde(default)]
    pub shortlist: LlmJudgeShortlistConfig,
    #[serde(default)]
    pub output: LlmJudgeOutputConfig,
    #[serde(default)]
    pub budget: LlmJudgeBudgetConfig,
}

/// Trigger configuration for when the LLM judge fires.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmJudgeTriggerConfig {
    /// Fire LLM judge when top-2 candidate score gap is below this threshold.
    #[serde(default = "default_llm_judge_score_gap_threshold")]
    pub score_gap_threshold: f64,
    /// Fire LLM judge when a rule has `llm_judge` = true (regardless of score gap).
    #[serde(default = "default_llm_judge_rule_triggered")]
    pub rule_triggered: bool,
}

const fn default_llm_judge_score_gap_threshold() -> f64 {
    5.0
}

const fn default_llm_judge_rule_triggered() -> bool {
    true
}

/// Shortlist configuration for LLM judge candidate presentation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmJudgeShortlistConfig {
    /// Maximum number of candidates to include in the shortlist.
    #[serde(default = "default_llm_judge_shortlist_size")]
    pub size: usize,
    /// Minimum deterministic score floor. Candidates below this are excluded.
    #[serde(default)]
    pub min_score: f64,
    /// Hard deny list applied after scoring (additional exclusions).
    #[serde(default)]
    pub deny: Vec<CandidateSelectorConfig>,
}

const fn default_llm_judge_shortlist_size() -> usize {
    5
}

impl Default for LlmJudgeShortlistConfig {
    fn default() -> Self {
        Self {
            size: default_llm_judge_shortlist_size(),
            min_score: 0.0,
            deny: Vec::new(),
        }
    }
}

/// Output configuration for LLM judge responses.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmJudgeOutputConfig {
    /// Request structured JSON output via model tool-calling / `response_format`.
    #[serde(default = "default_llm_judge_structured")]
    pub structured: bool,
    /// Maximum tokens for the reasoning response.
    #[serde(default = "default_llm_judge_max_tokens")]
    pub max_tokens: u32,
    /// Temperature for reasoning calls.
    #[serde(default = "default_llm_judge_temperature")]
    pub temperature: f64,
}

const fn default_llm_judge_structured() -> bool {
    true
}

const fn default_llm_judge_max_tokens() -> u32 {
    4096
}

const fn default_llm_judge_temperature() -> f64 {
    0.3
}

/// Budget configuration for LLM judge calls.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmJudgeBudgetConfig {
    /// Maximum estimated cost for the reasoning call. Abort if exceeded.
    #[serde(default)]
    pub max_estimated_cost: f64,
}

/// Optional prompt classifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClassifierConfig {
    pub provider: String,
    pub model: String,
}

/// Telemetry storage configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub database_path: Option<String>,
    #[serde(default)]
    pub events: TelemetryEventsConfig,
}

/// Session routing event log configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct TelemetryEventsConfig {
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub include_prompt: bool,
    #[serde(default)]
    pub include_judge_prompt: bool,
    #[serde(default)]
    pub include_excluded_candidates: bool,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
    #[serde(default)]
    pub prompt_preview_chars: usize,
}

impl Default for TelemetryEventsConfig {
    fn default() -> Self {
        Self {
            disabled: false,
            include_prompt: false,
            include_judge_prompt: false,
            include_excluded_candidates: true,
            candidate_limit: default_candidate_limit(),
            prompt_preview_chars: 0,
        }
    }
}

const fn default_candidate_limit() -> usize {
    8
}

/// Provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_estimated_cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_vault_path: Option<String>,
    #[serde(default)]
    pub introspection: ProviderIntrospectionConfig,
    #[serde(default)]
    pub resource_pools: Vec<ResourcePoolConfig>,
    #[serde(default)]
    pub controls: ProviderControlsConfig,
    #[serde(default)]
    pub virtual_variants: VirtualVariantsConfig,
    #[serde(default)]
    pub attribute_mappings: BTreeMap<String, BTreeMap<String, AttributeRequestMapping>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omit_request_fields: Vec<String>,
}

impl ProviderConfig {
    /// Returns virtual variants merged from declarative controls and legacy fields.
    #[must_use]
    pub fn effective_virtual_variants(&self) -> VirtualVariantsConfig {
        let mut variants = self.controls.virtual_variants.clone();
        append_missing_strings(
            &mut variants.service_tiers,
            &self.virtual_variants.service_tiers,
        );
        append_missing_strings(
            &mut variants.reasoning_efforts,
            &self.virtual_variants.reasoning_efforts,
        );
        variants
    }

    /// Returns request field mappings merged from declarative controls and legacy fields.
    #[must_use]
    pub fn effective_attribute_mappings(
        &self,
    ) -> BTreeMap<String, BTreeMap<String, AttributeRequestMapping>> {
        let mut mappings = self.controls.attribute_mappings.clone();
        for (attribute, values) in &self.attribute_mappings {
            let target = mappings.entry(attribute.clone()).or_default();
            for (value, mapping) in values {
                target
                    .entry(value.clone())
                    .and_modify(|existing| existing.merge_missing(mapping))
                    .or_insert_with(|| mapping.clone());
            }
        }
        mappings
    }

    /// Returns request fields to omit, merged from declarative controls and legacy fields.
    #[must_use]
    pub fn effective_omit_request_fields(&self) -> Vec<String> {
        let mut fields = self.controls.omit_request_fields.clone();
        append_missing_strings(&mut fields, &self.omit_request_fields);
        fields
    }
}

fn append_missing_strings(target: &mut Vec<String>, additions: &[String]) {
    let mut seen = target.iter().cloned().collect::<BTreeSet<_>>();
    for value in additions {
        if seen.insert(value.clone()) {
            target.push(value.clone());
        }
    }
}

/// Declarative provider controls for route variants and provider request fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderControlsConfig {
    #[serde(default)]
    pub virtual_variants: VirtualVariantsConfig,
    #[serde(default)]
    pub attribute_mappings: BTreeMap<String, BTreeMap<String, AttributeRequestMapping>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omit_request_fields: Vec<String>,
}

/// Provider-level virtual request-control variants to expose as routeable candidates.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VirtualVariantsConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service_tiers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<String>,
}

/// Configured provider resource pool used with live account usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResourcePoolConfig {
    pub id: String,
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub unit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refill_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_ms: Option<u64>,
    #[serde(default)]
    pub applies_to: ResourceSelectorConfig,
}

/// Configured resource selector.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceSelectorConfig {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub upstream_models: Vec<String>,
    #[serde(default)]
    pub providers: Vec<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// Provider live introspection configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProviderIntrospectionConfig {
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_introspection_catalog")]
    pub catalog: bool,
    #[serde(default)]
    pub account: bool,
    #[serde(default)]
    pub limits: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub usage_endpoints: Vec<String>,
}

impl Default for ProviderIntrospectionConfig {
    fn default() -> Self {
        Self {
            disabled: false,
            enabled: false,
            catalog: default_introspection_catalog(),
            account: false,
            limits: false,
            usage_endpoints: Vec::new(),
        }
    }
}

const fn default_introspection_catalog() -> bool {
    true
}

/// Provider request mapping applied when a selected model has a matching attribute.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttributeRequestMapping {
    #[serde(default)]
    pub request_fields: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub omit_request_fields: Vec<String>,
}

impl AttributeRequestMapping {
    fn merge_missing(&mut self, other: &Self) {
        for field in &other.omit_request_fields {
            if !self.omit_request_fields.contains(field) {
                self.omit_request_fields.push(field.clone());
            }
        }
        for (field, value) in &other.request_fields {
            self.request_fields
                .entry(field.clone())
                .or_insert_with(|| value.clone());
        }
    }
}

/// Supported provider kinds.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    #[default]
    OpenAiCompatible,
    Anthropic,
    OpenaiCodex,
}

impl ProviderKind {
    /// Parses a provider kind name.
    #[must_use]
    pub fn from_name(value: &str) -> Self {
        match value {
            "anthropic" => Self::Anthropic,
            "openai-codex" | "openai_codex" => Self::OpenaiCodex,
            _ => Self::OpenAiCompatible,
        }
    }

    /// Returns a debug-friendly name for this provider kind.
    #[must_use]
    pub const fn debug_name(&self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "openai-compatible",
            Self::Anthropic => "anthropic",
            Self::OpenaiCodex => "openai-codex",
        }
    }
}

/// Model configuration used by the router.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub input_cost_per_million: f64,
    #[serde(default)]
    pub output_cost_per_million: f64,
    #[serde(default)]
    pub quality: Option<u8>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub display_badges: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_estimated_cost: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata_overrides: Option<ModelMetadataOverridesConfig>,
}

/// User-provided model metadata overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadataOverridesConfig {
    #[serde(default = "default_metadata_override_mode")]
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_cost_per_million: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_cost_per_million: Option<f64>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

fn default_metadata_override_mode() -> String {
    "fallback".to_string()
}
