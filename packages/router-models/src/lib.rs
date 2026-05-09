#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Routing decision and prompt classification models for brouter.

use std::collections::BTreeMap;

use brouter_catalog_models::ResolvedModelMetadata;
use brouter_introspection_models::DynamicPolicyEffect;
use brouter_provider_models::{ModelCapability, ModelId};
use serde::{Deserialize, Serialize};

/// High-level prompt intent detected by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptIntent {
    General,
    Coding,
    Debugging,
    Summarization,
    Extraction,
    Planning,
    Creative,
    Math,
    Agentic,
}

/// Estimated reasoning level required for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningLevel {
    Low,
    Medium,
    High,
}

/// User or policy objective used when scoring candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingObjective {
    Cheapest,
    Fastest,
    Balanced,
    Strongest,
    LocalOnly,
}

impl std::str::FromStr for PromptIntent {
    type Err = ParsePromptIntentError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "general" => Ok(Self::General),
            "coding" | "code" => Ok(Self::Coding),
            "debugging" | "debug" => Ok(Self::Debugging),
            "summarization" | "summary" => Ok(Self::Summarization),
            "extraction" | "extract" => Ok(Self::Extraction),
            "planning" | "plan" => Ok(Self::Planning),
            "creative" => Ok(Self::Creative),
            "math" => Ok(Self::Math),
            "agentic" | "agent" => Ok(Self::Agentic),
            _ => Err(ParsePromptIntentError),
        }
    }
}

/// Error returned when parsing a prompt intent fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsePromptIntentError;

impl std::fmt::Display for ParsePromptIntentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("unknown prompt intent")
    }
}

impl std::error::Error for ParsePromptIntentError {}

impl RoutingObjective {
    /// Parses a routing objective from configuration or CLI input.
    #[must_use]
    pub fn from_name(value: &str) -> Self {
        match value {
            "cheapest" => Self::Cheapest,
            "fastest" => Self::Fastest,
            "strongest" => Self::Strongest,
            "local_only" | "local-only" | "local" => Self::LocalOnly,
            _ => Self::Balanced,
        }
    }
}

/// Candidate scoring weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub quality_weight: f64,
    pub balanced_cost_weight: f64,
    pub cheapest_cost_weight: f64,
    pub local_bonus: f64,
    pub strongest_quality_weight: f64,
    pub first_message_reasoning_bonus: f64,
    pub code_bonus: f64,
    pub reasoning_bonus: f64,
    pub policy_penalty: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            quality_weight: 1.0,
            balanced_cost_weight: 20.0,
            cheapest_cost_weight: 100.0,
            local_bonus: 10.0,
            strongest_quality_weight: 0.5,
            first_message_reasoning_bonus: 8.0,
            code_bonus: 15.0,
            reasoning_bonus: 20.0,
            policy_penalty: 30.0,
        }
    }
}

/// Context safety settings used before model scoring.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContextPolicy {
    pub safety_margin_ratio: f64,
    pub preserve_session_context_floor: bool,
    pub allow_context_downgrade: bool,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            safety_margin_ratio: 0.15,
            preserve_session_context_floor: true,
            allow_context_downgrade: false,
        }
    }
}

/// Runtime options for a single routing request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoutingOptions {
    pub allowed_models: Option<Vec<ModelId>>,
    pub profile: Option<String>,
    pub session_context_tokens: Option<u32>,
    pub dynamic_policy_effects: Vec<DynamicPolicyEffect>,
}

/// Candidate selector used by profile allow and deny policies.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSelector {
    pub models: Vec<ModelId>,
    pub upstream_models: Vec<String>,
    pub providers: Vec<brouter_provider_models::ProviderId>,
    pub capabilities: Vec<ModelCapability>,
    pub attributes: BTreeMap<String, String>,
}

impl CandidateSelector {
    /// Returns true when the selector does not constrain candidates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
            && self.upstream_models.is_empty()
            && self.providers.is_empty()
            && self.capabilities.is_empty()
            && self.attributes.is_empty()
    }
}

/// Profile deny policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateDenyRule {
    pub selector: CandidateSelector,
    pub reason: String,
    pub hard: bool,
    pub penalty: Option<f64>,
}

/// Named routing profile.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoutingProfile {
    pub objective: Option<RoutingObjective>,
    pub allow: Vec<CandidateSelector>,
    pub deny: Vec<CandidateDenyRule>,
    pub context_policy: Option<ContextPolicy>,
}

/// Candidate excluded before scoring, with an explainable reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedCandidate {
    pub model_id: ModelId,
    pub reason: String,
}

/// Configurable routing rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingRule {
    pub name: String,
    pub when_contains: Vec<String>,
    pub intent: Option<PromptIntent>,
    pub objective: Option<RoutingObjective>,
    pub prefer_capabilities: Vec<ModelCapability>,
    pub require_capabilities: Vec<ModelCapability>,
    pub prefer_attributes: BTreeMap<String, String>,
    pub require_attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub llm_judge: bool,
}

/// Prompt features extracted before candidate scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptFeatures {
    pub intent: PromptIntent,
    pub reasoning: ReasoningLevel,
    pub estimated_input_tokens: u32,
    pub estimated_output_tokens: u32,
    pub required_context_tokens: u32,
    pub required_capabilities: Vec<ModelCapability>,
    pub preferred_capabilities: Vec<ModelCapability>,
    pub required_attributes: BTreeMap<String, String>,
    pub preferred_attributes: BTreeMap<String, String>,
    pub matched_rules: Vec<String>,
    pub is_first_message: bool,
}

/// One scored router candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredCandidate {
    pub model_id: ModelId,
    pub score: f64,
    pub estimated_cost: f64,
    pub reasons: Vec<String>,
    /// Flat metadata fields used for LLM judge display.
    pub capabilities: Vec<ModelCapability>,
    pub provider: String,
    pub quality: u8,
    #[serde(default)]
    pub metadata: ResolvedModelMetadata,
}

/// Explainable model selection result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub selected_model: ModelId,
    pub objective: RoutingObjective,
    pub features: PromptFeatures,
    pub reasons: Vec<String>,
    pub candidates: Vec<ScoredCandidate>,
    #[serde(default)]
    pub excluded_candidates: Vec<ExcludedCandidate>,
    /// LLM-generated reasoning for why this model was selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ModelReasoning>,
}

/// LLM-generated reasoning attached to a routing decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelReasoning {
    /// The model used to generate the reasoning.
    pub model_id: ModelId,
    /// Free-text explanation of the routing decision.
    pub rationale: String,
    /// The model chosen by the LLM judge.
    pub chosen_model: ModelId,
    /// True when the LLM judge overrode the deterministic top pick.
    pub overridden: bool,
    /// Error message if reasoning generation failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// Re-export judge types so config package can use them without a circular dep.
pub use brouter_provider_models::ProviderId;

#[derive(Debug, Clone)]
pub struct JudgeConfig {
    pub model: ModelId,
    pub provider: Option<ProviderId>,
    pub system_prompt: Option<String>,
    pub trigger: JudgeTrigger,
    pub shortlist: JudgeShortlistConfig,
    pub output: JudgeOutput,
    pub max_estimated_cost: f64,
}

#[derive(Debug, Clone)]
pub struct JudgeTrigger {
    pub score_gap_threshold: f64,
    pub rule_triggered: bool,
}

/// Returns true when the judge should fire based on trigger config, score gap, and rule match.
#[must_use]
pub fn should_fire_trigger(trigger: &JudgeTrigger, top_2_gap: f64, rule_triggered: bool) -> bool {
    top_2_gap < trigger.score_gap_threshold || (trigger.rule_triggered && rule_triggered)
}

#[derive(Debug, Clone)]
pub struct JudgeShortlistConfig {
    pub size: usize,
    pub min_score: f64,
    pub deny: Vec<JudgeShortlistDeny>,
}

#[derive(Debug, Clone)]
pub struct JudgeShortlistDeny {
    pub models: Vec<ModelId>,
    pub upstream_models: Vec<String>,
    pub providers: Vec<ProviderId>,
    pub capabilities: Vec<ModelCapability>,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
pub struct JudgeOutput {
    pub structured: bool,
    pub max_tokens: u32,
    pub temperature: f64,
}

/// Session context summary passed to the judge.
#[derive(Debug, Default)]
pub struct JudgeSessionContext {
    /// Number of previous requests in the session.
    pub request_count: u32,
    /// Accumulated estimated cost so far.
    pub accumulated_cost: f64,
    /// Recent routing decisions (model IDs and intents).
    pub recent_decisions: Vec<RecentDecision>,
}

/// A lightweight recent routing decision for judge context.
#[derive(Debug)]
pub struct RecentDecision {
    pub model_id: String,
    pub intent: String,
}
