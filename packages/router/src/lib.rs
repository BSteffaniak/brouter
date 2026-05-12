#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Prompt analysis and deterministic model routing engine for brouter.

mod llm_judge;

use std::cmp::Ordering;
use std::collections::BTreeMap;

use brouter_api_models::{ChatCompletionRequest, ReasoningEffort};
use brouter_introspection_models::{DynamicPolicyEffect, ResourceSelector};
use brouter_provider_models::{ModelCapability, ModelId, RouteableModel};
use brouter_router_models::{
    CandidateDenyRule, CandidateSelector, ContextPolicy, ExcludedCandidate, PromptFeatures,
    PromptIntent, ReasoningLevel, RoutingDecision, RoutingObjective, RoutingOptions,
    RoutingProfile, RoutingRule, ScoredCandidate, ScoringWeights,
};
use thiserror::Error;

// Re-export judge types so config package can access them via the router crate.
#[allow(unused_imports)]
pub use brouter_router_models::{
    JudgeConfig, JudgeOutput, JudgeSessionContext, JudgeShortlistConfig, JudgeTrigger,
    RecentDecision, should_fire_trigger,
};

// Re-export llm_judge helpers for use by the server package.
pub use llm_judge::{
    DEFAULT_JUDGE_SYSTEM_PROMPT, build_judge_prompt, judge_request, parse_judge_response,
    top_2_score_gap,
};
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("no configured model can satisfy the request")]
    NoCandidate,
    #[error("LLM judge call failed: {source}")]
    Judge {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Deterministic prompt router.
#[derive(Debug, Clone)]
pub struct Router {
    models: Vec<RouteableModel>,
    objective: RoutingObjective,
    weights: ScoringWeights,
    rules: Vec<RoutingRule>,
    profiles: BTreeMap<String, RoutingProfile>,
    context_policy: ContextPolicy,
}

impl Router {
    /// Creates a deterministic router.
    #[must_use]
    pub fn new(models: Vec<RouteableModel>, objective: RoutingObjective) -> Self {
        Self::new_with_scoring(models, objective, ScoringWeights::default())
    }

    /// Creates a deterministic router with custom scoring weights.
    #[must_use]
    pub fn new_with_scoring(
        models: Vec<RouteableModel>,
        objective: RoutingObjective,
        weights: ScoringWeights,
    ) -> Self {
        Self {
            models,
            objective,
            weights,
            rules: Vec::new(),
            profiles: BTreeMap::new(),
            context_policy: ContextPolicy::default(),
        }
    }

    /// Creates a deterministic router with custom scoring weights and rules.
    #[must_use]
    pub fn new_with_rules(
        models: Vec<RouteableModel>,
        objective: RoutingObjective,
        weights: ScoringWeights,
        rules: Vec<RoutingRule>,
    ) -> Self {
        Self {
            models,
            objective,
            weights,
            rules,
            profiles: BTreeMap::new(),
            context_policy: ContextPolicy::default(),
        }
    }

    /// Creates a deterministic router with profiles and context policy.
    #[must_use]
    pub const fn new_with_policy(
        models: Vec<RouteableModel>,
        objective: RoutingObjective,
        weights: ScoringWeights,
        rules: Vec<RoutingRule>,
        profiles: BTreeMap<String, RoutingProfile>,
        context_policy: ContextPolicy,
    ) -> Self {
        Self {
            models,
            objective,
            weights,
            rules,
            profiles,
            context_policy,
        }
    }

    /// Returns configured routeable models.
    #[must_use]
    pub fn models(&self) -> &[RouteableModel] {
        &self.models
    }

    /// Returns configured routing rules.
    #[must_use]
    pub fn rules(&self) -> &[RoutingRule] {
        &self.rules
    }

    /// Selects a model for a chat completion request.
    ///
    /// # Errors
    ///
    /// Returns an error when no configured model satisfies the request's
    /// required capabilities and context window.
    pub fn route_chat(
        &self,
        request: &ChatCompletionRequest,
        is_first_message: bool,
    ) -> Result<RoutingDecision, RouterError> {
        self.route_chat_for_models(request, is_first_message, None)
    }

    /// Selects a model for a chat completion request from a restricted model set.
    ///
    /// # Errors
    ///
    /// Returns an error when no allowed configured model satisfies the request.
    pub fn route_chat_for_models(
        &self,
        request: &ChatCompletionRequest,
        is_first_message: bool,
        allowed_models: Option<&[ModelId]>,
    ) -> Result<RoutingDecision, RouterError> {
        self.route_chat_with_options(
            request,
            is_first_message,
            RoutingOptions {
                allowed_models: allowed_models.map(<[ModelId]>::to_vec),
                profile: None,
                session_context_tokens: None,
                dynamic_policy_effects: Vec::new(),
            },
        )
    }

    /// Selects a model using explicit routing options.
    ///
    /// # Errors
    ///
    /// Returns an error when no configured model satisfies the request and
    /// routing options.
    pub fn route_chat_with_options(
        &self,
        request: &ChatCompletionRequest,
        is_first_message: bool,
        options: RoutingOptions,
    ) -> Result<RoutingDecision, RouterError> {
        let prompt_text = request_prompt_text(request);
        let latest_user_text = request_latest_user_text(request);
        let classification_text = latest_user_text
            .as_deref()
            .filter(|text| !text.is_empty())
            .unwrap_or(prompt_text.as_str());
        let mut features =
            analyze_prompt_texts(classification_text, &prompt_text, request, is_first_message);
        let profile = options
            .profile
            .as_deref()
            .and_then(|profile| self.profiles.get(profile));
        let rule_objective = self.apply_rules(&classification_text.to_lowercase(), &mut features);
        let objective = profile
            .and_then(|profile| profile.objective)
            .unwrap_or(rule_objective);
        apply_context_policy(
            &mut features,
            profile
                .and_then(|profile| profile.context_policy)
                .unwrap_or(self.context_policy),
            options.session_context_tokens,
        );
        let explicit_model = if options.allowed_models.is_none() && !is_auto_model(&request.model) {
            Some(vec![ModelId::new(request.model.clone())])
        } else {
            None
        };
        let allowed_models = options.allowed_models.or(explicit_model);
        self.route_features_with_objective(
            features,
            objective,
            allowed_models.as_deref(),
            profile,
            &options.dynamic_policy_effects,
        )
    }

    /// Selects a model for precomputed prompt features.
    ///
    /// # Errors
    ///
    /// Returns an error when no configured model satisfies the requested
    /// capabilities and context window.
    pub fn route_features(&self, features: PromptFeatures) -> Result<RoutingDecision, RouterError> {
        let mut features = features;
        let objective = self.apply_rules("", &mut features);
        apply_context_policy(&mut features, self.context_policy, None);
        self.route_features_with_objective(features, objective, None, None, &[])
    }

    fn route_features_with_objective(
        &self,
        features: PromptFeatures,
        objective: RoutingObjective,
        allowed_models: Option<&[ModelId]>,
        profile: Option<&RoutingProfile>,
        dynamic_policy_effects: &[DynamicPolicyEffect],
    ) -> Result<RoutingDecision, RouterError> {
        let (mut candidates, excluded_candidates) = self.scored_candidates(
            &features,
            objective,
            allowed_models,
            profile,
            dynamic_policy_effects,
        );
        candidates.sort_by(compare_candidate_scores);
        let selected = candidates.first().ok_or(RouterError::NoCandidate)?;
        let selected_model = selected.model_id.clone();
        let reasons = selected.reasons.clone();
        Ok(RoutingDecision {
            selected_model,
            objective,
            features,
            reasons,
            candidates,
            excluded_candidates,
            request_controls: brouter_router_models::SelectedRequestControls::default(),
            reasoning: None,
        })
    }

    fn scored_candidates(
        &self,
        features: &PromptFeatures,
        objective: RoutingObjective,
        allowed_models: Option<&[ModelId]>,
        profile: Option<&RoutingProfile>,
        dynamic_policy_effects: &[DynamicPolicyEffect],
    ) -> (Vec<ScoredCandidate>, Vec<ExcludedCandidate>) {
        let mut candidates = Vec::new();
        let mut excluded = Vec::new();
        for model in &self.models {
            if let Some(reason) = candidate_exclusion_reason(
                model,
                features,
                objective,
                allowed_models,
                profile,
                dynamic_policy_effects,
            ) {
                excluded.push(ExcludedCandidate {
                    model_id: model.id.clone(),
                    reason,
                });
                continue;
            }
            let mut candidate = score_model(model, features, objective, self.weights);
            apply_soft_denies(model, profile, self.weights, &mut candidate);
            apply_dynamic_penalties(model, dynamic_policy_effects, &mut candidate);
            candidates.push(candidate);
        }
        (candidates, excluded)
    }

    fn apply_rules(&self, lower_prompt: &str, features: &mut PromptFeatures) -> RoutingObjective {
        let mut objective = self.objective;
        for rule in &self.rules {
            if !rule_matches(rule, lower_prompt, features.intent) {
                continue;
            }
            features.matched_rules.push(rule.name.clone());
            append_unique_capabilities(
                &mut features.required_capabilities,
                &rule.require_capabilities,
            );
            append_unique_capabilities(
                &mut features.preferred_capabilities,
                &rule.prefer_capabilities,
            );
            append_attributes(&mut features.required_attributes, &rule.require_attributes);
            append_attributes(&mut features.preferred_attributes, &rule.prefer_attributes);
            if let Some(rule_objective) = rule.objective {
                objective = rule_objective;
            }
        }
        objective
    }
}

/// Extracts prompt features from a chat request.
#[must_use]
pub fn analyze_chat_request(
    request: &ChatCompletionRequest,
    is_first_message: bool,
) -> PromptFeatures {
    let prompt_text = request_prompt_text(request);
    let latest_user_text = request_latest_user_text(request);
    let classification_text = latest_user_text
        .as_deref()
        .filter(|text| !text.is_empty())
        .unwrap_or(prompt_text.as_str());
    analyze_prompt_texts(classification_text, &prompt_text, request, is_first_message)
}

fn is_auto_model(model: &str) -> bool {
    matches!(model, "auto" | "brouter/auto")
        || model.starts_with("group:")
        || model.starts_with("profile:")
}

fn request_prompt_text(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .iter()
        .map(|message| message.content.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

fn request_latest_user_text(request: &ChatCompletionRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role.eq_ignore_ascii_case("user"))
        .map(|message| message.content.as_text())
}

fn analyze_prompt_texts(
    classification_prompt: &str,
    token_prompt: &str,
    request: &ChatCompletionRequest,
    is_first_message: bool,
) -> PromptFeatures {
    let lower_prompt = classification_prompt.to_lowercase();
    let intent = detect_intent(&lower_prompt);
    let reasoning = request.reasoning_effort.map_or_else(
        || detect_reasoning(&lower_prompt, intent, is_first_message),
        reasoning_level,
    );
    let mut required_capabilities = required_capabilities(intent, reasoning);
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        required_capabilities.push(ModelCapability::Tools);
    }
    if request.response_format.is_some() {
        required_capabilities.push(ModelCapability::Json);
    }

    let estimated_input_tokens = estimate_tokens(token_prompt);
    let estimated_output_tokens = request.max_tokens.unwrap_or(1_024);

    PromptFeatures {
        intent,
        reasoning,
        original_prompt: token_prompt.to_string(),
        estimated_input_tokens,
        estimated_output_tokens,
        required_context_tokens: estimated_input_tokens.saturating_add(estimated_output_tokens),
        required_capabilities,
        preferred_capabilities: Vec::new(),
        required_attributes: BTreeMap::new(),
        preferred_attributes: BTreeMap::new(),
        matched_rules: Vec::new(),
        is_first_message,
    }
}

fn detect_intent(prompt: &str) -> PromptIntent {
    if contains_any(
        prompt,
        &["debug", "stack trace", "error:", "exception", "panic"],
    ) {
        PromptIntent::Debugging
    } else if contains_any(
        prompt,
        &["code", "rust", "typescript", "function", "compile"],
    ) {
        PromptIntent::Coding
    } else if contains_any(prompt, &["summarize", "summary", "tl;dr"]) {
        PromptIntent::Summarization
    } else if contains_any(prompt, &["extract", "parse", "json", "schema"]) {
        PromptIntent::Extraction
    } else if contains_any(prompt, &["plan", "architecture", "design", "roadmap"]) {
        PromptIntent::Planning
    } else if contains_any(prompt, &["poem", "story", "creative", "rewrite"]) {
        PromptIntent::Creative
    } else if contains_any(prompt, &["calculate", "proof", "equation", "math"]) {
        PromptIntent::Math
    } else if contains_any(prompt, &["tool", "agent", "workflow", "execute"]) {
        PromptIntent::Agentic
    } else {
        PromptIntent::General
    }
}

fn detect_reasoning(prompt: &str, intent: PromptIntent, is_first_message: bool) -> ReasoningLevel {
    if contains_any(
        prompt,
        &[
            "deeply reason",
            "complex",
            "architecture",
            "prove",
            "root cause",
            "tradeoff",
        ],
    ) || matches!(intent, PromptIntent::Math | PromptIntent::Planning)
    {
        ReasoningLevel::High
    } else if is_first_message
        || matches!(
            intent,
            PromptIntent::Coding | PromptIntent::Debugging | PromptIntent::Agentic
        )
    {
        ReasoningLevel::Medium
    } else {
        ReasoningLevel::Low
    }
}

const fn reasoning_level(effort: ReasoningEffort) -> ReasoningLevel {
    match effort {
        ReasoningEffort::None | ReasoningEffort::Minimal | ReasoningEffort::Low => {
            ReasoningLevel::Low
        }
        ReasoningEffort::Medium => ReasoningLevel::Medium,
        ReasoningEffort::High | ReasoningEffort::Max => ReasoningLevel::High,
    }
}

fn required_capabilities(intent: PromptIntent, reasoning: ReasoningLevel) -> Vec<ModelCapability> {
    let mut capabilities = vec![ModelCapability::Chat];
    if matches!(intent, PromptIntent::Coding | PromptIntent::Debugging) {
        capabilities.push(ModelCapability::Code);
    }
    if matches!(reasoning, ReasoningLevel::High) {
        capabilities.push(ModelCapability::Reasoning);
    }
    capabilities
}

fn rule_matches(rule: &RoutingRule, lower_prompt: &str, intent: PromptIntent) -> bool {
    let intent_matches = rule.intent.is_none_or(|rule_intent| rule_intent == intent);
    let contains_matches = rule.when_contains.is_empty()
        || rule
            .when_contains
            .iter()
            .any(|needle| lower_prompt.contains(needle));
    intent_matches && contains_matches
}

fn append_unique_capabilities(
    capabilities: &mut Vec<ModelCapability>,
    additions: &[ModelCapability],
) {
    for capability in additions {
        if !capabilities.contains(capability) {
            capabilities.push(*capability);
        }
    }
}

fn append_attributes(
    attributes: &mut BTreeMap<String, String>,
    additions: &BTreeMap<String, String>,
) {
    for (key, value) in additions {
        attributes
            .entry(key.clone())
            .or_insert_with(|| value.clone());
    }
}

fn contains_any(prompt: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| prompt.contains(needle))
}

fn estimate_tokens(prompt: &str) -> u32 {
    let char_count = prompt.chars().count();
    let token_estimate = char_count.saturating_add(3) / 4;
    u32::try_from(token_estimate).unwrap_or(u32::MAX)
}

fn apply_context_policy(
    features: &mut PromptFeatures,
    context_policy: ContextPolicy,
    session_context_tokens: Option<u32>,
) {
    let current_requirement = features
        .estimated_input_tokens
        .saturating_add(features.estimated_output_tokens);
    let current_requirement =
        add_safety_margin(current_requirement, context_policy.safety_margin_ratio);
    let session_requirement = if context_policy.preserve_session_context_floor
        && !context_policy.allow_context_downgrade
    {
        session_context_tokens.unwrap_or_default()
    } else {
        0
    };
    features.required_context_tokens = current_requirement.max(session_requirement);
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn add_safety_margin(tokens: u32, margin_ratio: f64) -> u32 {
    if !margin_ratio.is_finite() || margin_ratio <= 0.0 {
        return tokens;
    }
    let multiplier = 1.0 + margin_ratio;
    if multiplier >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    let with_margin = (f64::from(tokens) * multiplier).ceil();
    if with_margin >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        with_margin as u32
    }
}

fn candidate_exclusion_reason(
    model: &RouteableModel,
    features: &PromptFeatures,
    objective: RoutingObjective,
    allowed_models: Option<&[ModelId]>,
    profile: Option<&RoutingProfile>,
    dynamic_policy_effects: &[DynamicPolicyEffect],
) -> Option<String> {
    if allowed_models.is_some_and(|allowed| !allowed.contains(&model.id)) {
        return Some("not in requested model group or allowlist".to_string());
    }
    if let Some(profile) = profile
        && !profile.allow.is_empty()
        && !profile
            .allow
            .iter()
            .any(|selector| selector_matches_model(selector, model))
    {
        return Some("not allowed by routing profile".to_string());
    }
    if let Some(deny) = hard_matching_deny(profile, model) {
        return Some(format!("denied by routing profile: {}", deny.reason));
    }
    if let Some(reason) = dynamic_exclusion_reason(model, dynamic_policy_effects) {
        return Some(reason);
    }
    if objective == RoutingObjective::LocalOnly && !model.has_capability(ModelCapability::Local) {
        return Some("local-only objective requires local capability".to_string());
    }
    if model.context_window < features.required_context_tokens {
        return Some(format!(
            "context_window {} below required context {}",
            model.context_window, features.required_context_tokens
        ));
    }
    if let Some(capability) = features
        .required_capabilities
        .iter()
        .find(|capability| !model.has_capability(**capability))
    {
        return Some(format!("missing required capability {capability:?}"));
    }
    if !attributes_match(&model.attributes, &features.required_attributes) {
        return Some("missing required routing attributes".to_string());
    }
    None
}

fn hard_matching_deny<'a>(
    profile: Option<&'a RoutingProfile>,
    model: &RouteableModel,
) -> Option<&'a CandidateDenyRule> {
    profile.and_then(|profile| {
        profile
            .deny
            .iter()
            .find(|deny| deny.hard && selector_matches_model(&deny.selector, model))
    })
}

fn selector_matches_model(selector: &CandidateSelector, model: &RouteableModel) -> bool {
    (selector.models.is_empty() || selector.models.contains(&model.id))
        && (selector.upstream_models.is_empty()
            || selector.upstream_models.contains(&model.upstream_model))
        && (selector.providers.is_empty() || selector.providers.contains(&model.provider))
        && selector
            .capabilities
            .iter()
            .all(|capability| model.has_capability(*capability))
        && attributes_match(&model.attributes, &selector.attributes)
}

fn dynamic_exclusion_reason(
    model: &RouteableModel,
    effects: &[DynamicPolicyEffect],
) -> Option<String> {
    effects.iter().find_map(|effect| match effect {
        DynamicPolicyEffect::Exclude { selector, reason }
            if resource_selector_matches_model(selector, model) =>
        {
            Some(reason.clone())
        }
        DynamicPolicyEffect::DisableAttribute {
            selector,
            key,
            value,
            reason,
        } if resource_selector_matches_model(selector, model)
            && model.attributes.get(key) == Some(value) =>
        {
            Some(reason.clone())
        }
        _ => None,
    })
}

fn resource_selector_matches_model(selector: &ResourceSelector, model: &RouteableModel) -> bool {
    (selector.providers.is_empty() || selector.providers.contains(&model.provider))
        && (selector.upstream_models.is_empty()
            || selector.upstream_models.contains(&model.upstream_model))
        && (selector.configured_models.is_empty() || selector.configured_models.contains(&model.id))
        && selector
            .capabilities
            .iter()
            .all(|capability| model.has_capability(*capability))
        && attributes_match(&model.attributes, &selector.attributes)
}

fn apply_soft_denies(
    model: &RouteableModel,
    profile: Option<&RoutingProfile>,
    weights: ScoringWeights,
    candidate: &mut ScoredCandidate,
) {
    let Some(profile) = profile else {
        return;
    };
    for deny in &profile.deny {
        if deny.hard || !selector_matches_model(&deny.selector, model) {
            continue;
        }
        let penalty = deny.penalty.unwrap_or(weights.policy_penalty);
        candidate.score -= penalty;
        candidate
            .reasons
            .push(format!("soft-deny penalty {penalty}: {}", deny.reason));
    }
}

fn score_model(
    model: &RouteableModel,
    features: &PromptFeatures,
    objective: RoutingObjective,
    weights: ScoringWeights,
) -> ScoredCandidate {
    let estimated_cost = estimate_cost(model, features);
    let mut score = base_quality_score(model, features, weights);
    let mut reasons = vec![format!("quality score {}", model.quality)];

    apply_objective(
        model,
        objective,
        estimated_cost,
        weights,
        &mut score,
        &mut reasons,
    );
    apply_preferred_capabilities(model, features, weights, &mut score, &mut reasons);
    apply_preferred_attributes(model, features, weights, &mut score, &mut reasons);
    apply_session_bias(features, weights, &mut score, &mut reasons);

    ScoredCandidate {
        model_id: model.id.clone(),
        score,
        estimated_cost,
        reasons,
        capabilities: model.capabilities.clone(),
        provider: model.provider.to_string(),
        quality: model.quality,
        attributes: model.attributes.clone(),
        display_badges: model.display_badges.clone(),
        metadata: model.metadata.clone(),
    }
}

fn base_quality_score(
    model: &RouteableModel,
    features: &PromptFeatures,
    weights: ScoringWeights,
) -> f64 {
    let mut score = f64::from(model.quality) * weights.quality_weight;
    if model.has_capability(ModelCapability::Code)
        && matches!(
            features.intent,
            PromptIntent::Coding | PromptIntent::Debugging
        )
    {
        score += weights.code_bonus;
    }
    if model.has_capability(ModelCapability::Reasoning)
        && matches!(features.reasoning, ReasoningLevel::High)
    {
        score += weights.reasoning_bonus;
    }
    score
}

fn apply_objective(
    model: &RouteableModel,
    objective: RoutingObjective,
    estimated_cost: f64,
    weights: ScoringWeights,
    score: &mut f64,
    reasons: &mut Vec<String>,
) {
    match objective {
        RoutingObjective::Cheapest => {
            *score -= estimated_cost * weights.cheapest_cost_weight;
            reasons.push("cheapest objective penalized cost".to_string());
        }
        RoutingObjective::Fastest => {
            if model.has_capability(ModelCapability::Local) {
                *score += weights.local_bonus;
                reasons.push("fastest objective preferred local model".to_string());
            }
        }
        RoutingObjective::Strongest => {
            *score += f64::from(model.quality) * weights.strongest_quality_weight;
            reasons.push("strongest objective boosted quality".to_string());
        }
        RoutingObjective::LocalOnly => {
            *score += weights.local_bonus / 2.0;
            reasons.push("local-only objective matched local model".to_string());
        }
        RoutingObjective::Balanced => {
            *score -= estimated_cost * weights.balanced_cost_weight;
            reasons.push("balanced objective considered cost".to_string());
        }
    }
}

fn apply_preferred_capabilities(
    model: &RouteableModel,
    features: &PromptFeatures,
    weights: ScoringWeights,
    score: &mut f64,
    reasons: &mut Vec<String>,
) {
    for capability in &features.preferred_capabilities {
        if model.has_capability(*capability) {
            *score += weights.reasoning_bonus / 2.0;
            reasons.push(format!("matched preferred capability {capability:?}"));
        }
    }
}

fn apply_preferred_attributes(
    model: &RouteableModel,
    features: &PromptFeatures,
    weights: ScoringWeights,
    score: &mut f64,
    reasons: &mut Vec<String>,
) {
    for (key, value) in &features.preferred_attributes {
        if model.attributes.get(key) == Some(value) {
            *score += weights.reasoning_bonus / 4.0;
            reasons.push(format!("matched preferred attribute {key}={value}"));
        }
    }
}

fn apply_dynamic_penalties(
    model: &RouteableModel,
    effects: &[DynamicPolicyEffect],
    candidate: &mut ScoredCandidate,
) {
    for effect in effects {
        if let DynamicPolicyEffect::Penalize {
            selector,
            penalty,
            reason,
        } = effect
            && resource_selector_matches_model(selector, model)
        {
            candidate.score -= *penalty;
            candidate
                .reasons
                .push(format!("dynamic policy penalty {penalty}: {reason}"));
        }
    }
}

fn attributes_match(
    model_attributes: &BTreeMap<String, String>,
    required_attributes: &BTreeMap<String, String>,
) -> bool {
    required_attributes
        .iter()
        .all(|(key, value)| model_attributes.get(key) == Some(value))
}

fn apply_session_bias(
    features: &PromptFeatures,
    weights: ScoringWeights,
    score: &mut f64,
    reasons: &mut Vec<String>,
) {
    if features.is_first_message && features.reasoning >= ReasoningLevel::Medium {
        *score += weights.first_message_reasoning_bonus;
        reasons.push("first complex message favored stronger model".to_string());
    }
}

fn estimate_cost(model: &RouteableModel, features: &PromptFeatures) -> f64 {
    let input_cost =
        f64::from(features.estimated_input_tokens) / 1_000_000.0 * model.input_cost_per_million;
    let output_cost =
        f64::from(features.estimated_output_tokens) / 1_000_000.0 * model.output_cost_per_million;
    input_cost + output_cost
}

fn compare_candidate_scores(left: &ScoredCandidate, right: &ScoredCandidate) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.estimated_cost.total_cmp(&right.estimated_cost))
        .then_with(|| left.model_id.cmp(&right.model_id))
}

#[cfg(test)]
mod tests {
    use brouter_api_models::{ChatMessage, MessageContent};
    use brouter_provider_models::{ModelId, ProviderId};

    use super::*;

    #[test]
    fn chooses_code_model_for_rust_prompt() {
        let router = Router::new(test_models(), RoutingObjective::Balanced);
        let decision = router
            .route_chat(&chat_request("debug this Rust compile error"), true)
            .expect("router should choose a model");

        assert_eq!(decision.selected_model, ModelId::new("coder"));
    }

    #[test]
    fn rule_can_force_local_only_for_private_prompts() {
        let router = Router::new_with_rules(
            test_models(),
            RoutingObjective::Balanced,
            ScoringWeights::default(),
            vec![RoutingRule {
                name: "private-local".to_string(),
                when_contains: vec!["secret".to_string()],
                intent: None,
                objective: Some(RoutingObjective::LocalOnly),
                prefer_capabilities: Vec::new(),
                require_capabilities: vec![ModelCapability::Local],
                prefer_attributes: BTreeMap::new(),
                require_attributes: BTreeMap::new(),
                llm_judge: false,
            }],
        );

        let decision = router
            .route_chat(&chat_request("summarize this secret"), true)
            .expect("router should choose local model");

        assert_eq!(decision.selected_model, ModelId::new("general"));
        assert_eq!(decision.objective, RoutingObjective::LocalOnly);
        assert_eq!(decision.features.matched_rules, vec!["private-local"]);
    }

    #[test]
    fn latest_user_message_drives_intent_over_system_context() {
        let router = Router::new_with_rules(
            priority_lane_models(),
            RoutingObjective::Balanced,
            ScoringWeights::default(),
            vec![RoutingRule {
                name: "planning-priority".to_string(),
                when_contains: Vec::new(),
                intent: Some(PromptIntent::Planning),
                objective: None,
                prefer_capabilities: Vec::new(),
                require_capabilities: Vec::new(),
                prefer_attributes: BTreeMap::from([(
                    "latency_class".to_string(),
                    "priority".to_string(),
                )]),
                require_attributes: BTreeMap::new(),
                llm_judge: false,
            }],
        );

        let decision = router
            .route_chat(
                &chat_request_with_system("plan debug architecture", "hello"),
                true,
            )
            .expect("router should choose a model");

        assert_eq!(decision.selected_model, ModelId::new("openai_max_strong"));
        assert!(decision.features.matched_rules.is_empty());
    }

    #[test]
    fn explicit_priority_trigger_can_prefer_priority_lane() {
        let router = priority_trigger_router();

        let decision = router
            .route_chat(&chat_request("urgent please answer quickly"), true)
            .expect("router should choose a model");

        assert_eq!(
            decision.selected_model,
            ModelId::new("openai_max_strong_priority")
        );
        assert_eq!(decision.features.matched_rules, vec!["explicit-priority"]);
    }

    #[test]
    fn explicit_priority_trigger_is_not_sticky_across_user_messages() {
        let decision = priority_trigger_router()
            .route_chat(
                &chat_request_with_messages(vec![
                    chat_message("user", "urgent please answer quickly"),
                    chat_message("assistant", "ok"),
                    chat_message("user", "hello"),
                ]),
                false,
            )
            .expect("router should choose a model");

        assert_eq!(decision.selected_model, ModelId::new("openai_max_strong"));
        assert!(decision.features.matched_rules.is_empty());
    }

    #[test]
    fn profile_hard_deny_excludes_matching_models() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "conserve".to_string(),
            RoutingProfile {
                objective: None,
                allow: Vec::new(),
                deny: vec![CandidateDenyRule {
                    selector: CandidateSelector {
                        attributes: BTreeMap::from([(
                            "latency_class".to_string(),
                            "priority".to_string(),
                        )]),
                        ..CandidateSelector::default()
                    },
                    reason: "priority lane disabled while quota is low".to_string(),
                    hard: true,
                    penalty: None,
                }],
                context_policy: None,
            },
        );
        let router = Router::new_with_policy(
            priority_lane_models(),
            RoutingObjective::Balanced,
            ScoringWeights::default(),
            Vec::new(),
            profiles,
            ContextPolicy::default(),
        );

        let decision = router
            .route_chat_with_options(
                &chat_request("urgent please answer quickly"),
                true,
                RoutingOptions {
                    profile: Some("conserve".to_string()),
                    ..RoutingOptions::default()
                },
            )
            .expect("router should choose non-denied model");

        assert_eq!(decision.selected_model, ModelId::new("openai_max_strong"));
        assert!(decision.excluded_candidates.iter().any(|excluded| {
            excluded.model_id == ModelId::new("openai_max_strong_priority")
                && excluded.reason.contains("priority lane disabled")
        }));
    }

    #[test]
    fn session_context_floor_prevents_unsafe_downgrade() {
        let router = Router::new(context_floor_models(), RoutingObjective::Balanced);

        let decision = router
            .route_chat_with_options(
                &chat_request("short follow-up"),
                false,
                RoutingOptions {
                    session_context_tokens: Some(140_000),
                    ..RoutingOptions::default()
                },
            )
            .expect("router should preserve session context floor");

        assert_eq!(decision.selected_model, ModelId::new("long_context"));
        assert!(decision.excluded_candidates.iter().any(|excluded| {
            excluded.model_id == ModelId::new("small_context")
                && excluded.reason.contains("below required context")
        }));
    }

    fn priority_trigger_router() -> Router {
        Router::new_with_rules(
            priority_lane_models(),
            RoutingObjective::Balanced,
            ScoringWeights::default(),
            vec![RoutingRule {
                name: "explicit-priority".to_string(),
                when_contains: vec!["urgent".to_string()],
                intent: None,
                objective: None,
                prefer_capabilities: Vec::new(),
                require_capabilities: Vec::new(),
                prefer_attributes: BTreeMap::from([(
                    "latency_class".to_string(),
                    "priority".to_string(),
                )]),
                require_attributes: BTreeMap::new(),
                llm_judge: false,
            }],
        )
    }

    fn chat_request(prompt: &str) -> ChatCompletionRequest {
        chat_request_with_messages(vec![chat_message("user", prompt)])
    }

    fn chat_request_with_system(system: &str, user: &str) -> ChatCompletionRequest {
        chat_request_with_messages(vec![
            chat_message("system", system),
            chat_message("user", user),
        ])
    }

    fn chat_request_with_messages(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "auto".to_string(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            metadata: None,
            extra: std::collections::BTreeMap::new(),
        }
    }

    fn chat_message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            extra: BTreeMap::new(),
        }
    }

    fn priority_lane_models() -> Vec<RouteableModel> {
        vec![
            RouteableModel {
                id: ModelId::new("openai_max_strong"),
                provider: ProviderId::new("openai_max"),
                upstream_model: "gpt-5.5".to_string(),
                context_window: 1_050_000,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: 98,
                capabilities: vec![ModelCapability::Chat, ModelCapability::Reasoning],
                attributes: BTreeMap::from([("latency_class".to_string(), "standard".to_string())]),
                display_badges: vec!["standard".to_string()],
                metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
            },
            RouteableModel {
                id: ModelId::new("openai_max_strong_priority"),
                provider: ProviderId::new("openai_max"),
                upstream_model: "gpt-5.5".to_string(),
                context_window: 1_050_000,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: 98,
                capabilities: vec![ModelCapability::Chat, ModelCapability::Reasoning],
                attributes: BTreeMap::from([("latency_class".to_string(), "priority".to_string())]),
                display_badges: vec!["priority".to_string()],
                metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
            },
        ]
    }

    fn context_floor_models() -> Vec<RouteableModel> {
        vec![
            RouteableModel {
                id: ModelId::new("small_context"),
                provider: ProviderId::new("cloud"),
                upstream_model: "small".to_string(),
                context_window: 100_000,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: 99,
                capabilities: vec![ModelCapability::Chat],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
            },
            RouteableModel {
                id: ModelId::new("long_context"),
                provider: ProviderId::new("cloud"),
                upstream_model: "long".to_string(),
                context_window: 200_000,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: 80,
                capabilities: vec![ModelCapability::Chat],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
            },
        ]
    }

    fn test_models() -> Vec<RouteableModel> {
        vec![
            RouteableModel {
                id: ModelId::new("general"),
                provider: ProviderId::new("local"),
                upstream_model: "general".to_string(),
                context_window: 8_192,
                input_cost_per_million: 0.0,
                output_cost_per_million: 0.0,
                quality: 60,
                capabilities: vec![ModelCapability::Chat, ModelCapability::Local],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
            },
            RouteableModel {
                id: ModelId::new("coder"),
                provider: ProviderId::new("cloud"),
                upstream_model: "coder".to_string(),
                context_window: 128_000,
                input_cost_per_million: 0.15,
                output_cost_per_million: 0.60,
                quality: 80,
                capabilities: vec![ModelCapability::Chat, ModelCapability::Code],
                attributes: BTreeMap::new(),
                display_badges: Vec::new(),
                metadata: brouter_catalog_models::ResolvedModelMetadata::default(),
            },
        ]
    }
}
