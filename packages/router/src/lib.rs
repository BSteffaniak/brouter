#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Prompt analysis and deterministic model routing engine for brouter.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use brouter_api_models::{ChatCompletionRequest, ReasoningEffort};
use brouter_provider_models::{ModelCapability, RouteableModel};
use brouter_router_models::{
    PromptFeatures, PromptIntent, ReasoningLevel, RoutingDecision, RoutingObjective, RoutingRule,
    ScoredCandidate, ScoringWeights,
};
use thiserror::Error;

/// Router error.
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("no configured model can satisfy the request")]
    NoCandidate,
}

/// Deterministic prompt router.
#[derive(Debug, Clone)]
pub struct Router {
    models: Vec<RouteableModel>,
    objective: RoutingObjective,
    weights: ScoringWeights,
    rules: Vec<RoutingRule>,
}

impl Router {
    /// Creates a deterministic router.
    #[must_use]
    pub fn new(models: Vec<RouteableModel>, objective: RoutingObjective) -> Self {
        Self::new_with_scoring(models, objective, ScoringWeights::default())
    }

    /// Creates a deterministic router with custom scoring weights.
    #[must_use]
    pub const fn new_with_scoring(
        models: Vec<RouteableModel>,
        objective: RoutingObjective,
        weights: ScoringWeights,
    ) -> Self {
        Self {
            models,
            objective,
            weights,
            rules: Vec::new(),
        }
    }

    /// Creates a deterministic router with custom scoring weights and rules.
    #[must_use]
    pub const fn new_with_rules(
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
        }
    }

    /// Returns configured routeable models.
    #[must_use]
    pub fn models(&self) -> &[RouteableModel] {
        &self.models
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
        allowed_models: Option<&[brouter_provider_models::ModelId]>,
    ) -> Result<RoutingDecision, RouterError> {
        let prompt_text = request_prompt_text(request);
        let latest_user_text = request_latest_user_text(request);
        let classification_text = latest_user_text
            .as_deref()
            .filter(|text| !text.is_empty())
            .unwrap_or(prompt_text.as_str());
        let mut features =
            analyze_prompt_texts(classification_text, &prompt_text, request, is_first_message);
        let objective = self.apply_rules(&classification_text.to_lowercase(), &mut features);
        let explicit_model = if allowed_models.is_none() && !is_auto_model(&request.model) {
            Some(vec![brouter_provider_models::ModelId::new(
                request.model.clone(),
            )])
        } else {
            None
        };
        let allowed_models = allowed_models.or(explicit_model.as_deref());
        self.route_features_with_objective(features, objective, allowed_models)
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
        self.route_features_with_objective(features, objective, None)
    }

    fn route_features_with_objective(
        &self,
        features: PromptFeatures,
        objective: RoutingObjective,
        allowed_models: Option<&[brouter_provider_models::ModelId]>,
    ) -> Result<RoutingDecision, RouterError> {
        let mut candidates = self.scored_candidates(&features, objective, allowed_models);
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
        })
    }

    fn scored_candidates(
        &self,
        features: &PromptFeatures,
        objective: RoutingObjective,
        allowed_models: Option<&[brouter_provider_models::ModelId]>,
    ) -> Vec<ScoredCandidate> {
        self.models
            .iter()
            .filter(|model| allowed_models.is_none_or(|allowed| allowed.contains(&model.id)))
            .filter(|model| model_satisfies_features(model, features, objective))
            .map(|model| score_model(model, features, objective, self.weights))
            .collect()
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
    matches!(model, "auto" | "brouter/auto") || model.starts_with("group:")
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

    PromptFeatures {
        intent,
        reasoning,
        estimated_input_tokens: estimate_tokens(token_prompt),
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

fn model_satisfies_features(
    model: &RouteableModel,
    features: &PromptFeatures,
    objective: RoutingObjective,
) -> bool {
    if objective == RoutingObjective::LocalOnly && !model.has_capability(ModelCapability::Local) {
        return false;
    }
    if model.context_window < features.estimated_input_tokens {
        return false;
    }
    features
        .required_capabilities
        .iter()
        .all(|capability| model.has_capability(*capability))
        && attributes_match(&model.attributes, &features.required_attributes)
}

fn score_model(
    model: &RouteableModel,
    features: &PromptFeatures,
    objective: RoutingObjective,
    weights: ScoringWeights,
) -> ScoredCandidate {
    let estimated_cost = estimate_cost(model, features.estimated_input_tokens);
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

fn estimate_cost(model: &RouteableModel, input_tokens: u32) -> f64 {
    let input_cost = f64::from(input_tokens) / 1_000_000.0 * model.input_cost_per_million;
    let output_tokens = 1_000.0;
    let output_cost = output_tokens / 1_000_000.0 * model.output_cost_per_million;
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
            },
        ]
    }
}
