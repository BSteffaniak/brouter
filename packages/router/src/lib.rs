#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Prompt analysis and deterministic model routing engine for brouter.

use std::cmp::Ordering;

use brouter_api_models::ChatCompletionRequest;
use brouter_provider_models::{ModelCapability, RouteableModel};
use brouter_router_models::{
    PromptFeatures, PromptIntent, ReasoningLevel, RoutingDecision, RoutingObjective,
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
        let prompt = request_prompt_text(request);
        let features = analyze_prompt(&prompt, request, is_first_message);
        self.route_features(features)
    }

    /// Selects a model for precomputed prompt features.
    ///
    /// # Errors
    ///
    /// Returns an error when no configured model satisfies the requested
    /// capabilities and context window.
    pub fn route_features(&self, features: PromptFeatures) -> Result<RoutingDecision, RouterError> {
        let mut candidates = self.scored_candidates(&features);
        candidates.sort_by(compare_candidate_scores);
        let selected = candidates.first().ok_or(RouterError::NoCandidate)?;
        let selected_model = selected.model_id.clone();
        let reasons = selected.reasons.clone();
        Ok(RoutingDecision {
            selected_model,
            objective: self.objective,
            features,
            reasons,
            candidates,
        })
    }

    fn scored_candidates(&self, features: &PromptFeatures) -> Vec<ScoredCandidate> {
        self.models
            .iter()
            .filter(|model| model_satisfies_features(model, features, self.objective))
            .map(|model| score_model(model, features, self.objective, self.weights))
            .collect()
    }
}

/// Extracts prompt features from a chat request.
#[must_use]
pub fn analyze_chat_request(
    request: &ChatCompletionRequest,
    is_first_message: bool,
) -> PromptFeatures {
    analyze_prompt(&request_prompt_text(request), request, is_first_message)
}

fn request_prompt_text(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .iter()
        .map(|message| message.content.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

fn analyze_prompt(
    prompt: &str,
    request: &ChatCompletionRequest,
    is_first_message: bool,
) -> PromptFeatures {
    let lower_prompt = prompt.to_lowercase();
    let intent = detect_intent(&lower_prompt);
    let reasoning = detect_reasoning(&lower_prompt, intent, is_first_message);
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
        estimated_input_tokens: estimate_tokens(prompt),
        required_capabilities,
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
        let request = ChatCompletionRequest {
            model: "auto".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("debug this Rust compile error".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            metadata: None,
        };

        let decision = router
            .route_chat(&request, true)
            .expect("router should choose a model");

        assert_eq!(decision.selected_model, ModelId::new("coder"));
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
            },
        ]
    }
}
