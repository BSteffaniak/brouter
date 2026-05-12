#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions, dead_code, unused_imports)]

//! LLM-based judge for model selection reasoning.
//!
//! Stage 3 of the routing pipeline: after deterministic scoring, an optional LLM
//! re-evaluates the shortlist and produces a natural-language rationale.

use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent};
use brouter_provider_models::{ModelCapability, ModelId};
use brouter_router_models::{
    JudgeConfig, JudgeOutput, JudgeSessionContext, JudgeShortlistConfig, JudgeTrigger,
    ModelReasoning, RecentDecision, RoutingObjective, ScoredCandidate,
};
use serde::Deserialize;
use std::collections::BTreeMap;

/// Default system prompt for the LLM judge.
pub const DEFAULT_JUDGE_SYSTEM_PROMPT: &str = r#"You are a routing advisor for brouter, a deterministic LLM router.

Your role: given the original prompt, a ranked shortlist of model candidates, and their deterministic scores, select the best model and explain your reasoning in 1-3 sentences.

Rules:
- You may only select from the provided shortlist. Do not invent or assume models.
- The deterministic scores are a strong signal, not a constraint. Use your judgment.
- CRITICAL: Do NOT use internal reasoning, chain-of-thought, or thinking blocks. Output JSON directly.
- Keep your explanation concise.

Output format (JSON, no additional text):
{"selected_model": "<model_id from shortlist>", "service_tier": "standard|priority|null", "reasoning_effort": "low|medium|high|null", "reasoning": "<1-3 sentence explanation>"}"#;

/// Output format expected from the LLM judge.
#[derive(Debug, Deserialize)]
struct JudgeResponse {
    #[serde(alias = "model", alias = "selected_model")]
    selected_model: String,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    reasoning: String,
}

/// Error response format from API providers.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
}

/// OpenAI-compatible response wrapper format.
#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
    #[serde(alias = "reasoning_content", default)]
    reasoning: Option<String>,
}

/// Normalizes a provider response to extract the raw judge JSON.
#[must_use]
pub fn normalize_judge_response(raw: &str) -> String {
    let trimmed = raw.trim();

    // Try to parse as OpenAI wrapped format (see OpenAiResponse struct).
    if let Ok(response) = serde_json::from_str::<OpenAiResponse>(trimmed)
        && let Some(choice) = response.choices.first()
    {
        // First try content field (normal response)
        if let Some(content) = choice.message.content.as_ref() {
            let content_str = content.trim();
            if !content_str.is_empty()
                && content_str.starts_with('{')
                && serde_json::from_str::<serde_json::Value>(content_str).is_ok()
            {
                return content_str.to_string();
            }
        }
        // Fallback: try reasoning field (some models put thinking here)
        if let Some(reasoning) = choice.message.reasoning.as_ref() {
            let reasoning_str = reasoning.trim();
            // Try to extract JSON from the reasoning text
            if let Some(json_start) = reasoning_str.find('{') {
                let potential_json = &reasoning_str[json_start..];
                if serde_json::from_str::<serde_json::Value>(potential_json).is_ok() {
                    // Extract just the JSON portion
                    if let Some(json_end) = potential_json.find('}') {
                        let json_only = &potential_json[..=json_end];
                        if serde_json::from_str::<JudgeResponse>(json_only).is_ok() {
                            return json_only.to_string();
                        }
                    }
                }
            }
        }
    }

    // Fall through to direct JSON or code fence extraction (handled by extract_json).
    trimmed.to_string()
}

/// Format capabilities as a comma-separated string.
fn format_capabilities(caps: &[ModelCapability]) -> String {
    caps.iter()
        .map(|c| format!("{c:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_attributes(attributes: &BTreeMap<String, String>) -> String {
    attributes
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Build the judge prompt for a routing decision.
#[must_use]
pub fn build_judge_prompt(
    prompt_text: &str,
    intent: &str,
    objective: RoutingObjective,
    candidates: &[ScoredCandidate],
    session: &JudgeSessionContext,
) -> String {
    let candidate_lines = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "  {}. {} | provider: {} | capabilities: [{}] | attributes: [{}] | badges: [{}] | quality: {} | est. cost: ${} | score: {}",
                i + 1,
                c.model_id,
                c.provider,
                format_capabilities(&c.capabilities),
                format_attributes(&c.attributes),
                c.display_badges.join(", "),
                c.quality,
                c.estimated_cost,
                c.score
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let recent_lines = if session.recent_decisions.is_empty() {
        "  (none)".to_string()
    } else {
        session
            .recent_decisions
            .iter()
            .map(|d| format!("  - {} (intent: {})", d.model_id, d.intent))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let resource_lines = if session.resource_summary.is_empty() {
        "  (no live quota data available)".to_string()
    } else {
        session
            .resource_summary
            .iter()
            .map(|line| format!("  - {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let intent_str = format!("{intent:?}");
    let objective_str = format!("{objective:?}");
    format!(
        "Original prompt: \"{prompt_text}\"\n\nIntent detected: {intent_str}\nObjective: {objective_str}\n\nSession context:\n  - previous requests: {count}\n  - accumulated cost: ${cost}\n  - recent decisions:\n{recent}\n\nLive provider resource/usage state:\n{resources}\n\nStage 2 candidates (ranked by deterministic score):\n{candidates}\n\nChoose the best safe route. Prefer standard service tier and lower reasoning when adequate; choose priority or high reasoning only when task urgency/complexity justifies spending that quota. Explain in JSON format:\n{{\"selected_model\": \"<model_id>\", \"service_tier\": \"standard|priority|null\", \"reasoning_effort\": \"low|medium|high|null\", \"reasoning\": \"<1-3 sentence explanation>\"}}",
        count = session.request_count,
        cost = session.accumulated_cost,
        recent = recent_lines,
        resources = resource_lines,
        candidates = candidate_lines,
    )
}

/// Parse the judge response and validate the model selection.
#[allow(clippy::must_use_candidate)]
pub fn parse_judge_response(
    raw: &str,
    shortlist: &[ScoredCandidate],
    judge_model: &ModelId,
) -> ModelReasoning {
    let trimmed = raw.trim();
    // Normalize provider-specific response formats (e.g., OpenAI wrapped format).
    let json_str = extract_json(&normalize_judge_response(trimmed));
    let parsed: Result<JudgeResponse, _> = serde_json::from_str(&json_str);

    match parsed {
        Ok(response) => {
            let raw_model = response.selected_model;
            // Try to normalize the model name to match shortlist
            let chosen_id = find_matching_candidate(&raw_model, shortlist)
                .unwrap_or_else(|| ModelId::new(raw_model.clone()));
            let top_pick = shortlist.first().map(|c| c.model_id.clone());
            let overridden = top_pick.as_ref().is_some_and(|top| top != &chosen_id);
            let valid = shortlist.iter().any(|c| c.model_id == chosen_id);

            if valid {
                ModelReasoning {
                    model_id: judge_model.clone(),
                    rationale: response.reasoning,
                    chosen_model: chosen_id,
                    service_tier: normalize_optional_choice(response.service_tier),
                    reasoning_effort: normalize_optional_choice(response.reasoning_effort),
                    overridden,
                    error: None,
                }
            } else {
                ModelReasoning {
                    model_id: judge_model.clone(),
                    rationale: format!(
                        "Judge returned unknown model '{raw_model}'. Falling back to deterministic top pick.",
                    ),
                    chosen_model: top_pick.unwrap_or_else(|| judge_model.clone()),
                    service_tier: normalize_optional_choice(response.service_tier),
                    reasoning_effort: normalize_optional_choice(response.reasoning_effort),
                    overridden: false,
                    error: Some(format!("invalid model '{raw_model}' returned by judge")),
                }
            }
        }
        Err(e) => {
            // Check if the raw response is an error from the API.
            let error_msg = serde_json::from_str::<ApiErrorResponse>(&json_str)
                .ok()
                .and_then(|r| (!r.error.message.is_empty()).then_some(r.error.message));
            let fallback = shortlist
                .first()
                .map_or_else(|| judge_model.clone(), |c| c.model_id.clone());
            let msg = error_msg.unwrap_or_else(|| format!("parse failure: {e}"));
            ModelReasoning {
                model_id: judge_model.clone(),
                rationale: format!("Failed to parse judge response: {msg}"),
                chosen_model: fallback,
                service_tier: None,
                reasoning_effort: None,
                overridden: false,
                error: Some(msg),
            }
        }
    }
}

fn normalize_optional_choice(value: Option<String>) -> Option<String> {
    let value = value?.trim().to_lowercase();
    (!value.is_empty() && value != "null" && value != "none" && value != "auto").then_some(value)
}

/// Attempts to find a matching candidate from the shortlist for the given model name.
/// Handles provider-prefixed model names (e.g., "anthropic/claude-3-opus-v1")
/// and tries to match against known shortlist models.
fn find_matching_candidate(model_name: &str, shortlist: &[ScoredCandidate]) -> Option<ModelId> {
    // 1. Try exact match
    let exact = ModelId::new(model_name);
    if shortlist.iter().any(|c| c.model_id == exact) {
        return Some(exact);
    }

    // 2. Try stripping provider prefix (e.g., "anthropic/claude-3-opus" -> "claude-3-opus")
    if let Some(slash_pos) = model_name.find('/') {
        let stripped = ModelId::new(&model_name[slash_pos + 1..]);
        if shortlist.iter().any(|c| c.model_id == stripped) {
            return Some(stripped);
        }
    }

    // 3. Try stripping version suffixes (e.g., "claude-3-opus-20240229" -> "claude-3-opus")
    let stripped = model_name
        .split('-')
        .take(3) // e.g., claude-3-opus
        .collect::<Vec<_>>()
        .join("-");
    let stripped_id = ModelId::new(&stripped);
    if shortlist.iter().any(|c| c.model_id == stripped_id) {
        return Some(stripped_id);
    }

    // 4. Try matching just the base model name
    for candidate in shortlist {
        let cand_str = candidate.model_id.as_str();
        // Check if candidate contains the model_name as a substring
        if cand_str.contains(model_name) || model_name.contains(cand_str) {
            return Some(candidate.model_id.clone());
        }
    }

    None
}

/// Build a chat completion request for the judge model.
#[must_use]
pub fn judge_request(
    config: &JudgeConfig,
    system_prompt: &str,
    user_prompt: &str,
) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: config.model.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(system_prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                extra: BTreeMap::new(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(user_prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                extra: BTreeMap::new(),
            },
        ],
        temperature: (config.output.temperature > 0.0).then_some(config.output.temperature),
        top_p: None,
        max_tokens: Some(config.output.max_tokens),
        reasoning_effort: None,
        stream: Some(false),
        tools: None,
        tool_choice: None,
        response_format: (config.output.structured)
            .then(|| serde_json::json!({"type": "json_object"})),
        metadata: None,
        extra: BTreeMap::new(),
    }
}

/// Returns the top-2 score gap from a sorted candidate list.
#[must_use]
pub fn top_2_score_gap(candidates: &[ScoredCandidate]) -> f64 {
    if candidates.len() >= 2 {
        let scores: Vec<f64> = candidates.iter().map(|c| c.score).collect();
        (scores[0] - scores[1]).abs()
    } else if candidates.len() == 1 {
        0.0
    } else {
        f64::MAX
    }
}

/// Extract JSON from a string that may contain markdown code fences.
fn extract_json(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let mut in_json = false;
    let mut json_lines = Vec::new();

    for line in &lines {
        if line.contains("```json") {
            in_json = true;
            continue;
        }
        if line.contains("```") && in_json {
            break;
        }
        if in_json {
            json_lines.push(*line);
        }
    }

    if json_lines.is_empty() {
        s.to_string()
    } else {
        json_lines.join("\n").trim().to_string()
    }
}
