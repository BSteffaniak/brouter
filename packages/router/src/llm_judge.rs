#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions, dead_code)]

//! LLM-based judge for model selection reasoning.
//!
//! Stage 3 of the routing pipeline: after deterministic scoring, an optional LLM
//! re-evaluates the shortlist and produces a natural-language rationale.

use brouter_api_models::{ChatCompletionRequest, ChatMessage, MessageContent};
use brouter_provider_models::{ModelCapability, ModelId};
use brouter_router_models::{ModelReasoning, RoutingObjective, ScoredCandidate};
use serde::Deserialize;
use std::collections::BTreeMap;

/// Default system prompt for the LLM judge.
pub const DEFAULT_JUDGE_SYSTEM_PROMPT: &str = r#"You are a routing advisor for brouter, a deterministic LLM router.

Your role: given the original prompt, a ranked shortlist of model candidates, and their deterministic scores, select the best model and explain your reasoning in 1-3 sentences.

Rules:
- You may only select from the provided shortlist. Do not invent or assume models.
- The deterministic scores are a strong signal, not a constraint. Use your judgment.
- Keep your explanation concise.

Output format (JSON, no additional text):
{"selected_model": "<model_id from shortlist>", "reasoning": "<1-3 sentence explanation>"}"#;

/// Output format expected from the LLM judge.
#[derive(Debug, Deserialize)]
struct JudgeResponse {
    selected_model: String,
    reasoning: String,
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

/// Configuration for invoking the LLM judge.
#[derive(Debug, Clone)]
pub struct JudgeConfig {
    /// Model ID to use for LLM judge calls.
    pub model: ModelId,
    /// Custom system prompt. None uses the default.
    pub system_prompt: Option<String>,
    /// Maximum tokens for the reasoning response.
    pub max_tokens: u32,
    /// Temperature for reasoning calls.
    pub temperature: f64,
    /// Maximum estimated cost for the reasoning call.
    pub max_estimated_cost: f64,
}

/// Trigger evaluation for the LLM judge.
#[derive(Debug, Clone, Copy)]
pub struct JudgeTrigger {
    /// Fire when top-2 score gap is below this threshold.
    pub score_gap_threshold: f64,
    /// Fire when a matched rule has `llm_judge` = true.
    pub rule_triggered: bool,
}

impl JudgeTrigger {
    /// Returns true when the judge should fire.
    #[must_use]
    pub fn should_fire(&self, top_2_gap: f64, rule_triggered: bool) -> bool {
        top_2_gap < self.score_gap_threshold || (self.rule_triggered && rule_triggered)
    }
}

impl JudgeConfig {
    /// Returns the effective system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> &str {
        self.system_prompt
            .as_deref()
            .unwrap_or(DEFAULT_JUDGE_SYSTEM_PROMPT)
    }
}

/// Format capabilities as a comma-separated string.
fn format_capabilities(caps: &[ModelCapability]) -> String {
    caps.iter()
        .map(|c| format!("{c:?}"))
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
                "  {}. {} | provider: {} | capabilities: [{}] | quality: {} | est. cost: ${} | score: {}",
                i + 1,
                c.model_id,
                c.provider,
                format_capabilities(&c.capabilities),
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

    let intent_str = format!("{intent:?}");
    let objective_str = format!("{objective:?}");
    format!(
        "Original prompt: \"{prompt_text}\"\n\nIntent detected: {intent_str}\nObjective: {objective_str}\n\nSession context:\n  - previous requests: {count}\n  - accumulated cost: ${cost}\n  - recent decisions:\n{recent}\n\nStage 2 candidates (ranked by deterministic score):\n{candidates}\n\nSelect the best model and explain in JSON format:\n{{\"selected_model\": \"<model_id>\", \"reasoning\": \"<1-3 sentence explanation>\"}}",
        count = session.request_count,
        cost = session.accumulated_cost,
        recent = recent_lines,
        candidates = candidate_lines,
    )
}

/// Parse the judge response and validate the model selection.
pub fn parse_judge_response(
    raw: &str,
    shortlist: &[ScoredCandidate],
    judge_model: &ModelId,
) -> ModelReasoning {
    let trimmed = raw.trim();
    let json_str = extract_json(trimmed);
    let parsed: Result<JudgeResponse, _> = serde_json::from_str(&json_str);

    match parsed {
        Ok(response) => {
            let raw_model = response.selected_model;
            let chosen_id = ModelId::new(raw_model.clone());
            let top_pick = shortlist.first().map(|c| c.model_id.clone());
            let overridden = top_pick.as_ref().is_some_and(|top| top != &chosen_id);
            let valid = shortlist.iter().any(|c| c.model_id == chosen_id);

            if valid {
                ModelReasoning {
                    model_id: judge_model.clone(),
                    rationale: response.reasoning,
                    chosen_model: chosen_id,
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
                    overridden: false,
                    error: Some(format!("invalid model '{raw_model}' returned by judge")),
                }
            }
        }
        Err(e) => {
            let fallback = shortlist
                .first()
                .map_or_else(|| judge_model.clone(), |c| c.model_id.clone());
            ModelReasoning {
                model_id: judge_model.clone(),
                rationale: format!("Failed to parse judge response: {e}"),
                chosen_model: fallback,
                overridden: false,
                error: Some(format!("parse failure: {e}")),
            }
        }
    }
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
            },
            ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(user_prompt.to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: Some(config.temperature),
        top_p: None,
        max_tokens: Some(config.max_tokens),
        reasoning_effort: None,
        stream: Some(false),
        tools: None,
        tool_choice: None,
        response_format: Some(serde_json::json!({"type": "json_object"})),
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
