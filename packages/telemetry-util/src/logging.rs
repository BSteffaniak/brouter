//! Structured observability utilities for brouter.
//!
//! This module provides macros and functions for consistent, structured logging
//! across all brouter components. Logs are designed to be:
//! - Machine-parseable (JSON)
//! - Human-readable (in dev mode)
//! - Comprehensive (no guessing what happened)

use serde::Serialize;
use std::time::{Duration, Instant};

/// Masks sensitive values in strings (API keys, tokens, etc.)
#[must_use]
pub fn mask_sensitive(value: &str) -> String {
    if value.len() <= 8 {
        "***".to_string()
    } else {
        format!("{}...{}", &value[..4], &value[value.len() - 4..])
    }
}

/// Masks an API key in a URL query string
#[must_use]
pub fn mask_api_key_in_url(url: &str) -> String {
    if url.contains("api_key=") {
        url.split("api_key=")
            .enumerate()
            .map(|(i, part)| {
                if i == 1 {
                    mask_sensitive(part.split('&').next().unwrap_or(part))
                } else {
                    part.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("api_key=")
    } else {
        url.to_string()
    }
}

/// A timer for measuring elapsed time in structured logs
pub struct Timer {
    start: Instant,
}

impl Timer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for a provider request
#[derive(Debug, Clone, Serialize)]
pub struct ProviderRequestMeta {
    pub provider_id: String,
    pub provider_kind: String,
    pub model_id: String,
    pub upstream_model: String,
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ProviderRequestMeta {
    /// Creates new metadata for a provider request
    #[must_use]
    pub fn new(
        provider_id: &str,
        provider_kind: &str,
        model_id: &str,
        upstream_model: &str,
        base_url: Option<&str>,
    ) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            provider_kind: provider_kind.to_string(),
            model_id: model_id.to_string(),
            upstream_model: upstream_model.to_string(),
            base_url: base_url.map(|s| s.to_string()),
            elapsed_ms: None,
            status: None,
            error: None,
        }
    }

    /// Adds timing information
    pub fn with_timing(mut self, elapsed_ms: u64) -> Self {
        self.elapsed_ms = Some(elapsed_ms);
        self
    }

    /// Adds response status
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    /// Adds error information
    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self
    }
}

/// Metadata for routing decisions
#[derive(Debug, Clone, Serialize)]
pub struct RoutingDecisionMeta {
    pub objective: String,
    pub intent: String,
    pub reasoning_level: String,
    pub candidates_count: usize,
    pub score_gap: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_triggered: Option<String>,
    pub selected_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_overridden: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_error: Option<String>,
}

impl RoutingDecisionMeta {
    /// Creates new metadata for a routing decision
    #[must_use]
    pub fn new(
        objective: &str,
        intent: &str,
        reasoning_level: &str,
        candidates_count: usize,
        score_gap: f64,
        selected_model: &str,
    ) -> Self {
        Self {
            objective: objective.to_string(),
            intent: intent.to_string(),
            reasoning_level: reasoning_level.to_string(),
            candidates_count,
            score_gap,
            rule_triggered: None,
            selected_model: selected_model.to_string(),
            judge_overridden: None,
            judge_model: None,
            judge_error: None,
        }
    }

    /// Adds rule that triggered the decision
    pub fn with_rule(mut self, rule: Option<&str>) -> Self {
        self.rule_triggered = rule.map(|s| s.to_string());
        self
    }

    /// Adds judge override information
    pub fn with_judge(
        mut self,
        judge_overridden: bool,
        judge_model: Option<&str>,
        judge_error: Option<&str>,
    ) -> Self {
        self.judge_overridden = Some(judge_overridden);
        self.judge_model = judge_model.map(|s| s.to_string());
        self.judge_error = judge_error.map(|s| s.to_string());
        self
    }
}

/// Metadata for judge invocations
#[derive(Debug, Clone, Serialize)]
pub struct JudgeInvocationMeta {
    pub judge_model: String,
    pub provider: String,
    pub candidate_count: usize,
    pub candidate_ids: Vec<String>,
    pub active_provider_count: usize,
    pub failed_providers: Vec<String>,
    pub prompt_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_length: Option<usize>,
    pub parse_success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
}

impl JudgeInvocationMeta {
    /// Creates new metadata for a judge invocation
    #[must_use]
    pub fn new(
        judge_model: &str,
        provider: &str,
        candidate_ids: &[String],
        active_provider_count: usize,
        failed_providers: &[String],
        prompt_length: usize,
    ) -> Self {
        Self {
            judge_model: judge_model.to_string(),
            provider: provider.to_string(),
            candidate_count: candidate_ids.len(),
            candidate_ids: candidate_ids.to_vec(),
            active_provider_count,
            failed_providers: failed_providers.to_vec(),
            prompt_length,
            response_length: None,
            parse_success: false,
            chosen_model: None,
            error: None,
            elapsed_ms: None,
        }
    }

    /// Records the judge succeeded
    pub fn with_success(mut self, response_length: usize, chosen_model: &str) -> Self {
        self.response_length = Some(response_length);
        self.parse_success = true;
        self.chosen_model = Some(chosen_model.to_string());
        self
    }

    /// Records the judge failed
    pub fn with_failure(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self
    }

    /// Adds timing
    pub fn with_timing(mut self, elapsed_ms: u64) -> Self {
        self.elapsed_ms = Some(elapsed_ms);
        self
    }
}
