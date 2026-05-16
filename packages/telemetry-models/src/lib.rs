#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Telemetry data models for brouter.

use brouter_provider_models::ModelId;
use serde::{Deserialize, Serialize};

/// Request telemetry captured after a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageEvent {
    pub timestamp_ms: u64,
    pub session_id: Option<String>,
    pub selected_model: ModelId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resource_pools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_model: Option<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_overridden: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routing_reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_used: Option<bool>,
    pub estimated_cost: f64,
    pub latency_ms: Option<u64>,
    pub status_code: Option<u16>,
    pub provider_error: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub success: bool,
}
