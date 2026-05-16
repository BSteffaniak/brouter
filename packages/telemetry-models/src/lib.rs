#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Telemetry data models for brouter.

use brouter_provider_models::ModelId;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_source: Option<String>,
    pub success: bool,
}

/// Structured session-scoped routing event captured during request handling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingEvent {
    pub timestamp_ms: u64,
    pub event_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub request_id: String,
    pub kind: RoutingEventKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    #[serde(default)]
    pub payload: Value,
}

/// Routing event kind.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RoutingEventKind {
    RouteDecision,
    JudgeInvocation,
    ControlsApplied,
    ProviderAttempt,
    FallbackAttempt,
    DynamicPolicyAdjustment,
    UserPreferenceApplied,
    IntrospectionRefresh,
}

impl RoutingEventKind {
    /// Returns this event kind as a stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RouteDecision => "route_decision",
            Self::JudgeInvocation => "judge_invocation",
            Self::ControlsApplied => "controls_applied",
            Self::ProviderAttempt => "provider_attempt",
            Self::FallbackAttempt => "fallback_attempt",
            Self::DynamicPolicyAdjustment => "dynamic_policy_adjustment",
            Self::UserPreferenceApplied => "user_preference_applied",
            Self::IntrospectionRefresh => "introspection_refresh",
        }
    }
}

/// Summary of events recorded for one brouter client session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionSummary {
    pub session_id: String,
    pub first_timestamp_ms: u64,
    pub last_timestamp_ms: u64,
    pub event_count: u64,
    pub request_count: u64,
}
