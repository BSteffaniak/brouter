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
    pub estimated_cost: f64,
    pub latency_ms: Option<u64>,
    pub status_code: Option<u16>,
    pub provider_error: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub success: bool,
}
