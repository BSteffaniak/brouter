#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Routing decision and prompt classification models for brouter.

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

/// Prompt features extracted before candidate scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptFeatures {
    pub intent: PromptIntent,
    pub reasoning: ReasoningLevel,
    pub estimated_input_tokens: u32,
    pub required_capabilities: Vec<ModelCapability>,
    pub is_first_message: bool,
}

/// One scored router candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredCandidate {
    pub model_id: ModelId,
    pub score: f64,
    pub estimated_cost: f64,
    pub reasons: Vec<String>,
}

/// Explainable model selection result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub selected_model: ModelId,
    pub objective: RoutingObjective,
    pub features: PromptFeatures,
    pub reasons: Vec<String>,
    pub candidates: Vec<ScoredCandidate>,
}
