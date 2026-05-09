#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Provider and model metadata types for brouter.

use std::collections::BTreeMap;

use brouter_catalog_models::ResolvedModelMetadata;
use serde::{Deserialize, Serialize};

/// Stable identifier for a configured provider.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProviderId(String);

impl ProviderId {
    /// Creates a provider identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the provider identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ProviderId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for ProviderId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for ProviderId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Stable identifier for a configured routeable model.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a model identifier.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the model identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ModelId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for ModelId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Capabilities a model may provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    Chat,
    Code,
    Json,
    Tools,
    Vision,
    Local,
    Reasoning,
    Embeddings,
}

impl std::str::FromStr for ModelCapability {
    type Err = ParseModelCapabilityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "chat" => Ok(Self::Chat),
            "code" => Ok(Self::Code),
            "json" => Ok(Self::Json),
            "tools" => Ok(Self::Tools),
            "vision" => Ok(Self::Vision),
            "local" => Ok(Self::Local),
            "reasoning" => Ok(Self::Reasoning),
            "embeddings" => Ok(Self::Embeddings),
            _ => Err(ParseModelCapabilityError),
        }
    }
}

/// Error returned when parsing a model capability fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseModelCapabilityError;

impl std::fmt::Display for ParseModelCapabilityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("unknown model capability")
    }
}

impl std::error::Error for ParseModelCapabilityError {}

/// Routeable model metadata used by router candidate selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteableModel {
    pub id: ModelId,
    pub provider: ProviderId,
    pub upstream_model: String,
    pub context_window: u32,
    pub input_cost_per_million: f64,
    pub output_cost_per_million: f64,
    pub quality: u8,
    pub capabilities: Vec<ModelCapability>,
    pub attributes: BTreeMap<String, String>,
    pub display_badges: Vec<String>,
    pub metadata: ResolvedModelMetadata,
}

impl RouteableModel {
    /// Returns true when the model declares the requested capability.
    #[must_use]
    pub fn has_capability(&self, capability: ModelCapability) -> bool {
        self.capabilities.contains(&capability)
    }
}
