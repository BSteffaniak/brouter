#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Serde-compatible fallback model catalog metadata models for brouter.

use serde::{Deserialize, Serialize};

/// Source type for model metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataSource {
    UserForcedOverride,
    ProviderApi,
    ProviderCache,
    UserVerifiedFallback,
    BrouterFallbackCatalog,
    UserConfig,
    Unknown,
}

/// User override behavior for configured model metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataOverrideMode {
    Force,
    #[default]
    Fallback,
    Validate,
}

impl MetadataOverrideMode {
    /// Parses an override mode from configuration.
    #[must_use]
    pub fn from_name(value: &str) -> Self {
        match value {
            "force" => Self::Force,
            "validate" => Self::Validate,
            _ => Self::Fallback,
        }
    }
}

/// Provenance for one resolved metadata field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataProvenance {
    pub source: MetadataSource,
    pub source_label: Option<String>,
    pub source_url: Option<String>,
    pub reason: Option<String>,
    pub verified_at_ms: Option<u64>,
}

impl MetadataProvenance {
    /// Creates field provenance for a metadata source.
    #[must_use]
    pub const fn new(source: MetadataSource) -> Self {
        Self {
            source,
            source_label: None,
            source_url: None,
            reason: None,
            verified_at_ms: None,
        }
    }
}

impl Default for MetadataProvenance {
    fn default() -> Self {
        Self::new(MetadataSource::Unknown)
    }
}

/// Resolved metadata and provenance attached to a routeable model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelMetadata {
    pub context_window: MetadataProvenance,
    pub max_output_tokens: Option<u32>,
    pub max_output_tokens_source: MetadataProvenance,
    pub cost: MetadataProvenance,
    pub capabilities: MetadataProvenance,
}

/// Curated fallback metadata for one provider model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModelMetadata {
    pub provider_kind: String,
    pub provider_family: String,
    pub upstream_model: String,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub input_cost_per_million: Option<f64>,
    pub output_cost_per_million: Option<f64>,
    pub capabilities: Vec<String>,
    pub source_label: String,
    pub source_url: Option<String>,
    pub verified_at_ms: Option<u64>,
}

/// Top-level fallback catalog data file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CatalogFile {
    #[serde(default)]
    pub models: Vec<CatalogModelMetadata>,
}
