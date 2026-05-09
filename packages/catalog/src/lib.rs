#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Curated fallback model catalog for brouter.

use brouter_catalog_models::{CatalogFile, CatalogModelMetadata};
use brouter_config_models::ProviderKind;
use thiserror::Error;

/// Fallback catalog lookup error.
#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("failed to parse embedded fallback catalog: {0}")]
    Parse(#[from] toml::de::Error),
}

/// In-memory fallback model catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct FallbackCatalog {
    models: Vec<CatalogModelMetadata>,
}

impl FallbackCatalog {
    /// Loads the embedded brouter fallback catalog.
    ///
    /// # Errors
    ///
    /// Returns an error when the embedded catalog TOML is invalid.
    pub fn embedded() -> Result<Self, CatalogError> {
        let mut models = Vec::new();
        for contents in [OPENAI_CATALOG, ANTHROPIC_CATALOG] {
            let file = toml::from_str::<CatalogFile>(contents)?;
            models.extend(file.models);
        }
        Ok(Self { models })
    }

    /// Returns a fallback catalog entry for the provider/model pair.
    #[must_use]
    pub fn find(
        &self,
        provider_kind: ProviderKind,
        provider_family: &str,
        upstream_model: &str,
    ) -> Option<&CatalogModelMetadata> {
        let provider_kind = provider_kind_name(provider_kind);
        self.models.iter().find(|model| {
            model.provider_kind == provider_kind
                && model.provider_family == provider_family
                && model.upstream_model == upstream_model
        })
    }

    /// Returns all embedded catalog entries.
    #[must_use]
    pub fn models(&self) -> &[CatalogModelMetadata] {
        &self.models
    }
}

impl Default for FallbackCatalog {
    fn default() -> Self {
        Self::embedded().expect("embedded fallback catalog should parse")
    }
}

/// Returns the canonical catalog name for a provider kind.
#[must_use]
pub const fn provider_kind_name(provider_kind: ProviderKind) -> &'static str {
    match provider_kind {
        ProviderKind::OpenAiCompatible => "open-ai-compatible",
        ProviderKind::Anthropic => "anthropic",
        ProviderKind::OpenaiCodex => "openai-codex",
    }
}

const OPENAI_CATALOG: &str = include_str!("../data/openai.toml");
const ANTHROPIC_CATALOG: &str = include_str!("../data/anthropic.toml");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_finds_openai_model() {
        let catalog = FallbackCatalog::embedded().expect("catalog should parse");
        let model = catalog
            .find(ProviderKind::OpenAiCompatible, "openai", "gpt-4.1")
            .expect("gpt-4.1 should be cataloged");

        assert_eq!(model.context_window, Some(1_047_576));
    }
}
