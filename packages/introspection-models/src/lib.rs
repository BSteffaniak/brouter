#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Generic live provider catalog/account introspection models for brouter.

use std::collections::BTreeMap;

use brouter_catalog_models::MetadataProvenance;
use brouter_provider_models::{ModelCapability, ModelId, ProviderId};
use serde::{Deserialize, Serialize};

/// Generic introspection request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntrospectionRequest {
    pub include_catalog: bool,
    pub include_account: bool,
    pub include_limits: bool,
}

impl Default for IntrospectionRequest {
    fn default() -> Self {
        Self {
            include_catalog: true,
            include_account: true,
            include_limits: true,
        }
    }
}

/// Generic provider introspection snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntrospectionSnapshot {
    pub provider: ProviderId,
    pub fetched_at_ms: u64,
    pub source: SnapshotSource,
    pub catalog: Option<ModelCatalogSnapshot>,
    pub account: Option<AccountSnapshot>,
    pub warnings: Vec<IntrospectionWarning>,
}

/// Snapshot source metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotSource {
    pub kind: SnapshotSourceKind,
    pub endpoint: Option<String>,
    pub label: Option<String>,
}

impl SnapshotSource {
    /// Creates a provider API source.
    #[must_use]
    pub fn provider_api(endpoint: impl Into<String>) -> Self {
        Self {
            kind: SnapshotSourceKind::ProviderApi,
            endpoint: Some(endpoint.into()),
            label: None,
        }
    }
}

/// Source kind for a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotSourceKind {
    ProviderApi,
    Cache,
    Runtime,
    Unknown,
}

/// Non-fatal introspection warning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntrospectionWarning {
    pub code: String,
    pub message: String,
}

impl IntrospectionWarning {
    /// Creates an introspection warning.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Generic model catalog snapshot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogSnapshot {
    pub models: BTreeMap<String, CatalogModel>,
}

/// Generic catalog model entry.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CatalogModel {
    pub upstream_model: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub fields: ModelMetadataFields,
    #[serde(default)]
    pub raw_capabilities: BTreeMap<String, String>,
    #[serde(default)]
    pub raw_parameters: BTreeMap<String, serde_json::Value>,
}

/// Generic model metadata fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadataFields {
    pub context_window: MetadataField<u32>,
    pub max_output_tokens: MetadataField<u32>,
    pub input_cost_per_million: MetadataField<f64>,
    pub output_cost_per_million: MetadataField<f64>,
    pub capabilities: MetadataField<Vec<ModelCapability>>,
    pub supported_parameters: MetadataField<Vec<String>>,
}

/// Value plus field provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataField<T> {
    pub value: Option<T>,
    pub provenance: MetadataProvenance,
}

impl<T> MetadataField<T> {
    /// Creates an empty metadata field.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            value: None,
            provenance: MetadataProvenance::default(),
        }
    }

    /// Creates a metadata field with a value and provenance.
    #[must_use]
    pub const fn new(value: T, provenance: MetadataProvenance) -> Self {
        Self {
            value: Some(value),
            provenance,
        }
    }
}

impl<T> Default for MetadataField<T> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Generic account state snapshot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub account_id: Option<String>,
    pub status: AccountStatus,
    #[serde(default)]
    pub pools: Vec<ResourcePool>,
}

/// Account availability status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountStatus {
    Available,
    Exhausted,
    Disabled,
    #[default]
    Unknown,
}

/// Generic resource pool used for quota, credits, allowances, and rate limits.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourcePool {
    pub id: String,
    pub scope: ResourceScope,
    pub kind: ResourceKind,
    pub unit: ResourceUnit,
    pub remaining: Option<f64>,
    pub total: Option<f64>,
    pub used: Option<f64>,
    pub refill_at_ms: Option<u64>,
    pub reset_at_ms: Option<u64>,
    pub expires_at_ms: Option<u64>,
    pub applies_to: ResourceSelector,
    pub provenance: MetadataProvenance,
}

impl ResourcePool {
    /// Returns remaining / total when both values are known and total is positive.
    #[must_use]
    pub fn remaining_ratio(&self) -> Option<f64> {
        let remaining = self.remaining?;
        let total = self.total?;
        (total > 0.0).then_some((remaining / total).clamp(0.0, 1.0))
    }
}

/// Resource scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceScope {
    Provider,
    Account,
    Model,
    Attribute,
    #[default]
    Unknown,
}

/// Resource kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    MonetaryCredit,
    SubscriptionAllowance,
    TokenBudget,
    RequestBudget,
    RateLimit,
    PriorityAllowance,
    #[default]
    Unknown,
}

/// Resource unit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceUnit {
    Usd,
    Tokens,
    Requests,
    RequestsPerMinute,
    TokensPerMinute,
    Percent,
    #[default]
    Unknown,
}

/// Generic selector describing which candidates a resource applies to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSelector {
    #[serde(default)]
    pub providers: Vec<ProviderId>,
    #[serde(default)]
    pub upstream_models: Vec<String>,
    #[serde(default)]
    pub configured_models: Vec<ModelId>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
    #[serde(default)]
    pub capabilities: Vec<ModelCapability>,
}

/// Generic dynamic policy effect produced from introspected state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DynamicPolicyEffect {
    Exclude {
        selector: ResourceSelector,
        reason: String,
    },
    Penalize {
        selector: ResourceSelector,
        penalty: f64,
        reason: String,
    },
    DisableAttribute {
        selector: ResourceSelector,
        key: String,
        value: String,
        reason: String,
    },
}
