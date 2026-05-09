#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Generic cache and dynamic policy helpers for live provider introspection.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use brouter_introspection_models::{
    AccountStatus, DynamicPolicyEffect, IntrospectionSnapshot, ResourcePool, ResourceSelector,
};
use brouter_provider_models::ProviderId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Introspection cache error.
#[derive(Debug, Error)]
pub enum IntrospectionError {
    #[error("failed to read introspection cache {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to write introspection cache {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse introspection cache {path}: {source}")]
    Parse {
        path: String,
        source: serde_json::Error,
    },
    #[error("failed to encode introspection cache: {0}")]
    Encode(#[from] serde_json::Error),
}

/// File cache of provider introspection snapshots.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IntrospectionCache {
    #[serde(default)]
    pub providers: BTreeMap<ProviderId, IntrospectionSnapshot>,
}

impl IntrospectionCache {
    /// Loads a cache file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed.
    pub fn load(path: &Path) -> Result<Self, IntrospectionError> {
        let contents =
            std::fs::read_to_string(path).map_err(|source| IntrospectionError::Read {
                path: path.display().to_string(),
                source,
            })?;
        serde_json::from_str(&contents).map_err(|source| IntrospectionError::Parse {
            path: path.display().to_string(),
            source,
        })
    }

    /// Saves a cache file.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache cannot be serialized or written.
    pub fn save(&self, path: &Path) -> Result<(), IntrospectionError> {
        let contents = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| IntrospectionError::Write {
                path: parent.display().to_string(),
                source,
            })?;
        }
        std::fs::write(path, contents).map_err(|source| IntrospectionError::Write {
            path: path.display().to_string(),
            source,
        })
    }

    /// Returns a fresh snapshot for a provider.
    #[must_use]
    pub fn fresh_snapshot(
        &self,
        provider: &ProviderId,
        max_age_ms: u64,
    ) -> Option<&IntrospectionSnapshot> {
        let snapshot = self.providers.get(provider)?;
        let age = now_millis().saturating_sub(snapshot.fetched_at_ms);
        (age <= max_age_ms).then_some(snapshot)
    }
}

/// Generic dynamic policy thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DynamicPolicyConfig {
    pub low_remaining_ratio: f64,
    pub critical_remaining_ratio: f64,
    pub low_remaining_penalty: f64,
    pub exclude_when_exhausted: bool,
}

impl Default for DynamicPolicyConfig {
    fn default() -> Self {
        Self {
            low_remaining_ratio: 0.25,
            critical_remaining_ratio: 0.05,
            low_remaining_penalty: 30.0,
            exclude_when_exhausted: true,
        }
    }
}

/// Builds generic dynamic policy effects from account snapshots.
#[must_use]
pub fn dynamic_policy_effects(
    snapshots: impl IntoIterator<Item = IntrospectionSnapshot>,
    config: DynamicPolicyConfig,
    disable_attributes_when_low: &BTreeMap<String, String>,
) -> Vec<DynamicPolicyEffect> {
    let mut effects = Vec::new();
    for snapshot in snapshots {
        let Some(account) = snapshot.account else {
            continue;
        };
        if config.exclude_when_exhausted && account.status == AccountStatus::Exhausted {
            effects.push(DynamicPolicyEffect::Exclude {
                selector: ResourceSelector {
                    providers: vec![snapshot.provider.clone()],
                    ..ResourceSelector::default()
                },
                reason: format!("provider {} account is exhausted", snapshot.provider),
            });
        }
        for pool in account.pools {
            append_pool_effects(&mut effects, &pool, config, disable_attributes_when_low);
        }
    }
    effects
}

fn append_pool_effects(
    effects: &mut Vec<DynamicPolicyEffect>,
    pool: &ResourcePool,
    config: DynamicPolicyConfig,
    disable_attributes_when_low: &BTreeMap<String, String>,
) {
    let Some(ratio) = pool.remaining_ratio() else {
        return;
    };
    if ratio <= config.critical_remaining_ratio {
        effects.push(DynamicPolicyEffect::Exclude {
            selector: pool.applies_to.clone(),
            reason: format!(
                "resource pool {} remaining ratio {ratio:.3} is critical",
                pool.id
            ),
        });
    } else if ratio <= config.low_remaining_ratio {
        effects.push(DynamicPolicyEffect::Penalize {
            selector: pool.applies_to.clone(),
            penalty: config.low_remaining_penalty,
            reason: format!(
                "resource pool {} remaining ratio {ratio:.3} is low",
                pool.id
            ),
        });
        for (key, value) in disable_attributes_when_low {
            effects.push(DynamicPolicyEffect::DisableAttribute {
                selector: pool.applies_to.clone(),
                key: key.clone(),
                value: value.clone(),
                reason: format!(
                    "resource pool {} remaining ratio {ratio:.3} is low",
                    pool.id
                ),
            });
        }
    }
}

/// Returns the current Unix timestamp in milliseconds.
#[must_use]
pub fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}
