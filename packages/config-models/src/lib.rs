#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Serde-compatible configuration models for brouter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Top-level brouter configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BrouterConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelConfig>,
}

/// HTTP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

impl ServerConfig {
    /// Returns the TCP bind address for the configured server.
    #[must_use]
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

const fn default_port() -> u16 {
    8080
}

/// Router behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouterConfig {
    #[serde(default = "default_objective")]
    pub default_objective: String,
    #[serde(default)]
    pub debug_headers: bool,
    #[serde(default)]
    pub classifier: Option<ClassifierConfig>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            default_objective: default_objective(),
            debug_headers: false,
            classifier: None,
        }
    }
}

fn default_objective() -> String {
    "balanced".to_string()
}

/// Optional prompt classifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClassifierConfig {
    pub provider: String,
    pub model: String,
}

/// Provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
}

/// Supported provider kinds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    OpenAiCompatible,
    Anthropic,
}

/// Model configuration used by the router.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub provider: String,
    pub model: String,
    pub context_window: u32,
    #[serde(default)]
    pub input_cost_per_million: f64,
    #[serde(default)]
    pub output_cost_per_million: f64,
    #[serde(default)]
    pub quality: Option<u8>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}
