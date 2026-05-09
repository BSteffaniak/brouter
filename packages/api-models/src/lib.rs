#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! OpenAI-compatible API models used at the brouter HTTP boundary.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Canonical brouter reasoning effort scale.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable provider-level extended thinking when possible.
    #[serde(alias = "off")]
    None,
    /// Minimal provider-level thinking.
    Minimal,
    /// Low provider-level thinking.
    Low,
    /// Medium provider-level thinking.
    Medium,
    /// High provider-level thinking.
    High,
    /// Maximum provider-level thinking.
    #[serde(alias = "xhigh")]
    Max,
}

impl ReasoningEffort {
    /// Returns true when provider-level extended thinking should be disabled.
    #[must_use]
    pub const fn is_none(self) -> bool {
        matches!(self, Self::None)
    }
}

/// Chat completion request compatible with the `OpenAI` chat completions API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ChatCompletionRequest {
    /// Returns true when the request asks for a streaming response.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI-compatible chat message content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<Value>),
    Null,
}

impl MessageContent {
    /// Converts message content into plain text for prompt classification.
    #[must_use]
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(extract_text_part)
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Null => String::new(),
        }
    }
}

fn extract_text_part(value: &Value) -> Option<String> {
    let object = value.as_object()?;
    match object.get("type").and_then(Value::as_str) {
        Some("text") => object
            .get("text")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        _ => None,
    }
}

/// OpenAI-compatible embeddings request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Response returned by `/v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

impl ModelListResponse {
    /// Creates a model list response.
    #[must_use]
    pub const fn new(data: Vec<ModelObject>) -> Self {
        Self {
            object: String::new(),
            data,
        }
    }

    /// Creates a model list response with the OpenAI-compatible object name.
    #[must_use]
    pub fn model_list(data: Vec<ModelObject>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

/// OpenAI-compatible model descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

impl ModelObject {
    /// Creates a model object with OpenAI-compatible defaults.
    #[must_use]
    pub fn new(id: impl Into<String>, owned_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: 0,
            owned_by: owned_by.into(),
        }
    }
}

/// OpenAI-compatible error response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorResponse {
    pub error: ErrorObject,
}

impl ErrorResponse {
    /// Creates an error response.
    #[must_use]
    pub fn new(message: impl Into<String>, error_type: impl Into<String>, code: u16) -> Self {
        Self {
            error: ErrorObject {
                message: message.into(),
                error_type: error_type.into(),
                code,
            },
        }
    }
}

/// OpenAI-compatible error object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorObject {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: u16,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn reasoning_effort_accepts_canonical_aliases() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "brouter/auto",
            "messages": [],
            "reasoning_effort": "off"
        }))
        .expect("request should deserialize");
        assert_eq!(request.reasoning_effort, Some(ReasoningEffort::None));

        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "brouter/auto",
            "messages": [],
            "reasoning_effort": "xhigh"
        }))
        .expect("request should deserialize");
        assert_eq!(request.reasoning_effort, Some(ReasoningEffort::Max));
    }

    #[test]
    fn preserves_unknown_chat_request_fields() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "brouter/auto",
            "messages": [],
            "parallel_tool_calls": false
        }))
        .expect("request should deserialize");

        assert_eq!(
            request.extra.get("parallel_tool_calls"),
            Some(&json!(false))
        );
        let serialized = serde_json::to_value(request).expect("request should serialize");
        assert_eq!(serialized["parallel_tool_calls"], json!(false));
    }
}
