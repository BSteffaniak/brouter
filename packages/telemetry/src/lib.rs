#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Telemetry storage primitives for brouter.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use brouter_provider_models::ModelId;
use brouter_telemetry_models::UsageEvent;
use switchy_database::{Database, DatabaseError, DatabaseValue, Row};
use thiserror::Error;

/// Telemetry storage error.
#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error("failed to initialize SQLite telemetry database: {0}")]
    InitSqlite(#[from] switchy_database_connection::InitSqliteRusqliteError),
    #[error("telemetry database error: {0}")]
    Database(#[from] DatabaseError),
}

/// Telemetry store used by the local service.
#[derive(Debug, Clone)]
pub struct TelemetryStore {
    backend: TelemetryBackend,
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self::memory()
    }
}

impl TelemetryStore {
    /// Creates an in-memory telemetry store.
    #[must_use]
    pub fn memory() -> Self {
        Self {
            backend: TelemetryBackend::Memory(Arc::new(Mutex::new(Vec::new()))),
        }
    }

    /// Creates a `SQLite` telemetry store using `switchy_database`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `SQLite` database cannot be opened or the
    /// telemetry schema cannot be initialized.
    pub async fn sqlite(path: &Path) -> Result<Self, TelemetryError> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let database = switchy_database_connection::init_sqlite_rusqlite(Some(path))?;
        database.exec_raw(CREATE_USAGE_EVENTS_TABLE).await?;
        for statement in ADD_USAGE_EVENT_COLUMNS {
            if let Err(error) = database.exec_raw(statement).await {
                drop(error);
            }
        }
        Ok(Self {
            backend: TelemetryBackend::Database(Arc::new(database)),
        })
    }

    /// Records a usage event.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing telemetry database cannot persist the
    /// event.
    pub async fn record(&self, event: &UsageEvent) -> Result<(), TelemetryError> {
        match &self.backend {
            TelemetryBackend::Memory(events) => {
                events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(event.clone());
                Ok(())
            }
            TelemetryBackend::Database(database) => {
                database
                    .exec_raw_params(
                        INSERT_USAGE_EVENT,
                        &[
                            DatabaseValue::Int64(
                                i64::try_from(event.timestamp_ms).unwrap_or(i64::MAX),
                            ),
                            optional_string_value(event.session_id.as_deref()),
                            DatabaseValue::String(event.selected_model.to_string()),
                            optional_string_value(event.provider.as_deref()),
                            optional_string_value(event.upstream_model.as_deref()),
                            optional_string_value(event.service_tier.as_deref()),
                            optional_string_value(event.reasoning_effort.as_deref()),
                            json_vec_value(&event.resource_pools),
                            optional_string_value(
                                event
                                    .judge_model
                                    .as_ref()
                                    .map(ToString::to_string)
                                    .as_deref(),
                            ),
                            optional_bool_value(event.judge_overridden),
                            optional_string_value(event.judge_error.as_deref()),
                            optional_string_value(event.judge_rationale.as_deref()),
                            json_vec_value(&event.routing_reasons),
                            optional_bool_value(event.fallback_used),
                            DatabaseValue::Real64(event.estimated_cost),
                            optional_u64_value(event.latency_ms),
                            optional_u16_value(event.status_code),
                            optional_string_value(event.provider_error.as_deref()),
                            optional_u64_value(event.prompt_tokens),
                            optional_u64_value(event.completion_tokens),
                            DatabaseValue::Bool(event.success),
                        ],
                    )
                    .await?;
                Ok(())
            }
        }
    }

    /// Returns all recorded usage events.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing telemetry database cannot be queried.
    pub async fn events(&self) -> Result<Vec<UsageEvent>, TelemetryError> {
        match &self.backend {
            TelemetryBackend::Memory(events) => Ok(events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()),
            TelemetryBackend::Database(database) => Ok(database
                .query_raw(SELECT_USAGE_EVENTS)
                .await?
                .iter()
                .map(row_to_usage_event)
                .collect()),
        }
    }

    /// Returns true when any event has been recorded for the session.
    ///
    /// # Errors
    ///
    /// Returns an error when the backing telemetry database cannot be queried.
    pub async fn has_session(&self, session_id: &str) -> Result<bool, TelemetryError> {
        match &self.backend {
            TelemetryBackend::Memory(events) => Ok(events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
                .any(|event| event.session_id.as_deref() == Some(session_id))),
            TelemetryBackend::Database(database) => {
                let rows = database
                    .query_raw_params(
                        SELECT_SESSION_EXISTS,
                        &[DatabaseValue::String(session_id.to_string())],
                    )
                    .await?;
                Ok(!rows.is_empty())
            }
        }
    }

    /// Returns the backing store kind for status/diagnostics.
    #[must_use]
    pub const fn backend_kind(&self) -> &'static str {
        match &self.backend {
            TelemetryBackend::Memory(_) => "memory",
            TelemetryBackend::Database(_) => "sqlite",
        }
    }
}

#[derive(Debug, Clone)]
enum TelemetryBackend {
    Memory(Arc<Mutex<Vec<UsageEvent>>>),
    Database(Arc<Box<dyn Database>>),
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

const CREATE_USAGE_EVENTS_TABLE: &str = "\
CREATE TABLE IF NOT EXISTS usage_events (\
    id INTEGER PRIMARY KEY AUTOINCREMENT,\
    timestamp_ms INTEGER NOT NULL,\
    session_id TEXT NULL,\
    selected_model TEXT NOT NULL,\
    provider TEXT NULL,\
    upstream_model TEXT NULL,\
    service_tier TEXT NULL,\
    reasoning_effort TEXT NULL,\
    resource_pools TEXT NULL,\
    judge_model TEXT NULL,\
    judge_overridden INTEGER NULL,\
    judge_error TEXT NULL,\
    judge_rationale TEXT NULL,\
    routing_reasons TEXT NULL,\
    fallback_used INTEGER NULL,\
    estimated_cost REAL NOT NULL,\
    latency_ms INTEGER NULL,\
    status_code INTEGER NULL,\
    provider_error TEXT NULL,\
    prompt_tokens INTEGER NULL,\
    completion_tokens INTEGER NULL,\
    success INTEGER NOT NULL\
)";

const ADD_USAGE_EVENT_COLUMNS: &[&str] = &[
    "ALTER TABLE usage_events ADD COLUMN provider TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN upstream_model TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN service_tier TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN reasoning_effort TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN resource_pools TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN judge_model TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN judge_overridden INTEGER NULL",
    "ALTER TABLE usage_events ADD COLUMN judge_error TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN judge_rationale TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN routing_reasons TEXT NULL",
    "ALTER TABLE usage_events ADD COLUMN fallback_used INTEGER NULL",
];

const INSERT_USAGE_EVENT: &str = "\
INSERT INTO usage_events (\
    timestamp_ms, session_id, selected_model, provider, upstream_model, service_tier,\
    reasoning_effort, resource_pools, judge_model, judge_overridden, judge_error,\
    judge_rationale, routing_reasons, fallback_used, estimated_cost, latency_ms,\
    status_code, provider_error, prompt_tokens, completion_tokens, success\
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

const SELECT_USAGE_EVENTS: &str = "\
SELECT timestamp_ms, session_id, selected_model, provider, upstream_model, service_tier,\
    reasoning_effort, resource_pools, judge_model, judge_overridden, judge_error,\
    judge_rationale, routing_reasons, fallback_used, estimated_cost, latency_ms,\
    status_code, provider_error, prompt_tokens, completion_tokens, success \
FROM usage_events ORDER BY id ASC";

const SELECT_SESSION_EXISTS: &str = "\
SELECT 1 FROM usage_events WHERE session_id = ? LIMIT 1";

fn optional_string_value(value: Option<&str>) -> DatabaseValue {
    DatabaseValue::StringOpt(value.map(ToOwned::to_owned))
}

fn optional_u64_value(value: Option<u64>) -> DatabaseValue {
    DatabaseValue::Int64Opt(value.map(|value| i64::try_from(value).unwrap_or(i64::MAX)))
}

fn optional_u16_value(value: Option<u16>) -> DatabaseValue {
    DatabaseValue::Int64Opt(value.map(i64::from))
}

fn optional_bool_value(value: Option<bool>) -> DatabaseValue {
    DatabaseValue::Int64Opt(value.map(i64::from))
}

fn json_vec_value(values: &[String]) -> DatabaseValue {
    if values.is_empty() {
        DatabaseValue::StringOpt(None)
    } else {
        DatabaseValue::String(serde_json::to_string(values).unwrap_or_else(|_| "[]".to_string()))
    }
}

fn row_to_usage_event(row: &Row) -> UsageEvent {
    UsageEvent {
        timestamp_ms: get_u64(row, "timestamp_ms"),
        session_id: optional_string_column(row, "session_id"),
        selected_model: ModelId::new(get_string(row, "selected_model")),
        provider: optional_string_column(row, "provider"),
        upstream_model: optional_string_column(row, "upstream_model"),
        service_tier: optional_string_column(row, "service_tier"),
        reasoning_effort: optional_string_column(row, "reasoning_effort"),
        resource_pools: vec_column(row, "resource_pools"),
        judge_model: optional_string_column(row, "judge_model").map(ModelId::new),
        judge_overridden: optional_bool_column(row, "judge_overridden"),
        judge_error: optional_string_column(row, "judge_error"),
        judge_rationale: optional_string_column(row, "judge_rationale"),
        routing_reasons: vec_column(row, "routing_reasons"),
        fallback_used: optional_bool_column(row, "fallback_used"),
        estimated_cost: row
            .get("estimated_cost")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0),
        latency_ms: optional_u64_column(row, "latency_ms"),
        status_code: row
            .get("status_code")
            .and_then(|value| value.as_i64())
            .and_then(|value| u16::try_from(value).ok()),
        provider_error: optional_string_column(row, "provider_error"),
        prompt_tokens: optional_u64_column(row, "prompt_tokens"),
        completion_tokens: optional_u64_column(row, "completion_tokens"),
        success: row
            .get("success")
            .is_some_and(|value| value.as_bool().unwrap_or_else(|| value.as_i64() == Some(1))),
    }
}

fn optional_string_column(row: &Row, column: &str) -> Option<String> {
    row.get(column)
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
}

fn optional_bool_column(row: &Row, column: &str) -> Option<bool> {
    row.get(column).and_then(|value| {
        value
            .as_bool()
            .or_else(|| value.as_i64().map(|value| value != 0))
    })
}

fn vec_column(row: &Row, column: &str) -> Vec<String> {
    row.get(column)
        .and_then(|value| {
            value
                .as_str()
                .and_then(|value| serde_json::from_str(value).ok())
        })
        .unwrap_or_default()
}

fn optional_u64_column(row: &Row, column: &str) -> Option<u64> {
    row.get(column)
        .and_then(|value| value.as_i64())
        .and_then(|value| u64::try_from(value).ok())
}

fn get_string(row: &Row, column: &str) -> String {
    row.get(column)
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_default()
}

fn get_u64(row: &Row, column: &str) -> u64 {
    row.get(column)
        .and_then(|value| value.as_i64())
        .and_then(|value| u64::try_from(value).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sqlite_store_persists_usage_events() {
        let path = std::env::temp_dir().join(format!("brouter-telemetry-{}.db", now_millis()));
        let store = TelemetryStore::sqlite(&path)
            .await
            .expect("sqlite store should initialize");
        let event = UsageEvent {
            timestamp_ms: now_millis(),
            session_id: Some("session-a".to_string()),
            selected_model: ModelId::new("model-a"),
            provider: Some("provider-a".to_string()),
            upstream_model: Some("upstream-a".to_string()),
            service_tier: Some("priority".to_string()),
            reasoning_effort: Some("high".to_string()),
            resource_pools: vec!["priority_pool".to_string()],
            judge_model: Some(ModelId::new("judge-a")),
            judge_overridden: Some(true),
            judge_error: None,
            judge_rationale: Some("best fit".to_string()),
            routing_reasons: vec!["reason-a".to_string()],
            fallback_used: Some(false),
            estimated_cost: 0.25,
            latency_ms: Some(42),
            status_code: Some(200),
            provider_error: None,
            prompt_tokens: Some(3),
            completion_tokens: Some(2),
            success: true,
        };

        store
            .record(&event)
            .await
            .expect("event should be recorded");

        assert!(
            store
                .has_session("session-a")
                .await
                .expect("session lookup should work")
        );
        let events = store.events().await.expect("events should load");
        assert_eq!(events, vec![event]);

        let _ = std::fs::remove_file(path);
    }
}
