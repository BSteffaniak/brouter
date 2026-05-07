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
        let database = switchy_database_connection::init_sqlite_rusqlite(Some(path))?;
        database.exec_raw(CREATE_USAGE_EVENTS_TABLE).await?;
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
                            DatabaseValue::Real64(event.estimated_cost),
                            optional_u64_value(event.latency_ms),
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
    estimated_cost REAL NOT NULL,\
    latency_ms INTEGER NULL,\
    success INTEGER NOT NULL\
)";

const INSERT_USAGE_EVENT: &str = "\
INSERT INTO usage_events (\
    timestamp_ms, session_id, selected_model, estimated_cost, latency_ms, success\
) VALUES (?, ?, ?, ?, ?, ?)";

const SELECT_USAGE_EVENTS: &str = "\
SELECT timestamp_ms, session_id, selected_model, estimated_cost, latency_ms, success \
FROM usage_events ORDER BY id ASC";

const SELECT_SESSION_EXISTS: &str = "\
SELECT 1 FROM usage_events WHERE session_id = ? LIMIT 1";

fn optional_string_value(value: Option<&str>) -> DatabaseValue {
    DatabaseValue::StringOpt(value.map(ToOwned::to_owned))
}

fn optional_u64_value(value: Option<u64>) -> DatabaseValue {
    DatabaseValue::Int64Opt(value.map(|value| i64::try_from(value).unwrap_or(i64::MAX)))
}

fn row_to_usage_event(row: &Row) -> UsageEvent {
    UsageEvent {
        timestamp_ms: get_u64(row, "timestamp_ms"),
        session_id: row
            .get("session_id")
            .and_then(|value| value.as_str().map(ToOwned::to_owned)),
        selected_model: ModelId::new(get_string(row, "selected_model")),
        estimated_cost: row
            .get("estimated_cost")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0),
        latency_ms: row
            .get("latency_ms")
            .and_then(|value| value.as_i64())
            .and_then(|value| u64::try_from(value).ok()),
        success: row
            .get("success")
            .is_some_and(|value| value.as_bool().unwrap_or_else(|| value.as_i64() == Some(1))),
    }
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
            estimated_cost: 0.25,
            latency_ms: Some(42),
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
