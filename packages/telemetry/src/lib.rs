#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Telemetry storage primitives for brouter.

use std::sync::{Arc, Mutex};

use brouter_telemetry_models::UsageEvent;

/// In-memory telemetry store used by the initial local service.
#[derive(Debug, Clone, Default)]
pub struct TelemetryStore {
    events: Arc<Mutex<Vec<UsageEvent>>>,
}

impl TelemetryStore {
    /// Records a usage event.
    pub fn record(&self, event: UsageEvent) {
        let mut events = self
            .events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        events.push(event);
    }

    /// Returns all recorded usage events.
    #[must_use]
    pub fn events(&self) -> Vec<UsageEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Returns true when any event has been recorded for the session.
    #[must_use]
    pub fn has_session(&self, session_id: &str) -> bool {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|event| event.session_id.as_deref() == Some(session_id))
    }
}
