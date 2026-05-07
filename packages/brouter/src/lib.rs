#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Umbrella crate for brouter packages.

#[cfg(feature = "api-models")]
pub use brouter_api_models as api_models;
#[cfg(feature = "config")]
pub use brouter_config as config;
#[cfg(feature = "config-models")]
pub use brouter_config_models as config_models;
#[cfg(feature = "provider")]
pub use brouter_provider as provider;
#[cfg(feature = "provider-models")]
pub use brouter_provider_models as provider_models;
#[cfg(feature = "router")]
pub use brouter_router as router;
#[cfg(feature = "router-models")]
pub use brouter_router_models as router_models;
#[cfg(feature = "server")]
pub use brouter_server as server;
#[cfg(feature = "telemetry")]
pub use brouter_telemetry as telemetry;
#[cfg(feature = "telemetry-models")]
pub use brouter_telemetry_models as telemetry_models;
