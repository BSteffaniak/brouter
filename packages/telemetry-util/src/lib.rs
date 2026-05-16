#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

//! Utility crate for structured observability logging in brouter.
//!
//! Provides macros and functions for consistent, structured logging across
//! all brouter components.

mod logging;
pub use logging::*;
