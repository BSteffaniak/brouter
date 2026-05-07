#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    match brouter_cli::run_cli().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("brouter error: {error:#}");
            ExitCode::from(1)
        }
    }
}
