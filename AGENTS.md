# AGENTS.md

Guidance for AI coding agents working on this repository.

## Project Structure

Rust workspace with crates under `packages/`. Domain data shared across crates
belongs in sibling `*-models` crates. Model crates contain only serde-friendly
wire/config types, type aliases, conversions, and small parsing helpers. Runtime
logic belongs in the matching implementation crate.

## Required Patterns

- Use Rust edition 2024.
- Use workspace dependencies in package `Cargo.toml` files.
- Add new external dependencies to the workspace root with `default-features = false`.
- Specify full dependency versions including patch versions.
- Prefer `BTreeMap`/`BTreeSet`; do not use `HashMap`/`HashSet`.
- Every crate must expose a `fail-on-warnings = []` feature.
- Every public API returning `Result` must document `# Errors`.
- Add `#[must_use]` to constructors and getters that return non-`Result`/non-`Option` values.
- Do not add `#[must_use]` to functions returning `Result` or `Option`.

## Required Crate Lints

Every `lib.rs` and binary entrypoint should include:

```rust
#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::multiple_crate_versions)]
```

## Verification After Code Changes

Run these commands before finishing any code change:

1. `cargo fmt`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo test`
4. `cargo machete --with-metadata` if available

Report exactly what was run and whether each command passed. If a required
command is skipped, explain why.
