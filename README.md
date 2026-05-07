# brouter

`brouter` is a local, OpenAI-compatible LLM router service. It sits between
clients and model providers, then chooses a provider/model for each request based
on prompt intent, required capabilities, session context, cost, latency, and
routing policy.

The initial implementation is Rust-first and follows the workspace/package
patterns used by the sibling projects in this checkout.

## Goals

- Provide a local OpenAI-compatible API surface.
- Route `auto` model requests to configured real models.
- Keep routing decisions explainable and deterministic by default.
- Support local providers such as Ollama and OpenAI-compatible servers.
- Leave room for an optional local LLM classifier without making it mandatory.

## Current endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/brouter/route/explain`
- `GET /v1/brouter/usage`

`/v1/chat/completions` supports non-streaming and streaming OpenAI-compatible
upstreams, fallback attempts for retryable failures, Anthropic non-streaming
conversion, configurable scoring/routing rules, and optional SQLite telemetry via
`switchy_database`.

## Development

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

## Example

```sh
cargo run -p brouter_cli -- serve --config brouter.toml
```

Then point an OpenAI-compatible client at:

```text
http://127.0.0.1:8080/v1
```
