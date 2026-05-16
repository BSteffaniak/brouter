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
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/brouter/route/explain`
- `GET /v1/brouter/usage`
- `GET /v1/brouter/usage/summary`
- `GET /v1/brouter/introspection`
- `POST /v1/brouter/introspection/refresh`
- `GET /v1/brouter/status`

`/v1/chat/completions` supports non-streaming and streaming OpenAI-compatible
upstreams, provider timeouts, fallback attempts for retryable failures,
provider cooldowns after repeated failures, Anthropic non-streaming and streaming
conversion, OpenAI-compatible embeddings forwarding, configurable scoring/routing rules, named routing profiles with allow/deny policy, context-window safety for session-aware model switching, cached live provider/account introspection with periodic and manual refresh, quota-aware dynamic policy, virtual service-tier/reasoning route variants for supported providers, a default opt-out LLM judge for close routing decisions, generic model/provider route attributes, and SQLite telemetry via
`switchy_database`. Successful chat responses include brouter headers such as
`x-brouter-selected-model`, `x-brouter-provider`, `x-brouter-service-tier`,
`x-brouter-reasoning-effort`, `x-brouter-resource-pools`,
`x-brouter-attributes`, and `x-brouter-display-badges`. `/v1/brouter/usage`
supports `session_id`, `model`, `success`, `since_ms`, and `until_ms` query
filters. `/v1/brouter/introspection` shows live/cache provider resource data and
`/v1/brouter/status` summarizes active defaults. `/metrics` exposes basic
Prometheus text metrics from telemetry events.

## Development

A normal Rust toolchain is enough for development. Nix is supported, but not
required.

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

## Build and run

Start with no config if you have a supported provider environment variable such as `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OLLAMA_HOST`:

```sh
cargo run -p brouter_cli -- serve
```

Or copy the example config, edit providers/models for your environment, then start
brouter:

```sh
cp brouter.example.toml brouter.toml
cargo run -p brouter_cli -- serve --config brouter.toml
```

For a release binary:

```sh
cargo build --release -p brouter_cli
./target/release/brouter serve --config brouter.toml
```

Then point an OpenAI-compatible client at:

```text
http://127.0.0.1:8080/v1
```

## Config validation

```sh
cargo run -p brouter_cli -- print-example-config > brouter.toml
cargo run -p brouter_cli -- check-config --config brouter.toml
cargo run -p brouter_cli -- check-config --strict --json --config brouter.toml
cargo run -p brouter_cli -- doctor --config brouter.toml
```

`check-config` reports non-fatal warnings for suspicious settings, including
unknown capabilities, missing provider environment variables, unknown rule
intents/objectives, OpenAI-compatible providers without a `base_url`, and
`local_only` rules without a local model. `--strict` turns those warnings into a
non-zero exit. `doctor` also checks provider `/models` reachability. The full
configuration schema is documented in `docs/config.md`.

## Common setups

Local Ollama-only:

```toml
[providers.ollama]
kind = "open-ai-compatible"
base_url = "http://localhost:11434/v1"
timeout_ms = 60000

[models.local]
provider = "ollama"
model = "qwen2.5-coder:7b"
context_window = 32768
capabilities = ["chat", "code", "local"]
```

OpenAI plus local fallback/private routing:

```toml
[router]
default_objective = "balanced"
provider_failure_threshold = 3
provider_cooldown_ms = 30000
# Optional global per-request/session budgets using router cost estimates.
# max_estimated_cost = 0.05
# max_session_estimated_cost = 1.00

[router.aliases]
fast = "fast_local"
strong = "strong_cloud"

[router.groups]
cloud = ["cheap_cloud", "strong_cloud"]

[[router.rules]]
name = "private-local"
when_contains = ["secret", "private key", "credentials"]
objective = "local_only"
require_capabilities = ["local"]

[providers.openai]
kind = "open-ai-compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
timeout_ms = 60000
# Optional provider-level per-request budget.
# max_estimated_cost = 0.05
```

Embeddings model:

```toml
[models.embedding_cloud]
provider = "openai"
model = "text-embedding-3-small"
context_window = 8192
input_cost_per_million = 0.02
capabilities = ["embeddings"]
```

Optional local server auth:

```toml
[server]
api_key_env = "BROUTER_API_KEY"
```

Authenticated requests must include either `Authorization: Bearer $BROUTER_API_KEY`
or `x-api-key: $BROUTER_API_KEY`.

For structured logs, set:

```sh
BROUTER_LOG_FORMAT=json RUST_LOG=info brouter serve --config brouter.toml
```

## Non-Nix service example

A typical Linux deployment can keep declarative config in `/etc/brouter` and
state in `/var/lib/brouter`:

```ini
[Unit]
Description=brouter local LLM router
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/brouter serve --config /etc/brouter/brouter.toml
EnvironmentFile=-/etc/brouter/brouter.env
Restart=on-failure
StateDirectory=brouter
DynamicUser=yes

[Install]
WantedBy=multi-user.target
```

The same TOML config format is used everywhere. Nix modules only generate that
TOML and systemd wiring for users who prefer declarative Nix deployments.

## Container image

On Linux, the flake can build an OCI/Docker image:

```sh
nix build .#container
```

Run the resulting image with a mounted config directory and optional env file:

```sh
docker load < result
docker run --rm -p 8080:8080 \
  --env-file ./brouter.env \
  -v "$PWD:/config:ro" \
  brouter:latest
```

## Nix usage

Nix is a first-class, fully supported deployment path, but it is not required for
building or running brouter.

The flake exposes:

- `packages.default` / `packages.brouter`
- `apps.default` / `apps.brouter`
- `checks.package` and `checks.example-config`
- `nixosModules.default`
- `homeManagerModules.default`

A NixOS configuration can generate the TOML config declaratively:

```nix
{
  inputs.brouter.url = "github:your-org/brouter";

  outputs = { nixpkgs, brouter, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        brouter.nixosModules.default
        {
          services.brouter = {
            enable = true;
            environmentFile = "/run/secrets/brouter.env";
            settings = {
              server = {
                host = "127.0.0.1";
                port = 8080;
              };
              telemetry.database_path = "/var/lib/brouter/brouter.db";
              providers.ollama = {
                kind = "open-ai-compatible";
                base_url = "http://localhost:11434/v1";
              };
              models.fast_local = {
                provider = "ollama";
                model = "qwen2.5-coder:7b";
                context_window = 32768;
                capabilities = [ "chat" "code" "local" ];
              };
            };
          };
        }
      ];
    };
  };
}
```

Runtime state, such as the telemetry database, should stay outside routing
policy. The NixOS module creates a systemd state directory and expects paths like
`/var/lib/brouter/brouter.db` for persisted telemetry.

For development with Nix, prefix the normal Cargo commands with `nix develop -c`:

```sh
nix develop -c cargo fmt
nix develop -c cargo clippy --all-targets -- -D warnings
nix develop -c cargo test
nix develop -c cargo run -p brouter_cli -- check-config --strict --config brouter.toml
nix develop -c cargo run -p brouter_cli -- doctor --config brouter.toml
```
