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

A normal Rust toolchain is enough for development. Nix is supported, but not
required.

```sh
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

## Build and run

Copy the example config, edit providers/models for your environment, then start
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
cargo run -p brouter_cli -- check-config --config brouter.toml
cargo run -p brouter_cli -- check-config --strict --config brouter.toml
```

`check-config` reports non-fatal warnings for suspicious settings, including
unknown capabilities, missing provider environment variables, unknown rule
intents/objectives, OpenAI-compatible providers without a `base_url`, and
`local_only` rules without a local model. `--strict` turns those warnings into a
non-zero exit.

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
```
