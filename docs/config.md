# brouter config schema

`brouter` uses one TOML file. Nix modules generate this same TOML; non-Nix deployments can write it directly.

## `[server]`

- `host` string, default `"127.0.0.1"`
- `port` integer, default `8080`
- `api_key_env` optional string; enables local API key auth from this environment variable
- `max_request_body_bytes` integer, default `1048576`
- `cors_allowed_origins` array of strings; use `["*"]` for permissive local-browser access

## `[router]`

- `default_objective`: `balanced`, `cheapest`, `fastest`, `strongest`, `local_only`
- `debug_headers` boolean
- `provider_failure_threshold` integer
- `provider_cooldown_ms` integer
- `max_estimated_cost` optional float; global per-request budget
- `max_session_estimated_cost` optional float; per-session budget using `x-brouter-session` or `metadata.session_id`
- `aliases` map of alias model IDs to configured model IDs
- `groups` map of group names to model ID arrays; clients can request `group:<name>`

## `[[router.rules]]`

- `name` string
- `when_contains` array of substrings matched against the latest user message
- `intent` optional string; detected from the latest user message, not system/developer context
- `objective` optional routing objective
- `prefer_capabilities` array
- `require_capabilities` array
- `prefer_attributes` map of attribute names to preferred values
- `require_attributes` map of attribute names to required values

## `[router.scoring]`

All fields are optional floats:

- `quality_weight`
- `balanced_cost_weight`
- `cheapest_cost_weight`
- `local_bonus`
- `strongest_quality_weight`
- `first_message_reasoning_bonus`
- `code_bonus`
- `reasoning_bonus`

## `[telemetry]`

- `database_path` optional path; omitted uses in-memory telemetry

## `[providers.<id>]`

- `kind`: `open-ai-compatible`, `anthropic`, or `openai-codex`
- `base_url` optional string; required for OpenAI-compatible providers
- `api_key_env` optional string
- `timeout_ms` optional integer
- `max_estimated_cost` optional float; provider-level per-request budget
- `auth_backend` optional string; `openai-codex` currently supports `sshenv`
- `auth_profile` optional string; sshenv profile containing ChatGPT/Codex tokens
- `auth_vault_path` optional string; sshenv vault path. If omitted, brouter uses `$BROUTER_AUTH_VAULT` or `~/.local/state/brouter/auth/vault`.
- `attribute_mappings` optional nested map from attribute name/value to provider request edits. Each mapping can add top-level `request_fields` or remove top-level `omit_request_fields`.

`openai-codex` expects brouter-owned sshenv keys such as
`BROUTER_OPENAI_CODEX_ACCESS_TOKEN`, `BROUTER_OPENAI_CODEX_REFRESH_TOKEN`,
`BROUTER_OPENAI_CODEX_EXPIRES_AT`, and `BROUTER_OPENAI_CODEX_ACCOUNT_ID`.
Create them with browser OAuth:

```sh
brouter auth openai-codex login --profile openai-max
```

For headless machines, opt into Codex device-code auth:

```sh
brouter auth openai-codex login --profile openai-max --headless
```

## `[models.<id>]`

- `provider` provider ID
- `model` upstream model name
- `context_window` integer
- `input_cost_per_million` float
- `output_cost_per_million` float
- `quality` optional integer 0-255
- `capabilities` array: `chat`, `code`, `json`, `tools`, `vision`, `local`, `reasoning`, `embeddings`
- `attributes` optional map of arbitrary routing dimensions, for example `latency_class = "priority"`
- `display_badges` optional array of compact badges exposed in `x-brouter-display-badges`; when omitted, brouter derives badges for known attributes such as `latency_class`
- `max_estimated_cost` optional float; model-level per-request budget

Example provider-specific request mapping:

```toml
[providers.openai.attribute_mappings.latency_class.priority.request_fields]
service_tier = "priority"

[providers.openai.attribute_mappings.latency_class.standard]
omit_request_fields = ["service_tier"]
```

A model with `attributes.latency_class = "priority"` will add `service_tier = "priority"` to OpenAI-compatible requests. The same routing attribute can also be used by `prefer_attributes` or `require_attributes` without making `service_tier` a brouter-specific concept.
