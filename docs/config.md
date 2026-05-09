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
- `default_profile` optional profile name used by `brouter/auto`
- `debug_headers` boolean
- `provider_failure_threshold` integer
- `provider_cooldown_ms` integer
- `max_estimated_cost` optional float; global per-request budget
- `max_session_estimated_cost` optional float; per-session budget using `x-brouter-session` or `metadata.session_id`
- `aliases` map of alias model IDs to configured model IDs
- `groups` map of group names to model ID arrays; clients can request `group:<name>`

Profiles can be selected with `model = "profile:<name>"`, `x-brouter-profile`, or request metadata `brouter_profile`. Profile allow rules are hard allowlists. Hard deny rules exclude candidates; soft deny rules apply a scoring penalty.

## `[router.context]`

- `safety_margin_ratio` float, default `0.15`; added to the estimated input plus output context requirement
- `preserve_session_context_floor` boolean, default `true`; keeps later requests in a session from falling back below the highest context requirement seen in that server process
- `allow_context_downgrade` boolean, default `false`; when true, session high-water context does not block smaller models if the current request fits

## `[router.profiles.<name>]`

- `objective` optional routing objective override
- `context` optional profile-specific context policy with the same fields as `[router.context]`
- `allow` array of candidate selectors
- `deny` array of candidate deny rules

Candidate selectors support:

- `models` array of model IDs
- `providers` array of provider IDs
- `capabilities` array
- `attributes` map of required model attributes

Deny rules use the same selector fields plus:

- `reason` string shown in route explanations
- `hard` boolean, default `true`; `false` keeps the candidate but penalizes its score
- `penalty` optional float for soft deny rules

Example:

```toml
[router]
default_profile = "conserve_openai"

[[router.profiles.conserve_openai.deny]]
attributes = { latency_class = "priority" }
reason = "priority lane disabled while quota is low"
hard = true

[[router.profiles.conserve_openai.deny]]
providers = ["openai"]
reason = "prefer free/local models before OpenAI"
hard = false
penalty = 25.0
```

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
- `policy_penalty`

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
- `context_window` optional integer; user-provided context metadata. If omitted, brouter tries user metadata overrides and then the built-in fallback catalog.
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

## `[models.<id>.metadata_overrides]`

Use metadata overrides when the user intentionally knows more than provider APIs or brouter's fallback catalog. `mode` controls precedence:

- `force`: override all other metadata sources.
- `fallback`: use only when provider/cache/user model fields do not provide a value.
- `validate`: reserved for conflict reporting; currently does not override.

Fields:

- `reason` optional explanation, strongly recommended for `force`
- `source_url` optional source URL, strongly recommended for `force`
- `verified_at_ms` optional Unix timestamp in milliseconds
- `context_window` optional integer
- `max_output_tokens` optional integer
- `input_cost_per_million` optional float
- `output_cost_per_million` optional float
- `capabilities` optional array

Example:

```toml
[models.gpt41.metadata_overrides]
mode = "force"
reason = "Provider API has not reflected the latest documented context window yet"
source_url = "https://platform.openai.com/docs/models"
verified_at_ms = 1762560000000
context_window = 1047576
max_output_tokens = 32768
capabilities = ["chat", "code", "json", "tools", "reasoning"]
```

## Built-in fallback catalog

When provider APIs do not expose context/pricing/capabilities and the user has not provided metadata, brouter can use an isolated curated fallback catalog in `packages/catalog/data`. This catalog is a last-resort source; route explanations include field provenance so users can see when fallback data was used.
