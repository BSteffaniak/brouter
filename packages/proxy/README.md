# brouter-proxy

Lightweight forwarding proxy for LLM API requests with structured observability.

## Features

- **Request Forwarding**: Forwards `/v1/*` requests to any backend
- **Trace IDs**: Every request gets a UUID for correlation
- **Structured Logging**: Full trace logging with request/response details
- **Environment-Based Config**: No config files needed

## Quick Start

```bash
# Build/start brouter and start the proxy in one terminal.
cargo run --package brouter_proxy -- up

# Configure pi/opencode to point to port 8581.
# Run pi or opencode and see backend/proxy traces in the same terminal.
```

For an installed binary, run:

```bash
brouter-proxy up
```

## CLI

| Command                 | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `brouter-proxy serve`   | Starts only the forwarding proxy                 |
| `brouter-proxy up`      | Builds/starts brouter and the proxy together     |
| `brouter-proxy up --no-build` | Uses `cargo run` for the backend instead of a prebuilt binary |

Useful options:

```bash
brouter-proxy up --config ./brouter.example.toml --backend-port 8582 --proxy-port 8581
brouter-proxy serve --backend http://127.0.0.1:8582 --port 8581
```

| Variable        | Default                 | Description               |
| --------------- | ----------------------- | ------------------------- |
| `PROXY_PORT`    | `8581`                  | Port to listen on         |
| `PROXY_BACKEND` | `http://127.0.0.1:8582` | Backend URL to forward to |

## Logging

- `RUST_LOG=info` - Basic request/response summary
- `RUST_LOG=debug` - Headers and status codes
- `RUST_LOG=trace` - Full request/response bodies (very verbose)

All logs include:

- `request_id` - UUID for correlating requests
- `method`, `path`, `backend_url`
- `content_type`, `content_length`
- `status`, `response_size`

## Configuration

## pi/opencode Integration

1. Start both services: `RUST_LOG=trace cargo run --package brouter_proxy -- up`
2. Update provider configs to use port `8581`
3. Run pi/opencode and observe multiplexed backend/proxy logs
