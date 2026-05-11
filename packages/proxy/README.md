# brouter-proxy

Lightweight forwarding proxy for LLM API requests with structured observability.

## Features

- **Request Forwarding**: Forwards `/v1/*` requests to any backend
- **Trace IDs**: Every request gets a UUID for correlation
- **Structured Logging**: Full trace logging with request/response details
- **Environment-Based Config**: No config files needed

## Quick Start

```bash
# Terminal 1: Start the proxy
cargo run --package brouter_proxy

# Terminal 2: Configure pi/opencode to point to port 8081 instead of 8080
# Run pi or opencode and see traces
```

## Configuration

| Variable        | Default                 | Description               |
| --------------- | ----------------------- | ------------------------- |
| `PROXY_PORT`    | `8081`                  | Port to listen on         |
| `PROXY_BACKEND` | `http://127.0.0.1:8080` | Backend URL to forward to |

## Logging

- `RUST_LOG=info` - Basic request/response summary
- `RUST_LOG=debug` - Headers and status codes
- `RUST_LOG=trace` - Full request/response bodies (very verbose)

All logs include:

- `request_id` - UUID for correlating requests
- `method`, `path`, `backend_url`
- `content_type`, `content_length`
- `status`, `response_size`

## pi/opencode Integration

1. Start the proxy: `RUST_LOG=trace cargo run --package brouter_proxy`
2. Update provider configs to use port `8081` instead of `8080`
3. Run pi/opencode and observe trace logs
