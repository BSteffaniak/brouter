# brouter-proxy

Lightweight forwarding proxy for LLM API requests with structured observability.

## Features

- **Request Forwarding**: Forwards `/v1/*` requests to any backend
- **Trace IDs**: Every request gets a UUID for correlation
- **Structured Logging**: Full trace logging with request/response details
- **Environment-Based Config**: No config files needed

## Usage

```bash
# Default: listen on 8081, forward to http://127.0.0.1:8080
brouter-proxy

# Custom port and backend
PROXY_PORT=9090 PROXY_BACKEND=http://my-brouter:8080 brouter-proxy

# Just change the backend
PROXY_BACKEND=http://production-brouter:8080 brouter-proxy
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_PORT` | `8081` | Port to listen on |
| `PROXY_BACKEND` | `http://127.0.0.1:8080` | Backend URL to forward to |

## Example Workflow

```bash
# Terminal 1: Start your brouter backend on port 8080
brouter serve

# Terminal 2: Start the proxy on port 8081, forwarding to brouter
RUST_LOG=trace PROXY_PORT=8081 PROXY_BACKEND=http://127.0.0.1:8080 brouter-proxy

# Terminal 3: Configure pi/opencode to use port 8081
# Edit ~/.config/pi/models.json or ~/.config/opencode/providers/brouter.json
# Change baseUrl to http://127.0.0.1:8081/v1

# Or for quick testing with env vars:
BROUTER_PORT=8081 opencode
```

## Logging

The proxy logs at different levels:

- `RUST_LOG=info` - Basic request/response info
- `RUST_LOG=debug` - More details including headers
- `RUST_LOG=trace` - Full request/response bodies (very verbose)

All logs include:
- `request_id` - UUID for correlating requests
- `method`, `path`, `backend_url`
- `content_type`, `content_length`
- `status`, `response_size`

## Docker

```bash
docker run -p 8081:8081 \
  -e PROXY_BACKEND=http://host.docker.internal:8080 \
  brouter-proxy
```
