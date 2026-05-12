# Running the Streamlit UI

The Streamlit UI is mostly a read-only dashboard that calls the UI API. The
operator pages can also send write requests for job retry, baseline ensure, and
repair actions.

## Install UI dependencies

```bash
uv sync --extra ui
```

## Start

When the UI API is not running and `--api-base-url` points to a local HTTP URL
(`http://127.0.0.1:<port>`, `http://localhost:<port>`, or `http://[::1]:<port>`),
starting the UI will automatically start the UI API in a subprocess.

You can still start the API manually:

```bash
uv run loreley api
```

Start Streamlit:

```bash
uv run loreley ui --api-base-url http://127.0.0.1:8000
```

## Auth Tokens

Set `LORELEY_API_WRITE_TOKEN` in the Streamlit process when you want operator
write buttons to work. Use the same value in the FastAPI UI API process, because
the UI sends it as `Authorization: Bearer ...` on POST requests.

If the UI auto-starts the local API subprocess, the subprocess inherits the
Streamlit environment. If you start the API manually, set
`LORELEY_API_WRITE_TOKEN` in both processes.

## Options

- `--api-base-url`: base URL of the UI API (also available via `LORELEY_UI_API_BASE_URL`)
- `--host`: Streamlit bind host (default: `127.0.0.1`)
- `--port`: Streamlit bind port (default: `8501`)
- `--headless`: run without opening a browser
- `--no-preflight`: skip preflight validation
- `--preflight-timeout-seconds`: network timeout used for UI/API reachability checks
