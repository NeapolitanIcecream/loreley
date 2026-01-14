## Running the Streamlit UI

The Streamlit UI is a read-only dashboard that calls the UI API.

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

## Options

- `--api-base-url`: base URL of the UI API (also available via `LORELEY_UI_API_BASE_URL`)
- `--host`: Streamlit bind host (default: `127.0.0.1`)
- `--port`: Streamlit bind port (default: `8501`)
- `--headless`: run without opening a browser


