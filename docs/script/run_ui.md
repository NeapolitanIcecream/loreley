## Running the Streamlit UI

The Streamlit UI is a read-only dashboard that calls the UI API.

## Install UI dependencies

```bash
uv sync --extra ui
```

## Start

Start the API first:

```bash
uv run loreley api

# legacy wrapper (still supported)
uv run python script/run_api.py
```

Then start Streamlit:

```bash
uv run loreley ui --api-base-url http://127.0.0.1:8000

# legacy wrapper (still supported)
uv run python script/run_ui.py --api-base-url http://127.0.0.1:8000
```

## Options

- `--api-base-url`: base URL of the UI API (also available via `LORELEY_UI_API_BASE_URL`)
- `--host`: Streamlit bind host (default: `127.0.0.1`)
- `--port`: Streamlit bind port (default: `8501`)
- `--headless`: run without opening a browser


