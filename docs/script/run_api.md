## Running the UI API

This script starts the **read-only** UI API based on FastAPI.

## Install UI dependencies

```bash
uv sync --extra ui
```

## Start

```bash
uv run loreley api

# legacy wrapper (still supported)
uv run python script/run_api.py
```

## Options

- `--host`: bind host (default: `127.0.0.1`)
- `--port`: bind port (default: `8000`)
- `--log-level`: override `LOG_LEVEL`
- `--reload`: enable auto-reload (development only)

## Logs

Logs are written to:

- `logs/ui_api/ui-api-<timestamp>.log`


