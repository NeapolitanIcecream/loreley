## Running the UI API

This command starts the **read-only** UI API based on FastAPI.

## Install UI dependencies

```bash
uv sync --extra ui
```

## Start

```bash
uv run loreley api
```

## Options

- `--host`: bind host (default: `127.0.0.1`)
- `--port`: bind port (default: `8000`)
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation
- `--reload`: enable auto-reload (development only)

## Logs

Logs are written to:

- `logs/ui_api/ui_api-YYYYMMDD-HHMMSS.log`


