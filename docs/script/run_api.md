# Running the UI API

This command starts the **read-only** UI API based on FastAPI.

## Install UI dependencies

```bash
uv sync --extra ui
```

## Start

```bash
uv run loreley api
```

This command requires `DATABASE_URL` to point at a Loreley database. Empty
databases are initialized automatically, and schema-version-5 databases are
migrated automatically, only when `DB_AUTO_MIGRATE=true`. With
`DB_AUTO_MIGRATE=false`, run `uv run loreley db migrate` before starting the
API.

## Options

- `--host`: bind host (default: `127.0.0.1`)
- `--port`: bind port (default: `8000`)
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation
- `--reload`: enable auto-reload (development only)
- `--no-preflight`: skip preflight validation
- `--preflight-timeout-seconds`: network timeout used for DB connectivity checks

## Logs

Logs are written to:

- `logs/{experiment_namespace}/ui_api/ui_api-YYYYMMDD-HHMMSS.log`
