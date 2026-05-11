# Running the UI API

This command starts the UI API based on FastAPI. Most routes are read-only.
Operator write routes and agent control routes can mutate database state and
require bearer-token auth.

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

## Auth Tokens

Set `LORELEY_API_WRITE_TOKEN` before using UI API POST routes such as job retry,
baseline ensure, repair schedule-one, or repair candidate actions.

Set `LORELEY_AGENT_API_TOKEN` before using `/api/v1/agent/*` routes.

Generate each token yourself:

```bash
python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
```

Use different values for the two tokens. See
[UI API](../loreley/api.md#api-tokens) for request examples.

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
