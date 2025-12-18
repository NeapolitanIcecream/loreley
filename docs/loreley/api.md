## UI API (FastAPI)

Loreley ships an optional **read-only** HTTP API used by the Streamlit dashboard.
The implementation lives in `loreley/api` and is intentionally scoped to observability:
it does not enqueue jobs, stop workers, or mutate the database.

## Install

The UI stack dependencies live under the `ui` extra in `pyproject.toml`.

```bash
uv sync --extra ui
```

## Run

Start the API via the CLI wrapper:

```bash
uv run python script/run_api.py
```

See also: [Running the UI API](../script/run_api.md)

## Configuration

The UI API relies on the standard Loreley settings (`loreley.config.Settings`), especially
database and logs configuration.

Common variables:

- `DATABASE_URL`
- `LOGS_BASE_DIR` (optional; logs are read from `<LOGS_BASE_DIR>/logs` or `<cwd>/logs`)
- `LOG_LEVEL`

## Versioning and prefix

All routes are served under the versioned prefix: `/api/v1`.

FastAPI also exposes OpenAPI docs by default:

- `/docs` (Swagger UI)
- `/redoc`

## Endpoints (v1)

- `GET /health`
- `GET /repositories`
- `GET /repositories/{repository_id}/experiments`
- `GET /experiments/{experiment_id}`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /commits`
- `GET /commits/{commit_hash}`
- `GET /archive/islands`
- `GET /archive/records`
- `GET /archive/snapshot_meta`
- `GET /graphs/commit_lineage`
- `GET /logs`
- `GET /logs/tail`

## Notes

- **Authentication**: there is no authentication layer. Deploy behind your internal network controls if exposing remotely.
- **Read-only contract**: treat this API as an observability surface, not a control plane.

