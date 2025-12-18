## Streamlit UI (loreley.ui)

Loreley ships an optional **read-only** Streamlit dashboard for observability.
It calls the [UI API](api.md) and renders tables, charts, and commit lineage graphs.

The UI stack is intentionally read-only: it does not enqueue jobs, stop workers, or mutate the database.

```mermaid
flowchart LR
  user[User] --> stApp[Streamlit UI]
  stApp --> api["UI API (FastAPI)"]
  api --> db[(PostgreSQL)]
  api --> logsDir[Logs directory]
```

## Install

The UI dependencies live under the `ui` extra in `pyproject.toml`.

```bash
uv sync --extra ui
```

## Run

Start the API first:

```bash
uv run python script/run_api.py
```

Then start Streamlit:

```bash
uv run python script/run_ui.py --api-base-url http://127.0.0.1:8000
```

See also:

- [Running the UI API](../script/run_api.md)
- [Running the Streamlit UI](../script/run_ui.md)

## Configuration

### UI variables

- `LORELEY_UI_API_BASE_URL`: Base URL for the UI API (default: `http://127.0.0.1:8000`).

### API runtime variables

The API relies on standard Loreley settings (database/logs). See:

- [Configuration](config.md)
- [UI API](api.md)

## Pages

The Streamlit UI is multi-page (implemented under `loreley/ui/pages`):

- **Overview**: quick KPIs, fitness trend, island table.
- **Experiments**: experiment list and selected experiment details.
- **Jobs**: job table with filters and a details panel.
- **Commits**: commit table with search; commit details with charts.
- **Archive**: island stats, snapshot metadata, record plots and table.
- **Graphs**: fitness scatter and commit lineage graph.
- **Logs**: browse role logs and tail a file.
- **Settings**: API health and safe settings (`Settings.export_safe()`).

## Notes

- **Caching**: the Streamlit UI caches API GET calls (default: ~60s); use the sidebar **Refresh data** button to clear cache.
- **Security**: there is no authentication layer. Deploy behind your internal network controls if exposing remotely.