# UI API (FastAPI)

Loreley ships an optional HTTP API used by the Streamlit dashboard. Most routes
are read-only observability routes. The operator console also exposes a small
set of write routes for local run repair: baseline ensure tasks, repair-pool
actions, and job retries.

The operator write API is unauthenticated by design. Run it only on trusted
local or internal networks.

## Install

The UI stack dependencies live under the `ui` extra in `pyproject.toml`.

```bash
uv sync --extra ui
```

## Run

Start the API:

```bash
uv run loreley api
```

See also: [Running the UI API](../script/run_api.md)

## Configuration

The UI API relies on the standard Loreley settings (`loreley.config.Settings`), especially
database and logs configuration. On startup it validates that the database contains an
`InstanceMetadata` marker (schema/version). If the marker is missing, seed it by starting
the scheduler/worker once (or via `uv run loreley reset-db --yes`). If the schema version
is mismatched, reset the schema with `uv run loreley reset-db --yes` (dev).

Common variables:

- `DATABASE_URL`
- `LOGS_BASE_DIR` (optional; logs are read from `<LOGS_BASE_DIR>/logs` or `<cwd>/logs`)
- `LOG_LEVEL`
- `EXPERIMENT_ID` (optional; used to resolve the experiment namespace for log browsing; if unset, the API falls back to the database marker)

## Versioning and prefix

All routes are served under the versioned prefix: `/api/v1`.

FastAPI also exposes OpenAPI docs by default:

- `/docs` (Swagger UI)
- `/redoc`

## Endpoints (v1)

- `GET /health`
- `GET /instance`
- `GET /jobs`
- `GET /jobs/page` (`status`, `job_kind`, `limit`, and `cursor` filters)
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/retry`
- `POST /jobs/retry-failed-stale`
- `GET /jobs/{job_id}/artifacts`
- `GET /jobs/{job_id}/artifacts/{artifact_key}`
- `GET /commits`
- `GET /commits/page`
- `GET /commits/{commit_hash}`
- `GET /archive/islands`
- `GET /archive/records`
- `GET /archive/records/page`
- `GET /archive/snapshot_meta`
- `GET /graphs/commit_lineage`
- `GET /logs`
- `GET /logs/tail`
- `GET /operator/status`
- `POST /operator/tasks/baseline-ensure`
- `GET /operator/tasks`
- `GET /operator/tasks/{task_id}`
- `GET /repair/pool`
- `POST /repair/schedule-one`
- `POST /repair/candidates/{candidate_id}/quarantine`
- `POST /repair/candidates/{candidate_id}/discard`
- `POST /repair/candidates/{candidate_id}/restore`

## Operator Writes

These routes mutate database state:

- `POST /jobs/{job_id}/retry` requeues a `FAILED` job, or a `RUNNING` job with
  stale or missing lease state. It resets the active lease fields, clears stale
  candidate/result metadata, sets the job back to `pending`, and writes the
  retry reason to `last_error`.
- `POST /jobs/retry-failed-stale` retries failed jobs that exhausted stale-lease
  recovery. The request body must be either `{"all": true}` or `{"limit": N}`.
- `POST /operator/tasks/baseline-ensure` creates an `operator_tasks` row and
  runs baseline ensure work in a FastAPI background task in the UI API process.
  It does not use Dramatiq. The API returns `409` when another baseline ensure
  task is already `pending` or `running`. A `pending` task older than 10 minutes
  is treated as stale, marked `failed`, and replaced by the new request.
- `POST /repair/schedule-one` calls the existing
  `FailedCandidateRepairSampler.schedule_one()` path, so the same eligibility,
  settings, repair-token budget, and campaign-baseline gates apply.
- `POST /repair/candidates/{candidate_id}/quarantine|discard|restore` updates
  repair-pool operator state. These actions fail with `409` while an active
  repair job exists for the candidate. Restore sets
  `lifecycle_status=active` and `repair_state=audit_only`.

## Operator Status

`GET /operator/status` returns one payload for the console status band:

- current `loreley.program.md` file state, sections, warnings, and hash;
- scheduler active or persisted campaign hash;
- current campaign baseline status, key, value, and failure summary;
- repair-pool counts by repair state, lifecycle status, and failure kind;
- job health, including unfinished jobs, pending ingestion, lease health, and
  counts by status and job kind.

## Notes

- **Authentication**: there is no authentication layer. Deploy the API only
  behind trusted local or internal network controls.
- **Write scope**: operator writes are intentionally narrow. They do not add
  authentication, restart processes, change environment variables, or bypass the
  scheduler and worker settings already in the database/runtime.
- **Job artifacts**: large, audit/debug oriented payloads (planning/coding prompts, raw outputs, evaluation logs) are stored on disk and referenced via `JobArtifacts`. The API exposes:
  - `GET /jobs/{job_id}/artifacts` as an index of available URLs
  - `GET /jobs/{job_id}/artifacts/{artifact_key}` for direct downloads
  Supported keys: `planning_prompt`, `planning_raw_output`, `planning_plan_json`, `coding_prompt`, `coding_raw_output`, `coding_execution_json`, `evaluation_json`, `evaluation_logs`.
