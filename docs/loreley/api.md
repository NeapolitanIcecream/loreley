# UI API (FastAPI)

Loreley ships an optional HTTP API used by the Streamlit dashboard. Most routes
are read-only observability routes. The operator console also exposes a small
set of write routes for local run repair: baseline ensure tasks, repair-pool
actions, and job retries.

The operator write API is unauthenticated by design. Run it only on trusted
local or internal networks.

The agent REST facade under `/api/v1/agent` can be protected separately with
`LORELEY_AGENT_API_TOKEN`. If that variable is unset, the agent facade remains
open for local development.

The Streamlit UI disables its operator write buttons until the matching
per-action confirmation checkbox is checked. That UI guard does not add
authentication or change these API routes.

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
`InstanceMetadata` marker (schema/version). Empty databases are initialized on
startup, and schema-version-5 databases are migrated automatically, only when
`DB_AUTO_MIGRATE=true`. With `DB_AUTO_MIGRATE=false`, run
`uv run loreley db migrate` before startup.

Common variables:

- `DATABASE_URL`
- `LOGS_BASE_DIR` (optional; logs are read from `<LOGS_BASE_DIR>/logs` or `<cwd>/logs`)
- `LOG_LEVEL`
- `EXPERIMENT_ID` (optional; used to resolve the experiment namespace for log browsing; if unset, the API falls back to the database marker)
- `LORELEY_AGENT_API_TOKEN` (optional; bearer token required only for
  `/api/v1/agent/*` when set)

## Versioning and prefix

All routes are served under the versioned prefix: `/api/v1`.

FastAPI also exposes OpenAPI docs by default:

- `/docs` (Swagger UI)
- `/redoc`

## Endpoints (v1)

- `GET /health`
- `GET /instance`
- `GET /jobs` (`status`, `job_kind`, `candidate_fate`, `evidence`, `limit`,
  and `offset` filters)
- `GET /jobs/page` (`status`, `job_kind`, `candidate_fate`,
  `evidence=has_evidence|agent_visible|none`, `limit`, and `cursor` filters)
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
- `GET /agent/capabilities`
- `GET /agent/status`
- `GET /agent/next-actions`
- `POST /agent/actions`
- `GET /agent/actions/{action_id}`
- `GET /agent/jobs/{job_id}/feedback`
- `GET /agent/commits/{commit_hash}/feedback`

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
  or a `running` task older than 6 hours is treated as stale, marked `failed`,
  and replaced by the new request.
- `POST /repair/schedule-one` calls the existing
  `FailedCandidateRepairSampler.schedule_one()` path, so the same eligibility,
  settings, repair-token budget, and campaign-baseline gates apply. Manual API
  requests and scheduler repair dispatch serialize the active-job cap check,
  token-budget check, and scheduling mutation on the instance metadata row.
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

## Agent REST Facade

The agent facade is a REST-first automation surface for control clients. It
wraps only the existing operator writes:

- `retry_job`
- `retry_failed_stale_jobs`
- `baseline_ensure`
- `repair_schedule_one`
- `repair_candidate_quarantine`
- `repair_candidate_discard`
- `repair_candidate_restore`

`POST /api/v1/agent/actions` accepts an action envelope with `action_type`,
`dry_run`, `idempotency_key`, `reason`, `expected_state`, and `params`. Dry-runs
validate parameters and focused current-state checks without calling write
services. Execute requests persist an `agent_actions` audit row and call the
matching existing service.

Structured errors are used only for agent routes:

```json
{
  "error_code": "precondition_failed",
  "message": "Expected state mismatch for status.",
  "retryable": false,
  "resource": {"type": "job", "id": "..."},
  "suggested_next_actions": []
}
```

See [Agent REST API](agent-api.md) for examples, auth behavior, idempotency, and
feedback endpoints.

## Notes

- **Authentication**: there is no authentication layer. Deploy the API only
  behind trusted local or internal network controls. The agent facade is the
  exception: set `LORELEY_AGENT_API_TOKEN` to require bearer-token auth for
  `/api/v1/agent/*`.
- **Streamlit confirmation**: the Streamlit operator pages require a checkbox
  confirmation before enabling each write button. Direct API clients still call
  the same unauthenticated POST routes.
- **Write scope**: operator writes are intentionally narrow. They do not add
  authentication, restart processes, change environment variables, or bypass the
  scheduler and worker settings already in the database/runtime.
- **Schema migration**: the agent facade adds the `agent_actions` table and bumps
  `INSTANCE_SCHEMA_VERSION` to `12`. Existing schema-version-5 databases should
  be upgraded with `uv run loreley db migrate`; `reset-db --yes` is only a
  destructive local fallback.
- **Job artifacts**: large, audit/debug oriented payloads (planning/coding prompts, raw outputs, evaluation logs) are stored on disk and referenced via `JobArtifacts`. The API exposes:
  - `GET /jobs/{job_id}/artifacts` as an index of available URLs
  - `GET /jobs/{job_id}/artifacts/{artifact_key}` for direct downloads
  Supported keys: `planning_prompt`, `planning_raw_output`, `planning_plan_json`, `coding_prompt`, `coding_raw_output`, `coding_execution_json`, `evaluation_json`, `evaluation_logs`.
