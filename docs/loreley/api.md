# UI API (FastAPI)

Loreley ships an optional HTTP API used by the Streamlit dashboard. Most routes
are read-only observability routes. The operator console also exposes a small
set of write routes for local run repair: baseline ensure tasks, repair-pool
actions, and job retries.

UI API write routes require bearer-token auth with `LORELEY_API_WRITE_TOKEN`.
Set the same token in the Streamlit UI process environment so operator POSTs
include `Authorization: Bearer ...`.

The agent REST facade under `/api/v1/agent` requires bearer-token auth with
`LORELEY_AGENT_API_TOKEN`.

The Streamlit UI disables its operator write buttons until the matching
per-action confirmation checkbox is checked. That UI guard is separate from
the bearer-token check enforced by the API.

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
- `LORELEY_API_WRITE_TOKEN` (required for UI API POST routes)
- `LORELEY_AGENT_API_TOKEN` (required for `/api/v1/agent/*`)

API preflight reports missing `LORELEY_API_WRITE_TOKEN` and
`LORELEY_AGENT_API_TOKEN` as warnings. The API can still serve read-only routes
without them. Protected POST routes return `write_auth_not_configured`, and
agent routes return `agent_auth_not_configured`, until the matching token is
set.

## API Tokens

Loreley does not issue `LORELEY_API_WRITE_TOKEN` or
`LORELEY_AGENT_API_TOKEN`. Generate each value yourself and store it as a
deployment secret:

```bash
python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
```

Use different values for the two tokens:

```bash
LORELEY_API_WRITE_TOKEN=<generated-write-token>
LORELEY_AGENT_API_TOKEN=<generated-agent-token>
```

Set `LORELEY_API_WRITE_TOKEN` in both the FastAPI UI API process and the
Streamlit UI process. The API checks it on POST routes, and the Streamlit UI
uses the same value when it sends operator POST requests.

Set `LORELEY_AGENT_API_TOKEN` in the FastAPI UI API process. Agent clients must
send it on every `/api/v1/agent/*` request:

```bash
curl -H "Authorization: Bearer $LORELEY_AGENT_API_TOKEN" \
  http://127.0.0.1:8000/api/v1/agent/capabilities
```

Direct clients that call UI API write routes must send
`LORELEY_API_WRITE_TOKEN`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs/retry-failed-stale \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LORELEY_API_WRITE_TOKEN" \
  -d '{"limit": 1}'
```

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
- `GET /jobs/{job_id}/evaluation-artifacts`
- `GET /jobs/{job_id}/evaluation-artifacts/{artifact_key}`
- `GET /jobs/{job_id}/usage`
- `GET /usage/summary`
- `GET /usage/events/page`
- `GET /commits`
- `GET /commits/page`
- `GET /commits/{commit_hash}`
- `GET /commits/{commit_hash}/evaluation-artifacts`
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
- `POST /repair/schedule-one` (deprecated no-op)
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
- `POST /repair/schedule-one` is retained as a deprecated compatibility route.
  It returns `scheduled=false` with a deprecation message and does not create a
  new repair job. New candidate-owned evaluator failures are handled by
  evaluator-guided in-loop rework inside the worker job.
- `POST /repair/candidates/{candidate_id}/quarantine|discard|restore` updates
  repair-pool operator state. These actions fail with `409` while an active
  repair job exists for the candidate. Restore sets
  `lifecycle_status=active` and `repair_state=audit_only`. The optional
  request `reason` is stored in a durable `operator_tasks` audit row and the
  response includes `operator_audit_task_id`.

## Job And Evidence Fields

Job, commit, archive, and graph rows include these operator-facing evidence
fields when the underlying data exists:

- `candidate_fate_label` and `candidate_fate_reason`: derived presentation
  state such as `elite_inserted`, `valid_not_elite`, `candidate_failed`,
  `repair_pending`, `discarded_for_sampling`, or `unknown`.
- `has_evaluation_evidence`: true when a non-hidden evaluator artifact exists
  for the candidate commit.
- `agent_visible_evidence_count`: number of artifacts whose visibility is
  `agent_visible`.
- `top_evaluation_diagnosis`: first agent-visible diagnostic message or summary.

`GET /jobs/{job_id}` also includes `latest_evaluation_attempt`. For phased
evaluations this contains the protocol, complete evaluator/campaign identity,
measurement cache key and fingerprint, whether a real measurement ran, reuse
kind, accepted measurement id, source attempt id, payload SHA-256, location-free
evidence hashes, and evaluator-slot wait/acquire/release telemetry. Local
artifact paths and secrets are not exposed. The payload also includes the
immutable per-job `attempt_ordinal`; retries keep earlier attempt evidence in
the database while this field remains a latest projection.

`GET /jobs` and `GET /jobs/page` can filter on:

- `candidate_fate=<label>`
- `evidence=has_evidence|agent_visible|none`

`GET /repair/pool` can filter on `repair_state`, `lifecycle_status`,
`failure_kind`, and `campaign_program_hash`.

## LLM Usage

LLM usage routes expose the `llm_usage_events` ledger added in schema version
13. They are read-only and support dashboard views, job debugging, and cost
exports.

- `GET /usage/summary` returns token and cost aggregates plus
  `by_cost_source`. Optional filters:
  `job_id`, `source`, `phase`, and `model`.
- `GET /usage/events/page` returns a cursor-paginated event page. Optional
  filters: `job_id`, `source`, `phase`, `model`, and `limit`.
- `GET /jobs/{job_id}/usage` returns all usage rows attributed to one job.

Usage rows distinguish provider-reported USD, Kilo catalog dollars, locally
estimated USD, unpriced token usage, and unavailable telemetry. Kilo catalog
values are not presented as provider invoices. The API does
not return prompts, raw transcripts, credentials, or full CLI event streams.

## Evaluation Artifacts

Evaluator-declared artifacts are stored separately from the fixed worker
artifacts. `GET /jobs/{job_id}/evaluation-artifacts` and
`GET /commits/{commit_hash}/evaluation-artifacts` return non-hidden artifact
metadata:

- `key`, `kind`, `mime_type`, `label`, and `summary`
- `visibility`: `agent_visible` or `human_only`
- `agent_projection`: `summary`, `manifest`, or `path`
- `size_bytes`, `sha256`, and structured `diagnostics`
- `download_url` when a stored payload is available

Hidden artifacts are not listed or downloadable through these routes. Direct
downloads use `GET /jobs/{job_id}/evaluation-artifacts/{artifact_key}`.

## Operator Status

`GET /operator/status` returns one payload for the console status band:

- current `loreley.program.md` file state, sections, warnings, and hash;
- scheduler active or persisted campaign hash;
- current campaign baseline status, key, value, and failure summary;
- legacy repair-pool counts by repair state, lifecycle status, and failure kind;
- job health, including unfinished jobs, pending ingestion, lease health, and
  counts by status and job kind.

## Agent REST Facade

The agent facade is a REST-first automation surface for control clients. It
wraps only the existing operator writes:

- `retry_job`
- `retry_failed_stale_jobs`
- `baseline_ensure`
- `repair_schedule_one` (deprecated no-op compatibility action)
- `repair_candidate_quarantine`
- `repair_candidate_discard`
- `repair_candidate_restore`

`POST /api/v1/agent/actions` accepts an action envelope with `action_type`,
`dry_run`, `idempotency_key`, `reason`, `expected_state`, and `params`. Dry-runs
validate parameters and focused current-state checks without calling write
services. Execute requests persist a pending `agent_actions` audit row before
calling side-effecting services, then update that row with the result.

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

- **Authentication**: UI API POST routes require `LORELEY_API_WRITE_TOKEN`;
  agent routes require `LORELEY_AGENT_API_TOKEN`. Keep the API behind trusted
  local or internal network controls; these bearer tokens are a process-local
  guard, not a multi-user authorization system.
- **Streamlit confirmation**: the Streamlit operator pages require a checkbox
  confirmation before enabling each write button. Direct API clients must still
  include the write bearer token.
- **Write scope**: operator writes are intentionally narrow. They do not add
  user management, restart processes, change environment variables, or bypass the
  scheduler and worker settings already in the database/runtime.
- **Schema migration**: the usage ledger adds the `llm_usage_events` table and
  bumps `INSTANCE_SCHEMA_VERSION` to `13`. Existing schema-version-5 or
  schema-version-12 databases should be upgraded with
  `uv run loreley db migrate`; `reset-db --yes` is only a destructive local
  fallback.
- **Job artifacts**: large, audit/debug oriented payloads (planning/coding prompts, raw outputs, evaluation logs) are stored on disk and referenced via `JobArtifacts`. The API exposes:
  - `GET /jobs/{job_id}/artifacts` as an index of available URLs
  - `GET /jobs/{job_id}/artifacts/{artifact_key}` for direct downloads
  Supported keys: `planning_prompt`, `planning_raw_output`, `planning_plan_json`, `coding_prompt`, `coding_raw_output`, `coding_execution_json`, `evaluation_json`, `evaluation_logs`.
- **Evaluation artifacts**: evaluator-declared artifacts are referenced through
  `EvaluationArtifactRecord` rows. Non-hidden records are available to UI API
  clients; Agent REST feedback endpoints include only `agent_visible` records.
