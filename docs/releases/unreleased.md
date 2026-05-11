# Unreleased

These notes cover changes merged after `v0.7.9-alpha`.

## Added

- Added the ADR 0048 failed-candidate repair pool MVP. Failed candidates are
  stored in a separate candidate ledger with structured evaluation attempts and
  sanitized DiagnosticCapsule evidence. They do not create `CommitCard` rows,
  do not enter the MAP-Elites archive, and are not visible to the normal
  sampler.
- Added disabled-by-default repair scheduling settings:
  `FAILED_CANDIDATE_REPAIR_ENABLED`,
  `FAILED_CANDIDATE_REPAIR_NORMAL_JOBS_PER_TOKEN`,
  `FAILED_CANDIDATE_REPAIR_MAX_TOKENS`,
  `FAILED_CANDIDATE_REPAIR_MAX_ACTIVE_JOBS`, and related repair limits.
- Added optional `loreley.program.md` campaign programs with parsed goal,
  metric, scope, policy projection, provenance hashes, evaluator payload
  propagation, and a worker scope gate.
- Added ADR 0050 campaign baseline bootstrap with a `campaign_baselines`
  source-of-truth table and `BASELINE_BOOTSTRAP_POLICY=required|warn`.
- Added evaluator-declared evaluation artifacts with bounded diagnostics,
  visibility controls (`agent_visible`, `human_only`, `hidden`), worker-managed
  payload storage, `evaluation_artifacts` database rows, UI API listing/download
  routes, and agent-visible feedback projections.
- Added the Loreley operator console API and Streamlit pages:
  `GET /api/v1/operator/status`, background baseline ensure tasks, repair-pool
  listing/actions, repair schedule-one, single job retry, and failed-stale bulk
  retry.
- Added an `operator_tasks` table for UI API background task state. Baseline
  ensure tasks run in the FastAPI UI API process, not Dramatiq. A partial
  unique index prevents overlapping active baseline ensure tasks.
- Added the Agent REST control facade under `/api/v1/agent`, including
  capabilities, agent-oriented status and next actions, audited dry-run/execute
  actions with idempotency, required `LORELEY_AGENT_API_TOKEN` bearer auth, and
  job/commit feedback endpoints.
- Added an `agent_actions` table for Agent REST action audit records.
- Added native data-preserving database migrations from schema version 5
  (`v0.7.9-alpha`) to schema version 12, including
  `uv run loreley db current`, `uv run loreley db migrate`, and
  `uv run loreley db validate`.

## Changed

- Bumped `INSTANCE_SCHEMA_VERSION` to `12`. Existing schema-version-5
  databases should be upgraded with `uv run loreley db migrate` after taking a
  Postgres backup. API/scheduler/worker startup also migrates automatically when
  `DB_AUTO_MIGRATE=true`; with `DB_AUTO_MIGRATE=false`, fresh initialization and
  upgrades require an explicit `uv run loreley db migrate` first. `reset-db
  --yes` remains only a destructive local fallback for disposable databases.
- The Streamlit UI now includes Campaign and Repair Pool pages. Jobs, Commits,
  Archive, and Graphs show clearer fate and evidence indicators where the API
  already has the data.
- Manual repair scheduling from the UI API and scheduler repair dispatch now
  serialize cap and repair-token budget checks with the scheduling mutation.
  Automatic scheduler repair dispatch uses the persisted token budget, so a
  scheduler restart keeps already earned repair capacity. Repair scheduling is
  one-generation for the MVP: only original failed candidates with
  `failed_depth=0` and no `repair_source_candidate_id` are eligible. The
  `FAILED_CANDIDATE_REPAIR_MAX_DEPTH` setting is retained for compatibility but
  no longer controls active scheduling.
- Baseline ensure from the Operator console now reruns failed or degraded
  same-key baseline rows while still reusing valid rows. UI API startup now
  marks only stale pending/running baseline ensure tasks failed, leaving recent
  active tasks alone for multi-process API deployments. Task-start failures are
  persisted when possible.
- `GET /api/v1/jobs` and `GET /api/v1/jobs/page` now support server-side
  `candidate_fate` and `evidence=has_evidence|agent_visible|none` filters. The
  Streamlit Jobs page passes those filters to the API before pagination.
- Job, commit, archive, and graph API rows now expose candidate fate and
  evaluation evidence indicators when data is available. Job and commit detail
  payloads include non-hidden evaluation artifact metadata and agent feedback.
- Streamlit operator write buttons now require explicit per-action checkbox
  confirmation before enabling Campaign baseline ensure, Jobs retry, Repair Pool
  schedule-one, and Repair Pool candidate state actions.
- `CAMPAIGN_PROGRAM_CHANGE_POLICY=approve` is rejected until an approval
  workflow exists. Use `locked` or `auto`.
- The Streamlit Jobs page job-kind filter includes `seed`.
- Direct repair candidate operator actions now persist a durable operator audit
  row with actor, action, reason, and state transition metadata.

## Security

- UI API POST routes require `LORELEY_API_WRITE_TOKEN`.
- The Agent REST facade requires `LORELEY_AGENT_API_TOKEN`; if unset, agent
  routes return `agent_auth_not_configured`.
