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
- Added the Loreley operator console API and Streamlit pages:
  `GET /api/v1/operator/status`, background baseline ensure tasks, repair-pool
  listing/actions, repair schedule-one, single job retry, and failed-stale bulk
  retry.
- Added an `operator_tasks` table for UI API background task state. Baseline
  ensure tasks run in the FastAPI UI API process, not Dramatiq. A partial
  unique index prevents overlapping active baseline ensure tasks.

## Changed

- Bumped `INSTANCE_SCHEMA_VERSION` to `11`. Existing development databases must
  be reset with `uv run loreley reset-db --yes` before running this version.
- The Streamlit UI now includes Campaign and Repair Pool pages. Jobs, Commits,
  Archive, and Graphs show clearer fate and evidence indicators where the API
  already has the data.
- Manual repair scheduling from the UI API and scheduler repair dispatch now
  serialize cap and repair-token budget checks with the scheduling mutation.
  Stale pending or running baseline ensure tasks are failed before a replacement
  task is created.

## Security

- The operator write API remains unauthenticated by design. Deploy the UI API
  only on trusted local or internal networks.
