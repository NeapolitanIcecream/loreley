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

## Changed

- Bumped `INSTANCE_SCHEMA_VERSION` to `9`. Existing development databases must
  be reset with `uv run loreley reset-db --yes` before running this version.
