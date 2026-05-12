# loreley.core.worker.artifacts

Cold-path artifact store for the evolution worker.

Artifacts are large, audit/debug oriented payloads (prompts, raw agent output, execution JSON, evaluation logs). They are written to disk and referenced from the database via `loreley.db.models.JobArtifacts`.

Evaluator-declared artifacts are stored in the same worker artifact tree but
are indexed separately in `EvaluationArtifactRecord` rows. They can be surfaced
to humans, hidden from API consumers, or marked as agent-visible evidence for
planning/coding feedback.

## Directory layout

Artifacts are written under:

- `<base>/logs/<experiment_namespace>/worker/artifacts/<job_id>/<run_token>/` when an experiment namespace is available
- `<base>/logs/worker/artifacts/<job_id>/<run_token>/` otherwise

Where `<base>` is `<LOGS_BASE_DIR>` when set, and `<cwd>` otherwise. The experiment namespace is derived from `EXPERIMENT_ID` when configured.
When `run_token` is unavailable, the worker falls back to the per-job directory without the extra segment.
Evaluator-declared payloads are copied or written below
`evaluation_artifacts/<artifact_key>/` inside that job/run directory.

## Files written

`write_job_artifacts(...)` writes the following files:

- `planning_prompt.txt`
- `planning_raw_output.txt`
- `planning_plan.json`
- `coding_prompt.txt`
- `coding_raw_output.txt`
- `coding_execution.json`
- `evaluation.json`
- `evaluation_logs.txt`

It returns a dict of absolute paths. `EvolutionJobStore.persist_success()` later persists those paths by inserting a `JobArtifacts` row with `session.add(...)`.

When an evaluator returns `EvaluationArtifact` declarations, Loreley also
materializes allowed payloads and stores artifact metadata:

- `key`, `kind`, `mime_type`, `label`, and `summary`
- `visibility`: `agent_visible`, `human_only`, or `hidden`
- `agent_projection`: `summary`, `manifest`, or `path`
- optional `diagnostics`, `metadata`, `size_bytes`, and `sha256`

Artifact paths must stay inside the evaluation worktree. During
materialization, unsupported MIME types, oversized payloads, multiple payload
sources, missing or unreadable paths, directories, non-files, and path escapes
are recorded as sanitized validation warnings. If the accepted declaration
still has a summary or diagnostics, Loreley keeps a metadata-only record;
otherwise it skips that artifact. Duplicate keys and malformed declarations are
skipped during artifact coercion before materialization.

Relevant settings:

- `WORKER_EVALUATION_ARTIFACTS_ENABLED`
- `WORKER_EVALUATION_ARTIFACT_ALLOWED_MIME_TYPES`
- `WORKER_EVALUATION_ARTIFACT_MAX_BYTES`
- `WORKER_EVALUATION_AGENT_FEEDBACK_MODE`
- `WORKER_EVALUATION_AGENT_FEEDBACK_MAX_ARTIFACTS`
- `WORKER_EVALUATION_AGENT_FEEDBACK_MAX_DIAGNOSTICS`
- `WORKER_EVALUATION_AGENT_FEEDBACK_MAX_CHARS`
- `WORKER_EVALUATION_ARTIFACT_AGENT_PATH_MIME_TYPES`
- `WORKER_EVALUATION_ARTIFACT_AGENT_PATH_MAX_BYTES`

## API access (optional UI API)

When the UI API is enabled, artifacts can be retrieved via:

- `GET /api/v1/jobs/{job_id}/artifacts` (URL index)
- `GET /api/v1/jobs/{job_id}/artifacts/{artifact_key}` (file download)
- `GET /api/v1/jobs/{job_id}/evaluation-artifacts` (non-hidden evaluator artifact metadata)
- `GET /api/v1/jobs/{job_id}/evaluation-artifacts/{artifact_key}` (non-hidden evaluator artifact download)
- `GET /api/v1/commits/{commit_hash}/evaluation-artifacts` (non-hidden evaluator artifact metadata for a commit)

Supported keys:

- `planning_prompt`
- `planning_raw_output`
- `planning_plan_json`
- `coding_prompt`
- `coding_raw_output`
- `coding_execution_json`
- `evaluation_json`
- `evaluation_logs`

These fixed artifact keys are separate from evaluator-declared artifact keys.
Agent REST feedback endpoints include only evaluator artifacts marked
`agent_visible`; `human_only` and `hidden` records stay out of agent feedback.

## Failure handling

Writing artifacts is best-effort: if the artifact store fails, the worker still persists the hot-path job result and logs a warning.
