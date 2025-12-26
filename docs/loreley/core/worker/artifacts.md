# loreley.core.worker.artifacts

Cold-path artifact store for the evolution worker.

Artifacts are large, audit/debug oriented payloads (prompts, raw agent output, execution JSON, evaluation logs). They are written to disk and referenced from the database via `loreley.db.models.JobArtifacts`.

## Directory layout

Artifacts are written under:

- `<LOGS_BASE_DIR>/logs/worker/artifacts/<job_id>/` when `LOGS_BASE_DIR` is set
- `<cwd>/logs/worker/artifacts/<job_id>/` otherwise

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

It returns a dict of absolute paths which the worker upserts into the `job_artifacts` table.

## API access (optional UI API)

When the UI API is enabled, artifacts can be retrieved via:

- `GET /api/v1/jobs/{job_id}/artifacts` (URL index)
- `GET /api/v1/jobs/{job_id}/artifacts/{artifact_key}` (file download)

Supported keys:

- `planning_prompt`
- `planning_raw_output`
- `planning_plan_json`
- `coding_prompt`
- `coding_raw_output`
- `coding_execution_json`
- `evaluation_json`
- `evaluation_logs`

## Failure handling

Writing artifacts is best-effort: if the artifact store fails, the worker still persists the hot-path job result and logs a warning.


