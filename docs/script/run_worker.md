# Running the worker

The worker is a Dramatiq consumer process attached to a single experiment. It executes
the planning/coding/evaluation pipeline for jobs dispatched by the scheduler.

## Start

Recommended usage with `uv`:

```bash
uv run loreley worker
```

Minimum required settings for a functional worker are:

- `EXPERIMENT_ID`
- `MAPELITES_EXPERIMENT_ROOT_COMMIT`
- `WORKER_REPO_REMOTE_URL`
- `WORKER_EVALUATOR_PLUGIN`

You also need database and Redis connectivity (`DATABASE_URL`, `TASKS_REDIS_URL`), and a planning/coding backend binary (defaults to the Kilocode CLI via `kilo` on `PATH`).
`WORKER_EVOLUTION_GLOBAL_GOAL` defaults to a generic improvement objective, but
you will usually want to override it with a repository-specific goal.

If `MAPELITES_CODE_EMBEDDING_MODEL` is not a `local-hash` variant, or trajectory
summarization is enabled, preflight also requires `OPENAI_API_KEY` or
`LORELEY_LLM_API_KEY`.

## Options

- `--no-preflight`: skip preflight validation.
- `--preflight-timeout-seconds`: network timeout used for DB/Redis connectivity checks.
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation.

## Queue naming

The worker consumes jobs from a single experiment-scoped queue derived from `EXPERIMENT_ID`.
The queue name is not configurable and is derived as:

`"loreley.evolution.{experiment_namespace}"`

`EXPERIMENT_ID` can be a UUID or a short slug. Slugs are mapped to a stable UUID (uuid5)
and the derived `experiment_namespace` is stable across processes.

## Logs

Logs are written to:

- `logs/{experiment_namespace}/worker/worker-YYYYMMDD-HHMMSS.log`

If a worker loses its lease or dies mid-job, the scheduler will eventually reclaim that `RUNNING` row. Use `uv run loreley status` and the [Job lease recovery](job_leases.md) runbook to inspect the current state and recover stuck jobs.

## Exit codes

- `0`: success (clean shutdown)
- `1`: configuration, startup, or preflight error
