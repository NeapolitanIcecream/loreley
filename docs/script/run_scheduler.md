# Running the scheduler

The scheduler is the long-running process that keeps the evolution pipeline moving:
it ingests completed jobs, samples new work from the MAP-Elites archive, and dispatches
jobs to the Dramatiq worker queue.

## Start

Recommended usage with `uv`:

```bash
uv run loreley scheduler              # continuous loop
uv run loreley scheduler --once       # single tick (cron / smoke tests)
uv run loreley scheduler --yes --once # non-interactive run
```

Minimum required settings for a functional scheduler are:

- `EXPERIMENT_ID`
- `MAPELITES_EXPERIMENT_ROOT_COMMIT`
- `MAPELITES_CODE_EMBEDDING_DIMENSIONS`
- `SCHEDULER_MAX_TOTAL_JOBS`

You also need database and Redis connectivity (`DATABASE_URL`, `TASKS_REDIS_URL`), and a repository checkout (`SCHEDULER_REPO_ROOT`, or it falls back to `WORKER_REPO_WORKTREE` / the current directory).
`WORKER_EVOLUTION_GLOBAL_GOAL` defaults to a generic improvement objective, but
you will usually want to override it with a repository-specific goal.

If `MAPELITES_CODE_EMBEDDING_MODEL` is not a `local-hash` variant, preflight also requires
`OPENAI_API_KEY` or `LORELEY_LLM_API_KEY` for embeddings.

On first start the scheduler performs a repo-state root scan at `MAPELITES_EXPERIMENT_ROOT_COMMIT`
and requires operator approval. In non-interactive environments, pass `--yes` or set
`SCHEDULER_STARTUP_APPROVE=true`.

## Options

- `--once`: execute a single scheduling tick and exit.
- `--yes`: auto-approve startup approval and start without prompting.
- `--no-preflight`: skip preflight validation.
- `--preflight-timeout-seconds`: network timeout used for DB/Redis connectivity checks.
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation.

## Logs

Logs are written to:

- `logs/{experiment_namespace}/scheduler/scheduler-YYYYMMDD-HHMMSS.log`

Each tick log also includes `reclaimed_pending` and `reclaimed_failed`, which show how many stale `RUNNING` jobs were recovered or failed during that tick.

For lease recovery triage and manual retry steps, see [Job lease recovery](job_leases.md).

## Exit codes

- `0`: success
- `1`: startup or preflight failure
- `2`: refused to start (e.g. lock contention)
