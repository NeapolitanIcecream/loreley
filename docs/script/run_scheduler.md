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

You also need database and Redis connectivity (`DATABASE_URL`, `TASKS_REDIS_URL`), and a writable repository checkout (`SCHEDULER_REPO_ROOT`, or it falls back to `WORKER_REPO_WORKTREE` / the current directory).
When the scheduler shares `WORKER_REPO_WORKTREE`, its git fetch/branch-update paths coordinate with the worker through the same cross-process repo lock.
`WORKER_EVOLUTION_GLOBAL_GOAL` defaults to a generic improvement objective, but
you will usually want to override it with a repository-specific goal.

Lease recovery is enabled by default. Tune it with:

- `SCHEDULER_STALE_RUNNING_RECLAIM_BATCH_SIZE`
- `SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS`

If `MAPELITES_CODE_EMBEDDING_MODEL` is not a `local-hash` variant, preflight also requires
either `OPENAI_API_KEY` / `LORELEY_LLM_API_KEY`, or dynamic auth via
`OPENAI_DYNAMIC_API_KEY_PROVIDER` plus `OPENAI_DYNAMIC_API_KEY_TTL_SECONDS`.

On first start the scheduler performs a repo-state root scan at `MAPELITES_EXPERIMENT_ROOT_COMMIT`
and requires operator approval. In non-interactive environments, pass `--yes` or set
`SCHEDULER_STARTUP_APPROVE=true`.

If you are upgrading an older development database created before lease recovery was added, reset it first:

```bash
uv run loreley reset-db --yes
```

## Options

- `--once`: execute a single scheduling tick and exit.
- `--yes`: auto-approve startup approval and start without prompting.
- `--no-preflight`: skip preflight validation.
- `--preflight-timeout-seconds`: network timeout used for DB/Redis connectivity checks.
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation.

## Logs

Logs are written to:

- `logs/{experiment_namespace}/scheduler/scheduler-YYYYMMDD-HHMMSS.log`

Each tick log also includes `reclaimed_pending` and `reclaimed_failed`, which show how many stale or malformed `RUNNING` jobs were requeued or failed during that tick.

Dispatch also includes stale `QUEUED` rows older than `WORKER_JOB_LEASE_TTL_SECONDS`. This repairs broker-loss or scheduler-restart cases where a row was persisted as queued but no worker started it.

For lease recovery triage and manual retry steps, see [Job lease recovery](job_leases.md).

## Exit codes

- `0`: success
- `1`: startup or preflight failure
- `2`: refused to start (e.g. lock contention)
