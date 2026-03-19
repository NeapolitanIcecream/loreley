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

You also need database and Redis connectivity (`DATABASE_URL`, `TASKS_REDIS_URL`), a non-empty evolution goal (the code ships a generic default, but real runs should set `WORKER_EVOLUTION_GLOBAL_GOAL` explicitly), and a repository checkout. In practice, set `SCHEDULER_REPO_ROOT` explicitly on first cold start: if it is unset, the scheduler falls back to `WORKER_REPO_WORKTREE`, which often is not a git repo yet before any worker prepares a base clone.

Under the default embedding settings, scheduler preflight also expects `OPENAI_API_KEY`. That requirement is lifted when you switch to a `local-hash...` embedding model.

On first start the scheduler performs a repo-state root scan at `MAPELITES_EXPERIMENT_ROOT_COMMIT`
and requires operator approval. In non-interactive environments, pass `--yes` or set
`SCHEDULER_STARTUP_APPROVE=true`.

At the end of a bounded run, the scheduler force-updates a **local** branch
`evolution/best/<experiment>` inside `SCHEDULER_REPO_ROOT`. It does not auto-push
that branch to the remote.

## Options

- `--once`: execute a single scheduling tick and exit.
- `--yes`: auto-approve startup approval and start without prompting.
- `--no-preflight`: skip preflight validation.
- `--preflight-timeout-seconds`: network timeout used for DB/Redis connectivity checks.
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation.

## Logs

Logs are written to:

- `logs/{experiment_namespace}/scheduler/scheduler-YYYYMMDD-HHMMSS.log`

## Exit codes

- `0`: success
- `1`: startup or preflight failure
- `2`: refused to start (e.g. lock contention)
