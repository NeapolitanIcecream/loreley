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

- `logs/scheduler/scheduler-YYYYMMDD-HHMMSS.log`

## Exit codes

- `0`: success
- `1`: startup or preflight failure
- `2`: refused to start (e.g. lock contention)
