# Running the worker

The worker is a Dramatiq consumer process attached to a single experiment. It executes
the planning/coding/evaluation pipeline for jobs dispatched by the scheduler.

## Start

Recommended usage with `uv`:

```bash
uv run loreley worker --experiment-id <EXPERIMENT_UUID>
```

If `WORKER_EXPERIMENT_ID` is set, you can omit `--experiment-id`.

## Options

- `--experiment-id`: attach this worker process to a single experiment UUID (overrides `WORKER_EXPERIMENT_ID`).
- `--no-preflight`: skip preflight validation.
- `--preflight-timeout-seconds`: network timeout used for DB/Redis connectivity checks.
- `--log-level`: global option (pass before the subcommand) that overrides `LOG_LEVEL` for this invocation.

## Queue naming

The worker consumes jobs from an experiment-scoped queue derived from the configured queue
prefix (`TASKS_QUEUE_NAME`) and the experiment UUID hex:

`"{TASKS_QUEUE_NAME}.{experiment_id.hex}"`

## Logs

Logs are written to:

- `logs/worker/worker-YYYYMMDD-HHMMSS.log`

## Exit codes

- `0`: success (clean shutdown)
- `1`: configuration, startup, or preflight error
