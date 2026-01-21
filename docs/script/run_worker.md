# Running the worker

The worker is a Dramatiq consumer process attached to a single experiment. It executes
the planning/coding/evaluation pipeline for jobs dispatched by the scheduler.

## Start

Recommended usage with `uv`:

```bash
uv run loreley worker
```

This command requires `EXPERIMENT_ID` to be set in the environment (or `.env`).

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

## Exit codes

- `0`: success (clean shutdown)
- `1`: configuration, startup, or preflight error
