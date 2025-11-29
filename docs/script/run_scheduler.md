# script/run_scheduler.py

Thin CLI wrapper for running the Loreley evolution scheduler.

## Purpose

- Configure global Loguru logging based on `loreley.config.Settings.log_level`.
- Delegate all CLI parsing and control flow to `loreley.scheduler.main.main`.
- Provide a convenient entrypoint for process managers and local development
  without having to remember the module path.

The underlying scheduling logic, MAP-Elites integration, and database
interaction are all implemented in `loreley.scheduler.main.EvolutionScheduler`.

## Behaviour

- Calls `get_settings()` to load `Settings` and derive the desired log level
  (via `LOG_LEVEL`).
- Resets Loguru sinks and installs a single stderr sink at the configured
  level, disabling backtraces and diagnosis in production-style runs.
- Binds an informational logger with module name `script.run_scheduler`.
- Forwards the CLI arguments to `loreley.scheduler.main.main(argv)`, which
  supports the `--once` flag to run a single scheduler tick.

Exit codes are determined by `loreley.scheduler.main.main`: on success it returns
`0`, while unexpected exceptions bubble up and cause a non-zero exit.

## CLI usage

Recommended usage with `uv`:

```bash
uv run python script/run_scheduler.py        # continuous loop
uv run python script/run_scheduler.py --once # single tick (cron / smoke tests)
```

The wrapper is equivalent to invoking the module directly:

```bash
uv run python -m loreley.scheduler.main        # continuous loop
uv run python -m loreley.scheduler.main --once # single tick
```

## Configuration

The script relies on `loreley.config.Settings`:

- `LOG_LEVEL` controls the global Loguru log level for the scheduler process.
- All scheduler-related fields (`SCHEDULER_*`) and database/Redis settings are
  consumed by `loreley.scheduler.main` and `loreley.tasks.broker`. See
  `docs/loreley/config.md` and `docs/loreley/scheduler/main.md` for details.


