# script/run_scheduler.py

Thin CLI wrapper for running the Loreley evolution scheduler.

## Purpose

- Expose a minimal CLI (`--help`, `--log-level`) that works even when required
  environment variables are unset.
- Configure global Loguru logging based on `loreley.config.Settings.log_level`
  with optional per-invocation overrides.
- Delegate scheduler-specific CLI parsing and control flow to
  `loreley.scheduler.main.main`.
- Provide a convenient entrypoint for process managers and local development
  without having to remember the module path.

The underlying scheduling logic, MAP-Elites integration, and database
interaction are all implemented in `loreley.scheduler.main.EvolutionScheduler`.

## Behaviour

- Parses wrapper CLI arguments (notably `--help` and `--log-level`) before
  loading configuration, so help output works without a valid environment.
- Calls `get_settings()` to load `Settings`; validation errors are printed to
  the console and the process exits with code `1`.
- Resets Loguru sinks and installs a stderr sink at the configured level,
  disabling backtraces and diagnosis in production-style runs.
- Resolves a log directory under `<BASE>/logs/scheduler` where `<BASE>` is:
  - `LOGS_BASE_DIR` (expanded as a path) when set.
  - the current working directory when `LOGS_BASE_DIR` is unset.
- Adds a rotating file sink at `scheduler-YYYYMMDD.log` inside that directory
  with `rotation="10 MB"` and `retention="14 days"`, so scheduler output is
  always persisted for later debugging.
- Imports `loreley.scheduler.main.main` lazily after logging is configured; any
  import failure is reported via console/log and exits with code `1`.
- Forwards any remaining CLI arguments to `loreley.scheduler.main.main(argv)`,
  which supports the `--once` flag to run a single scheduler tick.

Exit codes are determined by `loreley.scheduler.main.main`: on success it returns
`0`, while unexpected exceptions bubble up and cause a non-zero exit.

## CLI usage

Recommended usage with `uv`:

```bash
uv run python script/run_scheduler.py        # continuous loop
uv run python script/run_scheduler.py --once # single tick (cron / smoke tests)
uv run python script/run_scheduler.py --log-level DEBUG
```

The wrapper is equivalent to invoking the module directly:

```bash
uv run python -m loreley.scheduler.main        # continuous loop
uv run python -m loreley.scheduler.main --once # single tick
```

## Configuration

The script relies on `loreley.config.Settings`:

- `LOG_LEVEL` controls the global Loguru log level for the scheduler process.
- `LOGS_BASE_DIR` (optional) overrides the base directory used for scheduler
  log files; when unset, logs are written under `./logs/scheduler` relative to
  the current working directory.
- All scheduler-related fields (`SCHEDULER_*`) and database/Redis settings are
  consumed by `loreley.scheduler.main` and `loreley.tasks.broker`. See
  `docs/loreley/config.md` and `docs/loreley/scheduler/main.md` for details.

The `examples/evol_circle_packing.py` helper simply delegates to this script
when running the scheduler, so its runs use the same logging configuration and
log file locations.

## Failure handling

- Invalid or missing environment variables produce a short console message and
  exit code `1` instead of an unhandled exception.
- Errors while importing or launching the scheduler are logged and surfaced
  before the process exits, so misconfigured dependencies do not produce long
  stack traces.


