# script/run_worker.py

CLI wrapper that runs the Loreley evolution worker as a single Dramatiq
process.

## Purpose

- Expose a minimal CLI (`--help`, `--log-level`) that works even when required
  environment variables are unset.
- Configure global Loguru logging based on `loreley.config.Settings.log_level`
  with optional per-invocation overrides.
- Lazily initialise the Dramatiq Redis broker (`loreley.tasks.broker.broker`)
  and import `loreley.tasks.workers` so that the `run_evolution_job` actor is
  registered.
- Start a single-threaded Dramatiq `Worker` bound to the configured queue.

## Behaviour

On startup the script:

1. Parses CLI args (primarily `--help` and `--log-level`) before loading
   configuration, so help output works without a valid environment.
2. Calls `get_settings()` to load `Settings`; any validation error is printed to
   the console and the process exits with code `1` instead of crashing.
3. Configures Loguru to log to stderr using `LOG_LEVEL` (or `--log-level`) as
   the threshold.
4. Resolves a log directory under `<BASE>/logs/worker` where `<BASE>` is:
   - `LOGS_BASE_DIR` (expanded as a path) when set.
   - the current working directory when `LOGS_BASE_DIR` is unset.
5. Adds a rotating file sink at `worker-YYYYMMDD.log` inside that directory
   with `rotation="10 MB"` and `retention="14 days"`, so worker output is
   always persisted for later debugging.
6. Imports `loreley.tasks.broker` (which constructs and registers the Redis
   broker) and `loreley.tasks.workers` (which defines the `run_evolution_job`
   actor and its queue settings) after logging is configured; any import/startup
   failure is reported with a concise console message and exit code `1`.
7. Logs a short “worker online” message including `TASKS_QUEUE_NAME` and
   `WORKER_REPO_WORKTREE` (the base clone; per-job worktrees are created under
   `<WORKER_REPO_WORKTREE>-worktrees/`).
8. Creates a `dramatiq.Worker` with:
   - `broker` set to the global Redis broker instance.
   - `worker_threads=1` to ensure a single-threaded execution model.
9. Installs `SIGINT`/`SIGTERM` handlers that call `worker.stop()` for a
   graceful shutdown.
10. Starts the worker and blocks with `worker.join()` until the process is
   stopped.

Keyboard interrupts (`Ctrl+C`) are handled explicitly with a friendly shutdown
message.

## CLI usage

Typical usage with `uv`:

```bash
uv run python script/run_worker.py
uv run python script/run_worker.py --log-level DEBUG
```

The worker will begin consuming messages for the queue specified by
`TASKS_QUEUE_NAME` (default: `loreley.evolution`) in a single process with a
single worker thread. Jobs are expected to be created and dispatched by the
scheduler (`loreley.scheduler.main`).

`--help` works without configured environment variables; other configuration is
still read from `loreley.config.Settings`.

## Configuration

The script uses `loreley.config.Settings` for:

- **Logging**
  - `LOG_LEVEL`: global Loguru level for worker logs.
  - `LOGS_BASE_DIR` (optional): overrides the base directory used for worker
    log files; when unset, logs are written under `./logs/worker` relative to
    the current working directory.
- **Task queue / broker**
  - `TASKS_REDIS_URL` or (`TASKS_REDIS_HOST`, `TASKS_REDIS_PORT`,
    `TASKS_REDIS_DB`, `TASKS_REDIS_PASSWORD`, `TASKS_REDIS_NAMESPACE`).
  - `TASKS_QUEUE_NAME`: queue name for the `run_evolution_job` actor.
  - `TASKS_WORKER_MAX_RETRIES`, `TASKS_WORKER_TIME_LIMIT_SECONDS`: consumed
    by `loreley.tasks.workers` when configuring the actor.
- **Worker repository**
  - `WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`,
    `WORKER_REPO_WORKTREE_RANDOMIZE`, `WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN`,
    and related `WORKER_REPO_*` options used by
    `loreley.core.worker.repository.WorkerRepository`. The worker maintains a
    base clone at `WORKER_REPO_WORKTREE` and creates isolated per-job worktrees
    (removed after each job) under `<WORKER_REPO_WORKTREE>-worktrees/` so
    multiple worker processes can run concurrently on the same host.

For a full description of these settings, see `docs/loreley/config.md` and the
worker module documentation in `docs/loreley/tasks/workers.md`.

The `examples/evol_circle_packing.py` helper simply delegates to this script
when running the worker, so its runs use the same logging configuration and
log file locations.

## Failure handling

- Invalid or missing environment variables produce a short console message and
  exit code `1` instead of an unhandled exception.
- Errors while initialising the broker/worker modules are logged and surfaced
  before the process exits, so misconfigured Redis/DB credentials do not
  produce long stack traces.


