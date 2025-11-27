# script/run_worker.py

CLI wrapper that runs the Loreley evolution worker as a single Dramatiq
process.

## Purpose

- Configure global Loguru logging based on `app.config.Settings.log_level`.
- Initialise the global Dramatiq Redis broker (`app.tasks.broker.broker`).
- Import `app.tasks.workers` so that the `run_evolution_job` actor is
  registered.
- Start a single-threaded Dramatiq `Worker` bound to the configured queue.

## Behaviour

On startup the script:

1. Calls `get_settings()` to load `Settings`.
2. Configures Loguru to log to stderr using `LOG_LEVEL` as the threshold.
3. Imports `app.tasks.broker` (which constructs and registers the Redis
   broker) and `app.tasks.workers` (which defines the `run_evolution_job`
   actor and its queue settings).
4. Logs a short “worker online” message including `TASKS_QUEUE_NAME` and
   `WORKER_REPO_WORKTREE`.
5. Creates a `dramatiq.Worker` with:
   - `broker` set to the global Redis broker instance.
   - `worker_threads=1` to ensure a single-threaded execution model.
6. Installs `SIGINT`/`SIGTERM` handlers that call `worker.stop()` for a
   graceful shutdown.
7. Starts the worker and blocks with `worker.join()` until the process is
   stopped.

Keyboard interrupts (`Ctrl+C`) are handled explicitly with a friendly shutdown
message.

## CLI usage

Typical usage with `uv`:

```bash
uv run python script/run_worker.py
```

The worker will begin consuming messages for the queue specified by
`TASKS_QUEUE_NAME` (default: `loreley.evolution`) in a single process with a
single worker thread. Jobs are expected to be created and dispatched by the
scheduler (`app.scheduler.main`).

## Configuration

The script uses `app.config.Settings` for:

- **Logging**
  - `LOG_LEVEL`: global Loguru level for worker logs.
- **Task queue / broker**
  - `TASKS_REDIS_URL` or (`TASKS_REDIS_HOST`, `TASKS_REDIS_PORT`,
    `TASKS_REDIS_DB`, `TASKS_REDIS_PASSWORD`, `TASKS_REDIS_NAMESPACE`).
  - `TASKS_QUEUE_NAME`: queue name for the `run_evolution_job` actor.
  - `TASKS_WORKER_MAX_RETRIES`, `TASKS_WORKER_TIME_LIMIT_SECONDS`: consumed
    by `app.tasks.workers` when configuring the actor.
- **Worker repository**
  - `WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`,
    and related `WORKER_REPO_*` options used by
    `app.core.worker.repository.WorkerRepository`.

For a full description of these settings, see `docs/app/config.md` and the
worker module documentation in `docs/app/tasks/workers.md`.


