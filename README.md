## Loreley

Loreley is an automated MAP-Elites system that evolves entire git repositories. It continuously samples promising commits, asks external agents to implement and evaluate them, and stores the best-performing variants for later reuse.

---

## Key Concepts

- **Whole-repo evolution** – each individual is a real git commit, so outputs stay debuggable and integrable with normal tooling.
- **Learned feature space** – behaviour descriptors come from embeddings (plus optional PCA) instead of hand-picked heuristics.
- **Production-grade loop** – a Dramatiq/Redis worker fleet, PostgreSQL archive, and central scheduler keep the system running indefinitely.

Related systems: [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) style pipelines and open-source efforts such as [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) and [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

---

## What Ships With Loreley

- `loreley.config` – a single `Settings` object (pydantic-settings) for logging, database, Redis/Dramatiq, scheduler, worker repos, and MAP-Elites knobs.
- `loreley.db` – SQLAlchemy engine/session helpers plus ORM models for commits, metrics, and job state.
- `loreley.core.map_elites` – preprocessing, chunking, embeddings, dimensionality reduction, and `MapElitesManager`.
- `loreley.core.worker` – worktree lifecycle plus planning/coding/evaluation orchestration used by Dramatiq actors.
- `loreley.tasks` – Redis broker definition and the `run_evolution_job(job_id)` actor.
- `loreley.scheduler` – `EvolutionScheduler` that ingests, dispatches, measures, and schedules jobs.
- `script/run_scheduler.py`, `script/run_worker.py` – CLI shims that wire up logging, settings, and entrypoints.
- `docs/` – focused guides for configuration, scheduler behaviour, and worker operations.

---

## Requirements & Tooling

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- PostgreSQL + Redis (Dramatiq broker)
- Git (worktrees, LFS optional)
- Access to configured external planning/coding/evaluation agents

Logging uses Loguru with Rich renderers; secrets such as database or Redis URLs are redacted automatically.

---

## Quick Start

```bash
git clone <YOUR_FORK_OR_ORIGIN_URL> loreley
cd loreley
uv sync          # installs according to pyproject.toml / uv.lock
```

If you already have an environment, pin dependencies without creating a workspace:

```bash
uv sync --no-workspace
```

### Configure

All runtime settings come from environment variables consumed by `loreley.config.Settings`. Common examples:

- `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL`
- `DATABASE_URL` (or individual `DB_*`)
- `TASKS_REDIS_URL` / (host, port, db, password) + `TASKS_REDIS_NAMESPACE`, `TASKS_QUEUE_NAME`
- Scheduler knobs: `SCHEDULER_REPO_ROOT`, `SCHEDULER_POLL_INTERVAL_SECONDS`, `SCHEDULER_MAX_UNFINISHED_JOBS`, etc.
- Worker repo knobs: `WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`, `WORKER_REPO_ENABLE_LFS`, etc.
- MAP-Elites controls: `MAPELITES_*` for preprocessing, embeddings, dimensionality reduction, bounds, resolution, and fitness metrics.

See `docs/loreley/config.md` for the exhaustive list.

---

## Running Loreley

- **Scheduler loop**

  ```bash
  uv run python script/run_scheduler.py        # continuous loop
  uv run python script/run_scheduler.py --once # single tick
  uv run python -m loreley.scheduler.main [--once]
  ```

- **Worker process**

  ```bash
  uv run python script/run_worker.py
  ```

  The worker configures Loguru/Rich, initialises the Redis broker defined in `loreley.tasks.broker`, imports `loreley.tasks.workers`, and launches a single-threaded Dramatiq worker bound to `TASKS_QUEUE_NAME`.

Refer to `docs/script/run_scheduler.md` and `docs/script/run_worker.md` for deeper operational guidance.

---

## Project Layout

- `loreley/` – core services (`config`, `db`, `core/map_elites`, `core/worker`, `scheduler`, `tasks`)
- `script/` – CLI shims (`run_scheduler.py`, `run_worker.py`)
- `docs/` – module-level docs under `docs/loreley` and `docs/script`
- `pyproject.toml`, `uv.lock` – dependency definitions for `uv`
- `examples/` – self-contained optimisation examples used for testing and demos

---

## Examples

- **`examples/circle-packing`** – a geometric optimisation benchmark based on the classical
  [circle packing](https://en.wikipedia.org/wiki/Circle_packing) problem. The example defines:
  - a simple solution interface in `solution.py` that returns a set of equal-radius circles
    inside the unit square;
  - an evaluator plugin in `evaluate.py` that checks geometric validity (no overlap, inside
    bounds) and reports packing density and related metrics.

  You can wire this into the worker by pointing
  `WORKER_EVALUATOR_PYTHON_PATHS` at `examples/circle-packing` and setting
  `WORKER_EVALUATOR_PLUGIN=evaluate:plugin`, then letting MAP-Elites evolve better packings.

---

## License

See `LICENSE` for details.
