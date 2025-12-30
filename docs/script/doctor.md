# Doctor (environment checks)

This script performs quick preflight checks to reduce onboarding friction before you start the scheduler/worker processes.

## What it checks

- Database connectivity (PostgreSQL) using `DATABASE_URL` / `DB_*`.
- Redis connectivity (Dramatiq broker) using `TASKS_REDIS_URL` / `TASKS_REDIS_*`.
- Git availability (`WORKER_REPO_GIT_BIN` / `git`).
- For workers:
  - `WORKER_REPO_REMOTE_URL` is set.
  - `WORKER_EVALUATOR_PLUGIN` is set and importable (after applying `WORKER_EVALUATOR_PYTHON_PATHS`).
  - Planning/coding backend binaries are present when using the default Codex CLI backend (`WORKER_PLANNING_CODEX_BIN`, `WORKER_CODING_CODEX_BIN`).
  - Warns if `cursor-agent` is missing (only required if you use the Cursor backend).
- For UI/API:
  - Warns if UI extras (`fastapi`, `uvicorn`, `streamlit`) are not installed.

## Usage

```bash
uv run loreley doctor --role all

# legacy wrapper (still supported)
uv run python script/doctor.py --role all
```

Validate only one component:

```bash
uv run loreley doctor --role scheduler
uv run loreley doctor --role worker
uv run loreley doctor --role ui
```

Adjust network timeouts:

```bash
uv run python script/doctor.py --role all --timeout-seconds 5
```

Machine-readable output (CI):

```bash
uv run python script/doctor.py --role all --json
```

## Exit codes

- `0`: all checks passed (warnings allowed).
- `1`: one or more failures.
- `2`: warnings present and `--strict` was provided.


