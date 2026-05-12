# Doctor (environment checks)

This command performs quick preflight checks to reduce onboarding friction before you start the scheduler/worker processes.

## What it checks

- Database connectivity (PostgreSQL) using `DATABASE_URL` / `DB_*`.
- Database schema marker state. Current schemas pass; supported older schemas
  pass only when `DB_AUTO_MIGRATE=true`; otherwise the check tells you to run
  `uv run loreley db migrate`.
- Redis connectivity (Dramatiq broker) using `TASKS_REDIS_URL` / `TASKS_REDIS_*`.
- Git availability (`WORKER_REPO_GIT_BIN` / `git`).
- OpenAI-compatible credentials. Static keys use `OPENAI_API_KEY` /
  `LORELEY_LLM_API_KEY`; dynamic credentials use
  `OPENAI_DYNAMIC_API_KEY_PROVIDER` plus `OPENAI_DYNAMIC_API_KEY_TTL_SECONDS`.
- For schedulers:
  - `EXPERIMENT_ID`, `MAPELITES_EXPERIMENT_ROOT_COMMIT`,
    `MAPELITES_CODE_EMBEDDING_DIMENSIONS`, and `SCHEDULER_MAX_TOTAL_JOBS` are
    set.
  - The scheduler repository is a git checkout.
  - `loreley.program.md` can be parsed when present.
- For workers:
  - `EXPERIMENT_ID` and `MAPELITES_EXPERIMENT_ROOT_COMMIT` are set.
  - `WORKER_REPO_REMOTE_URL` is set.
  - `WORKER_EVALUATOR_PLUGIN` is set and importable (after applying `WORKER_EVALUATOR_PYTHON_PATHS`).
  - Planning/coding backend binaries are present for built-in backends (`kilo`, `codex`, `cursor-agent`).
  - Dynamic OpenAI token TTL is long enough for Kilocode planning/coding
    timeouts, or a warning is reported.
  - Unknown backend types fall back to a warning because binary checks are skipped.
- For UI/API:
  - Warns if UI extras (`fastapi`, `uvicorn`, `streamlit`) are not installed.
  - Warns when `LORELEY_API_WRITE_TOKEN` is unset. Read-only API routes still
    work, but UI API POST routes return `write_auth_not_configured`.
  - Warns when `LORELEY_AGENT_API_TOKEN` is unset. Non-agent routes still work,
    but `/api/v1/agent/*` returns `agent_auth_not_configured`.

## Usage

```bash
uv run loreley doctor --role all
```

Validate only one component:

```bash
uv run loreley doctor --role scheduler
uv run loreley doctor --role worker
uv run loreley doctor --role api
uv run loreley doctor --role ui
```

Adjust network timeouts:

```bash
uv run loreley doctor --role all --timeout-seconds 5
```

Machine-readable output (CI):

```bash
uv run loreley doctor --role all --json
```

## Exit codes

- `0`: all checks passed (warnings allowed).
- `1`: one or more failures.
- `2`: warnings present and `--strict` was provided.
