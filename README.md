## Loreley

> Whole-repository Quality‑Diversity optimization for real git codebases.

Loreley is a distributed system that **evolves entire git repositories** (the unit of search is a git commit). It continuously samples base commits, asks external planning/coding agents to implement repo-wide changes, evaluates the result with your evaluator, and stores metrics plus a MAP‑Elites archive in Postgres for later sampling and reuse.

![](./docs/assets/loreley.svg)

### Why use it

- **Whole-repo evolution**: cross-module refactors and “production-style” changes are first-class.
- **QD-native (MAP-Elites)**: keeps multiple high-performing but *different* solutions instead of a single champion line.
- **Learned behaviour space**: behaviour descriptors come from repo-state code embeddings (cached by git blob SHA), not hand-crafted heuristics.
- **Production loop**: scheduler + Redis/Dramatiq workers + Postgres, with preflight checks, logs, and reproducible git history.

### Quick start (local)

**Requirements**: Python 3.11+, [`uv`](https://github.com/astral-sh/uv), Git (worktrees), PostgreSQL, Redis, and an OpenAI-compatible API for embeddings and some summaries (`OPENAI_API_KEY`). You also need:

- **Planning/coding backend**: default is the `kilocode` CLI on `PATH` (override via `WORKER_PLANNING_BACKEND` / `WORKER_CODING_BACKEND`).
- **Evaluator plugin**: `WORKER_EVALUATOR_PLUGIN=module:callable` that runs unattended and returns structured metrics.

```bash
git clone <YOUR_FORK_OR_ORIGIN_URL> loreley
cd loreley
uv sync
docker compose up -d postgres redis

cp env.example .env
# Minimal required vars (in addition to defaults in env.example):
# - EXPERIMENT_ID=<uuid or slug>
# - SCHEDULER_MAX_TOTAL_JOBS=<positive integer>
# - OPENAI_API_KEY
# - MAPELITES_EXPERIMENT_ROOT_COMMIT=<git commit hash>
# - WORKER_REPO_REMOTE_URL=<git remote URL with push access>
# - WORKER_EVALUATOR_PLUGIN=module:callable
#
# Recommended:
# - SCHEDULER_REPO_ROOT=/abs/path/to/your/target-git-checkout
# - WORKER_EVOLUTION_GLOBAL_GOAL="..."
#
# Optional:
# - WORKER_EVALUATOR_PYTHON_PATHS=["/abs/path/to/plugin_dir"]
# - SCHEDULER_STARTUP_APPROVE=true  # skip interactive startup approval

uv run loreley doctor --role all
uv run loreley scheduler
uv run loreley worker
uv run loreley status
```

### Optional UI (read-only)

```bash
uv sync --extra ui
uv run loreley ui
```

![](./docs/assets/ui.jpeg)

### Documentation

- [`docs/index.md`](docs/index.md) (local)
- [Online docs](https://NeapolitanIcecream.github.io/loreley/)
- Key guides: [`docs/loreley/config.md`](docs/loreley/config.md), [`docs/script/run_scheduler.md`](docs/script/run_scheduler.md), [`docs/script/run_worker.md`](docs/script/run_worker.md)


