# Benchmarking

This repository includes an **optional** benchmark suite under `benchmarks/`.
Benchmarks are **not** collected by default when running `pytest` (CI runs `pytest` with `testpaths=["tests"]`), so you must invoke them explicitly.

### Install

```bash
uv sync --locked --all-extras
```

### Run benchmarks

```bash
uv run pytest benchmarks
```

### Run Postgres-backed hot-path benchmarks

The DB-backed repo-state / steady-ingest benchmarks are opt-in and skip unless
`DATABASE_URL` is set. A simple local setup is:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql+psycopg://loreley:loreley@localhost:5432/loreley
uv run pytest benchmarks/test_repo_state_db_hot_path.py benchmarks/test_manager_ingest_steady_db.py
```

### Save a baseline run

```bash
uv run pytest benchmarks --benchmark-autosave
```

Saved runs are stored under `.benchmarks/` (by default `file://./.benchmarks`), grouped by platform/interpreter.

### Compare against a saved run

```bash
uv run pytest benchmarks --benchmark-compare=0001
```

To gate regressions, you can also fail the suite when a statistic degrades beyond a threshold:

```bash
uv run pytest benchmarks --benchmark-compare=0001 --benchmark-compare-fail=mean:5%
```

For advanced comparisons across multiple saved runs, use the `pytest-benchmark` CLI:

```bash
uv run pytest-benchmark compare 'Linux-CPython-3.13-64bit/*'
```
