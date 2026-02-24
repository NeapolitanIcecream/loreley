## Benchmarking

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
pytest-benchmark compare 'Linux-CPython-3.13-64bit/*'
```

