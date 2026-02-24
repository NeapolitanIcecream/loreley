# Status

This command provides a single, high-level operational summary without needing to dig into the database manually.

## Usage

```bash
uv run loreley status
```

It displays:
- Experiment and root commit information.
- The number of unfinished and pending-ingestion jobs.
- Default island MAP-Elites statistics (occupied cells, coverage, QD score, normalized QD score).
- The current best-fitness commit.

## Options

- `--island-id`: Inspect a specific island. If omitted, uses the default island.
- `--json`: Print the status payload as JSON, useful for machine-readable integrations.
