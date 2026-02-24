# Archive stats

This command inspects the MAP-Elites archive statistics for a given island.

## Usage

```bash
uv run loreley archive stats
```

It outputs key metrics about the learned behaviour space for the target island:
- `island_id`
- `occupied`
- `cells` (total cell capacity)
- `coverage`
- `qd_score`
- `norm_qd_score`
- `best_fitness`

## Options

- `--island-id`: Inspect a specific island. If omitted, uses the default island.
- `--json`: Print stats as JSON.
