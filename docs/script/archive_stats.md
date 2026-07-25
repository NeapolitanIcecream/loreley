# Archive stats

This command inspects the MAP-Elites archive statistics for a given island.

## Usage

```bash
uv run loreley archive stats
```

It outputs key metrics about the learned behaviour space for the target island:

- `island_id`
- `occupied` (behaviour cells with at least one retained member)
- `elites` (all retained Pareto members)
- `cells` (total cell capacity)
- `coverage`
- `objective_count`
- `front_max_size`
- `primary_metric_name` / `primary_metric_direction`
- `best_primary_value` (an explicit operational projection, not archive admission)

## Options

- `--island-id`: Inspect a specific island. If omitted, uses the first ID in `MAPELITES_ISLANDS`.
- `--json`: Print stats as JSON.
