# Status

This command provides a single, high-level operational summary without needing to dig into the database manually.

If this command fails on an older development database with a missing lease column such as `lease_expires_at`, reset the schema first:

```bash
uv run loreley reset-db --yes
```

## Usage

```bash
uv run loreley status
```

It displays:
- Experiment and root commit information.
- The number of unfinished and pending-ingestion jobs.
- Job lease health for `RUNNING` jobs, including stale and recovery-exhausted counts.
- The current lease TTL, heartbeat interval, and max recovery budget.
- Default island MAP-Elites statistics (occupied cells, coverage, QD score, normalized QD score).
- The current best-fitness commit.

## Options

- `--island-id`: Inspect a specific island. If omitted, uses the default island.
- `--json`: Print the status payload as JSON, useful for machine-readable integrations.

## Lease health

The status payload includes a `job_leases` section. Use it to answer these questions quickly:

- Are there any `RUNNING` jobs right now?
- Did any `RUNNING` jobs stop heartbeating and become stale?
- Are there any malformed `RUNNING` rows with missing lease fields?
- Have any jobs already exhausted the automatic stale-recovery budget?

For step-by-step triage and manual retry instructions, see [Job lease recovery](job_leases.md).
