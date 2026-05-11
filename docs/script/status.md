# Status

This command provides a single, high-level operational summary without needing to dig into the database manually.

If this command fails on an older schema-version-5 database, migrate the schema first:

```bash
uv run loreley db migrate
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
- Current root baseline status when a matching campaign baseline is known,
  including status, metric, value, baseline key, and failure kind.

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

## Baseline status

When a campaign baseline row matches the current experiment root, campaign
program, evaluator identity, primary metric, runtime profile, and settings
fingerprint, `status` adds a baseline section.

In JSON output this appears as `baseline` with fields such as:

- `campaign_baseline_id`
- `baseline_key_hash`
- `root_baseline_metric`
- `root_baseline_value`
- `root_baseline_status`
- `baseline_campaign_program_hash`
- `failure_kind`
- `failure_summary`

If no active campaign program or matching baseline is known, `baseline` is
`null`.
