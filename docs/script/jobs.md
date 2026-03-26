# Managing jobs

Use these commands when you need to inspect or repair individual evolution jobs.

If these commands fail on an older development database with missing lease columns, reset the schema first:

```bash
uv run loreley reset-db --yes
```

## List recent jobs

List the most recent jobs:

```bash
uv run loreley jobs ls
```

The table includes:

- `status`
- `lease`: derived lease state for the current row
- `recovery`: the current `recovery_count`
- `base_commit`
- `completed_at`

Change the result size:

```bash
uv run loreley jobs ls --limit 50
```

Print the payload as JSON:

```bash
uv run loreley jobs ls --json
```

## Inspect one job

Print a single job with lease details:

```bash
uv run loreley jobs inspect JOB_ID
```

Print the payload as JSON:

```bash
uv run loreley jobs inspect JOB_ID --json
```

This command shows the current job status, timestamps, `recovery_count`, and a lease block with:

- `state`: `active`, `stale`, `missing`, or `none`
- `worker_id`
- `run_token`
- `heartbeat_at`
- `lease_expires_at`

## List stale-failed jobs

List jobs that exhausted the stale-lease recovery budget:

```bash
uv run loreley jobs ls --failed-stale
```

Machine-readable output:

```bash
uv run loreley jobs ls --failed-stale --json
```

Use `--limit` to change the default result size:

```bash
uv run loreley jobs ls --failed-stale --limit 50
```

The `--failed-stale` filter matches both stale-heartbeat failures and failures caused by missing lease metadata on a `RUNNING` row.

## Retry one job

Move one job back to `PENDING`:

```bash
uv run loreley jobs retry JOB_ID
```

Print the result as JSON:

```bash
uv run loreley jobs retry JOB_ID --json
```

Add an explicit retry reason:

```bash
uv run loreley jobs retry JOB_ID --reason "manual retry after worker host restart"
```

This command accepts:

- `FAILED` jobs
- `RUNNING` jobs whose lease state is `missing` or `stale`

It resets the execution lease fields, clears `result_commit_hash`, resets `recovery_count` to `0`, writes your retry reason to `last_error`, and sets `scheduled_at` to `now()` so the scheduler can dispatch the job again.

Use it after you fix the underlying cause. If the worker environment is still unstable, the job will likely exhaust the recovery budget again.

## Retry stale-failed jobs in bulk

Retry the most recent 10 jobs that exhausted the stale-recovery budget:

```bash
uv run loreley jobs retry --failed-stale --limit 10
```

Retry all matching jobs:

```bash
uv run loreley jobs retry --failed-stale --all
```

Machine-readable output:

```bash
uv run loreley jobs retry --failed-stale --limit 10 --json
```

This path is intentionally explicit. When you use `--failed-stale`, you must also provide either `--all` or `--limit N`.
