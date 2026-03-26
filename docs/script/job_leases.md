# Job lease recovery

Use this runbook when `RUNNING` jobs stop making progress, the scheduler starts reclaiming stale jobs, or a job hits the stale-recovery budget and ends in `FAILED`.

If the lease commands or status views fail on an older development database with missing lease columns, reset the schema first:

```bash
uv run loreley reset-db --yes
```

## What to watch

The scheduler tick log now includes two lease-recovery counters:

- `reclaimed_pending`: stale `RUNNING` jobs moved back to `PENDING`
- `reclaimed_failed`: stale `RUNNING` jobs that exceeded the recovery budget and were marked `FAILED`

Example:

```text
Scheduler tick ingested=0 reclaimed_pending=1 reclaimed_failed=0 dispatched=1 ...
```

For an on-demand snapshot, run:

```bash
uv run loreley status
uv run loreley status --json
```

`status` now reports a `job_leases` section with:

- `running`: current `RUNNING` job count
- `stale_running`: `RUNNING` jobs whose lease has already expired
- `running_without_lease`: `RUNNING` jobs missing `run_token`, `worker_id`, or `lease_expires_at`
- `recovery_exhausted_failed`: failed jobs that were dropped after exceeding the stale-recovery budget
- `lease_ttl_seconds`
- `heartbeat_interval_seconds`
- `max_recovery_attempts`

## Normal values

- `stale_running=0`
- `running_without_lease=0`
- `reclaimed_failed=0`

Short spikes in `reclaimed_pending` are acceptable after a worker crash or host restart. The count should fall back to zero on later ticks after replacement workers pick the jobs up.

## Triage

1. Check whether the scheduler is still ticking. If `stale_running > 0` and `reclaimed_pending` stays at `0`, the scheduler may be stopped or `SCHEDULER_STALE_RUNNING_RECLAIM_BATCH_SIZE=0`.
2. Check worker health. A growing `reclaimed_pending` count usually means workers are dying faster than they recover.
3. Check `running_without_lease`. Any non-zero value means the database contains pre-lease or partially written `RUNNING` rows. The scheduler now requeues these rows on the next reclaim pass. If the count stays non-zero, reclaim may be disabled or blocked.
4. Check `recovery_exhausted_failed`. Any non-zero value means at least one job was retried until the scheduler stopped requeueing it automatically.
5. Inspect a specific job with `uv run loreley jobs inspect JOB_ID` before retrying it.

## Inspect jobs that exhausted the recovery budget

Preferred path:

```bash
uv run loreley jobs ls --failed-stale
uv run loreley jobs ls --failed-stale --json
```

Fallback SQL:

```bash
psql "$DATABASE_URL" <<'SQL'
SELECT
  id,
  base_commit_hash,
  status,
  recovery_count,
  started_at,
  completed_at,
  last_error
FROM evolution_jobs
WHERE status = 'failed'
  AND recovery_count > 3
  AND (
    lower(coalesce(last_error, '')) LIKE 'lease expired after missing heartbeat;%'
    OR lower(coalesce(last_error, '')) LIKE 'lease metadata missing for running job;%'
  )
ORDER BY completed_at DESC NULLS LAST, created_at DESC;
SQL
```

Replace `3` with your configured `SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS` if you changed the default.

## Retry one stuck job

Preferred path:

```bash
uv run loreley jobs retry REPLACE_JOB_ID
```

Add `--json` for machine-readable output or `--reason "..."` to record why you requeued the job.

This command accepts both:

- `FAILED` jobs
- `RUNNING` jobs whose lease state is `missing` or `stale`

For multiple recovery-exhausted jobs, use:

```bash
uv run loreley jobs retry --failed-stale --limit 10
uv run loreley jobs retry --failed-stale --all
```

When you use `--failed-stale`, you must also provide either `--limit N` or `--all`.

## Retry a failed job manually with SQL

Only do this after you fix the underlying cause, for example a worker host restart loop, bad evaluator environment, or a repository checkout failure.

1. Confirm the scheduler is running.
2. Confirm no worker is still actively executing the job.
3. Requeue the job row:

```bash
psql "$DATABASE_URL" <<'SQL'
UPDATE evolution_jobs
SET
  status = 'pending',
  scheduled_at = now(),
  started_at = NULL,
  completed_at = NULL,
  heartbeat_at = NULL,
  lease_expires_at = NULL,
  run_token = NULL,
  worker_id = NULL,
  recovery_count = 0,
  last_error = 'manual retry after lease recovery',
  result_commit_hash = NULL
WHERE id = 'REPLACE_JOB_ID';
SQL
```

4. Watch the next scheduler tick and confirm the job moves through `PENDING -> QUEUED -> RUNNING`.

## Leave the job failed when

- the root cause is still unknown
- the worker environment is still unstable
- the job is no longer worth retrying

In that case, keep the row in `FAILED`, inspect the worker and scheduler logs under `logs/{experiment_namespace}/`, and create a fresh job only after you understand the failure mode.
