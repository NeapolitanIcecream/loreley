# Unreleased

These notes cover changes merged after `v0.7.7-alpha`.

## Highlights

- Automatic recovery for `RUNNING` jobs that stop heartbeating. The scheduler now requeues stale jobs until they exhaust a recovery budget, then marks them `FAILED`.
- New job repair CLI. Use `uv run loreley jobs inspect`, `uv run loreley jobs ls`, and `uv run loreley jobs retry` to inspect lease state and requeue stuck jobs.
- Better lease visibility in `uv run loreley status`. The status payload now includes a `job_leases` section with active, stale, malformed, and recovery-exhausted counts.
- Safer worker fencing. Worker success, failure, and candidate-publication writes now use `run_token`-scoped lease ownership checks so stale workers cannot overwrite a newer attempt.
- Shared repo lock coordination. When the scheduler and worker share `WORKER_REPO_WORKTREE`, their base-repo mutation paths use the same cross-process lock.

## Upgrade Notes

### Database schema reset required

Loreley still does not ship migrations. This change adds lease-recovery columns and indexes to `evolution_jobs` and bumps the instance schema version.

If you are upgrading an existing development database created before this change, reset it before running `status`, `scheduler`, or `worker`:

```bash
uv run loreley reset-db --yes
```

If you skip the reset, commands may fail with database errors such as a missing `lease_expires_at` column.

### New configuration knobs

Review these settings when you tune recovery behavior:

- `SCHEDULER_STALE_RUNNING_RECLAIM_BATCH_SIZE`
- `SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS`
- `WORKER_JOB_LEASE_TTL_SECONDS`
- `WORKER_JOB_HEARTBEAT_INTERVAL_SECONDS`

See [Configuration](../loreley/config.md) for details.

### New operational docs

Use these pages when operating the new lease-recovery flow:

- [Managing jobs](../script/jobs.md)
- [Job lease recovery](../script/job_leases.md)
- [Status](../script/status.md)
