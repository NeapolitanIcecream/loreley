# Worker Candidate Atomicity Report

Date: 2026-03-24

## Scope

This report verifies and fixes the worker failure mode where a candidate commit
can be pushed to the remote repository before Loreley durably records any
pointer to that candidate on the `EvolutionJob` row.

## Reproduction Before The Fix

### Hypothesis

If `EvolutionWorker.run()` pushes a branch and a later step fails
(`persist_success`, evaluator, or any post-push step), the system can end up in
this state:

- the remote branch already exists,
- the job is marked `FAILED`,
- the job row has no durable candidate pointer,
- the candidate commit becomes operationally "lost" to Loreley.

### Reproduction Tests

The defect was reproduced with two regression tests added to
`tests/core/worker/test_evolution.py`:

- `test_run_records_candidate_before_push_when_persist_success_fails_gh_candidate_orphan`
- `test_run_keeps_candidate_metadata_when_post_push_persistence_fails_gh_candidate_orphan`

### Command

```bash
uv run pytest -q tests/core/worker/test_evolution.py -k 'candidate_orphan'
```

### Observed Failure Before The Fix

Before the fix, the test run failed with two assertions:

1. `repo.push_branch` happened, but no `store.record_candidate[...]` event
   happened before it.
2. `store.recorded_candidates` stayed empty even though the worker had already
   created and pushed `candidate123`.

The failure trace showed this exact ordering:

- `repo.commit`
- `repo.push_branch`
- `store.persist_success`
- `store.mark_job_failed`

That confirmed the issue is real: a post-push failure leaves the system without
any durable candidate pointer.

### Why The Impact Is Major

This is not just an observability gap:

- Loreley loses the only structured link between the failed job and the pushed
  candidate.
- Operators cannot reliably recover, inspect, or re-ingest the pushed commit
  from normal job metadata.
- A transient database failure after push can permanently drop a valid
  candidate from Loreley's search history.

## Fix Design

The fix uses a minimal state machine on `EvolutionJob`:

- add `candidate_commit_hash`
- add `candidate_branch_name`
- add `candidate_published_at`

The worker flow is now:

1. create the local commit
2. record candidate metadata with `published=False`
3. push the branch
4. record candidate metadata again with `published=True`
5. run evaluation
6. persist success

This guarantees Loreley never publishes a remote candidate before it has at
least recorded a durable pointer to that candidate.

## Implementation Summary

- Added candidate metadata columns to `EvolutionJob`.
- Added `EvolutionJobStore.record_candidate_commit(...)`.
- Reset stale candidate metadata in `start_job(...)`.
- Split worker commit handling into local commit creation plus explicit
  publication.
- Exposed candidate metadata in job detail API / UI.
- Bumped `INSTANCE_SCHEMA_VERSION` from `3` to `4`.

## Verification After The Fix

### Targeted Regression Verification

```bash
uv run pytest -q tests/core/worker/test_evolution.py -k 'candidate_orphan'
```

Observed result:

```text
2 passed, 2 deselected in 0.31s
```

### Job Store Verification

```bash
uv run pytest -q tests/core/worker/test_job_store.py -k 'candidate or start_job'
```

Observed result:

```text
3 passed, 2 deselected in 0.25s
```

### Full Regression Verification

```bash
uv run pytest -q
```

Observed result:

```text
321 passed in 6.49s
```

## Operational Note

Loreley does not ship migrations; it relies on schema versioning plus
`reset-db` for incompatible ORM changes. Because this fix adds new columns, the
instance schema version was bumped to `4`. Existing development databases must
be reset before running the updated worker/scheduler:

```bash
uv run loreley reset-db --yes
```
