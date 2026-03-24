# Archive Ref Retention Investigation

Date: 2026-03-24

## Conclusion

The issue is real and high impact.

- `map_elites_archive_cells` stores only `commit_hash`; there is no separate durable ref attached to an archived commit.
- The scheduler samples base commits from archive commit hashes, not refs.
- `WorkerRepository.prune_stale_job_branches()` previously deleted stale remote job branches based only on age.
- `MapElitesIngestion._ensure_commit_available()` and worker checkout both fail once a commit can no longer be fetched from `origin`.

That creates a concrete failure mode:

1. A worker publishes a candidate commit on a per-job branch.
2. The commit becomes archived, or is still needed by an unfinished / pending-ingestion job.
3. TTL pruning deletes the job branch anyway.
4. After remote GC or a fresh clone / restart, the commit is no longer fetchable by hash.
5. Scheduler ingestion and future worker checkouts fail on "commit unavailable".

## Evidence Chain

- `loreley/db/models.py`: `MapElitesArchiveCell` persists `commit_hash` only.
- `loreley/core/map_elites/sampler.py`: scheduler samples base commits from archive commit hashes.
- `loreley/core/worker/repository.py`: stale job-branch pruning used only TTL + branch prefix before this fix.
- `loreley/scheduler/ingestion.py`: ingestion calls `require_commit(...)` and fails when the commit cannot be fetched.

## Experiment 1: Baseline Reproduction

Goal: prove that deleting the last remote ref really makes a worker-produced commit unavailable to a fresh clone.

Setup:

- Create a bare remote.
- Create `main`.
- Create one candidate commit reachable only from `evolution/job/demo/job-1`.
- Delete that remote job branch.
- Force remote reflog expiry and `git gc --prune=now`.
- Fresh-clone the remote and test object existence with `git cat-file -e <sha>^{commit}`.

Observed result:

```text
CANDIDATE=2d3588483be507e23c46c68c98528ae340b6e8ff
BEFORE_RC=0
AFTER_RC=128
AFTER_ERR=fatal: Not a valid object name 2d3588483be507e23c46c68c98528ae340b6e8ff^{commit}
```

Interpretation:

- Before deleting the job branch, the candidate commit was present.
- After deleting the branch and running remote GC, a fresh clone could no longer resolve the commit object.
- This matches the reported "commit unavailable" failure path.

## Test-First Regression Work

Added regression coverage in `tests/core/worker/test_repository.py`:

- `test_load_protected_job_branch_state_collects_archive_and_job_refs`
  - specifies which archive / job references must block pruning.
- `test_prune_stale_job_branches_skips_protected_commits`
  - specifies that a stale branch whose head commit is still protected must not be deleted.
- `test_prune_stale_job_branches_preserves_last_ref_for_protected_commit`
  - integration-style proof: after prune + remote GC, a fresh clone can still `require_commit(...)` when the commit is protected.

Initial run before the fix:

```text
uv run pytest -q tests/core/worker/test_repository.py -q
...
FAILED test_load_protected_job_branch_state_collects_archive_and_job_refs
FAILED test_prune_stale_job_branches_skips_protected_commits
```

The failures were expected:

- `_load_protected_job_branch_state()` did not exist.
- `prune_stale_job_branches()` still deleted protected stale branches.

## Implemented Fix

Chosen fix:

- Reuse the existing per-job remote branch as the durable ref while the commit is still needed.
- Do not create an extra archive-ref namespace yet.
- Prevent pruning of a job branch if its head commit is still referenced by:
  - `map_elites_archive_cells.commit_hash`
  - unfinished jobs (`pending`, `queued`, `running`)
  - succeeded-but-not-yet-ingested jobs

Why this route:

- It directly closes the destructive edge that loses the last reachable ref.
- It keeps ref count bounded by the set of still-live job branches.
- It avoids adding a second remote-ref lifecycle before we have evidence that the retention-based approach is insufficient.

Implementation summary:

- `WorkerRepository.prune_stale_job_branches()` now loads protected commit hashes and branch names before deleting anything.
- It resolves each stale branch head and skips deletion when that commit is still protected.
- If protected-ref lookup fails, pruning is skipped instead of risking destructive cleanup.
- This is a forward fix: it prevents new last-ref loss, but it cannot resurrect archive commits that were already garbage-collected before the patch.

## Validation After Fix

Targeted regression suite:

```text
uv run pytest -q tests/core/worker/test_repository.py -q
10 passed
```

Impacted worker/git suite:

```text
uv run pytest -q tests/core/worker/test_repository.py tests/core/worker/test_evolution.py tests/core/test_git.py -q
passed
```

Most important validation:

- `test_prune_stale_job_branches_preserves_last_ref_for_protected_commit`
  - creates a real bare remote,
  - pushes an old job-branch commit,
  - runs `prune_stale_job_branches()`,
  - forces remote GC,
  - fresh-clones the remote,
  - verifies `require_commit(...)` still resolves the protected commit.

That test demonstrates the repaired behavior on the actual failure surface.
