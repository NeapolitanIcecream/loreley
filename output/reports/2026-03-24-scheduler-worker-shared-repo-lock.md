# Scheduler/Worker Shared Repo Lock Report

Date: 2026-03-24

## Summary

The race is real and high impact.

- `EvolutionScheduler` defaults `repo_root` to `SCHEDULER_REPO_ROOT or WORKER_REPO_WORKTREE`, so the scheduler and worker can target the same base clone.
- `WorkerRepository` documents and implements a cross-process file lock for base-clone mutations.
- Scheduler startup, ingestion, and best-fitness branch update previously called `require_commit(...)` and `git branch -f ...` on the shared repo without acquiring that same lock.

That meant the scheduler could fetch or update refs while the worker was already inside its protected clone/fetch/worktree critical section. The failure modes are severe because they hit:

- scheduler startup root-commit resolution,
- result ingestion for completed jobs,
- best-fitness branch publication at bounded-run completion.

## Evidence

### Static confirmation

The unfixed behavior was confirmed from these call paths:

- `loreley/scheduler/main.py`
  - `_resolve_repo_root()` fell back to `WORKER_REPO_WORKTREE`.
  - `_startup_scan_and_validate_repo_state_approval()` called `require_commit(...)` directly.
  - `_create_best_fitness_branch()` called `require_commit(...)` and `git branch -f ...` directly.
- `loreley/scheduler/ingestion.py`
  - `_ensure_commit_available()` called `require_commit(...)` directly.
- `loreley/core/worker/repository.py`
  - `_repo_lock()` protected base clone mutation paths.
- `docs/loreley/core/worker/repository.md`
  - explicitly states clone/fetch/worktree bookkeeping relies on a cross-process file lock.

### Reproduction experiment

Executable regression specs were added in `tests/scheduler/test_shared_repo_locking.py`.

Experiment design:

1. A child process acquires the worker-style repo lock file for 0.5 seconds.
2. A scheduler code path is invoked against the same repo root.
3. The test measures whether the scheduler blocks on that lock before entering its Git mutation path.

Why this reproduces the bug:

- If the scheduler does not share the worker lock, it enters immediately.
- If it shares the lock, it must wait until the child process releases it.

Repair-before baseline command:

```bash
uv run pytest tests/scheduler/test_shared_repo_locking.py -q
```

Repair-before result:

- Exit code: `1`
- Failed tests: `3`
- Observed elapsed times while the worker lock was still held:
  - ingestion path: `7.33e-05s`
  - startup root scan path: `0.00198s`
  - best-fitness branch path: `0.000454s`
- Required minimum wait threshold: `0.4s`

Interpretation:

- All three scheduler paths bypassed the worker lock and entered immediately.
- This is enough to create real Git races whenever the worker is mutating the shared base clone.

## Repair

Implemented fix:

1. Added `loreley/core/repo_lock.py` as the shared cross-process repo-lock utility.
2. Switched `WorkerRepository` to reuse that shared implementation instead of keeping a private copy.
3. Wrapped scheduler Git mutation entry points with the shared repo lock:
   - scheduler startup root-commit `require_commit(...)`
   - scheduler ingestion `require_commit(...)`
   - scheduler best-fitness `require_commit(...)` + `git branch -f ...`
4. Updated docs to state that shared `SCHEDULER_REPO_ROOT == WORKER_REPO_WORKTREE` is coordinated by the same repo lock, and that the scheduler repo must be writable.

## Verification

Post-repair commands and results:

```bash
uv run pytest tests/scheduler/test_shared_repo_locking.py -q
```

- Exit code: `0`
- Result: `3 passed`

```bash
uv run pytest tests/scheduler -q
```

- Exit code: `0`
- Result: `34 passed`

```bash
uv run pytest tests/core/test_git.py tests/core/worker/test_repository.py -q
```

- Exit code: `0`
- Result: `12 passed`

Additional targeted compatibility check:

```bash
uv run pytest tests/core/worker/test_repository.py tests/scheduler/test_ingestion_resilience.py tests/scheduler/test_startup_approval.py -q
```

- Exit code: `0`
- Result: `31 passed`

## Conclusion

The issue was confirmed, reproduced, fixed, and verified.

The repaired behavior now guarantees that when the scheduler and worker share the same base repo, scheduler-side Git mutations serialize on the same cross-process lock that already protected worker clone/fetch/worktree bookkeeping.
