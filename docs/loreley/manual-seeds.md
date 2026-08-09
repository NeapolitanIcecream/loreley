# Manual seeds

Loreley can evaluate user-authored seed commits before agent-generated
evolution begins. A manual seed is a normal, archive-eligible campaign job: it
uses the configured scope gate, evaluator, measurement cache, provenance store,
candidate identity, and archive ingestion. Planning, coding, and trajectory
model calls are skipped.

## Prepare seed commits

Each seed must be one commit whose only parent is
`MAPELITES_EXPERIMENT_ROOT_COMMIT`. Publish every seed under a stable full ref
on `WORKER_REPO_REMOTE_URL`. The import verifies that the remote ref resolves
to the declared commit; workers fetch that same ref and fail if it drifts.

For example, publish refs such as:

```text
refs/heads/loreley-seeds/allocation
refs/heads/loreley-seeds/branch-layout
```

Only full `refs/heads/...` and lightweight `refs/tags/...` refs are accepted.
Do not delete or move them while the campaign can still resume.

## Manifest

Create a YAML or JSON manifest:

```yaml
schema_version: 1
seeds:
  - key: allocation
    commit: 0123456789abcdef0123456789abcdef01234567
    remote_ref: refs/heads/loreley-seeds/allocation
    summary: Reduce allocations on the hot path.
    island_id: main
    tags: [allocation, hot-path]
    metadata:
      design: reuse a bounded scratch buffer

  - key: branch-layout
    commit: 89abcdef0123456789abcdef0123456789abcdef
    remote_ref: refs/heads/loreley-seeds/branch-layout
    summary: Reorder the common branch.
```

`key` is the stable logical identity used for idempotent import. `commit`,
`remote_ref`, and `summary` are required. `goal`, `island_id`, `tags`, and
JSON-compatible `metadata` are optional. Metadata is persisted and can appear
in operator APIs, so it must not contain credentials or local paths.

## Import and run

Set the normal experiment, repository, database, campaign endpoint, and worker
settings, then stop the scheduler and run:

```bash
uv run loreley seeds import seeds.yaml
```

Import is a pre-start operation. Loreley obtains the experiment scheduler lock,
validates or migrates the database, verifies the instance marker and root,
checks every remote ref, and inserts the complete manifest atomically. Repeating
the same manifest returns the existing jobs. Reusing a key with a different
definition fails instead of changing prior evidence.

Imported jobs start as `STAGED`. Staged jobs count toward
`SCHEDULER_MAX_TOTAL_JOBS`, but not toward the algorithm's unfinished-job limit
`U`. On each scheduler tick, Loreley promotes staged seeds in manifest order up
to the available `U` capacity and remaining physical-job endpoint. This permits,
for example, eight seeds with `U=4` without changing search concurrency.

Start the scheduler and worker normally after the import. A failed terminal seed
is preserved and is not silently retried or replaced.

Workers evaluate the pinned seed in a detached checkout. Loreley records the
candidate as already available through its verified seed ref; it does not
create or claim a worker-generated publication branch.

## Scope

This interface intentionally covers archive-eligible initial seeds. It is not a
generic validation-only submission API. A seed must:

- be a previously authored, remotely fetchable commit;
- be a direct child of the configured experiment root;
- target a configured island;
- introduce a commit not already registered as a campaign candidate; and
- pass the same committed-diff scope gate and evaluator as generated candidates.

The direct-child rule keeps the initial lineage explicit. Later candidates can
inherit and recombine seed ideas through the normal sampler.
