# ADR 0052: Multi-island Pareto archives and native worker processes

## Status

Accepted.

## Context

Loreley's data model carries `island_id` through jobs, commit metadata, PCA
history, and archive rows, but the scheduler bootstraps and samples only one
default island. There is no deliberate cross-island information flow. The
manager also maintains a global `dict[commit, island]` that is never used to
answer a product query and cannot represent a commit retained by two islands.

Evaluators may return several metrics, but archive admission reads one configured
fitness metric, applies a sign, and delegates to `ribs.GridArchive`. Every
behavior cell therefore retains one scalar winner. Multiple metrics in the
database or UI do not make the search multi-objective.

The production worker command starts one single-threaded Dramatiq `Worker`.
Running several workers requires an external process manager or several manual
commands. The circle-packing example contains its own process supervisor, while
Dramatiq 2.0 already ships a multi-process master with spawn, signal propagation,
startup synchronization, and child-failure propagation.

The required design must remain small:

- one database is still one experiment;
- behavior descriptors and optimization objectives remain different concepts;
- no scoring DSL, migration service, per-worker queue, or general process
  framework is justified;
- old configuration is converted once instead of parsed forever.

## Decision

### Objective contract

`MAPELITES_OBJECTIVES` is an ordered JSON list:

```json
[
  {"name": "throughput", "direction": "max"},
  {"name": "p99_latency_ms", "direction": "min"}
]
```

The list is the authoritative objective name, order, and direction contract.
Every evaluator result admitted to the archive must contain every configured
metric, every value must be finite, and persisted metric direction must match the
contract. Missing metrics and direction conflicts are explicit rejection
reasons; they are not replaced by a fitness floor.

The first objective is the **primary operational objective**. Campaign baseline
reporting and the optional best-result branch may use it because those artifacts
must name one value. Archive admission, retention, migration, and sampling may
not use it as a scalar proxy for the full contract.

The ordered contract and its fingerprint are persisted in each island snapshot.
Scheduler startup eagerly validates every configured island before constructing
the dispatcher, and archive read paths apply the same check. A mismatch fails
rather than interpreting stored vectors under a new order.

### Pareto behavior archive

Loreley owns a small Pareto grid instead of adapting the scalar
`ribs.GridArchive`.

- Behavior measures still locate one fixed grid cell.
- Each cell stores at most `MAPELITES_PARETO_FRONT_MAX_SIZE` entries.
- Objective values are normalized internally so every dimension is maximized;
  raw `Metric` rows remain the evaluation source of truth.
- A candidate enters a cell only if no retained member dominates it.
- It removes members it dominates.
- Equivalent objective vectors keep one deterministic representative.
- If the non-dominated set exceeds capacity, standard crowding distance keeps
  boundary trade-offs and the most isolated interior points. Crowding is only a
  bounded-front diversity rule, not an optimization scalar.
- Base sampling first chooses a behavior cell, then chooses one Pareto member,
  so cells with larger fronts do not silently receive more probability.

Archive persistence uses one row per elite with primary key
`(island_id, cell_index, commit_hash)`, a uniqueness constraint on
`(island_id, commit_hash)`, and the complete ordered objective vector. The
behavior measures are the stored solution coordinates, so the old duplicate
`solution` payload is removed. A normal insertion replaces the affected cell
front atomically. A PCA refit replaces the island archive. This removes the old
one-row-per-cell assumption and the diff/upsert machinery built around it.

The manager-level `commit_to_island` map is deleted. Per-island archive indexes
are the in-memory truth, and the database key is the persisted truth.

### Island scheduling and migration

`MAPELITES_ISLANDS` is an ordered, non-empty list of unique island IDs and
replaces the separate default-island setting. The first ID is the CLI/API
default when an island is not specified.

Every configured island has independent PCA history, projection, and Pareto
archive state. A PCA refit is fail-closed: every retained elite must have a
complete source vector of the expected dimensionality, and a failed rebuild
leaves the prior projection and archive intact. Seed and normal scheduling use
a global round-robin position derived from the database-wide job count,
preserving global capacity and job budget while avoiding a first-island restart
bias. Normal scheduling starts only after every configured island has a usable
archive, so a faster cold start cannot consume another island's fixed budget.

Every `MAPELITES_MIGRATION_INTERVAL_JOBS` jobs in each island, the target job
receives one non-duplicate elite from another ready island as an inspiration
when at least one inspiration slot is configured. A per-island cadence prevents
the global interval from aliasing with round-robin scheduling and repeatedly
targeting only one island. The target base remains local and the resulting
candidate remains in the target island. The job records the donor island and
migrant commit. Setting the interval to zero disables migration.

This is the smallest island-model gene flow for Loreley's variation mechanism:
the coding agent synthesizes the target base and inspiration histories. Directly
copying a donor into the target archive would require target-space reprojection
and extra archive mutation without creating a new candidate. It is not included.

### Worker processes

`loreley worker --processes N` is the supported local worker-fleet command.

- `N=1` uses the existing programmatic worker path.
- `N>1` runs preflight and schema preparation once, then delegates lifecycle to
  Dramatiq's native master with `--processes N --threads 1`. It requests spawn
  when multiprocessing has not been initialized and otherwise reuses the
  already initialized context.
- Every child imports one narrow Loreley bootstrap that loads settings,
  configures process-unique logging, creates the experiment broker, and
  registers the actor.
- Multi-process mode forces PID-plus-random base-checkout paths. Per-job worktrees,
  database leases, run-token fencing, and the experiment queue keep task state
  isolated.
- A child failure stops siblings and produces a non-zero command exit through
  Dramatiq's existing behavior. Loreley does not add an automatic restart
  policy.
- The scheduler remains a separate command and the experiment keeps one shared
  queue. No per-worker queues are added.

The circle-packing example must use this command and delete its private worker
supervisor.

## Migration

Schema version 15 performs one native migration:

1. create the multi-elite archive table shape;
2. remove the incomplete scalar archive and mark successful candidate jobs for
   one-time reingestion, backfilling a missing result commit from the durable
   candidate commit when necessary and rebuilding fronts from all available
   successful candidates rather than only the previous scalar winners;
3. add migration provenance fields to evolution jobs;
4. persist the objective contract in island metadata; and
5. replace the old archive table.

`tools/migrate_v15_config.py` converts an environment file:

- `MAPELITES_DEFAULT_ISLAND_ID` to `MAPELITES_ISLANDS`, preserving the legacy
  `main` fallback when the old value is blank;
- `MAPELITES_FITNESS_METRIC` and
  `MAPELITES_FITNESS_HIGHER_IS_BETTER` to `MAPELITES_OBJECTIVES`;
- `MAPELITES_ARCHIVE_EPSILON` to `MAPELITES_PARETO_EPSILON`;
- removes scalar-archive-only floor, learning-rate, threshold, and QD-score
  settings.

Runtime code does not retain dual old/new configuration parsing.

## Consequences

- The archive preserves non-convex objective trade-offs within every behavior
  niche and remains bounded.
- Operational "best" views are explicitly primary-objective projections rather
  than claims about a unique multi-objective winner.
- Several islands consume one global job budget fairly and exchange search
  material without another service.
- The custom archive surface is smaller than the old `ribs` adapter plus
  multi-elite side storage, and the `ribs` dependency can be removed.
- A larger Pareto front increases archive rows and planning choices linearly
  with the configured per-cell bound.
- Changing objective order, direction, or names requires an explicit archive
  migration/rebuild.
- Local multi-process mode increases database connection and checkout disk
  budgets. Operations documentation must describe both.

## Rejected alternatives

- **Several metrics plus one scalar archive:** records data but does not perform
  multi-objective selection.
- **Weighted sums or one scalar island per objective:** miss non-convex fronts
  and conflate search populations with objectives.
- **One global Pareto front:** loses behavior-niche quality diversity.
- **Unbounded fronts:** eventually make storage and sampling costs unbounded.
- **Direct donor insertion:** adds reprojection and archive mutation without a
  new evaluated variation.
- **Dramatiq worker threads:** share mutable worker dependencies and do not give
  process isolation.
- **A Loreley-owned generic supervisor:** duplicates lifecycle code already
  maintained by Dramatiq.
- **Per-worker queues:** add routing configuration without improving the current
  shared-queue lease model.
