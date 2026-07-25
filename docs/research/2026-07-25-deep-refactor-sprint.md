# Loreley deep refactor sprint

This is the single checkpoint for the 2026-07-25 research, refactor, and
development sprint. It records decisions only when evidence changes what should
be built or verified.

## Research contract

### Target outcome and success evidence

Deliver a smaller, coherent Loreley architecture that:

1. runs several configured MAP-Elites islands as independent search
   populations and schedules work fairly across them;
2. moves useful search material between islands through an explicit, observable
   migration rule;
3. starts and supervises multiple isolated worker processes from one supported
   Loreley command;
4. uses every configured evaluation objective in archive admission and
   retention, preserving non-dominated trade-offs instead of reducing them to a
   hidden scalar; and
5. removes or folds technical debt encountered on those paths when the change
   lowers total concepts or eliminates a demonstrated failure mode.

Completion requires focused contract tests, the full test suite, a native
database migration test, command-level worker-supervision tests, a clean
Cremona comparison against the repository baseline, and a requirement-by-
requirement adversarial audit. Documentation and configuration examples must
describe the behavior that the tests exercise.

### Definitions and scope

- An **island** is an independently persisted behavior archive with its own PCA
  state and a scheduler-visible work stream. Merely accepting an `island_id`
  argument or storing rows under different IDs is not multi-island support.
- **Migration** is deliberate cross-island parent or inspiration flow recorded
  on the scheduled job. Accidental sharing through a global lookup is not
  migration.
- **Multi-objective optimization** means Pareto dominance over all configured
  objectives. Evaluating several metrics while retaining one scalar winner per
  cell is not sufficient.
- Each behavior-space cell may retain a bounded Pareto front. A deterministic
  diversity rule may prune a front at its configured capacity, but weighted-sum
  scalarization may not decide admission.
- The first configured objective may remain the named primary operational
  metric for baseline reporting and a single convenience branch. It must not
  become the archive's selection proxy.
- Worker parallelism means one foreground command owns N child worker
  processes, gives each an isolated repository base, propagates shutdown, and
  returns non-zero if a child fails unexpectedly. Increasing Dramatiq threads
  inside one process is out of scope because evaluator and agent execution
  require process isolation.
- One Postgres database remains one experiment. Multi-island support does not
  reintroduce multi-tenant experiment multiplexing.
- Existing user-owned `reports/` and ignored `.codex-workflows/` artifacts are
  outside the change set.

### Invariants

- A job and its resulting candidate have one target island.
- A commit may be retained by more than one island; archive membership is not a
  one-to-one commit property.
- All objective values used for dominance are finite, have an explicit
  direction, and appear in a stable configured order.
- A candidate missing any configured objective is not silently assigned a
  floor and cannot enter the archive.
- No cell contains two entries for the same commit.
- Every retained entry is non-dominated within its cell after capacity pruning.
- Scheduler capacity and the database-wide maximum job count remain global.
- A supervised worker child has a stable instance ID and a distinct repository
  base path.
- Shutdown does not leave a healthy sibling running after the supervisor exits.

### Insufficient outcomes

- Exposing more CLI flags without scheduling more than the default island.
- Treating behavior descriptors as optimization objectives.
- Keeping multiple evaluator metrics only for display.
- Running several worker threads in one mutable checkout.
- Copying the large circle-packing example supervisor into production without a
  reusable lifecycle boundary.
- Preserving old configuration branches indefinitely. Legacy configuration is
  handled by a one-time migration tool and then removed from runtime code.
- Lowering static-analysis thresholds, refreshing the debt baseline to hide a
  regression, or refactoring unrelated UI hotspots because they rank highly.

### Epistemic stance

Established by the current repository:

- `MapElitesManager` can lazily create state for arbitrary island IDs, and the
  database keys PCA/archive state by island.
- Scheduler bootstrap, root ingestion, status defaults, and normal scheduling
  choose one default island.
- `GridArchive` retains one scalar `objective` per behavior cell.
- Evaluators already return several named metrics with direction metadata.
- `run_worker()` deliberately creates one single-threaded Dramatiq worker.
- The circle-packing example contains a bespoke multi-process supervisor that
  is not available through the product CLI.
- The pre-change test baseline is 774 passed and 4 skipped.

Working hypotheses to test:

- A small repository-owned Pareto grid is simpler than adapting scalar
  `ribs.GridArchive` semantics around a parallel multi-elite store.
- Fair round-robin island scheduling plus bounded periodic cross-island
  inspirations provides an understandable island model without a separate
  migration service.
- A parent supervisor around the existing single-worker entrypoint preserves
  process and worktree isolation with less risk than changing Dramatiq worker
  threading.

Allowed terminal states are success, a concrete external blocker, or a bounded
inconclusive result that names an outcome-strength gap. A partially working
feature is not success.

### Boundaries and source policy

- Work only in this repository and local disposable test artifacts. Do not
  operate remote experiment hosts, production databases, Redis instances, or
  user processes.
- Network sources may provide current primary documentation for dependencies
  and standard algorithms. Repository code, tests, Git history, and ADRs remain
  authoritative for Loreley behavior.
- Keep KISS/YAGNI as decision rules: introduce no service, queue, table, or
  compatibility layer unless an invariant or measured hot path requires it.
- Use a native one-step schema migration and a one-time environment migration
  tool. Do not keep dual old/new configuration parsing in normal startup.
- Preserve unrelated user changes and the untracked `reports/` directory.

### Failure model and audit checks

- **Pseudo-islands:** only one island receives seed or normal jobs. Audit job
  distribution, independent state restoration, and recorded migration.
- **Cross-island corruption:** a global commit-to-island map evicts membership
  from another island. Audit the same commit retained in two islands.
- **Scalarization in disguise:** one objective decides admission or truncation.
  Audit opposing two-objective candidates in one cell.
- **Unbounded fronts:** non-dominated candidates grow without limit. Audit
  deterministic capacity pruning and boundary preservation.
- **Direction mistakes:** minimization metrics are compared as maximization.
  Audit mixed directions and migration/restoration.
- **Restart drift:** persisted fronts restore differently. Audit round-trip
  equality and objective-order validation.
- **Unfair scheduling:** the first non-empty island consumes all capacity. Audit
  multiple ticks and partially empty islands.
- **Unsafe parallelism:** workers share a base checkout, orphan siblings, or
  report success after a child crash. Audit paths, signals, and exit codes.
- **Migration contamination:** legacy archive rows cannot satisfy the new
  objective contract. Audit the version-14 to target migration and the one-time
  configuration migration.
- **Observability noise:** per-candidate or hot-loop logs explode with
  concurrency. Prefer bounded supervisor lifecycle and sampled aggregate
  signals.

## Initial checkpoint (superseded)

This section records the pre-implementation decision state. Later dated
checkpoints are authoritative for the current sprint state.

Current bottleneck and decisive next evidence:

- Confirm the smallest Pareto archive and scheduler boundaries that satisfy the
  invariants without preserving scalar `GridArchive` concepts.
- Re-run Cremona after excluding ignored agent workspaces; the first full scan
  was contaminated and is not an admissible baseline.

Approach registry:

| Family | Thesis | Evidence | Exact gap | Status | Reopening condition |
| --- | --- | --- | --- | --- | --- |
| Scalar archive plus metric display | Preserve `GridArchive` and expose more metrics | Current implementation already does most of this | Does not preserve Pareto trade-offs | Retired | Only if the multi-objective requirement is withdrawn |
| One weighted archive per objective/island | Approximate trade-offs through several scalar searches | Small change to current scalar archive | Misses non-convex fronts and conflates islands with objectives | Retired | Only for an explicitly approximate mode |
| Bounded Pareto front per behavior cell | Use dominance for admission and crowding only for capacity | Matches the exact multi-objective requirement and existing behavior grid | Must prove persistence, rebuild, and sampling stay simple | Active | Retire if implementation needs more concepts than a direct store |
| In-process Dramatiq threads | Raise `worker_threads` | Small surface change | Shares process state and does not provide checkout/process isolation | Retired | Only if all worker dependencies become thread-safe |
| Parent process supervisor | Own N existing single-worker children | Existing entrypoint is already a sound single-child boundary; example proves lifecycle demand | Must extract a small production lifecycle and test signal/failure behavior | Active | Retire on a demonstrated platform/process limitation |
| Independent islands without migration | Round-robin several archives | Storage skeleton already supports separate IDs | Fails explicit cross-island information flow | Retired | Only if migration is removed from the definition |
| Scheduler-selected migrant inspirations | Keep target archive independent; periodically add donor elites to jobs | Reuses existing inspiration contract and records lineage on jobs | Must prove fairness and avoid duplicate/self migration | Active | Retire if planning cannot load donor commit context |

Candidate result and audit status:

- No candidate implementation yet. The baseline suite passes.
- The initial Cremona result is rejected because it scanned ignored
  `.codex-workflows/` environments and reported 693 unrelated new findings.

Losses and retired directions:

- Treating current `island_id` plumbing as feature completion fails scheduler
  and migration evidence.
- Treating multiple `Metric` rows as multi-objective optimization fails archive
  admission evidence.
- The first Cremona scan is a measurement loss, not repository regression.

Next action and why:

- Finish independent algorithm, architecture, and concurrency audits; then
  freeze the ADR and write contract tests before changing production behavior.

## 2026-07-25 - Baseline and measurement correction

Approach family: evidence baseline.

Action: ran the complete suite under branch coverage and ran Cremona against
the repository.

Evidence, including negative or ambiguous results:

- `uv run coverage run -m pytest -q`: 774 passed, 4 skipped.
- Cremona scanned 930 files instead of the configured product scope because
  ignored `.codex-workflows/` trees were not explicitly excluded. The reported
  693 new findings are in disposable plugin copies and do not reflect Loreley.

Claim or decision changed:

- The prior `corroding` verdict is invalid. Cremona needs an explicit exclusion
  before it can serve as a regression gate.

Audit implication:

- Future audit reports must show the configured project scope and must not count
  ignored agent workspaces.

Registry update:

- No product architecture route changed.

Next action:

- Correct the audit scope, re-run, and use the clean report only to rank
  refactors on the feature path.

## 2026-07-25 - Architecture and implementation checkpoint

Approach family: bounded Pareto fronts, fair configured islands, migration
inspirations, and Dramatiq-native worker processes.

Action:

- Replaced the scalar archive adapter with a repository-owned bounded Pareto
  grid and removed the `ribs` dependency.
- Added a strict ordered objective contract with explicit `max`/`min`
  directions. Missing, duplicate, non-finite, boolean-valued, or
  direction-conflicting objectives are rejected instead of receiving a floor.
- Added configured islands, global fair seed/normal scheduling, independent PCA
  state, and periodic donor inspirations with persisted source lineage.
- Added `loreley worker --processes N`, using Dramatiq's spawn-based native
  master with one thread per child, PID-unique logs, and forced randomized base
  clones.
- Replaced the circle-packing example's private process supervisor with the
  product command and configured two objectives and two islands.
- Removed scalar QD score, fitness/solution aliases, and write-only planning
  archive context from active API, UI, worker, and scheduler paths.
- Added schema version 15 plus a one-time environment migration script. Runtime
  code does not parse old scalar configuration.

Evidence:

- Pareto contract and archive tests cover opposing non-dominated candidates,
  dominated rejection, mixed directions, equivalence, deterministic bounded
  crowding, batch insertion, persistence restoration, and invalid vectors.
- Scheduler tests cover fair multi-island batches, configured empty islands,
  per-island seed deficits, global capacity, and recorded migration provenance.
- Worker command tests cover native master arguments, one-thread process
  isolation, one-time schema preparation, child environment isolation, and
  PID-unique log paths.
- API/UI tests cover multi-member cell pagination, distinct occupied-cell
  counts, retained-elite counts, configured-but-empty islands, explicit primary
  projections, and front-aware heatmap aggregation.
- `uv run pytest -q` passed with `809 passed, 4 skipped`; the skipped tests
  required an explicit PostgreSQL test DSN.
- A disposable PostgreSQL 18 instance then exposed a real migration failure:
  PostgreSQL could not infer the fingerprint parameter type inside
  `jsonb_build_object`. Explicit `TEXT` casting fixed it.
- With the Postgres integration gate enabled, coverage execution passed with
  `813 passed` and no skips. The v5-to-v15 migration chain, schema validation,
  and idempotent second run all passed.
- Changed and new Python files pass Ruff's fatal/unused checks
  (`--select F,E9`). The repository has no committed Ruff configuration, so a
  whole-repository latest-Ruff default scan is not used as a newly invented
  acceptance threshold.

Decision changed:

- Migrant inspirations remain the smallest adequate island-model gene flow;
  direct archive copying is still rejected because target-island PCA spaces
  differ.
- The first objective remains an explicitly labelled operational projection
  for baselines, status, and one convenience branch. It is not used for Pareto
  admission, retention, migration, or sampling.
- Real PostgreSQL execution is a mandatory migration gate; SQL-string unit
  substitutes are insufficient.

Current bottleneck:

- The first post-change Cremona comparison reports 13 resolved hotspots but
  also 7 new and 6 worsened hotspots. The sprint is therefore not ready to
  commit. The active cycle is extracting cohesive helpers in the changed
  Pareto, snapshot, scheduling, worker, migration-tool, and UI paths until
  `--fail-on-regression` passes without updating the stored baseline.

## 2026-07-25 - Final adversarial and release checkpoint

Approach family: bounded Pareto fronts, independent migrating islands,
Dramatiq-native worker processes, and one-time v15 migration.

Action:

- Completed the structural-debt loop without changing
  `quality/refactor-baseline.json`.
- Ran a separate read-only adversarial audit after the first green suite and
  treated every counterexample as a rejected release candidate until fixed.
- Eagerly validate every configured island's objective-contract fingerprint
  before the scheduler constructs its dispatcher. API archive reads apply the
  same validation.
- Made PCA refit/rebuild fail closed: incomplete retained source vectors abort
  the refit and restore the prior reducer, projection, history, and archive.
- Made reingestion archive replacement atomic, including duplicate commit
  replacement and candidates that no longer survive Pareto admission.
- Backfill successful v15 jobs from a durable candidate commit when their
  result hash is absent, then mark them for Pareto reingestion.
- Made migration genuinely optional: zero inspiration slots suppress donor
  lineage and `MAPELITES_MIGRATION_INTERVAL_JOBS=0` disables gene flow.
- Made worker base paths process-unique even with a one-character random
  suffix, fixed direction-only and inline-comment dotenv migration, and removed
  the archive's redundant `(island_id, commit_hash)` index already supplied by
  its uniqueness constraint.

Adversarial losses and corrections:

- The initial multi-process path had only 16 possible base paths when the
  random suffix length was one. PID plus random suffix replaced that design.
- The initial v15 converter lost a direction-only legacy override and treated
  unquoted inline comments as values. Dynamic replacement anchors and
  quote-aware comment parsing replaced it.
- The initial PCA rebuild filtered retained elites with missing vectors,
  allowing silent archive loss. Complete-vector validation and rollback
  replaced it.
- The initial API and completed-budget scheduler paths could interpret retained
  objective vectors under a same-length but changed contract. Read-time and
  pre-dispatch validation replaced it.
- The initial duplicate-ingest path could detach the old commit before a
  non-admission result, diverging memory and persistence. Full atomic archive
  replacement now covers reingestion.
- The first final Cremona attempt found one new hotspot in the migration tool.
  The release candidate remained rejected until quote-state parsing was split
  into a narrow transition helper and the unchanged baseline passed.

Final evidence:

- `LORELEY_POSTGRES_TEST_DSN=... uv run coverage run -m pytest -q`:
  **866 passed**, no skips, against a disposable PostgreSQL instance.
- The PostgreSQL v15 migration tests cover the real v5-to-v15 chain, schema
  validation, candidate-hash backfill, and an idempotent second migration run.
- `uv lock --check`, `git diff --check`, changed/new Python Ruff
  `--select F,E9`, `loreley worker --help`, and `mkdocs build --strict` pass.
- Cremona reports signal health `full`, **0 new**, **0 worsened**, and
  **22 resolved** hotspots against the stored baseline. The repository verdict
  remains `strained` because pre-existing debt outside this sprint remains; the
  current scope does not regress it.
- The independent final audit reports **0 blocker** and **0 major** findings
  across Pareto admission, island independence/migration, one-command worker
  processes, one-time configuration conversion, and PostgreSQL v15 rebuild.

Final decisions:

- PostgreSQL remains the only supported product database and migration dialect;
  adding SQLite migration compatibility would create unsupported runtime debt.
- The first objective remains only an explicitly labelled operational
  projection. No scalar QD score or scalar admission fallback returns.
- Existing unrelated Cremona routing items remain a future refactor queue, not
  a reason to expand this bounded feature sprint.

Stop condition:

- Feature invariants, real-database migration, failure atomicity, documentation,
  tests, adversarial review, and the no-regression structural gate all pass.
  This sprint is ready to commit.
