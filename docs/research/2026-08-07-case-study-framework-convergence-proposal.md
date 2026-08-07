# Framework Convergence Proposal After Three Case Studies

Date: 2026-08-07

Status: implementation handoff. This proposal contains only changes with a
defined implementation and acceptance path. Open research questions are listed
separately in
[the deferred research register](2026-08-07-case-study-deferred-research.md).

## Task

Finish moving the remaining generic improvements learned from the three case
studies into Loreley so future experiments need target adapters, not a second
orchestration system. Preserve existing databases and one-shot evaluator
plugins. Do not rerun a case study as part of this work.

Completion requires framework code, migrations, configuration, tests, and user
documentation. A feature present only in `tools/*_experiment` does not satisfy
completion.

## Scope

The framework should own:

- restart-stable sampling and duplicate-recipe provenance;
- exact-source and evaluator-relevant candidate identity;
- safe reuse of expensive measurements;
- evaluator concurrency independent of search and model-worker concurrency;
- identity-aware progress and stopping;
- phase-specific model configuration, preflight, and usage reporting; and
- concise commit metadata derived from existing agent reports.

Target adapters should continue to own repository scope, seeds, build commands,
correctness checks, corpora, benchmarks, metric definitions, precision rules,
and promotion thresholds.

The implementation must not add a model-request proxy, a default request cap, a
local price reconstruction path for Kilo, Zstandard-specific schema fields, or
a generic hidden-data system.

## Current implementation state

### Already merged on `main`

- Per-job and per-run-token Kilo state isolation.
- Kilo workspace verification, headless tool policy, and optional `--pure`.
- POSIX process-group timeout cleanup.
- Root-plus-descendant Kilo session usage and catalog-cost aggregation.
- Preservation of unpriced Kilo costs without local reconstruction.
- Campaign constraints take precedence over generic agent validation advice.

### Framework foundation delivered by this closeout

- Separate planning and coding Kilo model settings.
- Explicit trajectory-summary provider, model, API surface, thinking mode, and
  reasoning effort, with preflight validation and usage recording.
- Deterministic commit messages from `coding.summary`, then `plan.summary`, with
  no commit-summary LLM call.
- Persistent per-island sampling ordinals, restart-stable RNG derivation,
  order-insensitive recipe hashes, bounded recipe cooldown, and explicit reuse
  provenance.
- Exact source-tree hashing and reuse of a passed result under the same
  evaluator and campaign contract.
- Evaluator-provided candidate identity, a contract-scoped identity key,
  identity persistence, and duplicate archive-admission prevention.
- Full change summaries and migrations 16-18.

These items are not remaining work packages. The implementation agent must
preserve them and must not reimplement them from historical experiment scripts.

### Foundation regression contract

Keep the three identities distinct while implementing the remaining work:

1. commit hash for ancestry and reproducibility;
2. Git tree hash for exact source equality before evaluation; and
3. evaluator identity for target-defined behavioral or artifact equality.

Run these focused regressions before and after the new implementation:

```bash
uv run pytest \
  tests/core/map_elites/test_sampler.py \
  tests/core/worker/test_candidate_identity.py \
  tests/core/worker/test_commit_summary.py \
  tests/core/worker/test_evolution.py \
  tests/core/worker/test_job_store.py \
  tests/core/worker/test_trajectory.py \
  tests/scheduler/test_ingestion_resilience.py \
  tests/db/test_migration_registry.py \
  tests/db/test_migration_v0016.py \
  tests/db/test_migration_v0017.py \
  tests/db/test_migration_v0018.py \
  tests/test_cli_config_dump.py \
  tests/test_preflight.py
```

These invariants must remain true:

- restarting a scheduler does not restart its sampling stream;
- the recent recipe cooldown survives a restart and records unavoidable reuse;
- identical Git trees reuse only a passed result with the same evaluator name,
  evaluator version, and campaign-program hash;
- failed results never populate a reusable result;
- evaluator-equivalent commits do not occupy duplicate archive entries;
- trajectory summarization fails preflight when enabled without a resolvable
  model route;
- planning, coding, trajectory, and embedding usage retain their phase and model;
  and
- old databases migrate forward without deleting jobs, candidates, metrics, or
  artifacts.

## Work package 1: move measurement reuse into the evaluator contract

### Problem

Loreley can reuse an exact Git tree before evaluation and can deduplicate an
evaluator identity after evaluation. It cannot avoid an expensive measurement
when two different source trees produce the same binary, because the current
one-shot evaluator reveals `candidate_identity` only after the complete plugin
call. The Zstandard harness therefore implemented its own binary lock, accepted
measurement index, and cache.

### Required behavior

Add an optional phased evaluator protocol while retaining the existing one-shot
plugin API:

1. **Prepare and identify** performs source-specific gates required before
   reuse, builds the candidate when needed, and returns an evaluator identity
   plus preparation artifacts.
2. **Measure** performs the expensive cacheable work when no accepted
   measurement exists.
3. **Finalize** combines source-specific preparation evidence with either the
   new or reused measurement into the normal `EvaluationOutcome`.

The exact public names may follow existing evaluator conventions, but all three
states must be explicit. Do not reuse an entire source evaluation merely because
the binary matches: source-level scope, tests, or portability checks may still
differ.

Persist accepted measurements in framework tables. The cache key must include:

- normalized evaluator-provided candidate identity;
- evaluator name and version;
- campaign-program hash; and
- a plugin-supplied measurement-contract fingerprint when the evaluator version
  does not already cover corpus, benchmark, build mode, and metric protocol.

Only a completed, passed, plugin-marked-cacheable measurement may be reused.
Failures, timeouts, incomplete artifacts, precision rejections, and results from
a different contract must miss the cache.

Serialize the first measurement of one key across workers. Prefer a
PostgreSQL-backed lock whose ownership ends automatically with the database
session. After acquiring the lock, check the cache again before measuring.

Every reuse must create a new evaluation-attempt record that points to the
original accepted measurement and records `measurement_reused=true`. Copying
metrics without provenance is not acceptable.

### Acceptance tests

- Two different trees with one evaluator identity run source-specific prepare
  twice but expensive measure once.
- Two concurrent workers racing on one identity produce one measurement and one
  reuse.
- A failed, timed-out, or imprecise first measurement is never reused.
- Changing evaluator version, campaign program, or measurement fingerprint
  forces a new measurement.
- A one-shot legacy plugin behaves exactly as before.
- A reused result exposes original attempt, artifact hashes, and cache key in
  the API and database.

## Work package 2: make evaluator concurrency a framework control

### Problem

Loreley exposes the algorithm limit on unfinished jobs and the physical model
worker count, but the case studies enforced evaluator lanes with target-specific
file locks. The three controls have different meanings:

- `U`: maximum unfinished jobs visible to the search algorithm;
- `W`: physical planning/coding worker processes; and
- `E`: simultaneous evaluator measurements allowed by the host or service.

Coupling them changes archive dynamics or measurement quality when only machine
parallelism should change.

### Required behavior

Add an optional framework setting such as
`WORKER_EVALUATOR_MAX_CONCURRENCY`. Enforce it across processes and hosts for one
experiment and evaluator contract. Use a database-backed slot or equivalent
lease that is released after normal exit, exception, timeout, or worker death.
Do not implement it as a process-local semaphore.

For a phased evaluator, the plugin must declare whether the limit covers the
whole evaluator or only the measurement phase. For a legacy one-shot plugin,
the limit covers the complete plugin call.

Record slot number, wait time, acquisition time, and release outcome in the
evaluation attempt. Surface effective `U`, known local `W`, configured `E`, and
current evaluator waiters in `loreley status` and `loreley config dump` without
pretending that a distributed global worker count is known when it is not.

### Acceptance tests

- `W=4, E=1` never overlaps evaluator calls while planning/coding remains
  concurrent.
- `W=4, E=4` permits four evaluator calls.
- Changing `U` does not silently change `E`, and changing `E` does not alter the
  sampler's unfinished-job limit.
- A killed evaluator releases or expires its slot and does not deadlock the
  campaign.
- Wait time is persisted and visible in status output.

## Work package 3: identity-aware progress and stopping

### Problem

Physical jobs overstated useful Zstandard coverage. The V19 scripts had to query
the database directly to count source trees, binaries, real measurements, and
cache reuse, and to stop after a target number of new binaries.

### Required behavior

Extend the normal status surface with:

- terminal, succeeded, failed, running, and queued jobs;
- distinct passed Git trees;
- distinct passed evaluator identities;
- real measurements and measurement reuses;
- archive entries and unique evaluator identities represented in the archive;
- occupied coordinates; and
- failure counts by stable failure kind.

Add an optional at-least endpoint such as
`SCHEDULER_MAX_UNIQUE_EVALUATION_IDENTITIES`. Once the count reaches the target,
the scheduler must stop new dispatch and drain existing unfinished jobs. Report
the target and any bounded asynchronous overshoot. Do not manufacture an exact
physical-job count from an identity endpoint.

The feature is inactive when evaluators do not provide identities. Preflight
must reject an identity endpoint that cannot be satisfied by the configured
evaluator contract.

### Acceptance tests

- repeated commits and trees with one evaluator identity increment physical job
  counts but not the unique-identity count;
- reaching the identity target stops new dispatch and drains existing work;
- status clearly distinguishes archive entries, occupied coordinates, Git
  trees, evaluator identities, and measurements; and
- scheduler restart preserves the endpoint decision.

## Work package 4: finish model and embedding configuration DFX

Keep the current phase-specific settings and make the effective routing visible
before a campaign starts. `loreley doctor --role worker` and config dump must
show, without secrets:

- planning backend, provider mode, model, and variant/reasoning setting;
- coding backend, provider mode, model, and variant/reasoning setting;
- trajectory-summary provider, API surface, model, thinking mode, and reasoning
  effort;
- embedding provider route, model, and dimensions; and
- that commit summaries make no model call.

`text-embedding-3-small` remains the default. Keep deterministic local-hash
embeddings available for tests and offline examples, but emit a prominent
preflight warning when they are used for an optimization campaign. Require an
explicit acknowledgement setting if a campaign wants to continue with them.

Continue treating Kilo session-tree cost as authoritative for Kilo. Report
provider-reported, catalog, locally estimated, unpriced, and unavailable costs
as different categories. Never turn a zero or missing Kilo cost into a local
price estimate.

## Work package 5: retire generic harness responsibilities

After work packages 1-4 land, document the boundary for future case studies.
A target adapter may contain:

- seed patches and scope rules;
- build, correctness, compatibility, and benchmark commands;
- corpus construction and hidden-data handling;
- target-specific precision and result thresholds; and
- report-specific selection rules.

It must not reimplement Kilo routing, usage aggregation, process cleanup,
sampling restart logic, source-tree reuse, evaluator identity persistence,
measurement caching, evaluator concurrency, or identity-aware status.

Do not delete historical experiment code or evidence in this work. Mark it as
frozen and stop extending it after equivalent framework functionality exists.

## Verification and exit gates

Run the foundation regressions above, all new tests for work packages 1-4, then:

```bash
uv run pytest
uv run mkdocs build --strict
```

Before completion, verify:

- database migration from the last released schema and idempotent migration on
  the current schema;
- no API key, base URL credential, machine-local absolute path, or experiment
  corpus entered tracked fixtures or documentation;
- legacy one-shot evaluators and campaigns without candidate identities retain
  their behavior;
- the Zstandard target adapter can express its binary identity and measurement
  cache through the new public contract without importing Loreley internals;
- all reuse records are hash-linked to the original accepted evidence; and
- no target-specific metric or promotion rule entered the core framework.

The work is complete only when a minimal zero-model integration test runs two
source-distinct candidates with one evaluator identity, performs one expensive
measurement, records two source attempts, admits one evaluator identity, and
stops at an identity endpoint without target-harness database queries.
