# Loreley v15 experimental validation sprint

Date: 2026-07-26

Status: Complete

## Research contract

### Target claims

This sprint tests whether the v15 changes provide useful behavior beyond unit
correctness:

1. A bounded Pareto front preserves real quality/runtime trade-offs that a
   primary-objective scalar archive would discard in the same behavior niche.
2. Configured islands receive fair search work, retain independent state, and
   exchange usable donor context with explicit provenance when migration is
   enabled.
3. `loreley worker --processes N` provides one-command process parallelism with
   isolated workers and lower makespan on a fixed workload.
4. A model/backend choice can be justified by observed valid-patch rate,
   objective quality, latency, and proxy-accounted cost rather than reputation.

### Scope and invariants

- The empirical task is the repository's deterministic 26-circle packing
  example. Candidates must return exactly 26 non-overlapping circles inside the
  unit square and stay below the evaluator's runtime limit.
- Quality is `sum_radii` (maximize); latency is repeated `runtime_p50_ms`
  (minimize). Runtime comparisons use repeated measurements and report noise.
- Algorithm comparisons replay the same evaluated candidates. A different
  candidate set is not accepted as evidence that Pareto selection beats scalar
  selection.
- Island comparisons keep the total job budget and agent/model fixed.
- Parallel comparisons keep the task set and backend fixed. Worker startup,
  distinct PIDs/workspaces, duplicate execution, failures, and makespan are all
  measured.
- PostgreSQL and Redis are disposable local services. No production data or
  remote experiment database is in scope.
- API secrets remain environment-only and are never written to artifacts,
  command output, git, or prompts.

### Required evidence

The strongest successful result requires:

- at least one valid, measured same-niche pair where neither objective vector
  dominates the other, both are retained by the v15 front, and the scalar
  primary baseline retains only one;
- a replayable island trace showing fair target allocation, a donor from a
  different ready island, persisted provenance, local target membership, and no
  donor when migration is disabled;
- identical parallel workloads for `N=1` and `N>1`, with distinct worker PIDs
  and workspaces, no duplicate job completion, and a reported makespan ratio;
- a coding-agent/model bake-off with the same prompt and repository state,
  actual evaluator outcomes, response usage, latency, and proxy pricing;
- an end-to-end Loreley run using the selected backend/model, including
  scheduler, one-command worker pool, evaluator, archive persistence, usage
  records, and a generated report;
- adversarial review of the harness, measurements, and conclusions.

### Insufficient outcomes

- Existing unit tests alone.
- Synthetic objective vectors presented as empirical optimization evidence.
- Different prompts or job budgets compared as though they were controlled.
- Counting several metric rows as proof of multi-objective selection.
- Counting worker processes without measuring completed work and isolation.
- A single successful model sample used to claim general superiority.
- A migration row that never reaches agent context.
- A benchmark conclusion that ignores failed or invalid candidates.
- Spending the budget without changing a decision.

### Epistemic stance and terminal states

The implementation and 827-test baseline are established facts. Whether the
features improve empirical search behavior is an open hypothesis.

Allowed terminal states are:

- supported within the tested task and budget;
- refuted by a reproducible counterexample;
- bounded inconclusive, with the exact statistical or systems gap recorded;
- blocked by proxy, GitHub, or local-service state that cannot be repaired
  inside the authorized scope.

No desired empirical improvement is assumed.

## Resource and source policy

- Hard API cap: **$100**, equivalent to **50,000,000 New API quota points** at
  the proxy's documented 500,000 quota points per US dollar.
- Operational stop: $90 attributed spend. The final $10 is reserved for
  delayed accounting, a decisive repeat, or review repair.
- Codex-session tokens and local CPU/Docker use do not count toward the cap.
- Per-response token usage is attributed with the proxy's `/api/pricing`
  model/completion ratios. The token appears shared, so global balance deltas
  are retained as a conservative but contaminated cross-check.
- Pricing source:
  <https://docs.newapi.pro/en/docs/guide/console/settings/rate-settings>.
- Token-usage endpoint source:
  <https://doc.newapi.pro/en/api/token-usage/>.
- External sources may explain API behavior and standard experimental methods.
  Repository code, commits, evaluator results, database rows, and captured
  process evidence are authoritative for Loreley claims.

## Adaptive experiment portfolio

| Family | Thesis | First decisive test | Initial budget | Status | Reopening condition |
| --- | --- | --- | ---: | --- | --- |
| Coding-agent/model bake-off | A strong, medium, and cheap model have measurably different valid-patch/cost frontiers | Same repository, prompt, evaluator, and one independent run per profile/model; repeat only contenders | $18 | Complete | Reopen a rejected model only if failure is provider/tool incompatibility |
| Same-candidate Pareto replay | v15 retains measured trade-offs lost by scalar primary selection | Force evaluated candidates into one controlled behavior cell and replay both policies | $4 | Complete | Reopen if the objective contract or archive dominance logic changes |
| Migration causal trace | Migration changes usable planning context while preserving island identity and fairness | Fixed seeds/jobs with migration off/on; inspect persisted lineage and captured prompts | $20 | Complete for mechanism; quality effect inconclusive | Reopen quality-effect claim only with repeated non-seed campaigns |
| One-command parallel A/B | Native multi-process mode reduces makespan without isolation failure | Identical deterministic workload at N=1 and N=4; confirm process isolation in the live pool | $8 | Complete | Reopen on worker-process or broker changes |
| End-to-end selected configuration | The chosen model/backend exercises the complete product path within acceptable cost/failure bounds | Bounded circle-packing run with real scheduler, worker pool, evaluator, archive, usage, and report | $40 | Complete | Expand only if a new model/backend is being selected |
| Reserve/adversarial repeat | A surprising result is not a one-off or harness artifact | Repeat the most decision-sensitive cell after audit | $10 | Partly used for live failure-path confirmation | Reopen only for a merge-blocking review finding |

Budget allocations are ceilings, not quotas. Funds move toward experiments with
the highest expected decision value.

## Failure model

- Proxy model aliases may route differently or omit usable token accounting.
- Kilo may succeed conversationally without editing the requested worktree.
- Runtime noise may manufacture false Pareto trade-offs.
- Candidate behavior embeddings may place trade-offs in different cells,
  invalidating a direct scalar/Pareto comparison.
- Seed scheduling may look fair while normal scheduling is not.
- Migration provenance may exist without donor content reaching planning.
- Provider rate limiting may make N=4 slower without invalidating local worker
  parallelism.
- Shared API-token activity may contaminate global quota deltas.
- A report may silently omit failures, duplicate jobs, or expensive retries.
- Experiment-only code may accidentally leak credentials or become product
  compatibility debt.

Each item is an explicit audit check.

## Experiments and results

All machine-readable inputs, raw rows, traces, logs, reports, incidents, and
derived conclusions are under `artifacts/2026-07-26-v15-validation/`.
[`conclusions.json`](artifacts/2026-07-26-v15-validation/conclusions.json) is the
compact claim ledger, and
[`incidents.json`](artifacts/2026-07-26-v15-validation/incidents.json) retains
failed and superseded runs. Complete per-process debug directories are packaged
with hashes in
[`log-archive-manifest.json`](artifacts/2026-07-26-v15-validation/log-archive-manifest.json).

### Model and coding-agent selection

The direct-API probe was only a compatibility screen. It rejected three of four
one-shot outputs as invalid and sent the surviving candidate plus the other
models through the same Kilo task, repository commit, editable scope, and
100-run evaluator.

| Kilo model | Agent edit | Valid | `sum_radii` | Agent seconds | Attributed cost |
| --- | --- | --- | ---: | ---: | ---: |
| `gpt-5.4-mini` | yes | yes | 2.4389662674 | 85.626 | $0.0867366 |
| `deepseek-v4-flash` | yes | yes | 2.4389662674 | 217.568 | $0.5497128 |
| `gpt-5.4` | yes | yes | 1.4768432292 | 102.654 | $0.2460300 |
| `claude-sonnet-4-6` | no; timeout | baseline only | 0.25 | 241.722 | $1.8459540 |

`gpt-5.4-mini` was selected because it matched the best observed quality at
39% of DeepSeek's latency and 16% of its attributed cost. This is an
operational choice for this task, not a general model ranking: the sample per
alternative is small, and a later bounded Mini task also timed out.

The Kilo calibration found two integration defects before the comparison was
accepted: the chat-completions gateway used an incompatible adapter, and the
headless agent could enter interactive tools. Both failures, their costs, and
the fixes are in the incident ledger. A Kilo result counted only when it left a
tracked diff and that diff passed the independent evaluator.

### Same-candidate Pareto replay

Three measured implementations were evaluated in one Docker-constrained
environment with 31 independent timing samples of 1,000 calls each. Median
bootstrap intervals used 20,000 resamples.

| Candidate | `sum_radii` (max) | runtime p50 ms (min) | median 95% bootstrap CI |
| --- | ---: | ---: | --- |
| repository baseline | 0.25 | 0.001560605 | [0.001544772, 0.001565897] |
| DeepSeek candidate | 2.4389662674 | 0.101259361 | [0.100328039, 0.102409012] |
| Mini candidate | 2.4389662674 | 0.000095707 | [0.000092249, 0.000096791] |

The controlled baseline/DeepSeek pair shared the same behavior measure. Neither
dominated the other: DeepSeek improved quality while the baseline was faster.
The v15 front retained both, whereas a primary-quality scalar baseline retained
only DeepSeek. Adding Mini correctly removed both because it matched DeepSeek's
quality and was faster than both candidates. This supports Pareto retention
without relying on synthetic objective vectors or preserving dominated points.

### Multi-island scheduling and migration

The system harness launched the real scheduler, PostgreSQL persistence, Redis
broker, one-command worker pool, evaluator, and archive manager. A deterministic
delayed backend isolated scheduler behavior from provider variance.

The final eight-job migration run produced:

- four `alpha` and four `beta` jobs;
- two seed and two normal jobs per island;
- separate persisted state and archive rows for both islands;
- one `alpha ← beta` and one `beta ← alpha` migration;
- donor hashes present in both persisted inspirations and captured planning
  prompts;
- four worker PIDs, eight worktrees, and no duplicate phase execution.

The disabled controls had no donor provenance. Two rejected preliminary runs
found real defects: normal scheduling began before every island was ready, and
a global every-second-job migration cadence aliased with two-island
round-robin, sending both donors to one island. The implementation now gates
normal cold start on all configured islands and counts migration cadence per
target island.

This establishes genuine independent island execution and causal donor
delivery. It does not establish that migration improves objective quality; that
claim remains bounded inconclusive until repeated matched non-seed campaigns
are available.

### One-command parallel A/B

The `processes=1` and `processes=4` runs used the same eight-job workload,
delays, islands, job limits, migration setting, database shape, and evaluator.
Only the worker process count changed.

| Configuration | Job execution window | Scheduler/worker wall time | PIDs | Worktrees | Failures / duplicates |
| --- | ---: | ---: | ---: | ---: | --- |
| `--processes 1` | 25.701889 s | 29.694509 s | 1 | 8 | 0 / 0 |
| `--processes 4` | 8.452003 s | 15.068000 s | 4 | 8 | 0 / 0 |

The job-window speedup was **3.041×** and the end-to-end wall-time speedup was
**1.970×**. Startup, scheduler polling, and shutdown dominate the short
workload's wall time. Every job had exactly one planning and one coding event
in a distinct worktree.

The first four-process attempt also exposed a multiprocessing-context collision:
queued logging initialized `spawn` before Dramatiq unconditionally requested
`--use-spawn`. Reusing an initialized context avoided that collision but left a
`fork` caller able to copy cached, non-randomized worker settings. The final
entrypoint temporarily forces `spawn` for the Dramatiq master without asking
Dramatiq to set it a second time, then restores the caller's prior context.
A bounded regression smoke explicitly initialized the worker parent to `fork`
before launching the real two-process master. Four jobs completed 2/2 across
the islands using two PIDs and four distinct coding workspaces, with no
duplicate phase event, in 10.125 seconds wall time. The compact record is
[`fork-parent-worker-pool-regression.json`](artifacts/2026-07-26-v15-validation/fork-parent-worker-pool-regression.json);
its first invalid database-endpoint attempt remains in the incident ledger.

### Selected-model end-to-end campaign

The selected configuration ran four real Mini jobs with one command,
`--processes 2`, two islands, one attempt per phase, and a hard total-job cap.
All four jobs completed planning, coding, evaluation, branch publication,
ingestion, archive persistence, and usage recording:

- status: 4 succeeded, 0 failed;
- allocation: `alpha=2`, `beta=2`;
- isolation: two PIDs, four distinct worktrees, eight unique phase events;
- report: generated JSON and Markdown successfully;
- best `sum_radii`: 2.5, compared with root 0.25 and historical best
  2.003569804;
- usage: eight events and 640,030 recorded tokens.

The run found that the gateway leaves Kilo's provider cost at zero. Loreley
previously treated that placeholder as authoritative and bypassed an explicit
pricing table. The fix preserves positive provider costs but allows a matching
explicit rule to price non-empty zero-cost Kilo usage. A separate failure-path
confirmation persisted two `estimated` events totaling $0.3277944. That
confirmation's coding phase timed out at 360 seconds, and therefore also
disproved any zero-failure-rate interpretation of the four-job success.

The same failed task exposed an empty-campaign terminal bug: after all jobs
failed, the scheduler crashed while creating a primary-objective branch.
Absence of a retained candidate is now a logged, clean terminal state; genuine
branch creation errors still fail.

## Implementation decisions changed by evidence and review

The experiments and current-head review caused nine product changes that were
not justified by the original unit suite alone:

1. sanitize non-finite PCA diagnostics before JSON persistence;
2. restore an island manager from the durable snapshot when ingestion rolls
   back;
3. wait for every configured island before normal cold-start scheduling;
4. apply migration cadence per target island;
5. temporarily force a spawn multiprocessing context for the worker pool;
6. make Kilo gateway/headless execution and timeout usage recovery reliable;
7. price Kilo zero-cost placeholders with explicit rules and terminate cleanly
   when a campaign retains no candidate;
8. keep one bounded seed probe after PCA warmup until an island archive is
   usable; and
9. ignore zero-span objectives when assigning crowding-distance boundaries.

The same review also aligned configured island IDs with their persisted
64-character boundary and preserved the legacy `main` fallback for blank
default-island values in the one-shot migration tool.

The implementation did not add a compatibility layer for legacy scalar or
single-island semantics. Existing configuration is handled by the committed
one-shot migration command documented in the architecture work.

## Final verification and structural-debt gate

The final suite ran with the PostgreSQL migration tests enabled: **865 passed,
0 failed, 0 skipped** in 40.92 seconds. Coverage.py measured 82.70% statement
coverage and 57.36% branch coverage (78.76% combined).

Cremona then scanned 279 Python files against the committed refactor baseline
using that coverage report and git history. Signal health was `full`; the
branch introduced **0 new** and **0 worsened** structural-debt signals while
resolving **22** baseline signals. The baseline was not rewritten. Eight
`refactor_now` and eighteen `refactor_soon` findings remain as pre-existing
repository debt, so the verdict is `strained`, not debt-free. The compact
machine-readable record is
[`structural-audit.json`](artifacts/2026-07-26-v15-validation/structural-audit.json).

## Budget closeout

The conservative attributed total is **$3.844143** (1,922,071.5 quota points),
or **3.844%** of the $100 cap. The remaining cap is **$96.155857**. Cache reads
were charged as ordinary input because the proxy discount was not known. The
shared token's global balance delta was larger and is retained only as a
contaminated cross-check.

The budget was a ceiling, not a spending target. More calls would refine model
failure-rate estimates but would not change the selected backend, system
design, bug fixes, or merge decision. Stopping preserves the reserve and follows
the contract rule that spend must change a decision.

## Reproduction

The deterministic system experiments cost no API budget:

```bash
docker compose up -d postgres redis
uv run python tools/run_v15_pareto_replay.py \
  --output docs/research/artifacts/2026-07-26-v15-validation/pareto-replay.json
uv run python tools/run_v15_system_experiment.py \
  --label v15-parallel-p1 --processes 1 \
  --output docs/research/artifacts/2026-07-26-v15-validation/system-parallel-p1.json \
  --trace docs/research/artifacts/2026-07-26-v15-validation/system-parallel-p1-trace.jsonl
uv run python tools/run_v15_system_experiment.py \
  --label v15-parallel-p4 --processes 4 \
  --output docs/research/artifacts/2026-07-26-v15-validation/system-parallel-p4.json \
  --trace docs/research/artifacts/2026-07-26-v15-validation/system-parallel-p4-trace.jsonl
uv run python tools/run_v15_system_experiment.py \
  --label v15-islands-migration --processes 4 --migration-interval 2 \
  --output docs/research/artifacts/2026-07-26-v15-validation/system-islands-migration.json \
  --trace docs/research/artifacts/2026-07-26-v15-validation/system-islands-migration-trace.jsonl
```

The live command consumes proxy budget and requires `LLM_API_KEY` and
`LLM_BASE_URL`; secrets are forwarded only to the worker subprocess and are
never serialized:

```bash
uv run python tools/run_v15_system_experiment.py \
  --label v15-live-gpt54mini --backend kilocode \
  --model gpt-5.4-mini --processes 2 \
  --max-total-jobs 4 --max-unfinished-jobs 2 \
  --timeout-seconds 1200 \
  --report-dir docs/research/artifacts/2026-07-26-v15-validation/v15-live-gpt54mini-report \
  --output docs/research/artifacts/2026-07-26-v15-validation/system-live-gpt54mini.json \
  --trace docs/research/artifacts/2026-07-26-v15-validation/system-live-gpt54mini-trace.jsonl
```

## Final checkpoint

- Pareto retention: supported within the measured task.
- Multi-island independence and migration mechanism: supported.
- One-command multi-process execution: supported with measured speedup.
- Selected Kilo/Mini end-to-end path: supported, with observed timeout risk.
- Migration quality benefit: bounded inconclusive.
- Algorithm, debt, and parallelism audits reported no blocker or major finding.
- Final verification: 862 tests passed; structural audit found no regression
  and resolved 22 baseline signals.
- All failed and superseded experiments remain in the incident ledger.
- No credential was written to an artifact.
