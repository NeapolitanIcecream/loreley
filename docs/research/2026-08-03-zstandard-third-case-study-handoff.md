# Loreley Third Case Study Handoff: Zstandard

Date: 2026-08-03

Status: Historical handoff, completed. Zstandard V19 is now the third public case study. This document preserves the original selection process, measurements, and requirements for a paper-grade experiment.

Execution results are in the [Zstandard V19 report](2026-08-07-zstandard-gpt-v19-case-study-report.md) and the [Top-10 supplement](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md). V19 ran 220 physical jobs. Of the 211 successful jobs, 167 produced distinct release binaries. Expanded validation selected generation-4 candidate `fe39bee8`; it later measured +1.173% compression on the original holdout and +0.891% on a newly generated sealed corpus. Its holdout score was unknown at selection, but that corpus had already been opened for the preregistered Top-3 winner. The fresh-corpus recipe was chosen after the candidate. The preregistered protocol itself selected a manual seed at +1.019%. This run included only the quality-diversity arm. Champion-sequential search, root-independent search, and cross-architecture replication remain paper work and cannot be inferred from V19.

Recipient: the agent responsible for the `markdown-it-py` and `python-pathspec` experiments.

## Assignment

Use [Zstandard](https://github.com/facebook/zstd) as Loreley's third case study. Design the protocol, implement and calibrate the evaluator, run the experiment, and assemble the evidence.

Submit a reviewable protocol and evaluator calibration before freezing the formal run. Do not start a large search using the preliminary parameters in this document. The preliminary work fixes the research questions and evidence requirements; pilots must determine the corpora, compression levels, run duration, job budget, and compute environment.

## Decisions already made

1. Use Zstandard for the third case study. RocksDB remains a later option for a collaboration with dedicated Linux NVMe resources.
2. Demonstrate that Loreley can evaluate a non-Python project. Python is only the evaluator integration interface; the target repository and its build and evaluation pipeline are C/C++.
3. Produce evidence for both a public case study and a research paper. A public runnable demo is not a deliverable.
4. A paper-grade result requires equal-budget search controls, hidden validation, and a sealed holdout. Another 64-job Loreley-only campaign would not supply the controls missing from the first two cases.
5. Do not select a runner-up after a holdout failure. A finite set may be tested together only if its size, selection rule, and multiple-testing treatment are frozen before the holdout is opened.

## Why Zstandard

| Criterion | Evidence | Experimental implication |
| --- | --- | --- |
| Different target from the two Python libraries | Zstandard is a compression library with about 139,000 lines of C/C++ | Tests language-independent evaluator integration and low-level performance edits |
| Correct changes have several constraints | Changes must preserve decoding, format compatibility, API/ABI behavior, compression ratio, and memory limits | The evaluator must check performance and all constraints |
| Fast evaluation | A local one-second benchmark for one corpus and one level took 2.07 seconds; `make check` took 23.20 seconds | Repeated measurements and search controls fit within a practical budget |
| Low short-run noise on the pilot host | Across 15 runs of a fixed release binary, compression CV was 0.477% and decompression CV was 0.561% | A roughly one-minute evaluator may distinguish a 1% effect, subject to recalibration on the formal host |
| Intrinsically multi-objective | Compression speed, decompression speed, compressed size, and peak memory can conflict | Provides a direct test of Pareto archives and quality-diversity search |
| Easier parallelism than storage systems | The core benchmark is in memory and does not require one NVMe device per worker | Parallel lanes may scale by CPU core after frequency and memory-bandwidth interference tests |

Zstandard remains a difficult target because its mature C/C++ implementation combines low-level optimization, format compatibility, cross-corpus generalization, and multiple constraints. Repository size must not be presented as the amount of code changed by each candidate. The final report must state the files, call paths, and diff size actually affected.

## Reproducible preliminary measurements

Environment:

- Zstandard commit: `82d322c4973d9e2968d94047a40892bc6d9a9bdf`;
- reported version: 1.6.0;
- 658 tracked files;
- 275 C/C++ source or header files, about 138,614 lines;
- host: 14-core Apple M4 Pro, 24 GB RAM, Apple Clang 21;
- pilot corpus: 103 files under the repository's `lib/` directory, totaling 3,310,200 bytes.

| Operation | Local wall time |
| --- | ---: |
| Clean release build | 2.79 s |
| Incremental build after editing `lib/compress/zstd_compress.c` | 1.30 s |
| `make check` | 23.20 s |
| One corpus, level 1, `-i1 -T1` | 2.07 s |
| One corpus, levels 1 through 5, `-i1 -T1` | 13.61 s |

Fifteen level-1 runs using a fixed final release binary produced:

| Metric | Mean | Sample standard deviation | CV |
| --- | ---: | ---: | ---: |
| Compression speed | 608.27 MB/s | 2.90 MB/s | 0.477% |
| Decompression speed | 1,826.29 MB/s | 10.25 MB/s | 0.561% |

Using the larger 0.561% CV, a two-sided 5% significance level, and independent baseline and candidate samples, a 1% effect requires seven runs per group for at least 80% power and eight runs per group for at least 90% power. The resulting local estimates are:

- one corpus and one level, including build and `make check`: about 53 to 58 seconds;
- one corpus and levels 1 through 5: about 3.6 to 4.0 minutes;
- one low-precision training pass over levels 1 through 5: about 38 seconds.

These figures justify further evaluator design. They do not establish the noise level for the formal corpora, other levels, another compiler, or another host.

The pilot also found an approximately 7% performance difference between the release binary and a different build configuration produced during `make check`. Correctness and performance builds must therefore use isolated artifacts. Each evaluation must record the compiler, flags, link mode, CPU affinity, source commit, and binary hash. A correctness command must not replace the binary being measured.

## Questions the case study must answer

The formal protocol must answer at least four questions:

1. Can Loreley find a measurable performance improvement on a frozen Zstandard revision while passing correctness and compatibility checks?
2. Does the improvement generalize from training corpora to undisclosed content types, file sizes, and compression levels?
3. Under the same model and budget, does Loreley's quality-diversity search outperform independent proposals from the root and repeated edits to the current champion?
4. Does the archive preserve useful lineages or Pareto trade-offs, or can a simpler search explain the final result?

The first two questions support factual claims for the case study. The last two concern the search method and belong in the paper. Negative answers are valid outcomes. Do not change the primary question or winning rule after observing results.

## Required protocol

### Frozen environment

Freeze and record:

- the upstream URL and revision, initially `82d322c4973d9e2968d94047a40892bc6d9a9bdf`; if the revision changes, document the reason and recalibrate;
- compiler and version, build flags, link mode, and CPU-feature policy;
- separate construction of correctness and performance binaries;
- hashes for the evaluator, image or host setup, corpus manifest, and scripts;
- CPU governor, turbo policy, affinity, evaluator lanes, machine model, and operating system;
- model, agent backend, prompt, request guard, timeout, embedding provider, and embedding dimension;
- budgets for model requests, tokens, cash, candidate evaluations, device-hours, and wall time.

Do not run builds or other work that changes CPU frequency, cache state, or memory bandwidth during a formal benchmark. Establish safe lane concurrency with root-versus-root interference measurements; core count alone is not sufficient.

### Edit scope and anti-cheating rules

By default, allow edits only to product source and headers under `lib/**`. Protect:

- `programs/**`;
- `tests/**`;
- the evaluator, corpora, result parser, and experiment configuration;
- compiler flags, benchmark parameters, and output format;
- `loreley.program.md`, `.loreleyignore`, and other experiment-control files.

If a build file must be editable, add it individually to an allowlist and add checks against disabling safety checks, changing CPU flags, or substituting the benchmark binary.

Reject candidates that:

- modify protected files;
- branch on corpus names, paths, known hashes, or input byte patterns;
- skip input, reduce the workload, or forge benchmark output;
- change the public API, frame format, or compatibility outside the protocol;
- compare root and candidate builds with different compiler settings.

A static scope check cannot detect every corpus-specific edit. Hidden corpora and manual edit audits are also required.

### Corpus partitions

Use at least three partitions:

1. `training`: public corpus groups whose scores may be returned to the agent;
2. `validation`: hidden corpora used only to promote and select finalists under frozen rules;
3. `sealed holdout`: opened once after the search, finalist set, and selection rule have been frozen.

The partitions must not be random slices of one corpus. Cover source code, JSON or other structured text, natural-language text, binary data, and several file sizes. A pilot should decide whether dictionary, streaming, small-block, or multithreaded modes belong in scope.

For each corpus, record provenance, license, hashes, bytes, file count, and leakage checks. The target repository's `lib/` files may support evaluator development but cannot be the only formal corpus.

Validation results must not be returned to the model. No holdout corpus, aggregate, or cell result may be inspected before the finalist set is frozen. If the same operator stores the holdout, use an encryption or access boundary and record the first unsealing time.

### Metrics and outcome rules

Retain, for each workload cell:

- compression throughput;
- decompression throughput;
- compressed bytes or compression ratio;
- peak memory;
- correctness, compatibility, and scope-gate status;
- the order, timestamp, host, and binary hash for every repetition.

Do not store only a composite score. Before the search, define:

- one primary endpoint;
- the objective vector used by the Pareto archive;
- the maximum permitted regression in each cell;
- constraints or Pareto treatment for compressed size and peak memory;
- finalist count and selection function;
- treatment of multiple candidates, metrics, or profiles;
- criteria for `strong`, `modest-positive`, `negative`, and `invalid` outcomes.

A strong result should require a preregistered candidate to meet a practical improvement threshold on the sealed primary endpoint, exclude no improvement with its confidence interval, and pass every correctness, compatibility, scope, compression-ratio, memory, and cell-level regression limit. If the claim is "the point estimate improved by at least 1%," the point estimate must reach 1% and the lower confidence bound must exceed zero. If the claim is "the true improvement is at least 1%," the lower confidence bound must exceed 1%. The formal host's noise study must determine whether 1% is a suitable threshold.

A Pareto trade-off may be a useful secondary result. A throughput increase obtained by materially reducing compression ratio is not an unconditional speedup.

### Correctness and compatibility gates

Every scored candidate must pass:

- a clean or auditable incremental build;
- upstream `make check`;
- compression and decompression round trips on training corpora;
- cross-decoding between root and candidate;
- scope and benchmark-output sanity checks.

Promoted candidates and finalists also require the selected medium or long tests, sanitizers, fuzzers, legacy decoding, and API/ABI checks. Zstandard's [TESTING.md](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/TESTING.md) lists short, medium, long, sanitizer, fuzzer, legacy, and cross-platform tests. A 23-second `make check` run is not a complete upstream validation.

Calibrate the memory gate on the largest supported input shape and leave safety margin. The first `python-pathspec` winner passed a small training shape but failed the reference allocation gate; this case must not repeat that selection error.

### Measurement

Before formal search, run a noise study on the target host:

- 30 to 50 or more root repetitions;
- several minimum benchmark durations, corpora, levels, cold and warm states, and times of day;
- randomized interleaved or paired root-versus-root trials to estimate pairwise variance;
- single-lane and multi-lane tests for frequency, cache, and memory-bandwidth interference;
- a new power calculation for the final practical-effect threshold.

Measure root and candidate in randomized, interleaved A/B order. Report raw repetitions, effect sizes, and confidence intervals rather than a single CLI summary. If variance differs materially across corpora or levels, allocate repetitions by cell or preregister a robust aggregation rule.

### Staged evaluator

The formal protocol must specify thresholds and budgets. The initial design is:

| Gate | Candidates | Purpose |
| --- | --- | --- |
| 0 | Every candidate | Scope, build, `make check`, round trip, and output checks |
| 1 | Candidates that pass Gate 0 | Inexpensive training corpora and levels; returns a search score |
| 2 | Candidates promoted under a frozen rule | More repetitions and corpus groups, plus maximum-input memory checks |
| 3 | Frozen finalist set | Hidden validation, broader tests, and edit audit; selects the confirmation target under a frozen rule |
| 4 | Frozen target or preregistered set | Sealed holdout, formal statistics, compatibility, sanitizer or fuzzer checks, and a second architecture when available |

Gate 1 may begin with the approximately 38-second pass over levels 1 through 5, or a shorter variant. Choose it using false-promotion, false-rejection, and queue-throughput measurements. Do not pay for final 1% precision on every search candidate, and do not use a low-precision training score as the paper's performance claim.

### Equal-budget search controls

Run at least three arms:

1. Loreley quality-diversity search;
2. sequential edits to the current champion;
3. independent best-of-N proposals from the root.

Use the same repository, model, visible training feedback, evaluator gates, total model requests or other preregistered equivalent budget, and accounting rules in all arms. Unless the protocol specifies a different primary budget, report jobs, requests, tokens, and evaluator device-hours together.

Manual seeds are optional. If used, provide the same seed information and seed budget to every arm, or separate seeded and unseeded experiments. Do not give optimization directions only to the Loreley arm.

Run the arms as independent campaigns. Existing archive entries from one Loreley campaign are not a post hoc baseline. Freeze the number of campaign replicates and random seeds. Candidate jobs within one stateful campaign are not independent method-level replicates. If the budget allows only one campaign per arm, describe the comparison without a significance claim about the search algorithms. Preserve random seeds, stopping rules, failures, and unfinished jobs.

## Preliminary evaluator budget

The table includes evaluator device time only. It excludes generation, queuing, image or container setup, promotion tests, and final holdout work. Each search arm incurs its own cost.

| Evaluator design point | 256 evaluations | 1,024 evaluations | 28,000 evaluations |
| --- | ---: | ---: | ---: |
| One pass over levels 1 through 5, about 38 s | 2.70 h | 10.81 h | 295.56 h |
| One-level 1% measurement, about 53 to 58 s | 3.77–4.12 h | 15.08–16.50 h | 412.22–451.11 h |
| Levels 1 through 5 at 1% precision, about 3.6 to 4.0 min | 15.36–17.07 h | 61.44–68.27 h | 1,680–1,867 h |

Three equal-budget arms require about three times these evaluator totals. Most candidates should stop at Gate 1; only a preregistered fraction should advance.

CPU-bound device-hours cannot be converted directly to wall time. Estimate parallel speedup only after multi-lane interference tests pass, using independent physical cores or hosts. The roughly ten-minute model-job time from the first two cases does not establish C/C++ generation throughput; measure it in the pilot.

The formal job count is not fixed. Specify a pilot budget, a minimum publishable budget, and an expansion budget, with the question each can answer. The 64-job size of the first two cases is not a default for this case.

## Execution and deliverables

### Stage A: protocol

Submit a preregistration-style plan containing:

- frozen revision and environment;
- corpus partitions and licenses;
- workload matrix;
- primary endpoint, Pareto objectives, and regression limits;
- budgets and fairness rules for the three search arms;
- campaign replicates, analysis unit, and random seeds;
- finalist, holdout, and multiple-testing rules;
- threat model, failure classes, and stopping rules;
- expected device-hours, model usage, cash, and wall time.

### Stage B: evaluator and calibration

Implement and test:

- hermetic builds and binary provenance;
- protected-file and scope gates;
- correctness, round-trip, and compatibility checks;
- benchmark parsing and raw-result storage;
- corpus access boundaries;
- interleaved root and candidate measurements;
- fake candidates, known regressions, no-op changes, and benchmark-cheating fixtures;
- a 30-to-50-run baseline noise report;
- an interference report for the planned evaluator lanes.

### Stage C: pilot

Run a small pilot that exercises the full path and checks:

- whether the C/C++ agent produces valid edits before timeout;
- whether candidate build caches invalidate correctly;
- ranking agreement between Gates 1 and 2;
- evaluator throughput and queue pressure;
- whether tokens, requests, failure rate, and wall time fit the planned budget;
- whether the model specializes to corpora, edits tests, or bypasses the benchmark.

Use the pilot to freeze the protocol and budget. Do not combine it with the confirmatory run.

### Stage D: formal run

After freezing hashes and selection rules, run the three search arms. Preserve the complete campaign databases, candidate commits, ancestry, inspiration edges, model usage, raw evaluator runs, failures, and environment events. Record every protocol deviation before applying a correction; do not silently repair a confirmatory run and continue treating it as unchanged.

### Stage E: validation and evidence package

Stop model calls and freeze the finalist set before hidden validation and the sealed holdout. Deliver:

- root, finalist, and reported candidate commits and diffs;
- budget, valid-candidate count, best score, and gate counts for each arm;
- raw primary-endpoint measurements, effect size, and confidence interval;
- the full workload matrix, not only the winning cell;
- compressed size, memory, correctness, compatibility, and cross-architecture results;
- principal ancestry and inspiration edges, archive retention, and edit taxonomy;
- classification of parameter tuning, local optimization, structural changes, and corpus specialization;
- model requests, tokens, cash, device-hours, and wall time;
- failed and invalid candidate classes and protocol deviations;
- an English case-study report and a Chinese factual summary for promotional use;
- the final `strong`, `modest-positive`, `negative`, or `invalid` classification and claim boundary.

The evidence package should make candidate diffs, ancestry, protocol, performance distributions, and resource accounting inspectable. It does not require a one-click public demo.

## Go/no-go criteria

Start a large formal search only when:

- the evaluator can distinguish the minimum practical effect on the target host;
- root and no-op candidates do not show a systematic false gain;
- correctness and performance binaries cannot contaminate one another;
- hidden-validation and holdout boundaries have been implemented and tested;
- search-arm budgets and seed policy are frozen;
- multi-lane CPU and memory-bandwidth interference has been measured;
- pilot valid-job rate and duration leave enough budget for validation.

Pause and revise the protocol before continuing if:

- the repetitions required for a 1% threshold make search throughput impractical;
- build configurations still produce unexplained performance differences;
- corpus licensing, leakage control, or holdout sealing cannot be audited;
- planned correctness checks fail to reject known-bad fixtures;
- multi-lane and single-lane measurements rank candidates differently;
- pilot gains mainly come from benchmark specialization or relaxed compression constraints.

## Decisions left to the experiment owner

The selection work did not determine:

- whether to retain the preliminary upstream revision;
- the target host, primary architecture, and second validation architecture;
- corpus sources, licenses, and partitioning;
- compression levels, API modes, file sizes, and dictionary or streaming scope;
- the primary endpoint, Pareto aggregation, regression thresholds, and memory limit;
- repetition counts and promotion rates for Gates 1 through 4;
- model, prompt, manual-seed policy, and jobs per arm;
- exact commands for full tests, sanitizers, fuzzers, and legacy compatibility;
- pilot, minimum publishable, and expansion budgets.

Use the pilot data to make these decisions. Record the rationale and rejected alternatives in the protocol.

## Related material

- Overall promotion and paper plan: [2026-08-03-loreley-promotion-plan.md](2026-08-03-loreley-promotion-plan.md)
- First case study: [2026-08-02-markdown-it-py-deepseek-case-study.md](2026-08-02-markdown-it-py-deepseek-case-study.md)
- Second case study: [2026-08-03-pathspec-deepseek-case-study.md](2026-08-03-pathspec-deepseek-case-study.md)
- Zstandard benchmark mode: [programs/zstd.1.md](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/programs/zstd.1.md)
- Zstandard build-to-build benchmark: [tests/automated_benchmarking.py](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/tests/automated_benchmarking.py)
- Zstandard testing guide: [TESTING.md](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/TESTING.md)

## Transfer summary

> Use Zstandard as Loreley's third case study. First design and freeze the protocol described here, then implement and calibrate the evaluator, run a small pilot, and finally execute equal-budget quality-diversity, champion-sequential, and root-independent search arms. Keep training, hidden validation, and sealed holdout separate; record build provenance; protect the benchmark and corpora; and report compression speed, decompression speed, compressed size, memory, correctness, and compatibility. On the preliminary M4 Pro host, a one-level evaluator designed to detect a 1% effect took about 53 to 58 seconds, while levels 1 through 5 took about 3.6 to 4.0 minutes. Recalibrate these values on the target host. Deliver commits, diffs, ancestry, raw measurements, statistics, resource accounting, and a claim boundary. A public demo is not required.
