# Case study: Loreley evolved a 25.14% `python-pathspec` speedup

Date: 2026-08-03

Scope: one 64-job campaign on `cpburnz/python-pathspec` commit
`6568072c2703c72796cd02467feb924540157c92`

## Result in one minute

Loreley combined six small, human-written optimization seeds with 58
Kilo/DeepSeek evolution jobs. The final validated winner,
`9d977f0a73d58aec73fa36516c07cbb0ec879347`, ran the separate reference
workloads **1.2514x as fast as the fixed root**. It passed the complete upstream
test suite and semantic checks, changed only permitted source files, and stayed
below the 0.05 MiB peak-allocation limit.

| Measure | Result |
|---|---:|
| Jobs | 64 total: 6 seeds + 58 evolution |
| Successful / failed | 45 / 19 |
| Best manual seed on training | 1.1227x |
| Final winner on training | 1.2536x |
| Final winner on reference | **1.2514x** |
| Final winner peak allocation | 0.04354 MiB |
| Campaign wall time | 3.91 hours |
| Generation usage | 241.63M tokens, 3,977 requests |
| Embedding usage | 258,055 tokens |
| Generation cost | **$2.4856** |

The defensible statement is:

> In a fixed 64-job `python-pathspec` case study, Loreley evolved a weak manual
> seed through four Kilo/DeepSeek generations into a candidate that was 25.14%
> faster on disjoint reference workloads and passed correctness, semantics,
> edit-scope, and allocation checks.

The winner was selected after the campaign's initial training pick failed final
allocation validation. The reference workloads had therefore been revealed
before this candidate was validated. This case study supports a system
capability claim, not a clean prospective success claim.

## What was tested

The target was the pure-Python simple backend used by `PathSpec` and
`GitIgnoreSpec`. Agents could edit only `pathspec/**/*.py`. Every scored
candidate had to:

- pass the complete upstream suite: 197 tests, 276 skips, and 142 subtests;
- reproduce deterministic root outputs;
- preserve custom Pattern behavior and public APIs;
- stay at or below 0.05 MiB measured peak allocation; and
- improve five compile and matching workloads.

Training used balanced root/candidate process ordering. Reference validation
changed pattern counts, path counts, and salts and remained sealed until the
initial training pick was frozen. A candidate evaluation took about 16 seconds;
the full test suite itself took about 0.16 seconds. There was no model-driven
baseline arm beyond the root measurements required to calculate speedups.

The campaign used Kilo with `deepseek-v4-flash` for planning and coding. Archive
diversity used external `text-embedding-3-small` embeddings with 1,536
dimensions, not local hash embeddings. Algorithm concurrency was four
unfinished jobs, physical concurrency was four model workers, and evaluation
used one serialized lane.

## How the winner evolved

The candidate `9d977f0a73d58aec73fa36516c07cbb0ec879347` was a normal evolution job,
not a hand-written repair. Its parentage comes from the campaign database's
recorded `base_commit_hash`; inspiration edges are ideas supplied to an agent,
not ancestry.

Using the seed as generation 0, the candidate is generation 4:

| Generation | Job | Training result | Contribution |
|---:|---:|---:|---|
| 0 | 6 | 0.9978x | Move `from_lines` filtering and factory dispatch into C-level iterators. |
| 1 | 10 | 1.0721x | Bind batch hot-path calls and escape contiguous literal runs together. |
| 2 | 14 | 1.0866x | Replace `groupdict()` with `lastgroup` and reduce repeated attribute reads. |
| 3 | 18 | 1.1921x | Precompute stock regexes and call `regex.search()` directly, with a custom-Pattern fallback. |
| 4 | 38 | 1.2536x | Flatten patterns into pre-bound matcher tuples and remove hot-loop dispatch. |

The branch remained in the MAP-Elites archive between jobs 18 and 38 while 20
other jobs explored different branches. Loreley later sampled it again and
produced the final step. This is the archive retaining and revisiting a useful
line, rather than a single champion being edited repeatedly.

The final candidate changed five files with 127 additions and 51 deletions.
Its optimizations form one causal sequence: reduce construction overhead, bind
batch operations, remove per-match dictionaries and wrappers, then flatten the
remaining dispatch table. Individual changes were not separately ablated, so
the exact contribution of each step is unknown.

## Reference results

| Scenario | Speedup |
|---|---:|
| Compile 150 gitignore patterns | 1.3673x |
| `GitIgnoreSpec` match, 150 patterns | 1.2796x |
| `PathSpec` match, 150 patterns | 1.2384x |
| `PathSpec` match, 2 patterns | 1.1550x |
| `PathSpec` match, 40 patterns | 1.2265x |
| **Geometric mean** | **1.2514x** |

The small training-to-reference gap was 0.21 percentage points. All five
scenarios improved. Reference peak allocation was 0.04354 MiB.

## Why the initial training pick was rejected

The preregistered rule first selected
`59316e902c113ef9f4fcc47c276515772c86977c`, the feasible training candidate
with the highest throughput. It reached 1.2633x on training and 1.2619x on
reference. Its allocation was 0.04331 MiB while compiling 100 training patterns
but grew to 0.06472 MiB with 150 reference patterns, exceeding the fixed 0.05
MiB gate. It was therefore rejected.

The final winner used 0.02942 MiB on training and remained below the limit at
reference scale. A future confirmatory design should measure allocation on the
largest intended shape during training or preregister a safety margin.
Selecting only by throughput below a small-shape allocation limit was the
experiment-design error.

## Cost, failures, and request limit

The campaign completed at 16.4 terminal jobs per hour, including the six fast
seed jobs. Median end-to-end job duration was 10.0 minutes. Generation used
241,634,477 tokens. Recorded generation cost was $2.4856.

The 19 failures remained part of the result: five failed during planning,
thirteen produced no effective repository change, and one exceeded the
training allocation gate. No failure was caused by the request limit or a raw
HTTP 429/5xx response.

The per-job request guard should remain at 160:

- the median model-driven job used 66 requests;
- no job reached 160;
- one 157-request job produced a valid 1.2327x training candidate, showing that
  the former limit of 128 was too low; and
- failed jobs used at most 65 requests, so raising the guard would not have
  rescued them.

An independent 20-minute coding timeout remains the backstop. A higher limit,
such as 192, is justified only after multiple jobs reach 160 while still making
verified code-and-test progress.

## Project defect found during the run

Three planning failures left four Kilo descendants alive after their jobs had
become terminal. The direct Kilo process was killed on timeout, but its child
processes survived and could continue API requests.

Loreley's Kilo backend now launches each invocation in a separate process group
on POSIX hosts. A timeout terminates the group, waits for a bounded grace
period, and escalates to a group kill if needed. A regression test launches a
real grandchild process and verifies that timeout cleanup removes it.

## What ran end to end

The campaign used Loreley's scheduler, MAP-Elites archive and sampler, Kilo
planning and coding backends, Git candidate commits, external embeddings,
evaluator ingestion, and recorded lineage. The target adapter supplied the
repository contract, deterministic benchmark, and seed patches; it did not
replace search, archive selection, agent execution, commit ingestion, or
evaluation.

The one-off harness and raw run artifacts contain machine-local operational
details and are intentionally excluded from the repository.

## Claim boundary

This study covers one repository revision, one host, synthetic deterministic
workloads, and a human-seeded search. The final winner was selected after the
initial training pick's reference result was known, and no attempt was made to
establish upstream maintainability or production workload impact. Together
with the earlier `markdown-it-py` result, it motivates further preregistered
replications; it does not estimate Loreley's average effect across repositories.
