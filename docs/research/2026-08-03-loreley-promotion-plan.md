# Loreley Promotion and Research Plan

Date: 2026-08-03

Last updated: 2026-08-10

Status: Internal working document, excluded from the MkDocs site. Update this file when publication decisions, literature findings, case-study evidence, or experiment designs change.

## Objectives

The promotion and research work should produce:

1. an English technical article and a conservative Chinese translation;
2. a research paper suitable for public submission to arXiv;
3. auditable accounts of the `markdown-it-py`, `python-pathspec`, and Zstandard case studies;
4. a first public release consisting of the article, case-study evidence, and community posts, without waiting for paper-grade search controls;
5. an intake path for teams that have automated evaluators, valuable optimization targets, and sufficient compute.

A public runnable demo is not a prerequisite. The public evidence should instead expose candidate diffs, ancestry, evaluation protocols, aggregate results, resource accounting, and claim limits.

## Documentation language

Project documentation is English by default. This applies to engineering proposals, research notes, experiment reports, handoffs, reference material, and maintenance documentation.

Chinese is retained only for material explicitly written for Chinese-language promotion:

- the Chinese launch article and its distribution copy;
- Chinese claim sheets and editorial working files used to maintain that article;
- Chinese partnership or community copy intended for a Chinese audience;
- the link labels needed to expose those documents from the README and documentation navigation.

An English source and a Chinese publication may coexist. Supporting research and engineering records remain in English even when they discuss the Chinese publication.

## Current release assessment

The existing evidence supports a first round of promotion. A larger Zstandard result, paper-grade search controls, and cross-architecture replication are not release blockers. The purpose of the first release is to explain what Loreley does, state what each case demonstrates, and find design partners willing to provide repositories, evaluators, and compute. It does not claim that quality-diversity search is better than every simpler search strategy.

The release package is complete:

- unified evidence report: [2026-08-07-loreley-case-study-evidence-report.md](2026-08-07-loreley-case-study-evidence-report.md);
- Zstandard V19 report: [2026-08-07-zstandard-gpt-v19-case-study-report.md](2026-08-07-zstandard-gpt-v19-case-study-report.md);
- Zstandard Top-10 validation, fresh-corpus, and holdout supplement: [2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md);
- launch claim sheet: [2026-08-loreley-launch-claim-sheet.md](../marketing/2026-08-loreley-launch-claim-sheet.md);
- English article: [2026-08-loreley-launch-article-en.md](../marketing/2026-08-loreley-launch-article-en.md);
- Chinese article: [2026-08-loreley-launch-article-zh.md](../marketing/2026-08-loreley-launch-article-zh.md);
- summaries and community copy: [2026-08-loreley-launch-copy-kit.md](../marketing/2026-08-loreley-launch-copy-kit.md);
- partnership intake: [loreley-design-partner-brief.md](../marketing/loreley-design-partner-brief.md) and the public GitHub issue form;
- four data-driven figures and their SVG sources: [marketing assets](../marketing/assets/loreley-search-loop.png).

README, the documentation home page, and MkDocs navigation point to the three case studies and V19. V13 is retained only as historical evidence about infrastructure and binary equivalence.

## Publication status and next work

The GitHub and documentation-site material was published through [PR #54](https://github.com/NeapolitanIcecream/loreley/pull/54). The Pages build used source commit `018c144` and produced `gh-pages` commit `c66dc01`. The [deployment run](https://github.com/NeapolitanIcecream/loreley/actions/runs/31249188262) succeeded. The home page, both articles, the unified evidence report, the candidate-diff index, and the partnership page passed HTTP and title checks.

The fixed-Top-10 holdout evidence was merged through [PR #60](https://github.com/NeapolitanIcecream/loreley/pull/60). It adds a post-selection comparison on the original holdout without changing the preregistered winner.

Public pages:

- Chinese article: <https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>
- English article: <https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>
- unified evidence report: <https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/>
- candidate diffs: <https://neapolitanicecream.github.io/loreley/marketing/candidates/>
- partnership brief: <https://neapolitanicecream.github.io/loreley/marketing/loreley-design-partner-brief/>

Next actions:

1. publish to Chinese channels, then distribute the Chinese short copy, English article, and English community posts;
2. add an FAQ from reader questions and triage partnership submissions;
3. continue the paper track with equal-budget quality-diversity, champion-sequential, and root-independent arms, independent search replicates, x86-64 reproduction, and studies of finalist selection under noisy objectives.

No external community post has been made from this repository workflow.

## Positioning

### Intended users

Loreley is not scoped by implementation language. Its current evaluator interface is Python, but an evaluator plugin may invoke builds, tests, containers, hardware benchmarks, or remote systems written in any language.

The intended users are engineering and research teams that have:

- automated correctness gates and a numerical objective;
- a codebase or algorithm worth repeated optimization;
- enough value in a successful change to justify search;
- enough evaluation capacity for the target.

### Technical thesis

The state space of a software repository is very large, while states that build, pass tests, and improve a real objective are sparse. Coding agents use source semantics, repository context, and execution feedback to propose coherent edits across files. This makes repository states available to evaluator-guided search.

Relative to AlphaEvolve, Loreley applies evaluator-guided program evolution to existing software repositories. A Git commit is a reproducible source representation and an ancestry node. The repository is the candidate, and the search retains several valid lineages that can be revisited or combined.

The Zstandard experiments require a more precise identity model. A commit is not necessarily a distinct evaluator-relevant state. In V13, 48 passed reports from six Git trees compiled to one ARM64 executable. V19 therefore used evaluator-provided release-binary identity for archive admission, endpoint accounting, and finalist freezing. Its 211 successful jobs produced 167 distinct binaries; 44 successful jobs reproduced an existing binary, and 19 of those reused cached measurements after the cache was enabled.

For compiled targets, many repository states collapse into the same artifact or evaluator behavior. Loreley should record commit, tree, artifact, and evaluation identity separately.

### SATLUTION scale

Report both units:

- about 70 repository-evolution cycles;
- about 400 candidate evaluations per cycle.

Under the comparison used here, each candidate evaluation is close to a Loreley job, so the total is about 28,000 evaluations. A reference to tens of thousands of iterations is acceptable only when the unit is stated. Do not equate a cycle, a code candidate, a solver-instance evaluation, and a full benchmark round.

## Case-study roles

### `markdown-it-py`

- 64 jobs: 8 manual seeds and 56 model-driven evolution jobs;
- the selected candidate improved throughput by 6.75% on an independent 28-document validation set;
- all 28 documents improved;
- this is the strongest clean validation result in the current package.

### `python-pathspec`

- 64 jobs: 6 manual seeds and 58 model-driven evolution jobs;
- the final valid candidate improved throughput by 25.14% on reference workloads;
- it was selected after the initial training choice failed the reference allocation gate;
- it demonstrates multi-generation editing and archive lineage retention, but it is not a clean prospective success.

### Zstandard

- V19 ran 220 physical jobs: 8 manual seeds and 212 evolution jobs;
- 211 successful jobs produced 167 distinct release binaries;
- the preregistered winner improved single-threaded compression throughput by 1.019% on a sealed holdout, with a 95% confidence interval from +0.962% to +1.076%;
- decompression was neutral, compressed size was unchanged, and peak RSS increased by 0.063 MiB;
- the preregistered winner was a nine-line manual seed, so the primary result demonstrates retention, ranking, and independent validation, not an evolved candidate beating the best seed;
- a post hoc Top-10 expansion found a generation-4 candidate that improved compression by 0.891% on a separately sealed fresh corpus, with a 95% confidence interval from +0.522% to +1.261%;
- a later fixed-Top-10 comparison on the original holdout found all ten candidates modest-positive, with a median compression gain of 1.116% and point estimates from +0.856% to +1.239%;
- generation-3 `5ee53426` and generation-4 `fe39bee8` ranked first and second descriptively by compression lower bound, at +1.228% and +1.173%, respectively;
- the original holdout had already been revealed for the preregistered winner, so the fixed-Top-10 comparison is post-selection sensitivity evidence and does not revise that winner.

Zstandard supplies evidence for a mature C system, binary-aware candidate identity, separated training, validation, and holdout data, statistical measurement of a small performance effect, and the effect of finalist breadth on noisy rankings.

All three cases establish system behavior only on their frozen targets. They do not estimate Loreley's average effect on arbitrary repositories.

## Unified metrics for the first two cases

Speedup is `candidate throughput / root throughput`. A `1.0675x` ratio is a 6.75% increase in work per unit time. For a fixed workload, the corresponding latency reduction is `1 - 1 / speedup`, or 6.33%. Do not interchange these percentages.

### Performance and validity

| Metric | `markdown-it-py` | `python-pathspec` |
| --- | ---: | ---: |
| Frozen upstream revision | `97aff4f564e` | `6568072c2703` |
| Campaign | 64 jobs: 8 seeds + 56 evolution | 64 jobs: 6 seeds + 58 evolution |
| Terminal outcomes | 54 succeeded, 10 failed; 84.4% success | 45 succeeded, 19 failed; 70.3% success |
| Best manual seed on training | `1.032328x`, +3.23% | `1.1227x`, +12.27% |
| Final candidate on training | `1.069911x`, +6.99% | `1.2536x`, +25.36% |
| Final validation or reference result | `1.067538x`, +6.75% | `1.2514x`, +25.14% |
| Equivalent fixed-work latency reduction | 6.33% | 20.09% |
| Training-to-final gap | 0.237 percentage points | about 0.21 percentage points |
| Validation or reference coverage | 28 documents; 28/28 improved | 5 scenarios; 5/5 improved |
| Per-item range | `1.007149x` to `1.171532x` | `1.1550x` to `1.3673x` |
| Peak allocation | 3.488990 MiB, 0.099% below root | 0.04354 MiB, below the 0.05 MiB gate |
| Correctness and scope | Full tests, output and semantic checks, wheel, installed CLI/API, and scope passed | 197 tests, 276 skips, 142 subtests, semantic, API, and scope checks passed |
| Evidence grade | Winner frozen before the independent 28-document validation; preregistered strong outcome | Final candidate chosen after the initial candidate failed reference allocation; capability and mechanism evidence |

For `markdown-it-py`, raw per-document speedups range from `1.007149x` to `1.171532x`, with a median of `1.068380x`. The aggregate geometric mean is `1.067538x`. These are distinct statistics.

For `python-pathspec`, the initial candidate scored `1.2633x` on training and `1.2619x` on reference workloads, but used 0.06472 MiB peak allocation on the reference shape, above the 0.05 MiB gate. The final candidate scored `1.2536x` and 0.02942 MiB on training, then `1.2514x` and 0.04354 MiB on reference workloads. Its performance and validity results stand, but the reference data had already been revealed.

| `python-pathspec` reference scenario | Final speedup |
| --- | ---: |
| Compile 150 gitignore patterns | `1.3673x` |
| `GitIgnoreSpec` match, 150 patterns | `1.2796x` |
| `PathSpec` match, 150 patterns | `1.2384x` |
| `PathSpec` match, 2 patterns | `1.1550x` |
| `PathSpec` match, 40 patterns | `1.2265x` |
| Geometric mean | `1.2514x` |

The concise reports for these cases do not provide confidence intervals for the primary speedups or variance across independent searches. Similar training and final point estimates do not replace either measurement.

### Search process

| Metric | `markdown-it-py` | `python-pathspec` |
| --- | ---: | ---: |
| Winner | Job 26, generation 4 | Job 38, generation 4 |
| Commit | `b10adb6fad0d` | `9d977f0a73d5` |
| Training gain over best seed | 3.64% throughput | 11.66% throughput |
| Final diff | 5 files, +54/-14 | 5 files, +127/-51 |
| Archive | 36 entries, 18 island/cell coordinates | 28 entries, 19 island/cell coordinates |
| Lineage evidence | Combined inspiration from seeds 3, 4, and 8 over four generations | A generation-3 branch was retained while 20 other jobs entered the archive, then sampled again to produce the winner |

The final candidates exceeded the best manual seeds and show that the archive retained and revisited lineages. Neither experiment ran equal-budget root-independent or champion-sequential model arms. They do not establish a causal advantage for quality-diversity search.

### Cost and runtime

| Metric | `markdown-it-py` | `python-pathspec` |
| --- | ---: | ---: |
| Reported generation cost | $2.0833 | $2.4856 |
| Mean generation cost per evolution job | $0.0372 | $0.0429 |
| Mean generation cost per scheduled job | $0.0326 | $0.0388 |
| Embedding cash cost | Not reported | Not reported |
| Host and labor cost | Not monetized | Not monetized |
| Campaign wall time | 4.35 h | 3.91 h |
| Terminal throughput | 14.73 jobs/h | 16.4 jobs/h |
| Median job time | 10.02 min | 9.98 min |
| Generation requests | 3,792 | 3,977 |
| Generation tokens | 215,349,501 | 241,634,477 |
| Embedding tokens | 199,343 | 258,055 |

Together, the two campaigns ran 128 jobs: 14 manual seeds and 114 evolution jobs. There were 99 successes and 29 failures. Generation used 7,769 requests and 456,983,978 tokens; embeddings used 457,398 tokens. Campaign wall time summed to 8.26 hours. The reports record $4.5689 of generation cost, averaging $0.0401 per evolution job and $0.0357 per scheduled job.

Embedding, host, development, and manual-seed costs were not monetized. The $4.5689 total is not an all-in cost. The two speedups must not be averaged because the workloads, baselines, and distributions differ.

### Publication checks

The previously identified inconsistencies are resolved in the public-facing reports:

1. The old `python-pathspec` evidence report retains a $105.57 proxy ledger, including $105.00 from reservation fallbacks. The final case study and unified report use the $2.4856 request-level proxy estimate and state that it is not a provider bill or all-in cost.
2. The `markdown-it-py` per-document median is `1.068380x`. The `1.067538x` value is the geometric mean.
3. The `python-pathspec` primary outcome is reported as invalid under the original preregistration. The later candidate is described as the final valid candidate and as capability evidence, not as a clean prospective result.

## Zstandard V19 result

The formal evidence is in the [V19 report](2026-08-07-zstandard-gpt-v19-case-study-report.md) and [Top-10 supplement](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md). V19 used upstream revision `82d322c4973d9e2968d94047a40892bc6d9a9bdf` and experiment root `5b3fe474e4df572a7588be7abf3d8b6bd4b6010e`. It covers single-threaded levels 1, 3, and 5 on one Apple-silicon host.

### Preregistered holdout

All ratios are `candidate throughput / root throughput`.

| Holdout metric | Ratio or delta | 95% confidence interval |
| --- | ---: | ---: |
| Compression throughput | `1.01019x`, +1.019% | `1.00962–1.01076x`, +0.962% to +1.076% |
| Decompression throughput | `1.00010x`, +0.010% | `0.99890–1.00130x`, -0.110% to +0.130% |
| Combined throughput | `1.00513x`, +0.513% | `1.00441–1.00585x`, +0.441% to +0.585% |
| Worst measured cell | `0.99817x`, -0.183% | Not reported separately |
| Maximum compressed-size ratio | `1.00000x` | Not applicable |
| Peak RSS delta | +0.063 MiB | Not applicable |

The holdout used 12 interleaved root/candidate pairs. Compression and combined intervals exclude `1.0x`; the decompression interval crosses it and is reported as neutral. The candidate passed upstream checks, release build, bidirectional root/candidate decoding, compressed-size and RSS gates, and the conclusion audit.

The preregistration defined a strong result as a compression point estimate of at least `1.02x` with a lower 95% bound of at least `1.01x`. The observed point estimate was `1.01019x`, so the outcome is `modest-positive`. A 1.019% throughput improvement corresponds to about a 1.01% fixed-work latency reduction, not 1.019%.

The winner was commit `7b9aef38ecd4` with release binary `e7e9ef6b060f…`. It was manual seed 5. The change unrolled a scalar histogram update loop by four bytes in `lib/compress/hist.c`, with eight insertions and one deletion relative to root. It introduced no corpus-specific branch, format change, dependency, or broad rewrite.

### Search, identity, and throughput

| Metric | Result |
| --- | ---: |
| Physical terminal jobs | 220: 8 seeds + 212 evolution |
| Successes and failures | 211 / 9; 95.9% success |
| Distinct successful release binaries | 167; 79.1% of successful jobs |
| Successful jobs reproducing a binary | 44; 20.9% |
| Measurement-cache reuse | 19 jobs; 25 duplicate binaries were remeasured before the cache was enabled |
| Final archive | 13 entries; 11/64 coordinates occupied |
| First 128 jobs | 125 successes, 3 failures, 102 distinct binaries; 3.19 runner-hours |
| Efficient continuation | 92 physical jobs added 65 distinct binaries; 2.12 runner-hours |
| Full search | 5.31 runner-hours; 41.4 physical jobs/hour |
| Median evaluator time for a measured job | 186.7 s |
| Median end-to-end job time | 333.6 s |
| Median evaluator time for cache reuse | 21.6 s |

Four evaluator lanes were calibrated with root-versus-root trials and alternated root-first and candidate-first order. Relative to one lane they produced roughly four times the wall-clock throughput. Aggregate root/root performance was `0.99993x` with a 95% interval of `0.99705–1.00282x`. This found no lane-level paired bias large enough to explain the 1% holdout effect on this host. It does not establish that four lanes are safe on every host.

The nine failures were four upstream or candidate-test failures, four measurement-precision rejections, and one compressed-size gate failure. Failed and precision-invalid results did not enter the archive or measurement cache.

V19 recorded commit, Git tree, and evaluator identity separately. Archive and finalist operations deduplicated by release binary, so the V13 finalist-duplication problem did not recur. Source proposals still converged to evaluator-equivalent states: 44 of 211 successful jobs reproduced an existing executable. Measurement reuse saves evaluation work but does not improve proposal diversity.

### Top-10 sensitivity follow-up

The preregistered protocol validated only the top three training candidates and selected the manual seed. After preserving that result, the experiment expanded the frozen training ranking to the top ten. It reused the first three validation reports and ran eight validation rounds for ranks 4 through 10. The seven new validations consumed 2,348 seconds, about 39 minutes, and no model calls.

Training rank 10, generation-4 candidate `fe39bee8`, became the expanded validation winner at `1.01234x` compression with a lower 95% bound of `1.01156x`. After selecting it, the experiment chose a generator recipe and seed procedure, generated a new disjoint 16 MiB corpus, and sealed that corpus before a 12-round fixed-candidate measurement.

| Fresh-corpus metric | Ratio or delta | 95% confidence interval |
| --- | ---: | ---: |
| Compression throughput | `1.00891x`, +0.891% | `1.00522–1.01261x`, +0.522% to +1.261% |
| Decompression throughput | `0.99830x`, -0.170% | `0.99471–1.00191x`, -0.529% to +0.191% |
| Combined throughput | `1.00359x`, +0.359% | `1.00004–1.00716x`, +0.004% to +0.716% |
| Maximum compressed-size ratio | `1.00000x` | Not applicable |
| Peak RSS delta | +0.031 MiB | Not applicable |

The candidate's four-generation lineage combined a zero-literal fast path, compression hot-path edits, and an eight-byte histogram unroll. Relative to root it changed three files, +33/-16. The candidate remained positive on the new corpus, but its construction was candidate-aware. The result does not establish that the evolved candidate beats the manual seed, because they were not compared on the same new corpus under a frozen head-to-head protocol.

The follow-up also shows that fine-grained training ranks were unstable. The top ten compression lower bounds differed by only 0.276 percentage points, less than the uncertainty in the training estimates. A future finalist rule should cover at least ten candidates or use a preregistered effect band or adaptive-racing rule.

### Fixed Top-10 original-holdout comparison

The candidate identities and ordering were fixed before further measurement. The comparison reused the registered winner's 12-round holdout report and measured the other nine candidates for 12 rounds each on the same original holdout. It used one lane, made no model calls, and retained the original winner label.

All nine new measurements passed. Every candidate met the V19 modest-positive rule, and every compression lower bound remained above root. The median compression gain was 1.116%, and point estimates ranged from +0.856% to +1.239%. None met the strong rule because no point estimate reached `1.02x`.

| Candidate | Relation to search | Holdout compression | 95% confidence interval | Descriptive rank by lower bound |
| --- | --- | ---: | ---: | ---: |
| `5ee53426` | generation-3 evolved descendant of the registered seed | `1.01228x`, +1.228% | `1.01125–1.01330x`, +1.125% to +1.330% | 1 |
| `fe39bee8` | generation-4 expanded-validation winner | `1.01173x`, +1.173% | `1.01102–1.01245x`, +1.102% to +1.245% | 2 |
| `7b9aef38` | preregistered manual-seed winner | `1.01019x`, +1.019% | `1.00962–1.01076x`, +0.962% to +1.076% | 5 |

The first two expanded-validation candidates became the first two descriptive holdout candidates in reverse order. Their intervals overlap, and the median holdout interval width was 0.327 percentage points. The result supports generalization of the fixed Top 10 as a group; it does not establish that `5ee53426` is better than the other leaders.

The original holdout had already been revealed for `7b9aef38`. This comparison is therefore post-selection sensitivity evidence, even though the candidate set was fixed before the nine new measurements. It cannot be reported as a new blinded winner. The nine measurements consumed 4,404 seconds, or 73.4 minutes, of local compute.

### V19 usage and cost

V19 recorded 52,653,004 tokens, including cached input and embeddings. The Kilo catalog covered all 424 generation sessions: $57.8499 for planning and $2.3973 for coding, totaling $60.2472. This is a model-catalog estimate, not provider-billed spend. The 303 embedding events have token counts but no recorded price.

The V19 amount is not directly comparable with the proxy-calculated DeepSeek generation-cost estimates for `markdown-it-py` and `python-pathspec`. Do not add all three values into a project cash-spend total. Use the [unified evidence report](2026-08-07-loreley-case-study-evidence-report.md) for the cost definitions.

Formal Top-3 validation took about 16.7 minutes, and the registered holdout took 8.1 minutes. The Top-10 expansion took 39.1 minutes, the fresh-corpus measurement took 8.1 minutes, and the nine new fixed-Top-10 holdout measurements took 73.4 minutes. The three follow-ups used local evaluation only and generated no model tokens.

### Public use and claim boundary

V19 is not the headline for the largest speedup, and it does not provide a new blinded comparison between evolved candidates and manual seeds. It provides five forms of evidence:

- evaluator integration with a mature non-Python C repository;
- paired measurement, confidence intervals, sealed data, and correctness gates for an approximately 1% effect;
- separate commit, tree, binary, and measurement identities;
- a documented limitation in Top-3 finalist breadth; and
- post-selection evidence that the fixed Top 10 generalized as a group, with evolved candidates in the first two descriptive ranks.

Permitted primary statement:

> On a frozen Zstandard revision and one Apple-silicon host, Loreley evaluated 167 distinct release binaries and selected a nine-line candidate under a preregistered protocol. On a sealed holdout, it improved single-threaded compression throughput by 1.019%, with a 95% confidence interval from +0.962% to +1.076%, while decompression remained neutral and compressed size was unchanged.

Permitted supplementary statement:

> A post hoc expansion from the top three to the top ten training candidates identified a generation-4 candidate. On a separately sealed fresh corpus it improved compression throughput by 0.891%, with a 95% confidence interval from +0.522% to +1.261%. The corpus recipe was chosen after the candidate was selected.

The supplementary statement must retain the post hoc shortlist expansion and different-corpus qualifications.

Permitted fixed-Top-10 statement:

> In a later fixed-candidate comparison on the original holdout, all ten Zstandard training finalists remained positive, with a median compression gain of 1.116% and point estimates from +0.856% to +1.239%. The holdout had already been revealed for the preregistered winner, so this is post-selection sensitivity evidence rather than a new blinded winner.

If the descriptive ordering is reported, state that it is ranked by the compression lower bound, that `5ee53426` and `fe39bee8` were evolved candidates, and that the leading intervals overlap.

Do not claim a 2% improvement, a portable Zstandard speedup, a new blinded evolutionary winner, a significant difference between the leading Top-10 candidates, quality-diversity beating simpler search, likely upstream acceptance, or 220 distinct program behaviors.

## Historical Zstandard V13 record

V13 is superseded by V19 for all public summaries. It is retained to document the failures that motivated sampler restart handling, measurement isolation, and binary identity.

The historical report is [2026-08-06-zstandard-reliable-qd-case-study-report.md](2026-08-06-zstandard-reliable-qd-case-study-report.md). It describes a 256-logical-job quality-diversity campaign on upstream revision `82d322c4973d9e2968d94047a40892bc6d9a9bdf`. The original holdout-selected source was `893bf5cf9e02a703ff530116ce990c3f4dae6ad6`. Later audit showed that this commit, finalist `c1f1852b`, and earlier commit `817af317` produced the same ARM64 executable. Across the campaign, 48 passed reports from six Git trees used that binary.

### Historical result

| Holdout metric | Ratio or delta | 95% confidence interval |
| --- | ---: | ---: |
| Compression throughput, geometric mean of levels 1/3/5 | `1.010442x`, +1.044% | `1.008802–1.012085x`, +0.880% to +1.209% |
| Decompression throughput | `0.999817x`, -0.018% | `0.998067–1.001570x`, -0.193% to +0.157% |
| Combined throughput | `1.005116x`, +0.512% | `1.003455–1.006780x`, +0.345% to +0.678% |
| Compressed-size ratio at levels 1/3/5 | `1.000000x` | Not applicable |
| Peak RSS delta | +0.03125 MiB | Not applicable |

The holdout used 12 interleaved root/candidate pairs with at least three seconds per benchmark. The compression and combined intervals exclude `1.0x`; the decompression interval crosses it. These intervals cover sampling uncertainty on one host, one binary, and one corpus. They do not cover search replication, architecture variation, or the probability that Loreley finds the result.

The preregistered strong rule required a compression point estimate of at least +2% and a lower bound of at least +1%. V13 was therefore `modest-positive`.

### Historical search audit

| Metric | Result |
| --- | ---: |
| Logical campaign | 256 jobs: 7 seeds + 249 evolution |
| Physical terminal rows | 274, including 18 prospectively excluded after lifecycle failures |
| Passed candidate evaluations | 205; 80.08% of logical jobs |
| Distinct Git trees | 100 |
| Distinct ARM64 binaries | 55 |
| Tree duplicate rate | 51.22% |
| Binary duplicate rate | 73.17% |
| Final archive | 12 source/tree entries, 10 binaries, 10/64 coordinates |
| First appearance of the measured binary | `817af317`, logical completion 71 |
| Equivalent frozen finalist | `c1f1852b`, logical completion 118 |
| Holdout-selected source | `893bf5cf`, logical completion 245 |
| Active runner time | 15.72 h |
| Provider-recorded DeepSeek generation cost | $4.8277 |

The first 128 logical completions already contained the measured binary. The second 128 jobs increased distinct trees from 51 to 100, binaries from 33 to 55, archive entries from 11 to 12, and occupied coordinates from 9 to 10, but did not produce the final measured executable. Its $2.6407 generation cost belongs in the full campaign ledger but cannot be attributed to discovering the holdout result.

The 205 passed evaluations represented only 100 trees and 55 binaries. Forty-eight passed reports, or 23.4%, measured the eventual holdout binary. Their training compression estimates ranged from +0.887% to +1.672%. The training leader was the maximum of these repeated measurements, which makes selection bias directly observable.

Three source finalists represented only two binaries. `c1f1852b` and `893bf5cf` compiled byte-for-byte identically on the M4 host, while their separate validation runs measured +0.961% and +1.133%. That difference came from measurement noise, not from the source edit. The final generation of `893bf5cf` changed only a non-AArch64 branch and could not explain the Apple-silicon result.

V13 ran only the quality-diversity arm. Its lineage and inspiration edges show that the mechanism ran, but they do not show that it outperformed a simpler search.

### Historical runtime incidents

Three incidents remain relevant:

1. a Codex app restart invalidated inherited file descriptors, and 18 physical rows were excluded under a frozen lifecycle rule;
2. macOS `launchd` background policy reduced fixed-root throughput by 4.4 to 6.4 times and caused 11 precision failures before the process policy and measurement lock were corrected;
3. the first post-reveal audit run falsely rejected expected plaintext; the failure and fix were retained, and the corrected audit passed without changing selection or holdout data.

The 31.1-hour database window includes pauses, incident analysis, and recovery. It is not evaluator device time. V13 recorded 286,838,265 observed tokens, including 4,478,202 reasoning-output tokens, and $4.8277 of provider-recorded generation cost. Embeddings, host use, development, and manual seeds were not fully monetized.

### Historical claim boundary

V13 may be described only as a fixed-binary result on one revision and one Apple-silicon host: the campaign produced the measured ARM64 binary by logical completion 71; on the sealed holdout it improved single-threaded compression throughput by about 1%, left decompression essentially neutral, and preserved compressed size. The full 256-job campaign recorded less than $5 in provider-reported generation cost.

Do not attribute the result to the generation-3 source edit, the last 128 jobs, or cross-lineage inspiration. Do not claim a 2% gain, cross-architecture validity, general Zstandard acceleration, likely upstream acceptance, or an advantage over simpler search.

Do not reopen the holdout to compare `817af317`, `c1f1852b`, and `893bf5cf` on the current host; they produce the same executable. Record the measured artifact as binary SHA-256 `2cddc94…` and retain the three source identities for first observation, frozen finalist, and protocol-selected holdout source.

## Related work, 2025–2026

Existing Loreley documentation is not an authoritative literature review. Research positioning must be rebuilt from papers, official research pages, and maintained implementations, with particular attention to work published from 2025-08 through 2026-08.

The review must establish:

1. how the candidate unit and evaluator changed from FunSearch and AlphaEvolve to recent systems;
2. recent evidence on sample efficiency, search strategy, repository scale, runtime feedback, baselines, and trace analysis;
3. which systems already use MAP-Elites, islands, full repositories, or enterprise optimization;
4. the closest comparison for each part of Loreley's design; and
5. which comparative claims require controlled experiments.

### From local functions to engineered repositories

| Work | Candidate and evaluator | Implication for Loreley |
| --- | --- | --- |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6), Nature 2023 | Evolves one Python function inside a human-written scaffold and accumulates executable candidates | Establishes the LLM-plus-evaluator-plus-evolution pattern for a controlled local object |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131), 2025-06 | Extends the object to whole files, hundreds of lines, arbitrary languages, multiple metrics, and expensive external evaluation | Establishes the general LLM-plus-evolution-plus-evaluator design from which Loreley starts |
| [EvoGit](https://arxiv.org/abs/2506.02049), 2025-06 | Uses Git branches, commits, and merges to organize coding agents and lineages | Closest general-purpose precedent for Git-based agent lineages; its objective is collaborative software construction rather than evaluator-guided QD optimization |
| [SATLUTION](https://arxiv.org/abs/2509.07367), 2025-09 | Evolves a complete C/C++ SAT solver repository for 70 cycles and evaluates each solver revision on 400 instances | Full-repository optimization with roughly 28,000 solver-instance executions, specialized to SAT solving |
| [ABCEvo](https://arxiv.org/abs/2604.15082), DAC 2026 | Three agent roles modify an approximately 1.2-million-line, 4,000-file ABC repository with compilation, eight flows, suites, and formal equivalence | Full-repository optimization specialized to an EDA tool and its QoR pipeline |
| [CodeEvolve](https://arxiv.org/abs/2605.04677), 2026-05 | Optimizes enterprise Java and Apex using profiling, component graphs, MCTS, builds, tests, and performance evaluation | General software setting, but the evolutionary unit is a selected method or writable code block with surrounding read-only context |
| [Vesper](https://arxiv.org/abs/2605.15221), ICML AI for Science Workshop 2026 | Treats a repository branch as a candidate, runs coding agents in Git worktrees, and retains branches with an island model | Closest coding-agent harness design; evaluated on Circle Packing and explicitly omits MAP-Elites |
| [HORIZON](https://arxiv.org/abs/2606.28279), 2026-06 | Expands a Markdown harness into a hardware repository in isolated Git worktrees and retains accepted commit traces | Full-repository evolution specialized to hardware-design artifacts and hardware evaluators |

### Archives and open-ended lineages

| Work | Method | Implication for Loreley |
| --- | --- | --- |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), ICLR 2026 | Coding agents modify their own system and maintain an archive of improvement paths | Supports studying stepping stones, but the search object is the agent itself |
| [ShinkaEvolve](https://openreview.net/forum?id=lKEdGCoDNC), ICLR 2026 | Adaptively chooses parents, models, and prompts and rejects low-novelty proposals | Loreley must compare its fixed or learned diversity mechanisms with simpler adaptive sampling |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | Provides open-source islands, MAP-Elites, and multi-objective program evolution | Islands and MAP-Elites alone are not research contributions |
| [GEAR](https://arxiv.org/abs/2605.13874), 2026-05 | Maintains a population of machine-learning research states and compares it with single-path AutoResearch under the same compute budget | Direct evidence for multi-state agent search in one experimental domain and a useful model for Loreley's future controlled comparison |
| [Evolutionary Ensemble of Agents](https://arxiv.org/abs/2605.09018), 2026-05 | Co-evolves code solutions and agent guidance or skills | Distinguishes searching the target repository from searching the agent strategy; Loreley currently focuses on the former |

### Evaluators as validation pipelines

FunSearch primarily scores a local function. AlphaEvolve supports several metrics and expensive external computation. SATLUTION, ABCEvo, and HORIZON combine builds, benchmark suites, formal checks, edit policies, and commit acceptance. A repository evaluator therefore has four separate duties: reject invalid states, compute optimization objectives, constrain editable scope, and separate search feedback from final validation.

Loreley's Python interface is a scheduler boundary, not a language restriction. The paper should define an evaluator as a protocol that may invoke arbitrary builds, containers, hardware, or remote services. The Zstandard case demonstrates staged gates, hidden configurations, and sealed data.

### Experimental evidence on search design

- [Self-Evolving Coding Agents](https://arxiv.org/abs/2608.03392), 2026-08,
  organizes the field by the object that evolves, when evolution occurs, and
  which software evidence drives it. Use this taxonomy to distinguish
  repository-state search from agent, memory, tool, model, and collaboration
  evolution.
- [Simple Baselines are Competitive with Code Evolution](https://arxiv.org/abs/2602.16805), 2026-02, reports that independent sampling and sequential rewriting can match or exceed more elaborate evolutionary systems on some tasks. Any Loreley claim about relative search efficiency therefore needs equal-budget root-independent and champion-sequential controls.
- [What Do Evolutionary Coding Agents Evolve?](https://arxiv.org/abs/2605.20086), 2026-05, finds that gains can come from parameter tuning, recombination, overfitting, or reintroduction of existing code. Publish ancestry, edit taxonomy, replay, and holdout results rather than only a winner score.
- [HORIZON](https://arxiv.org/abs/2606.28279) reports reward hacking and over-solving risks. Keep visible training feedback, hidden validation, and a sealed holdout separate.
- [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189), 2025-10, identifies automated performance verification as a condition for agent-driven search in systems research. This supports defining the user by evaluator readiness.

### Enterprise use

Google DeepMind's [May 2026 AlphaEvolve update](https://deepmind.google/blog/alphaevolve-impact/) describes AlphaEvolve as a regular infrastructure tool and reports a 20% reduction in write amplification from improved Spanner LSM compaction heuristics, along with finance, semiconductor, logistics, and life-science cases.

Google announced [AlphaEvolve on Google Cloud](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/alphaevolve-on-cloud/) on 2026-07-09 through the Gemini Enterprise Agent Platform. Evaluator-guided program evolution is already an enterprise service category.

An LSM case cannot claim novelty from optimizing LSM alone. A future RocksDB study would need public, auditable repository search, equal-budget controls, and lineages across throughput, write amplification, and tail latency.

### Paper position

The paper studies a general-purpose quality-diversity search system for existing Git repositories. The closest full-repository systems in the reviewed AlphaEvolve line are specialized to SAT, EDA software, or hardware design. The closest general-purpose systems either organize agent lineages without evaluator-guided QD search or restrict evolution to selected functions and code blocks. Vesper is the closest harness-level comparison: it uses coding agents, repository branches, worktrees, and islands, but evaluates one Circle Packing setting and omits MAP-Elites.

Loreley's paper contributions are:

- a target-independent contract that connects an existing repository, a coding-agent backend, and an arbitrary external evaluator;
- complete Git commits as candidate states, with recorded ancestry and cross-lineage inspiration;
- separate source, tree, artifact, and evaluator identities for deduplication and measurement reuse;
- repository-state behaviour descriptors, MAP-Elites cells, bounded Pareto fronts, and multiple islands;
- asynchronous execution with recorded model, token, evaluation, device-hour, and wall-time usage; and
- three fixed-repository studies covering Python libraries and a compiled C/C++ system.

The paper should explain these contributions directly and give most of its space to the method, implementation, experiments, and observed search behavior. Related work should locate the closest comparisons once. Evidence qualifications belong with the affected result and in the limitations section instead of being repeated throughout the introduction and conclusion.

An equal-budget advantage over independent or champion-sequential search remains a separate empirical claim.

## Third-case selection and pilot record

Zstandard was selected and V19 is complete. This section retains the RocksDB screen, alternative candidates, and preliminary Zstandard measurements. RocksDB remains a possible collaboration case when dedicated storage hardware is available.

Selection criteria included clean and incremental build time, correctness-gate time, short-benchmark noise, validation time, evaluations per day, parallelism constraints, and device-hours for 256, 1,024, and 28,000 evaluations.

### RocksDB pilot

The 2026-08-03 pilot used RocksDB revision `4b35e9966c821b7bf29de3b042f405f30acc635e` (`db_bench` 11.9.0) on a 14-core Apple M4 Pro with 24 GB RAM, local SSD, and Apple Clang 21. This was not a dedicated Linux NVMe host, so the results address throughput feasibility only.

| Operation | Wall time |
| --- | ---: |
| Clean release `db_bench` build | 66.59 s |
| Clean debug build with three relevant test binaries | 84.66 s |
| Incremental release rebuild after five compaction/write source edits | 2.54 s |
| Incremental rebuild of the three debug test binaries | 8.82 s |
| `compaction_job_test`, 35 tests | 3.16 s |
| `db_write_test`, 52 tests | 7.48 s |
| `write_controller_test`, 4 tests | 0.64 s |

A typical five-file candidate required about 22.6 seconds for incremental release and debug builds plus 91 tests. Edits to high-fanout headers could approach clean-build time.

The ten-second write/compaction proxy used a fixed seed, four write threads, a 64 MiB write buffer, no compression, and a wait for background compaction. Five runs processed about 1.18 to 1.24 GB and produced 3.96 to 4.02 GB of cumulative compaction writes.

| Run | Operations per second | Full wall time |
| ---: | ---: | ---: |
| 1 | 313,445 | 15.12 s |
| 2 | 322,100 | 15.22 s |
| 3 | 316,817 | 15.17 s |
| 4 | 307,507 | 15.16 s |
| 5 | 310,012 | 15.18 s |

Mean throughput was 313,976 operations per second, with a sample standard deviation of about 5,743 and a CV of 1.83%. A single 15-second proxy can reject large regressions but cannot reliably distinguish a 1% to 3% gain. Five repeats take about 75.9 seconds and can support a training promotion threshold near 5%.

Estimated candidate cost:

- one smoke run with builds and 91 tests: about 38 seconds;
- a five-repeat training evaluation: about 99 seconds;
- a broad-change upper bound near a clean rebuild: about four minutes;
- at 99 seconds, 256, 1,024, and 28,000 evaluations require about 7, 28, and 770 device-hours.

Under ideal independent-storage scaling, 28,000 evaluations would take about 24 hours on 32 workers or 12 hours on 64. Several I/O jobs sharing one drive are not independent workers.

The pilot passed the throughput screen but not the validity screen. The database fit in memory, the run lasted ten seconds, and there were no mixed reads and writes, tail-latency measurements, hidden configurations, or dedicated Linux NVMe results.

#### Cost of measuring a 1% RocksDB effect

The power estimate used a two-sided 5% significance level, 80% or 90% power, the observed 1.829% CV, 15.17 seconds per benchmark, and 22.64 seconds for one incremental build and test gate.

| Design | 80% power | Total time | 90% power | Total time |
| --- | ---: | ---: | ---: | ---: |
| Root mean fixed from extensive history | 29 candidate runs | 7.7 min | 38 candidate runs | 10.0 min |
| Interleaved paired A/B, assuming 0.5 adjacent-run correlation | 29 pairs, 58 runs | 15.0 min | 38 pairs, 76 runs | 19.6 min |
| Independent root and candidate samples | 54 per group, 108 runs | 27.7 min | 72 per group, 144 runs | 36.8 min |

A fixed historical root mean does not account for temperature, SSD state, or background-load drift. The preferred paper design is randomized interleaved A/B with directly measured pairwise variance. Under the provisional 0.5 correlation, one workload cell costs 15 to 20 minutes, nine to twelve times the training evaluator. Applying that precision to all 28,000 evaluations would require about 7,000 to 9,200 device-hours. Three workload cells would cost about 45 to 60 minutes per candidate before full tests.

The CV came from only five repeats. Under an independent normal-noise assumption, its standard-deviation 95% interval is about 1.10% to 5.26%. Sample size scales approximately with `CV^2 / effect^2`; a true 3% CV would increase the estimate by roughly 2.7 times. A formal run needs 30 to 50 baseline repeats across seeds, thermal states, orders, and 10-, 30-, 60-, and 120-second workloads.

Final precision should not be paid for every search candidate. Use a short multi-fidelity gate for training, paired measurement for a frozen small finalist set, and multiple workloads with a sealed holdout for the final candidate.

### Faster alternatives considered

| Candidate | Speed evidence | Research value | Main limitation | Decision |
| --- | --- | --- | --- | --- |
| Zstandard | Measured locally; in-memory CPU benchmark | Mature C library with natural speed, size, and memory trade-offs | Requires hidden corpora, anti-specialization checks, and fixed builds | Selected |
| SQLite | Official Cachegrind suite runs about 30,000 SQL statements and reports at least seven reproducible significant digits, allowing 0.05%–0.1% micro-optimization measurement | Database kernel and query execution | Cachegrind is a CPU proxy; full correctness is complex; x86 Linux timing was not measured | Precision-oriented alternative |
| DuckDB | Official runner supports repeated runs and result checks; a 2024 project report put the then-current full suite below 35 seconds | Full analytical database with varied workloads and operators | Build cost and current suite time require measurement | Stronger system target after a pilot |
| CaDiCaL or another SAT solver | Fast build and functional suites covering API, CNF, proof, trace, and model-based checks | Formal validity and an important domain | Performance needs many hard instances and timeouts; overlaps SATLUTION | Not a speed-driven alternative |

SQLite's precision figures come from its [CPU measurement guide](https://sqlite.org/cpu.html), which also states that Cachegrind measures a CPU proxy rather than real I/O latency.

### Zstandard pilot

The 2026-08-03 pilot used upstream revision `82d322c4973d9e2968d94047a40892bc6d9a9bdf`, version 1.6.0, on the same M4 Pro host. The repository had 658 tracked files and about 138,614 lines across 275 C/C++ source or header files.

Zstandard's [benchmark mode](https://github.com/facebook/zstd/blob/dev/programs/zstd.1.md) repeatedly compresses and decompresses inputs in memory. The project also maintains [automated build-to-build benchmarking](https://github.com/facebook/zstd/blob/dev/tests/automated_benchmarking.py) with a 1% regression threshold.

| Operation | Wall time |
| --- | ---: |
| Clean release build | 2.79 s |
| Incremental build after one core compression-source edit | 1.30 s |
| `make check` | 23.20 s |
| One corpus, one level, `-i1` | 2.07 s |
| One corpus, levels 1 through 5, `-i1` | 13.61 s |

On a fixed release binary, 15 repeats gave a compression CV of 0.477% and a decompression CV of 0.561%. Under the preliminary independent-sample calculation, a one-level evaluator for a 1% effect cost 53 to 58 seconds, and levels 1 through 5 cost 3.6 to 4.0 minutes.

The proposed design protected `programs/`, `tests/`, evaluators, and corpora; used public training and hidden validation corpora; treated compression speed, decompression speed, compressed size, and peak memory as separate objectives; added round-trip and compatibility gates; and reserved cross-architecture measurements for finalists.

Zstandard was selected because its high-precision evaluator was roughly 4 to 15 times faster than the provisional RocksDB design, required no dedicated NVMe device, and exposed a direct multi-objective problem.

## Controlled comparison track

The first paper draft is a systems and empirical preprint based on the three completed case studies. It can describe Loreley's method, implementation, results, and cross-case findings without making a relative search-efficiency claim.

A later comparative section can test whether Loreley quality-diversity search uses a fixed budget more effectively than:

1. Loreley quality-diversity search;
2. sequential editing of the current champion;
3. independent best-of-N proposals from the root.

For every arm, freeze the target, model, visible training feedback, evaluator gates, budgets, seed policy, validation and holdout partitions, and winner rule. Report model requests, tokens, candidate evaluations, distinct evaluator identities, evaluator device-hours, and wall time.

V19 completed three-stage data separation, binary-aware archive admission, a distinct-binary endpoint, and binary-aware finalist freezing. It ran only the quality-diversity arm. Existing ancestry cannot substitute for the two controls or for cross-architecture replication.

Further experiments can address four questions raised by V19:

1. Across at least three independent searches, what are the discovery rate and time-to-first-useful-candidate distribution?
2. How much does evolution contribute under no-seed, weak-seed, and current manual-seed conditions?
3. Can Top-10 validation, an effect band, or adaptive racing avoid a Top-3 miss at acceptable cost?
4. Should noisy-objective archive admission use incumbent/challenger remeasurement, confidence-bound dominance, or another rule?

Tree identity and evaluator-state identity are now explicit system choices. Test phased measurement reuse and evaluator-defined identity on a second compiled or generated-artifact target instead of applying another post hoc collapse to V19.

## Publication material

### Articles

English title:

> Searching Real Code Repositories with Coding Agents

English subtitle:

> Results from 348 Loreley jobs on `markdown-it-py`, `python-pathspec`, and Zstandard

The Chinese article uses a direct translation of this title and subtitle. Its source is [2026-08-loreley-launch-article-zh.md](../marketing/2026-08-loreley-launch-article-zh.md).

The final editorial rules apply to both languages:

- syntax may change, but editing must not add a rhetorical function;
- do not add sentences merely to announce, recap, transition, or tell the reader what to take away;
- do not force experimental facts into a problem-example-lesson narrative;
- do not make headings more personal or dramatic than their content;
- do not add attitude, emotion, rhetorical questions, metaphors, or punchlines to create an authorial voice;
- use established Chinese technical terms in the Chinese article and retain English only for code identifiers or terms without a stable translation;
- retain evidence grades, selection timing, and uncertainty because they are findings, not decorative connective text;
- remove the same editorialization from the English source rather than passing it through translation.

Writing and translation workflow:

1. establish the system definition, setup, results, evidence limits, and related work in English;
2. remove each sentence that adds no fact, method, condition, evidence boundary, or action;
3. preserve the order and propositions in Chinese while correcting only unnatural Chinese syntax;
4. compare metrics, selection rules, costs, terms, and links across both versions.

The current article structure is:

1. system definition and three-case table;
2. search model, evaluator interface, and quality-diversity archive;
3. the three formal case sections;
4. resource use and cost;
5. FunSearch, AlphaEvolve, SATLUTION, ABCEvo, CodeEvolve, and HORIZON;
6. evidence limits, equal-budget controls, and integration requirements.

### Evidence package

Each case should expose:

- root and selected source diffs, canonical artifact hash, and known source/artifact equivalence classes;
- principal ancestry and inspiration edges;
- training and validation metrics;
- evaluator, scope gates, and selection rules;
- failed-candidate counts;
- tokens, requests, cost, and wall time;
- known limitations and prohibited claims.

The [unified report](2026-08-07-loreley-case-study-evidence-report.md) supplies the text package. Four published figures cover the principal lineages and statistics. The [candidate-diff index](../marketing/candidates/README.md) stores canonical patches for `markdown-it-py`, `python-pathspec`, the Zstandard registered winner, and the evolved Zstandard follow-up.

### Paper

Paper title:

> Loreley: Repository-Scale Program Evolution with Quality-Diversity Search

The first public preprint is an English systems and empirical paper. Its main
sections cover Loreley's repository-level QD method and implementation, a
matched Zstandard policy experiment, three earlier capability cases, related
work, limitations, and reproducibility. The controlled experiment separates
archive engagement from endpoint efficacy: retained alternatives were reused,
but Loreley QD did not beat Sequential Champion at the tested 48-job horizon.

The related-work section should compare candidate scope, agent interface, archive design, evaluator generality, and experimental domain in about one page. It should not turn the introduction, result sections, or conclusion into a sequence of priority disclaimers. A future controlled comparison can strengthen the paper without blocking the first complete draft.

The submitted source is available in [`paper/main.tex`](../../paper/main.tex),
with a checked-in bibliography, data-derived figures, public formal records,
and build instructions in [`paper/README.md`](../../paper/README.md). The
matched-study validator replays winner selection, endpoint mapping, primary
and secondary contrasts, sensitivity analyses, and public lineage counts.
Internal reviews and submission bundles are retained as local archives under
`output/` rather than published as project documentation.

## Decision log

### 2026-08-03

- Removed a public runnable demo from the release prerequisites.
- Confirmed that the Python evaluator interface does not restrict target languages.
- Assigned the first two cases to clean validation evidence and mechanism/lineage evidence.
- Corrected SATLUTION scale to report both cycles and candidate evaluations.
- Defined the technical thesis around agent proposals among sparse valid repository states.
- Decided to rebuild the recent literature review from primary sources.
- Piloted RocksDB as the initial third-case candidate.
- Withdrew priority claims based only on repository scale, Git, enterprise code, or hardware repositories.
- Narrowed the paper candidate to a general quality-diversity search system over complete Git repositories, subject to controls.
- Added AlphaEvolve's Spanner case and enterprise service to the business context.
- Measured a 99-second five-repeat RocksDB training evaluator with 1.83% short-workload CV.
- Estimated 15 to 20 minutes for a paired 1% RocksDB measurement, pending a larger noise study.
- Compared Zstandard, SQLite, DuckDB, and CaDiCaL and selected a CPU-bound target for faster precise evaluation.
- Measured a 53-to-58-second one-level Zstandard evaluator and a 3.6-to-4.0-minute levels-1-to-5 evaluator.
- Selected Zstandard for the third case and retained RocksDB for a later storage collaboration.
- Wrote the Zstandard handoff with equal-budget controls, hidden validation, and a sealed holdout.

### 2026-08-05

- Unified throughput, fixed-work latency, validation/reference, memory, search, resource, and evidence-grade reporting for the first two cases.
- Confirmed `markdown-it-py` as the primary clean validation result.
- Restricted the `python-pathspec` 25.14% result to capability and lineage evidence because candidate selection occurred after reference disclosure.
- Recorded and resolved the old `python-pathspec` proxy cost, the `markdown-it-py` median/geometric-mean mismatch, and `python-pathspec` outcome naming.
- Added the proxy-calculated generation-cost estimates: $2.0833 and $2.4856, totaling $4.5689 without embeddings, host, or labor.

### 2026-08-06

- Completed Zstandard V13 with 256 logical jobs and a `modest-positive` holdout result of +1.044% compression, 95% CI +0.880% to +1.209%.
- Found that the holdout-selected source and two earlier sources compiled to the same ARM64 executable.
- Established that the measured executable first appeared at logical completion 71; the second 128 jobs increased diversity but did not change the measured binary.
- Recorded $4.8277 provider-reported generation cost and 15.72 active runner-hours.
- Retained the result as a fixed-binary claim and rejected claims about the last source edit, 2%, portability, or method superiority.
- Recorded lifecycle exclusions, macOS background throttling, and the audit false positive.
- Added binary/evaluator-state deduplication to archive, finalist, and paper requirements.
- Corrected the usage table to include 4,478,202 reasoning-output tokens.

### 2026-08-07

- Replaced V13 with V19 as the public Zstandard case and retained V13 only as historical infrastructure evidence.
- V19 completed 220 physical jobs, with 211 successes, 167 distinct release binaries, and 5.31 runner-hours.
- Classified the preregistered manual-seed result as `modest-positive`: +1.019% compression, 95% CI +0.962% to +1.076%, neutral decompression, and unchanged compressed size.
- Confirmed binary-aware archive admission, endpoint accounting, finalist freezing, and measurement reuse. Forty-four successful jobs reproduced an existing binary; 19 reused measurements.
- Preserved the preregistered result, then ran the Top-10 sensitivity follow-up. Candidate `fe39bee8` achieved +0.891% compression, 95% CI +0.522% to +1.261%, on a separately sealed corpus.
- Added finalist breadth and noisy-objective archive policy to the paper questions.
- Recorded 52,653,004 tokens and a $60.2472 Kilo catalog estimate, with embeddings unpriced and no claim of provider-billed or all-in spend.
- Decided that the evidence supported promotion before new search controls or a larger Zstandard number.
- Fixed the `markdown-it-py` per-document median at `1.068380x`.
- Completed the claim sheet, both articles, copy kit, partnership brief, public intake, four SVG/PNG figures, and README/MkDocs entry points.
- Stored four canonical candidate patches and verified them byte for byte against the experiment repositories.
- Passed strict MkDocs, relative-link, SVG/XML, issue-template YAML, PNG-dimension, absolute-path, desktop, and mobile checks.
- Revised reader-facing material after reviews found internal editorial notes, abstract framing, and artificial narrative structure in the articles.
- Rewrote the English source first, then revised the Chinese translation from that source.

### 2026-08-08

- Compared the direct Chinese translation supplied by the user with the repository rewrite and two independent reviews.
- Identified structural editorialization, rather than formality or literal translation, as the main writing problem.
- Adopted the rule that syntax may change but editing must not add rhetorical function.
- Reorganized the English source as a formal technical account: system definition, result table, cases, costs, related work, evidence limits, and integration.
- Revised the Chinese article from the English information structure using established Chinese technical terms.
- Updated the copy kit, README, and MkDocs labels.
- Passed strict MkDocs, diff checks, 1440-pixel desktop rendering, and 390-pixel mobile rendering.
- Passed CI and Cremona structural checks on the publication branch, then merged [PR #54](https://github.com/NeapolitanIcecream/loreley/pull/54).
- Verified the deployed home page, both articles, unified report, candidate index, and partnership page.

### 2026-08-09

- Set English as the default language for project documentation.
- Kept explicitly Chinese promotional artifacts and their navigation labels in Chinese.
- Rewrote two engineering proposals, this research plan, and the Zstandard handoff in English.

### 2026-08-10

- Added the fixed-Top-10 comparison on the original Zstandard holdout while preserving the preregistered winner.
- Recorded that all ten candidates were modest-positive, with a median compression gain of 1.116% and point estimates from +0.856% to +1.239%.
- Recorded evolved candidates `5ee53426` and `fe39bee8` in the first two descriptive ranks by compression lower bound.
- Classified the comparison as post-selection sensitivity evidence because the holdout had already been revealed for the registered winner.
- Updated the articles, launch copy, figures, entry points, partnership material, handoff, release note, and claim boundaries to use the new evidence.
- Reorganized the Zstandard section in both articles around the same sequence as the first two cases: search budget and result, candidate changes, selection protocol, and artifact identity. Removed the evidence-stage table from the article while retaining it in the formal reports.
- Revised both Zstandard-facing figures to foreground the descriptive holdout leader, `5ee53426` at +1.228%, while retaining the 10/10 group result and post-selection boundary as secondary information.
- Set paper v0.1 as an English systems and empirical preprint centered on method, implementation, and the three completed studies. Equal-budget search controls remain a later comparative track.
- Classified the closest full-repository AlphaEvolve descendants as domain-specific SAT, EDA, or hardware systems. Recorded Vesper as the closest coding-agent harness comparison and CodeEvolve as function- or block-level enterprise optimization.
- Added Vesper and GEAR to the literature map and directed the paper to keep evidence qualifications with the affected result and in one limitations section.
- Completed the first English paper draft under the provisional title
  *Loreley: Quality-Diversity Search over Complete Git Repositories*.
- Organized the paper around the repository-level QD method, system identity
  model, experimental protocol, three results, and cross-case findings; kept
  prior-art positioning to one related-work section.
- Generated the Zstandard Top-10 figure directly from the checked-in evidence
  JSON and completed factual, citation, LaTeX-log, text-extraction, and
  eight-page visual checks.
- Added the August 2026 *Self-Evolving Coding Agents* survey to the literature
  map and paper after the final recency check.
- Prepared the v0.1 internal review package with the compiled paper, source,
  bibliography, figure inputs, claim-to-evidence matrix, and reviewer response
  form.
- Added `paper/review_notes.md` as the maintained record of paper status,
  contribution claims, changes, uncertainties, and requested review focus.
