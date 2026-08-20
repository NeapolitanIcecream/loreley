# Loreley Case-Study Evidence Report

Date: 2026-08-07

Scope: three fixed-repository case studies. The Zstandard result reported here
replaces an earlier DeepSeek campaign in the aggregate evidence story. The
earlier run remains historical evidence for infrastructure failures and does
not contribute the Zstandard result reported here.

## Executive summary

Loreley completed three end-to-end repository optimization case studies. They
cover two Python libraries and one C systems repository, use independent
evaluation outside the coding agent, and preserve successful candidates as Git
commits with recorded ancestry.

| Repository | Search | Independently measured result | Evidence status |
| --- | --- | --- | --- |
| `markdown-it-py` | 64 jobs; 8 seeds and 56 evolution jobs | Generation-4 winner was **6.75% faster** on a separate 28-document corpus; 28/28 documents improved | Candidate frozen before validation; endpoint fixed before model calls |
| `python-pathspec` | 64 jobs; 6 seeds and 58 evolution jobs | Generation-4 final candidate was **25.14% faster** on five reference scenarios; 5/5 improved | Capability evidence; candidate selected after the registered pick failed the revealed allocation gate |
| Zstandard | 220 jobs; 8 seeds and 212 evolution jobs; 167 distinct successful release binaries | Validation-selected generation-4 candidate `fe39bee8` improved original-holdout compression by **1.173%** (95% CI **+1.102% to +1.245%**) and new-corpus compression by **0.891%** (95% CI **+0.522% to +1.261%**) | Candidate selected before its original-holdout score was known, but that corpus was already open; new corpus was sealed before measurement, with its recipe chosen after candidate fixation; the preregistered protocol separately selected a manual seed |

Across the three reports, 348 terminal jobs produced 310 successful outcomes
and 38 failures. The reported campaign or active-runner times sum to 13.57
hours; this is not an all-in elapsed-time measurement. The counts use each case
study's terminal-job definition and do not estimate the probability that
Loreley will succeed on another repository.

The evidence supports a bounded claim: Loreley can use external coding agents,
repository-level evaluation, a quality-diversity archive, and recorded Git
lineage to find validated improvements in fixed repositories. It does not yet
show that quality-diversity search beats a simpler same-budget search strategy,
that the observed gains transfer across machines, or that the average result
over repositories is positive.

## What each case establishes

### `markdown-it-py`: the frozen validation result

The selected candidate combined ideas from several manual seeds over four
generations. It measured 1.0699x on training and 1.0675x on a separate
28-document corpus. All 28 documents improved, peak allocation fell slightly,
and the candidate passed the full test, semantic, wheel, installed CLI/API, and
scope checks.

This is the clearest promotion result because the candidate was frozen before
the validation corpus was used. Its limits are one repository, one search run,
human-written seeds, and no same-budget model-search baseline.

### `python-pathspec`: the clearest archive-lineage example

The final candidate followed a four-generation branch that remained in the
archive while 20 other jobs explored other branches. The branch reduced
construction overhead, removed repeated dictionaries and wrappers, and finally
flattened hot-loop matcher dispatch. It measured 1.2514x on five reference
scenarios and stayed below the allocation limit.

The initially registered training pick failed the larger reference allocation
shape. The final candidate was chosen after that failure revealed the reference
workload. Its performance and lineage are valid, but the selection process is
post-hoc and cannot be presented as a prospective confirmation.

### Zstandard: the systems and measurement case

This was a fresh run after repairing the main defects exposed by the earlier
DeepSeek campaign. It used eight independent seeds, four model workers, four
calibrated evaluator lanes, evaluator-provided binary identity, real OpenAI
embeddings, separate planning and coding models, headless Kilo sessions, and
disjoint training, validation, and holdout corpora.

The preregistered protocol selected manual seed 5. On the 12-round sealed
holdout, that nine-line histogram-loop change measured:

| Holdout metric | Root ratio | 95% interval |
| --- | ---: | ---: |
| Compression throughput | 1.01019x | 1.00962-1.01076x |
| Decompression throughput | 1.00010x | 0.99890-1.00130x |
| Combined throughput | 1.00513x | 1.00441-1.00585x |
| Maximum compressed-size ratio | 1.00000x | Not applicable |
| Peak RSS delta | +0.063 MiB | Not applicable |

The registered result is modest-positive rather than strong because the strong
rule required at least a 2% compression point gain and a 1% lower confidence
bound.

The registered Top-3 shortlist did not contain the best candidate on the
validation split. After the original conclusion was preserved, a Top-10
supplement fixed the candidate set before its seven new validation
measurements. Training rank 10, `fe39bee8`, became the validation winner. Its
expanded-validation compression ratio was 1.01234x, with a
1.01156-1.01312x interval; that interval uses the selection set and is not
selection-adjusted. It is a generation-4 candidate whose lineage combines
zero-literal and histogram hot paths.

After `fe39bee8` was fixed, a deterministic recipe and seed were chosen, and a
new disjoint corpus was generated and sealed before measurement. Compression
was 1.00891x, with a 1.00522-1.01261x interval. This is fixed-candidate evidence
on new data, but the corpus recipe was not fixed before the candidate was
known.

Because the Top-10 identities were already fixed, a later addendum measured the
entire set on the original holdout. All ten met the modest-positive rule. Their
median compression gain was 1.116%, and their point estimates ranged from
0.856% to 1.239%. Generation-3 candidate `5ee53426` ranked first descriptively,
with a 1.01228x point estimate and a 1.01125-1.01330x interval; `fe39bee8`
ranked second and the registered winner ranked fifth. The leading intervals
overlapped. `fe39bee8` had been selected on expanded validation before its own
original-holdout score was measured, so its 1.01173x result is candidate-level
out-of-sample evidence. Since the corpus had already been revealed for the
registered winner, it is not an untouched study-level holdout, and the Top-10
comparison does not create a new blinded winner.

## Search efficiency and identity

The Zstandard search ended after 220 physical jobs and 167 distinct successful
release binaries. It completed in 5.31 active runner-hours, or 41.4 terminal
jobs per runner-hour. For jobs that ran a real benchmark, median evaluator time
was 186.7 seconds and median end-to-end time was 333.6 seconds. Evaluation
occupied a median 64.9% of job time.

The first 128 jobs contained 102 distinct successful binaries. A continuation
added 65 binaries in 92 physical jobs. Across the full run, 44 successful jobs
compiled to an existing binary. Twenty-five were measured before the binary
cache was enabled; 19 later repeats reused an accepted report and reduced
median evaluator time from 186.7 to 21.6 seconds.

This result changes the unit that matters for compiled targets. A Git commit is
the reproducible source and ancestry node, while the evaluator-provided binary
identity determines whether a performance measurement is new. Both identities
must remain visible.

## Measurement uncertainty

The calibrated four-lane root/root experiment measured a maximum aggregate
log bias of 0.000314, about 0.031%. Pairing root and candidate measurements
therefore removed most host-wide contention bias even though absolute
throughput under four lanes was lower than under one lane.

Fine-grained training order remained less reliable than the final effect claim:

- the training Top-10 compression lower bounds spanned 0.276 percentage points;
- their median point-to-lower-bound distance was 0.541 percentage points;
- fixed eight-round validation reduced that median distance to 0.129 percentage
  points; and
- the validation winner had ranked tenth on training.

These numbers apply only to a narrow, training-selected frontier. They do not
measure the full-campaign training-validation correlation. They show that
four/eight-round training is adequate for screening and gross precision
rejection but not for treating tiny ordering differences as final effect-size
evidence. Independent validation carried the winner decision.

## Usage and cost

| Case study | Reported tokens | Dollar record | Cost interpretation |
| --- | ---: | ---: | --- |
| `markdown-it-py` | 215.35M generation; 0.20M embedding | $2.0833 | Proxy-calculated estimate under recorded public pricing; no provider bill; embedding and host unpriced |
| `python-pathspec` | 241.63M generation; 0.26M embedding | $2.4856 | Proxy-calculated estimate under recorded public pricing; no provider bill; embedding and host unpriced |
| Zstandard | 52.65M total, including cached input and embeddings | $60.2472 | Kilo model-catalog estimate, not provider-billed spend; embeddings unpriced |

The three dollar figures must not be summed as an all-in cost. Zstandard's Kilo
catalog estimate and the two proxy-calculated DeepSeek estimates have different
accounting paths. The Python estimates are reproducible from
`paper/evidence/python_generation_cost_audit.json`; its source records contain
no provider-billed cost. Across the reports, the raw token counters sum to
510.09M, but token prices and cache treatment differ by provider.

## What the experiments changed

The case studies exposed generic framework defects rather than only target
adapter defects.

| Finding | Framework status at this closeout |
| --- | --- |
| Kilo child processes survived timeouts | Fixed on `main` with process-group cleanup |
| Kilo state, workspace, and usage could leak across jobs | Fixed on `main` with per-job/run-token isolation and workspace checks |
| Interactive Kilo tools could block unattended jobs | Fixed on `main` with a headless policy and `--pure` capability checks |
| Cost reconstruction disagreed with Kilo | Fixed on `main`; Loreley reads root and descendant Kilo session aggregates and preserves unpriced costs as unpriced |
| Commit subjects made a low-value extra LLM call | Implemented in the framework in this closeout; reuse coding summary, then planning summary |
| Planning, coding, and trajectory summarization silently shared model defaults | Implemented in the framework in this closeout with phase-specific models, explicit trajectory provider/reasoning settings, and preflight |
| Scheduler restarts replayed sampling recipes | Implemented in the framework in this closeout with persistent per-island ordinals and recipe cooldown |
| Exact Git trees were generated and evaluated repeatedly | Implemented in the framework in this closeout with source-tree hashing and contract-scoped result reuse |
| Source-distinct commits compiled to the same executable | Candidate identity and archive deduplication are implemented in the framework in this closeout; early measurement reuse remains harness-only |
| Evaluator concurrency was enforced by target-specific file locks | Not yet implemented in the framework |
| Unique-binary endpoints and identity-aware progress required experiment scripts | Not yet implemented in the framework |
| Noisy-objective archive admission lacks a general policy | Deferred research question |

The immediate implementation work is defined in
[the framework convergence proposal](2026-08-07-case-study-framework-convergence-proposal.md).
Questions without an agreed solution are kept out of that proposal and recorded
in [the deferred research register](2026-08-07-case-study-deferred-research.md).

## Claim boundary

The public evidence can state:

> In three fixed-repository case studies, Loreley produced validated candidates
> with a 6.75% `markdown-it-py` throughput gain, a post-hoc 25.14%
> `python-pathspec` throughput gain, and a preregistered 1.019% Zstandard
> holdout compression gain. The Zstandard follow-up selected generation-4
> candidate `fe39bee8` on expanded validation. It then measured +1.173% on the
> original holdout, before its own score there was known, and +0.891% on a newly
> generated corpus sealed before measurement. The original corpus had already
> been opened for the preregistered winner; the new-corpus recipe was chosen
> after `fe39bee8` was known. In the broader fixed-Top-10 comparison, all ten
> candidates were positive on the original holdout, with a median gain of
> 1.116%; this comparison does not create a new blinded winner.

Every percentage must retain its selection status and workload scope. The
evidence does not support an average cross-repository speedup, a seed-free
claim, a quality-diversity advantage over simpler search, cross-platform
Zstandard performance, or upstream acceptance.

## Evidence links

- [`markdown-it-py` case study](2026-08-02-markdown-it-py-deepseek-case-study.md)
- [Static candidate source diffs](../marketing/candidates/README.md)
- [`python-pathspec` case study](2026-08-03-pathspec-deepseek-case-study.md)
- [Zstandard registered report](2026-08-07-zstandard-gpt-v19-case-study-report.md)
- [Zstandard Top-10 validation, fresh-corpus, and holdout supplement](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)

The corresponding machine-readable repository artifacts are
`reports/zstandard-gpt-v19-evidence.json` and
`reports/zstandard-gpt-v19-top10-validation-supplement.json`.
