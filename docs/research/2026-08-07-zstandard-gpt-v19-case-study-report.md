# Loreley Zstandard V19 Case Study: A 1.02% Holdout Compression Gain

Date: 2026-08-07

Scope: one fixed-repository campaign on Zstandard, one Apple-silicon host, and
one sealed corpus family.

## Result in one minute

The final candidate improved single-thread Zstandard compression throughput by
**1.019%** on the sealed holdout corpus. Its 95% confidence interval was
**+0.962% to +1.076%**. Decompression was neutral at +0.010%, and combined
compression-and-decompression throughput improved by 0.513%.

| Holdout metric | Root ratio | 95% interval |
| --- | ---: | ---: |
| Compression throughput | 1.01019× | 1.00962–1.01076× |
| Decompression throughput | 1.00010× | 0.99890–1.00130× |
| Combined throughput | 1.00513× | 1.00441–1.00585× |
| Worst measured cell | 0.99817× | — |
| Maximum compressed-size ratio | 1.00000× | — |
| Peak RSS delta | +0.063 MiB | — |

The candidate passed upstream checks, release compilation, bidirectional
root/candidate decoding, compressed-size limits, and RSS limits. The conclusion
audit passed.

This is a preregistered **modest-positive** result. It is not a strong result:
the strong rule required a compression point estimate of at least 1.02× and a
lower 95% bound of at least 1.01×.

## What was run

The benchmark root was Zstandard commit `5b3fe474`, based on upstream commit
`82d322c4`. Training, validation, and holdout used disjoint sealed corpora. Each
split contained eight 2 MiB files spanning source code, records, natural text,
and structured binary data. Performance covered compression levels 1, 3, and 5
with one thread.

Loreley used:

- eight independent, root-based manual seeds;
- MAP-Elites with a 3D 4×4×4 archive and per-cell Pareto fronts;
- compression lower 95%, decompression lower 95%, and worst-cell speedup as
  the three objectives;
- four maximum unfinished jobs, four physical agent workers, and four evaluator
  lanes;
- Kilo with `gpt-5.6-sol` for planning and `gpt-5.6-luna` for coding and
  trajectory summaries, all at maximum reasoning effort;
- OpenRouter `text-embedding-3-small` embeddings at 1,536 dimensions; and
- a headless Kilo profile that denied interactive tools and `suggest`.

Commit subjects reused the coding or planning report instead of making another
LLM call. Agents could inspect and edit source, but compilation, tests,
cross-decoding, and performance measurement belonged to the evaluator.

## Search and selection

The search ended at its registered unique-binary endpoint. It produced 220
terminal jobs: 211 succeeded and nine failed. The successful jobs represented
167 distinct release binaries. The final archive held 13 Pareto entries across
11 of 64 possible coordinates.

The training rule froze the top three unique binaries by compression lower 95%,
then used decompression lower 95%, worst-cell speedup, diff size, and commit hash
as tie-breakers. All three passed eight-round validation:

| Frozen finalist | First completion | Training compression lower 95% | Validation compression | Validation lower 95% | Validation decompression |
| --- | ---: | ---: | ---: | ---: | ---: |
| `44a22df0` | 5 | 1.00984× | 1.00987× | 1.00905× | 0.99754× |
| `f80c8619` | 195 | 1.00944× | 1.01069× | 1.00923× | 1.00115× |
| `7b9aef38` | 6 | 1.00903× | 1.01066× | **1.00994×** | 1.00025× |

Validation selected `7b9aef38` because it had the highest compression lower
bound. Only this candidate was evaluated on the 12-round holdout.

The earlier training throughput leader, `8ec126ea`, measured 1.01275× combined
throughput but ranked eighth under the registered uncertainty-aware rule. It was
not promoted from a favorable point estimate.

## Winner and lineage

The winner is release binary `e7e9ef6b060f…`, represented by commit
`7b9aef38ecd4`. It is manual seed 5, not a later evolved candidate. Its complete
lineage is one step:

1. root `5b3fe474`;
2. seed `7b9aef38`, which unrolls the scalar histogram update loop four bytes at
   a time.

The patch changes `lib/compress/hist.c` by eight insertions and one deletion. It
adds a four-byte loop and retains the scalar tail. It contains no corpus-specific
logic, generated code, dependency, format change, or broad rewrite.

The search did produce a competitive evolved finalist. `f80c8619` is a
generation-4 candidate with three evolution steps after its seed. It combined
histogram and compression hot-path changes and validated at 1.01069× compression
and 1.00115× decompression. Its compression lower bound was 0.071 percentage
points below the winning seed, so the frozen rule did not select it.

This distinction matters: the case study proves that Loreley preserved,
ranked, and independently validated a real improvement, and that evolution
reached a comparable region. It does not prove that this campaign evolved a
better binary than its strongest manual seed.

## Top-10 sensitivity follow-up

After preserving the registered conclusion, a separate analysis expanded
validation from the deterministic training Top 3 to the Top 10. The three
existing reports were reused and training ranks 4-10 were measured for eight
rounds on the same validation split. Training rank 10, `fe39bee8`, became the
expanded validation winner at 1.01234x compression, with a 1.01156x lower 95%
bound.

That result was post-hoc with respect to finalist count, so a new confirmation
protocol was sealed before testing `fe39bee8` on a newly generated disjoint
corpus. Its 12-round compression result was 1.00891x, with a 95% interval of
1.00522-1.01261x. Combined throughput was 1.00359x, with a
1.00004-1.00716x interval. The candidate passed correctness, cross-decoding,
size, and RSS gates.

`fe39bee8` is a generation-4 candidate first observed at logical completion 57.
Its lineage combines a zero-literal fast path, a compression hot-path evolution,
and an eight-byte histogram update unroll. This follow-up establishes a separate
evolved-candidate result. It does not relabel the registered holdout winner or
establish a head-to-head result because the two candidates were confirmed on
different fresh corpora. The complete method and evidence are in the
[Top-10 validation supplement](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md).

## Evaluation reliability and search efficiency

The fresh evaluator configuration used four calibrated lanes. For the 201 jobs
that ran a real evaluation rather than exact-binary reuse, median evaluator time
was 186.7 seconds and median end-to-end job time was 333.6 seconds. Evaluation
therefore occupied a median 64.9% of job time. The four active workers completed
the two search stages in 5.31 runner-hours, or 41.4 physical jobs per hour.

The first 128 jobs contained 102 unique successful binaries. The efficient
continuation added 65 unique binaries in 92 physical jobs and 2.12 runner-hours,
or 30.7 new unique binaries per hour. Nineteen exact-binary repeats reused a
previously accepted, hash-linked evaluator report and did not rerun cross-decode
or benchmark. Their median evaluator time was 21.6 seconds, versus 186.7 seconds
for real measurements.

Across the full campaign, 211 successful jobs map to 167 binaries, so 44
successful jobs repeated an existing executable identity. Twenty-five of those
repeats were measured before the binary-measurement cache was enabled; 19 were
short-circuited afterward. This separates search diversity from repeated source
commits that compile to the same executable.

The nine terminal failures were:

- four upstream or candidate test failures;
- four measurement-precision rejections; and
- one benchmark failure caused by the compressed-size limit.

Failed or imprecise results did not populate the measurement cache or enter the
archive. Final validation took about 5.6 minutes per finalist; the single
holdout evaluation took 8.1 minutes.

## Usage and cost

The campaign recorded 52,653,004 tokens, including embeddings and cached input.
Kilo's SQLite session trees provided complete catalog-cost coverage for all 424
generation sessions.

| Phase | Model | Events | Uncached input | Cached input | Output | Reasoning output | Kilo catalog estimate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Planning | `gpt-5.6-sol` | 212 | 6,547,054 | 10,642,848 | 463,630 | 196,143 | $57.8499 |
| Coding | `gpt-5.6-luna` | 212 | 6,178,958 | 25,369,844 | 452,351 | 92,768 | $2.3973 |
| Embedding | `text-embedding-3-small` | 303 | 2,709,408 | 0 | 0 | 0 | Unpriced |

The $60.2472 total is a Kilo model-catalog estimate, not provider-billed spend.
The embedding route returned token counts but no price, so the report does not
claim an all-provider dollar total.

## Claim and limitations

This case study supports a narrow claim: on one fixed Zstandard repository and
one Apple-silicon host, Loreley selected an explainable nine-line candidate whose
sealed holdout compression gain was about 1.02%, with neutral decompression,
unchanged compressed size, and negligible RSS change.

It does not establish a universal Zstandard gain, cross-platform performance,
a 2% improvement, or superiority over expert seed design. The winner came from
the initial manual seed set. A separately sealed follow-up found a generation-4
candidate with a positive fresh-corpus compression gain, but did not compare it
head to head with the registered winner. Together, the results provide promotion
evidence for the system's end-to-end search, identity-aware evaluation,
uncertainty-aware selection, and sealed validation workflow while leaving seed
dependence and fine-grained frontier ranking as open questions.
