# Loreley Zstandard Case Study: A 1.04% Holdout Compression Gain

Date: 2026-08-06

Scope: one 256-logical-job campaign on Zstandard commit
`82d322c4973d9e2968d94047a40892bc6d9a9bdf`

## Result in one minute

Loreley evolved a Zstandard patch that improved single-thread compression throughput by **1.044%** on a fresh holdout corpus. The 95% confidence interval was **+0.880% to +1.209%**. Overall compression-and-decompression throughput improved by **0.512%**, with a 95% interval of **+0.345% to +0.678%**.

The winner preserved compressed size, passed upstream checks and bidirectional cross-version decoding, and added 0.031 MiB of peak RSS. Its decompression point estimate was effectively neutral at -0.018%.

| Holdout metric | Root ratio | 95% interval |
| --- | ---: | ---: |
| Compression throughput | 1.01044× | 1.00880–1.01209× |
| Decompression throughput | 0.99982× | 0.99807–1.00157× |
| Combined throughput | 1.00512× | 1.00345–1.00678× |
| Worst measured cell | 0.99775× | — |
| Maximum compressed-size ratio | 1.00000× | — |
| Peak RSS delta | +0.031 MiB | — |

This is a preregistered **modest-positive** result. It is not a strong result because the holdout compression gain was below the strong threshold of 2%, and its lower confidence bound was below the strong threshold of 1% gain.

## What was tested

The fixed target was Zstandard commit `82d322c`. Loreley used MAP-Elites quality-diversity search with:

- DeepSeek v4 Flash through Kilo for planning and coding;
- OpenAI `text-embedding-3-small` embeddings at 1,536 dimensions;
- four physical agent workers, one evaluator lane, and four maximum unfinished jobs;
- a 3D 4×4×4 archive with a Pareto front in each cell; and
- seven manually written, independent seed directions.

The evaluator owned all compilation, tests, cross-decoding, and benchmarks. Agents could inspect and edit C sources but could not run private builds or performance tools. Training, validation, and holdout each used disjoint sealed corpora with eight 2 MiB files spanning source code, records, natural text, and structured binary data. Performance covered compression levels 1, 3, and 5 with one thread.

The campaign ended at 256 logical jobs. Eighteen physical rows caused by a documented process-lifecycle failure were fixed exclusions, so the database contains 274 terminal rows. Of the 256 logical jobs, 208 reached a reliable archive admission decision:

| Job accounting | Count |
| --- | ---: |
| Physical terminal rows | 274 |
| Fixed lifecycle exclusions | 18 |
| Logical jobs | 256 |
| Archive-decision-qualified jobs | 208 |
| Passed candidate evaluations | 205 |
| Reliably rejected candidates | 3 |
| Measurement-precision invalid | 34 |
| Did not reach evaluation | 14 |
| Unique feasible Git trees | 100 |

The final archive had 12 entries across 10 occupied coordinates. Archive size and unique occupied coordinates are reported separately because multiple Pareto entries can share a coordinate.

## Why the result is not just measurement noise

The search score did contain selection bias. The training leader, `c1f1852b`, measured a 1.672% compression gain during search but only 0.961% on validation. Its validation confidence interval still excluded zero, but the drop shows why a training maximum should not be presented as the result.

The three highest-ranked unique training trees were frozen before validation. Validation then selected `893bf5cf`, which had ranked third during training:

| Frozen finalist | Training compression | Validation compression | Validation lower 95% |
| --- | ---: | ---: | ---: |
| `c1f1852b` | 1.01672× | 1.00961× | 1.00374× |
| `e1c2cfad` | 1.01477× | 1.01024× | 1.00923× |
| `893bf5cf` | 1.01141× | 1.01133× | 1.01013× |

Only `893bf5cf` was then evaluated on holdout. Its 12-pair holdout compression interval remained entirely above 1.0, and its combined-throughput interval also remained entirely above 1.0. No runner-up was tested after holdout. The independent result therefore rejects the explanation that the observed benefit was entirely training noise, while also showing that the realistic gain is about 1%, not the larger training maximum.

## Winner and lineage

The winner is `893bf5cf9e02a703ff530116ce990c3f4dae6ad6`. It is an end-to-end evolved candidate, not a manual seed. With the seed treated as generation 0, it is generation 3:

1. `50be35e7` — manual seed: specialize sparse CCtx fast hash-table filling;
2. `205228b6` — first evolution: skip zero-frequency entropy-cost work;
3. `c1f1852b` — second evolution: combine the validated compression changes, including a four-byte scalar histogram update; and
4. `893bf5cf` — third evolution: keep generic decoder repeat offsets in local variables rather than repeatedly loading and storing the state array.

The final patch changes five C files, with 55 insertions and 40 deletions. Its lineage is causal and reviewable: it combines small hot-path changes on compression and decompression rather than introducing corpus-specific logic, generated code, dependencies, or a broad rewrite.

The final child used `c1f1852b` as its parent and `e1c2cfad` plus `74492608` as inspirations. This is a concrete example of the QD archive contributing ideas from more than one branch to a later candidate.

## Reliability and operational lessons

Two infrastructure defects materially slowed the campaign but were kept out of the method result:

- A Codex app restart left inherited standard file descriptors invalid. Eighteen affected rows were prospectively frozen as lifecycle exclusions before recovery on the same database.
- Launching the evaluator as a macOS Background service reduced fixed-root throughput by 4.4–6.4× and caused systematic precision failures. Switching the service to Interactive and using a shared-agent/exclusive-measurement lock restored the historical fixed-root regime without loosening thresholds. All 11 extension precision failures occurred before this repair; none occurred afterward.

Across the complete logical campaign, 34 jobs were rejected for insufficient measurement precision. They remain in reliability and resource accounting but are not treated as evidence for or against the search method. The conclusion audit verifies the exact endpoint, fixed exclusions, source and evaluator identities, model closure before hidden-data reveal, three frozen finalists, one frozen validation winner, and one holdout target.

The first post-reveal audit invocation also exposed a narrow audit-tool bug: the generic pre-reveal verifier treated the expected extracted plaintext as drift. The failed invocation and fix were preserved, while the frozen selections, holdout result, conclusion, and sealed archive hashes remained unchanged. The corrected post-reveal audit passed.

The first-to-last database window was 31.1 hours because it includes pauses, incident analysis, and recovery. Median job row lifetime, including queueing, was 17.9 minutes; the 90th percentile was 27.7 minutes. Final local evaluation took 16.8 minutes for three validation candidates and 8.1 minutes for the single holdout target.

## Usage and cost

Kilo's database-backed session aggregates reported **$4.8277** of DeepSeek generation cost with complete cost coverage for all 493 token-bearing Kilo events.

| Usage source | Events | Input tokens | Cached input | Output tokens | Reported cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kilo / DeepSeek | 493 | 15,698,195 | 263,211,904 | 2,282,463 | $4.8277 |
| OpenAI embeddings | 126 | 1,167,501 | 0 | 0 | Unpriced |

Eighteen Kilo invocations had no model session and zero tokens; they are not missing spend. Total observed tokens across generation and embeddings were 286,838,265. The embedding provider did not return a price, so $4.8277 is the fully covered generation cost, not a claimed all-provider total.

## Claim

This case study supports a narrow, useful claim: on one fixed Zstandard repository and one Apple-silicon host, Loreley used a 256-job quality-diversity campaign to evolve an explainable, format-preserving C optimization with a repeatable holdout compression gain of about 1%, neutral decompression, and less than $5 of provider-reported generation cost.

It does not establish a universal Zstandard gain, a 2% improvement, or cross-repository generality. Together with the preserved negative and infrastructure evidence, it is suitable as a modest-positive case study and as evidence that independent validation is necessary when promoting search results.
