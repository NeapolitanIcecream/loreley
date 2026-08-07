# Loreley Zstandard Case Study: A 1.04% Holdout Compression Gain

Date: 2026-08-06

Status: historical infrastructure evidence. The aggregate promotion story now
uses the [fresh V19 case study](2026-08-07-zstandard-gpt-v19-case-study-report.md).
This report is retained because it exposed candidate-identity, restart, and
measurement-isolation defects; its result is not counted as a fourth case.

Scope: one 256-logical-job campaign on Zstandard commit
`82d322c4973d9e2968d94047a40892bc6d9a9bdf`

## Result in one minute

Loreley evolved a Zstandard ARM64 executable that improved single-thread compression throughput by **1.044%** on a fresh holdout corpus. The 95% confidence interval was **+0.880% to +1.209%**. Overall compression-and-decompression throughput improved by **0.512%**, with a 95% interval of **+0.345% to +0.678%**.

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
| Unique feasible ARM64 binaries | 55 |

The final archive had 12 source entries across 10 occupied coordinates, but only 10 unique ARM64 binaries. Archive admission used Git commits rather than evaluator-equivalent binaries, which overstated executable diversity.

## Why the result is not just measurement noise

The search score did contain selection bias. The training leader, `c1f1852b`, measured a 1.672% compression gain during search but only 0.961% on validation. Post-run binary audit showed that it produced the same ARM64 executable as the eventual holdout source and 46 other passed candidates. Its training score was the maximum of 48 measurements of that binary, not evidence for a distinct faster executable.

The three highest-ranked unique training trees were frozen before validation. Validation then selected `893bf5cf`, which had ranked third during training:

| Frozen finalist | ARM64 binary | Training compression | Validation compression | Validation lower 95% |
| --- | --- | ---: | ---: | ---: |
| `c1f1852b` | `2cddc94…` | 1.01672× | 1.00961× | 1.00374× |
| `e1c2cfad` | `5e7d2d5…` | 1.01477× | 1.01024× | 1.00923× |
| `893bf5cf` | `2cddc94…` | 1.01141× | 1.01133× | 1.01013× |

The first and third source finalists compiled byte-for-byte to binary SHA-256 `2cddc94eb7cbc1650c99237935e0bca7ae6416f61528f38647b5e0f2f2e0d391`. Their validation difference is measurement variation, not generalization from the later source edit. Giving the same binary two finalist slots was a protocol defect.

Only the source selected by the frozen rule, `893bf5cf`, was then evaluated on holdout. Its 12-pair holdout compression and combined-throughput intervals remained entirely above 1.0. That result applies to binary `2cddc94…`, regardless of which equivalent source wrapper names it. It rejects the explanation that the benefit was entirely training noise while placing the realistic effect near 1%, not the selected training maximum.

The 48 passed training measurements of this binary ranged from +0.887% to +1.672%. Seven were collected before the exclusive measurement lock; their standard deviation was 0.310 percentage points. The 41 locked measurements ranged from +0.887% to +1.176%, with a 0.065-point standard deviation. The old +1.672% leader was a pre-lock outlier. This post-hoc repeatability audit explains the training inflation but does not replace the sealed holdout result.

The locked lower-95% statistic was less stable: it ranged from 0.99899× to 1.01082×, with a 0.250-percentage-point standard deviation. The training evaluator used four rounds, extending to eight only when a component log half-width exceeded 1.5%; that was adequate to reject grossly unstable measurements but not to resolve close candidates by lower confidence bound. Fine-grained training order is therefore not an effect-size claim. The stricter, fixed validation and 12-pair holdout measurements carry the result.

## Measured winner and lineage

The measured winner is ARM64 binary `2cddc94…`. It first appeared at logical completion 71 as source commit `817af317acf69699eff2942ee745a1055fb85ad8`, an end-to-end generation-3 candidate:

On that first observation it measured +0.906% compression, with a +0.028% lower 95% bound; decompression was -0.296%. This is the canonical training observation when candidates are deduplicated by executable identity rather than repeatedly sampled by source commit.

1. `50be35e7` — manual seed: specialize sparse CCtx fast hash-table filling;
2. `9e91d9e0` — first evolution: unroll the scalar histogram update four bytes at a time;
3. `7762b68d` — second evolution: skip zero-frequency entropy-cost work; and
4. `817af317` — third evolution: add the missing sparse double-hash-table fill specialization.

This earliest equivalent source changes four compression C files, with 39 insertions and 33 deletions. Its lineage is causal and reviewable: it combines small compression hot-path changes without corpus-specific logic, generated code, dependencies, or a broad rewrite.

The frozen finalist `c1f1852b` appeared at completion 118 and compiled to the same binary from a smaller four-file source diff. The protocol-selected source `893bf5cf` appeared at completion 245; its extra decoder edit was in a non-AArch64 branch and did not change the measured executable. It remains part of the search history, but the holdout gain cannot be attributed to that final edit. The first 128 jobs had already found the measured binary; the extension to 256 increased coverage from 33 to 55 unique binaries but did not improve the final executable.

## Reliability and operational lessons

Two infrastructure defects materially slowed the campaign but were kept out of the method result:

- A Codex app restart left inherited standard file descriptors invalid. Eighteen affected rows were prospectively frozen as lifecycle exclusions before recovery on the same database.
- Launching the evaluator as a macOS Background service reduced fixed-root throughput by 4.4–6.4× and caused systematic precision failures. Switching the service to Interactive and using a shared-agent/exclusive-measurement lock restored the historical fixed-root regime without loosening thresholds. All 11 extension precision failures occurred before this repair; none occurred afterward.

Across the complete logical campaign, 34 jobs were rejected for insufficient measurement precision. They remain in reliability and resource accounting but are not treated as evidence for or against the search method. The conclusion audit verifies the exact endpoint, fixed exclusions, source and evaluator identities, model closure before hidden-data reveal, three frozen finalists, one frozen validation winner, and one holdout target.

The first post-reveal audit invocation also exposed a narrow audit-tool bug: the generic pre-reveal verifier treated the expected extracted plaintext as drift. The failed invocation and fix were preserved, while the frozen selections, holdout result, conclusion, and sealed archive hashes remained unchanged. The corrected post-reveal audit passed.

### Post-experiment framework fixes

A read-only replay of the 249 logical evolution rows found 178 unique unordered `(base, inspirations)` recipes. All 71 repeated recipe rows repeated a recipe seen within the previous 64 evolution rows; the most common recipe appeared seven times. Among 205 passed evaluations, only 100 Git trees were unique.

Loreley was changed after this experiment so the sampler derives randomness from a persistent per-island job ordinal instead of restarting one process-local seeded stream. It now cools down the most recent 64 recipes with bounded resampling, tells agents not to reproduce inspiration trees, and reuses a passed evaluation for an exact Git-tree match under the same evaluator and campaign contract. Evaluator-provided executable identity remains the archive admission invariant. These changes do not alter or retroactively filter the V13 evidence reported above.

The first-to-last database window was 31.1 hours because it includes pauses, incident analysis, and recovery. Median job row lifetime, including queueing, was 17.9 minutes; the 90th percentile was 27.7 minutes. Final local evaluation took 16.8 minutes for three validation candidates and 8.1 minutes for the single holdout target.

## Usage and cost

Kilo's database-backed session aggregates reported **$4.8277** of DeepSeek generation cost with complete cost coverage for all 493 token-bearing Kilo events.

| Usage source | Events | Input tokens | Cached input | Output tokens | Reasoning output | Reported cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kilo / DeepSeek | 493 | 15,698,195 | 263,211,904 | 2,282,463 | 4,478,202 | $4.8277 |
| OpenAI embeddings | 126 | 1,167,501 | 0 | 0 | 0 | Unpriced |

Eighteen Kilo invocations had no model session and zero tokens; they are not missing spend. Total observed tokens across generation and embeddings were 286,838,265. The embedding provider did not return a price, so $4.8277 is the fully covered generation cost, not a claimed all-provider total.

## Claim

This case study supports a narrow, useful claim: on one fixed Zstandard repository and one Apple-silicon host, Loreley found an explainable, format-preserving ARM64 executable by logical job 71, with a repeatable holdout compression gain of about 1%, neutral decompression, and less than $5 of provider-reported generation cost across the full 256-job campaign.

It does not establish a universal Zstandard gain, a 2% improvement, or cross-repository generality. Together with the preserved negative and infrastructure evidence, it is suitable as a modest-positive case study and as evidence that independent validation is necessary when promoting search results.
