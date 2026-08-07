# ADR 0053: Prevent restart recipe replay and repeated exact-tree evaluation

Date: 2026-08-06

Context: A long QD campaign repeatedly generated and evaluated equivalent candidates. The sampler's seeded RNG lived only in one scheduler process, so each restart replayed its initial choices. Historical `(base, inspirations)` recipes were not excluded, and agents could reproduce an inspiration tree exactly. Archive-level executable identity protected MAP-Elites from some duplicate admissions, but it acted after generation and evaluation.

Decision: Derive sampler randomness from the configured seed, island ID, and persistent per-island job ordinal. Persist an order-insensitive recipe hash and cool down a bounded number of recent recipes, with bounded resampling and an explicit unavoidable-reuse flag. Tell planning and coding agents that inspirations are evidence rather than target snapshots. Before evaluation, reuse a prior passed result only for an exact Git tree under the same evaluator name/version and campaign program. Keep evaluator-provided candidate identity as the archive admission invariant.

Consequences: Scheduler restarts no longer replay the start of the random stream, common lineage recipes are less likely to recur, and exact source-tree repeats avoid another benchmark. The cooldown is not a permanent ban: small archives can continue making progress and expose reuse in provenance. Source-distinct candidates that compile to the same binary remain separate before evaluation because source-tree identity is not a substitute for evaluator-defined phenotype identity.
