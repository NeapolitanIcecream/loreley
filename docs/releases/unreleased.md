# Unreleased

These notes cover changes merged after `v0.9.0-alpha`.

## Added

- Per-job Kilo state isolation and optional `kilo run --pure` support, including
  preflight capability checks.
- A [64-job `markdown-it-py` case study](../research/2026-08-02-markdown-it-py-deepseek-case-study.md)
  in which Loreley found a separately validated 6.75% throughput improvement.
- A [64-job `python-pathspec` case study](../research/2026-08-03-pathspec-deepseek-case-study.md)
  documenting a four-generation 25.14% diagnostic candidate and the
  allocation-selection failure that invalidated the preregistered outcome.
- A [fresh Zstandard V19 case study](../research/2026-08-07-zstandard-gpt-v19-case-study-report.md)
  in which the registered winner improved sealed-holdout compression by 1.02%
  with neutral decompression, plus a Top-10 follow-up that confirmed a
  generation-4 candidate on a new corpus.
- A [three-case-study evidence report](../research/2026-08-07-loreley-case-study-evidence-report.md)
  that keeps prospective, post-hoc, and supplemental results distinct.
- Phase-specific Kilo planning and coding models, plus explicit provider,
  thinking, and reasoning controls for trajectory summaries.
- Evaluator-provided candidate identities and exact Git-tree identities for
  archive deduplication and safe contract-scoped result reuse.
- A public `phased-v1` evaluator protocol that separates source preparation,
  reusable measurement, and finalization. Accepted measurements use canonical
  contract keys, hash-linked provenance, and source-attempt links.
- PostgreSQL-backed evaluator capacity `E`, independent of scheduler capacity
  `U` and configured worker processes `W`, with persisted contracts and
  waiter/acquisition/release telemetry.
- Shared campaign progress for CLI, operator API, and scheduler decisions,
  including separate counts for source trees, evaluator identities, real
  measurements, reuse, archive entries, and occupied coordinates.
- An optional restart-stable, at-least unique-evaluation-identity endpoint that
  drains in-flight jobs while retaining the physical job cap.
- Secret-free effective model-route diagnostics and explicit acknowledgement
  for non-semantic local-hash embedding fixtures.
- First-class, idempotent manual-seed manifests backed by pinned remote refs.
  Seeds use the normal evaluator and archive pipeline while skipping model
  agents; staged admission preserves the independent unfinished-job limit.
- Project-neutral fixed-sample and anytime-valid intervals, preweighted
  stratified estimates, adaptive sample/time budgets, and canonical measurement
  contract fingerprints for evaluator plugins. Adaptive results distinguish
  valid inference, completion of the declared evidence target, and evidence
  that is safe for a final decision.

## Fixed

- Include descendant Kilo sessions and reasoning output in usage accounting,
  while keeping repeated coding invocations distinct.
- Preserve bounded token/request failure reason codes through agent retries.
- Fetch candidate commits explicitly when narrow clone refspecs omit their
  branches.
- Terminate the complete Kilo process group after a POSIX timeout so descendant
  processes cannot continue making API requests after their job fails.
- Read provider-reported Kilo tokens and cost from session-tree aggregates,
  avoiding descendant double counting and local price reconstruction.
- Keep Kilo workers non-interactive even with native provider configuration by
  disabling interactive and suggestion tools in the injected headless profile.
- Make campaign constraints override generic planning and coding validation
  advice, and allow noisy root calibration to retry after cooldown.
- Derive sampling from persistent per-island ordinals and cool down recent
  `(base, inspirations)` recipes so scheduler restarts do not replay a campaign.
- Reuse coding or planning summaries for commit messages instead of making a
  separate commit-summary model request.
- Terminate evaluator process groups and their compiler/benchmark descendants
  on timeout or worker death.
- Classify Kilo SQLite session-tree dollars as catalog values instead of
  provider-reported invoices, and expose cost-source aggregates through the
  usage API.
- Preserve evaluation evidence per attempt across job retries instead of
  replacing artifact rows by job key, and verify stored measurement evidence
  before cache acceptance and reuse.
- Evaluate manual seeds in detached checkouts without requiring or fabricating
  a worker publication branch.
