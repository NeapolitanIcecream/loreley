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
- A [256-logical-job Zstandard case study](../research/2026-08-06-zstandard-reliable-qd-case-study-report.md)
  in which a third-generation candidate improved holdout compression
  throughput by 1.04% with neutral decompression.

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
