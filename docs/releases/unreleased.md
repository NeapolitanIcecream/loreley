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

## Fixed

- Include descendant Kilo sessions and reasoning output in usage accounting,
  while keeping repeated coding invocations distinct.
- Preserve bounded token/request failure reason codes through agent retries.
- Fetch candidate commits explicitly when narrow clone refspecs omit their
  branches.
- Terminate the complete Kilo process group after a POSIX timeout so descendant
  processes cannot continue making API requests after their job fails.
