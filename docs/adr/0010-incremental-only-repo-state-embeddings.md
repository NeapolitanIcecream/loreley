# ADR 0010: Incremental-only repo-state embeddings (after bootstrap)

Date: 2026-01-08

Context: Full repo-tree enumeration is expensive and introduces large, bursty cache writes; the MAP-Elites pipeline primarily operates on single-parent commits produced from known base commits.
Decision: After bootstrapping the experiment root commit aggregate, repo-state embeddings run in incremental-only mode; commits that cannot be incrementally derived fail fast.
Constraints: Bootstrapping computes and persists the root commit aggregate once; non-incremental commits include merge commits, diff failures, or changes to root ignore files.
Consequences: Runtime ingestion cost becomes O(changed_files) with stable aggregates; unsupported commit shapes are rejected explicitly rather than silently falling back to a full recompute.


