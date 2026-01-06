# ADR 0004: `.loreleyignore` for embedding scope

Date: 2026-01-06

Context: Users need a way to control which repository files participate in embeddings without changing Git tracking rules.
Decision: Support a repository-root `.loreleyignore` file and parse it using the existing best-effort `.gitignore`-style matcher.
Details: When `commit_hash` is specified, read `.gitignore` and `.loreleyignore` strictly from that commit; do not fall back to the working tree.
Details: Apply `.gitignore` rules first, then `.loreleyignore` rules, so `.loreleyignore` may override via negation (`!pattern`).
Constraints: Matching remains root-only and best-effort (no nested `.gitignore` files and no global excludes).
Consequences: Ignore file changes can alter eligibility broadly, so incremental repo-state aggregation is disabled when either ignore file changes.

