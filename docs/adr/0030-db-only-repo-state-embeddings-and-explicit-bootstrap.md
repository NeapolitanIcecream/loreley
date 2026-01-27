# ADR 0030: DB-only repo-state embeddings and explicit bootstrap

Date: 2026-01-27

Status: Accepted

Decision

- Remove the cache backend switch; repo-state embeddings always use the DB-backed file cache and aggregate table.
- Split repo-state embedding into explicit bootstrap (full recompute + persist) and runtime incremental APIs.
- Runtime ingestion is incremental-only and fails fast when no aggregate hit or incremental derivation is available.
- Bootstrap is the only full-recompute path and is invoked explicitly by scheduler initialisation.
