# ADR 0041: Split MAP-Elites manager modules

Date: 2026-02-14

Status: Superseded in part by ADR 0052

ADR 0052 removes the compatibility re-export module. Callers now import
`MapElitesManager` from `loreley.core.map_elites.manager` and record types from
`loreley.core.map_elites.types`.

Decision

- Split MAP-Elites implementation into focused modules: `types.py`, `archive_ops.py`, `db_ops.py`, `rebuild.py`, and `manager.py`.
- Keep `MapElitesManager` as the only orchestration entry point; helper modules hold pure/archive/db/rebuild details.
- Keep `loreley.core.map_elites.map_elites` as a compatibility shim that re-exports public symbols.
- Enforce one-way dependencies: `types -> archive_ops/db_ops -> rebuild -> manager`, with `map_elites.py` as re-export only.
- Preserve behavior and persistence semantics (PCA warmup, refit rebuild, snapshot updates) while simplifying control flow in `ingest()`.
