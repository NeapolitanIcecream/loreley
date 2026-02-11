# ADR 0037: Sampler lightweight cell/commit sampling

Date: 2026-02-11

Status: Accepted

Decision

- Scheduling samples only `cell_index` and `commit_hash` for base/neighbor selection.
- `MapElitesManager.get_cell_commits(island_id)` provides a lightweight cell→commit map without materializing full archive records.
- `MapElitesSampler.schedule_job()` uses `get_cell_commits()` directly and persists `EvolutionJob` rows using only commit hashes.
- Full archive records remain available via `get_records()` for UI and inspection use cases.

