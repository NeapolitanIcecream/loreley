# ADR 0042: Sample ingest INFO logs and gate progress to TTY

Date: 2026-02-17

Status: Accepted

Decision

- Add `MAPELITES_INGEST_INFO_LOG_EVERY` (`mapelites_ingest_info_log_every`) with default `20`.
- Emit hot-path INFO logs in `MapElitesManager.ingest()` on a fixed sample cadence.
- Always keep warning/error diagnostics unsampled; keep stage-metrics INFO on sampled cadence or key lifecycle events.
- Render Rich progress bars only when `sys.stdout.isatty()` is true in chunking and code-embedding paths.
- Keep behavior fail-fast and deterministic with simple, local sampling logic.
