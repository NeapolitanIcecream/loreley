# ADR 0039: PCA epochs and archive rebuild on refit

Date: 2026-02-12

Status: Accepted

Decision

- Treat the PCA projection as an epoch-scoped coordinate system for MAP-Elites behaviour measures.
- When PCA refits (epoch increments), rebuild the island archive by reprojecting existing elites into the new coordinates and replacing persisted `map_elites_archive_cells` rows.
- Persist `samples_since_fit` in `map_elites_states.snapshot` so `MAPELITES_DIMENSION_REDUCTION_REFIT_INTERVAL` works across one-shot reductions.
- Align successive projections with an orthogonal Procrustes rotation on elite anchors (when available) to reduce axis flips/rotations between epochs.

