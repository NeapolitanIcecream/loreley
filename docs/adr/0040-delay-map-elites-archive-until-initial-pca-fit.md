# ADR 0040: Delay MAP-Elites archive until initial PCA fit

Date: 2026-02-12

Status: Accepted

Decision

- Treat the pre-fit PCA period as a warmup phase: persist PCA history/projection state but do not insert commits into MAP-Elites archives.
- Once the first PCA projection is fitted (epoch 0), seed a fresh archive by projecting warmup commits into the fitted coordinates and persist archive cells via a full replace.
- Keep seed jobs as the only scheduling mechanism while the archive is empty; ensure seed scheduling considers `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES` so enough warmup data exists to fit PCA.

