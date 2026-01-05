# ADR 0003: Fixed embedding dimensions per experiment

Date: 2026-01-04
Status: Accepted

Context: Repo-state embeddings aggregate file vectors and require a single, consistent dimensionality; supporting multiple dimensions in the DB cache introduced batch-local selection and could crash aggregation.
Decision: Treat `MAPELITES_CODE_EMBEDDING_DIMENSIONS` as an experiment-scoped invariant (default: 2) and always read/write file-embedding cache rows filtered by that dimension.
Constraints: Changing embedding dimensions results in a new experiment config hash and a new pipeline signature; existing cache rows with other dimensions are intentionally ignored.
Consequences: Dimension mismatch during aggregation is eliminated and cache logic is simplified; opting into a different dimensionality is an explicit configuration change that starts a new experiment lifecycle.


