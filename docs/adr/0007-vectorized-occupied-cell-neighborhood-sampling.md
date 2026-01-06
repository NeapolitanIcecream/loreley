# ADR 0007: Vectorized occupied-cell neighbourhood sampling for MAP-Elites

Date: 2026-01-06

Context: The scheduler samples inspirations from nearby MAP-Elites archive cells. Enumerating the full Chebyshev ball scales as (2r+1)^d and can stall scheduling at higher dimensionality or radius.
Decision: The sampler computes Chebyshev (L∞) distances from the base cell to occupied archive cells in a single NumPy vectorized pass and samples inspirations by increasing radius.
Decision: The neighbourhood walk draws candidates within the initial radius as one randomized pool, then processes subsequent radii as shells to avoid redundant rescans.
Consequences: Scheduling cost becomes O(N * d + N * R) (occupied cells N, radius steps R); large grids remain responsive; `_neighbor_indices` stays as a small-grid helper used by tests.


