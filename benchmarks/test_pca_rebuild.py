from __future__ import annotations

import numpy as np
import pytest

from loreley.config import Settings
from loreley.core.map_elites.dimension_reduction import DimensionReducer, PcaHistoryEntry


def _make_history(*, sample_count: int, feature_count: int) -> tuple[PcaHistoryEntry, ...]:
    rng = np.random.default_rng(0)
    entries: list[PcaHistoryEntry] = []
    for i in range(int(sample_count)):
        vector = tuple(float(v) for v in rng.standard_normal(int(feature_count)))
        entries.append(
            PcaHistoryEntry(
                commit_hash=f"c{i:04d}",
                vector=vector,
                embedding_model="code",
            )
        )
    return tuple(entries)


@pytest.mark.benchmark(group="pca")
def test_pca_fit_projection(benchmark, settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 8
    settings.mapelites_dimensionality_min_fit_samples = 32
    settings.mapelites_feature_normalization_warmup_samples = 32
    settings.mapelites_dimensionality_history_size = 256
    settings.mapelites_dimensionality_refit_interval = 0
    settings.mapelites_dimensionality_penultimate_normalize = False
    settings.mapelites_dimensionality_seed = 0

    history = _make_history(sample_count=128, feature_count=64)
    reducer = DimensionReducer(settings=settings, history=history)

    # Warm once to build caches (e.g., the history matrix).
    assert reducer._fit_projection() is not None  # type: ignore[attr-defined]

    projection = benchmark(reducer._fit_projection)  # type: ignore[attr-defined]
    assert projection is not None

