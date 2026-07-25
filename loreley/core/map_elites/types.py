"""Core data structures for MAP-Elites manager orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .pareto_archive import ParetoGridArchive

if TYPE_CHECKING:
    from .code_embedding import CommitCodeEmbedding
    from .dimension_reduction import FinalEmbedding, PCAProjection, PcaHistoryEntry
    from .preprocess import PreprocessedFile
    from .repository_state_embedding import RepoStateEmbeddingStats

__all__ = [
    "Vector",
    "CommitEmbeddingArtifacts",
    "MapElitesRecord",
    "MapElitesInsertionResult",
    "IslandState",
]

Vector = tuple[float, ...]


@dataclass(slots=True, frozen=True)
class CommitEmbeddingArtifacts:
    """Lightweight container for intermediate embedding artifacts."""

    repo_state_stats: RepoStateEmbeddingStats | None
    preprocessed_files: tuple[PreprocessedFile, ...]
    code_embedding: CommitCodeEmbedding | None
    final_embedding: FinalEmbedding | None

    @property
    def file_count(self) -> int:
        if self.repo_state_stats is not None:
            return int(self.repo_state_stats.files_aggregated)
        return len(self.preprocessed_files)

    @property
    def chunk_count(self) -> int:
        # Repo-state embeddings do not retain chunk-level artifacts.
        return 0


@dataclass(slots=True, frozen=True)
class MapElitesRecord:
    """Snapshot of one Pareto elite retained in a behavior cell."""

    commit_hash: str
    island_id: str
    cell_index: int
    objective_values: Vector
    objective_scores: Vector
    measures: Vector
    timestamp: float
    primary_metric_value: float | None = None
    primary_metric_name: str | None = None
    primary_metric_higher_is_better: bool | None = None
    campaign_baseline_id: str | None = None
    baseline_key_hash: str | None = None
    baseline_status: str | None = None
    delta_from_root_baseline: float | None = None
    candidate_fate_label: str | None = None
    candidate_fate_reason: str | None = None

    @property
    def dimensions(self) -> int:
        return len(self.measures)

    @property
    def primary_objective_value(self) -> float:
        return float(self.objective_values[0])

    @property
    def primary_objective_score(self) -> float:
        return float(self.objective_scores[0])


@dataclass(slots=True, frozen=True)
class MapElitesInsertionResult:
    """Outcome of attempting Pareto admission for one commit."""

    status: int
    delta: float
    record: MapElitesRecord | None
    artifacts: CommitEmbeddingArtifacts
    message: str | None = None

    @property
    def inserted(self) -> bool:
        return self.status > 0 and self.record is not None


@dataclass(slots=True)
class IslandState:
    """Mutable projection and Pareto-archive state for one independent island."""

    archive: ParetoGridArchive
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    history: tuple[PcaHistoryEntry, ...] = field(default_factory=tuple)
    projection: PCAProjection | None = None
    samples_since_fit: int = 0
    commit_to_index: dict[str, int] = field(default_factory=dict)
    index_to_commits: dict[int, tuple[str, ...]] = field(default_factory=dict)
