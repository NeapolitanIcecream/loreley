"""Core data structures for MAP-Elites manager orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ribs.archives import GridArchive

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
    """Snapshot of a single elite stored inside an archive cell."""

    commit_hash: str
    island_id: str
    cell_index: int
    fitness: float
    measures: Vector
    solution: Vector
    timestamp: float

    @property
    def dimensions(self) -> int:
        return len(self.measures)


@dataclass(slots=True, frozen=True)
class MapElitesInsertionResult:
    """Wraps the outcome of adding a commit to the archive."""

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
    """Mutable bookkeeping attached to each island.

    Invariants:
    - `archive` is the source of truth for occupied cells and elite payload.
    - `index_to_commit` and `commit_to_index` are inverse mappings for occupied
      cells maintained by manager/archive helpers.
    - Manager-level `commit_to_island` must agree with per-island mappings.
    - Any archive replace/clear operation must keep archive payload and mappings
      consistent before the next read path (`get_records/get_cell_commits`).
    """

    archive: GridArchive
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    history: tuple[PcaHistoryEntry, ...] = field(default_factory=tuple)
    projection: PCAProjection | None = None
    samples_since_fit: int = 0
    commit_to_index: dict[str, int] = field(default_factory=dict)
    index_to_commit: dict[int, str] = field(default_factory=dict)
