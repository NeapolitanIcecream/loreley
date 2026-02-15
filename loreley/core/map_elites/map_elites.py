"""Backward-compatible exports for MAP-Elites manager APIs."""

from __future__ import annotations

from .manager import MapElitesManager
from .types import CommitEmbeddingArtifacts, IslandState, MapElitesInsertionResult, MapElitesRecord, Vector

__all__ = [
    "Vector",
    "CommitEmbeddingArtifacts",
    "MapElitesInsertionResult",
    "MapElitesManager",
    "MapElitesRecord",
    "IslandState",
]
