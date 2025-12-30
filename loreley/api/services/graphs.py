"""Graph builders for commit lineage visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, Metric


@dataclass(frozen=True, slots=True)
class CommitNode:
    commit_hash: str
    parent_commit_hash: str | None
    island_id: str | None
    experiment_id: UUID | None
    created_at: datetime | None
    author: str | None
    message: str | None
    metric_value: float | None
    fitness: float | None
    extra: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CommitEdge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True, slots=True)
class CommitGraph:
    nodes: list[CommitNode]
    edges: list[CommitEdge]
    truncated: bool
    metric_name: str | None


def build_commit_lineage_graph(
    *,
    experiment_id: UUID,
    max_nodes: int = 500,
    mode: str = "parent_chain",
    settings: Settings | None = None,
) -> CommitGraph:
    """Build a simple commit-parent graph for the given experiment.

    Currently supported modes:
    - parent_chain: edges from parent -> child when parent is known in the same result set.
    """

    effective_settings = settings or get_settings()
    metric_name = (effective_settings.mapelites_fitness_metric or "").strip() or None
    higher_is_better = bool(effective_settings.mapelites_fitness_higher_is_better)
    direction = 1.0 if higher_is_better else -1.0
    fitness_floor = float(effective_settings.mapelites_fitness_floor)

    limit = max(1, min(int(max_nodes), 5000))
    mode = (mode or "parent_chain").strip()

    with session_scope() as session:
        stmt = (
            select(CommitCard)
            .where(CommitCard.experiment_id == experiment_id)
            .order_by(CommitCard.created_at.desc())
            .limit(limit)
        )
        commits = list(session.execute(stmt).scalars())

        metric_map: dict[str, float] = {}
        if metric_name and commits:
            commit_ids = [c.id for c in commits]
            metric_stmt = (
                select(Metric.commit_card_id, Metric.value)
                .where(Metric.commit_card_id.in_(commit_ids), Metric.name == metric_name)
            )
            for commit_card_id, value in session.execute(metric_stmt).all():
                if commit_card_id and value is not None:
                    metric_map[str(commit_card_id)] = float(value)

    commit_set = {c.commit_hash for c in commits}
    nodes: list[CommitNode] = []
    edges: list[CommitEdge] = []

    for c in commits:
        raw = metric_map.get(str(c.id))
        if raw is None:
            fitness = fitness_floor
        else:
            fitness = raw * direction
        nodes.append(
            CommitNode(
                commit_hash=c.commit_hash,
                parent_commit_hash=c.parent_commit_hash,
                island_id=c.island_id,
                experiment_id=c.experiment_id,
                created_at=c.created_at,
                author=c.author,
                message=getattr(c, "subject", None),
                metric_value=raw,
                fitness=fitness,
                extra={},
            )
        )

    if mode == "parent_chain":
        for c in commits:
            parent = (c.parent_commit_hash or "").strip()
            if parent and parent in commit_set:
                edges.append(CommitEdge(source=parent, target=c.commit_hash, kind="parent"))

    truncated = len(commits) >= limit
    return CommitGraph(nodes=nodes, edges=edges, truncated=truncated, metric_name=metric_name)


