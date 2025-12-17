"""Graph endpoints for UI visualizations."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from loreley.api.schemas.graphs import CommitGraphEdgeOut, CommitGraphNodeOut, CommitGraphOut
from loreley.api.services.graphs import build_commit_lineage_graph

router = APIRouter()


@router.get("/graphs/commit_lineage", response_model=CommitGraphOut)
def commit_lineage(
    experiment_id: UUID,
    max_nodes: int = Query(default=500, ge=1, le=5000),
    mode: str = Query(default="parent_chain"),
) -> CommitGraphOut:
    graph = build_commit_lineage_graph(
        experiment_id=experiment_id,
        max_nodes=max_nodes,
        mode=mode,
    )
    nodes = [
        CommitGraphNodeOut(
            commit_hash=n.commit_hash,
            parent_commit_hash=n.parent_commit_hash,
            island_id=n.island_id,
            experiment_id=n.experiment_id,
            created_at=n.created_at,
            author=n.author,
            message=n.message,
            metric_name=graph.metric_name,
            metric_value=n.metric_value,
            fitness=n.fitness,
            is_elite=False,
            extra=dict(n.extra or {}),
        )
        for n in graph.nodes
    ]
    edges = [CommitGraphEdgeOut(source=e.source, target=e.target, kind=e.kind) for e in graph.edges]
    return CommitGraphOut(
        experiment_id=experiment_id,
        metric_name=graph.metric_name,
        mode=mode,
        max_nodes=int(max_nodes),
        truncated=bool(graph.truncated),
        nodes=nodes,
        edges=edges,
    )


