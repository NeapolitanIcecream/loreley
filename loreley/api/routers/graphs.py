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
    nodes: list[CommitGraphNodeOut] = []
    for n in graph.nodes:
        node = CommitGraphNodeOut.model_validate(n)
        node.metric_name = graph.metric_name
        nodes.append(node)
    edges = [CommitGraphEdgeOut.model_validate(e) for e in graph.edges]
    return CommitGraphOut(
        experiment_id=experiment_id,
        metric_name=graph.metric_name,
        mode=mode,
        max_nodes=int(max_nodes),
        truncated=bool(graph.truncated),
        nodes=nodes,
        edges=edges,
    )


