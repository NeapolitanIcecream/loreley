"""Graph endpoints for UI visualizations."""

from __future__ import annotations

from fastapi import APIRouter, Query

from loreley.api.schemas.graphs import CommitGraphEdgeOut, CommitGraphNodeOut, CommitGraphOut
from loreley.api.services.graphs import build_commit_lineage_graph

router = APIRouter()


@router.get("/graphs/commit_lineage", response_model=CommitGraphOut)
def commit_lineage(
    max_nodes: int = Query(default=500, ge=1, le=5000),
    mode: str = Query(default="parent_chain"),
) -> CommitGraphOut:
    graph = build_commit_lineage_graph(
        max_nodes=max_nodes,
        mode=mode,
    )
    nodes: list[CommitGraphNodeOut] = []
    for n in graph.nodes:
        node = CommitGraphNodeOut.model_validate(n)
        node.primary_metric_name = graph.primary_metric_name
        nodes.append(node)
    edges = [CommitGraphEdgeOut.model_validate(e) for e in graph.edges]
    return CommitGraphOut(
        primary_metric_name=graph.primary_metric_name,
        primary_metric_higher_is_better=bool(
            graph.primary_metric_higher_is_better
        ),
        mode=mode,
        max_nodes=int(max_nodes),
        truncated=bool(graph.truncated),
        nodes=nodes,
        edges=edges,
    )
