"""Graph builders for commit lineage visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from loreley.api.services.candidate_fates import load_candidate_fates_for_commits
from loreley.api.services.evidence import load_evidence_indicators_by_commit_hash
from loreley.config import Settings, get_settings, resolve_objective_contract
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, Metric


@dataclass(frozen=True, slots=True)
class CommitNode:
    commit_hash: str
    parent_commit_hash: str | None
    island_id: str | None
    created_at: datetime | None
    author: str | None
    message: str | None
    primary_metric_value: float | None
    has_evaluation_evidence: bool
    agent_visible_evidence_count: int
    top_evaluation_diagnosis: str | None
    candidate_fate_label: str | None
    candidate_fate_reason: str | None
    seed_portfolio_hash: str | None
    seed_direction_id: str | None
    seed_admission_lane: str | None
    seed_admission_reason: str | None
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
    primary_metric_name: str
    primary_metric_higher_is_better: bool


def build_commit_lineage_graph(
    *,
    max_nodes: int = 500,
    mode: str = "parent_chain",
    settings: Settings | None = None,
) -> CommitGraph:
    """Build a simple commit-parent graph for the current instance.

    Currently supported modes:
    - parent_chain: edges from parent -> child when parent is known in the same result set.
    """

    base_settings = settings or get_settings()
    primary = resolve_objective_contract(base_settings).primary
    metric_name = primary.name

    limit = max(1, min(int(max_nodes), 5000))
    mode = (mode or "parent_chain").strip()

    with session_scope() as session:
        stmt = (
            select(CommitCard)
            .order_by(CommitCard.created_at.desc())
            .limit(limit + 1)
        )
        commits = list(session.execute(stmt).scalars())
        truncated = len(commits) > limit
        commits = commits[:limit]

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
    evidence = load_evidence_indicators_by_commit_hash([c.commit_hash for c in commits])
    fates = load_candidate_fates_for_commits(commits)
    nodes: list[CommitNode] = []
    edges: list[CommitEdge] = []

    for c in commits:
        raw = metric_map.get(str(c.id))
        indicator = evidence.get(c.commit_hash)
        fate = fates.get(c.commit_hash)
        nodes.append(
            CommitNode(
                commit_hash=c.commit_hash,
                parent_commit_hash=c.parent_commit_hash,
                island_id=c.island_id,
                created_at=c.created_at,
                author=c.author,
                message=getattr(c, "subject", None),
                primary_metric_value=raw,
                has_evaluation_evidence=(
                    bool(indicator.has_evaluation_evidence) if indicator is not None else False
                ),
                agent_visible_evidence_count=(
                    int(indicator.agent_visible_evidence_count) if indicator is not None else 0
                ),
                top_evaluation_diagnosis=(
                    indicator.top_evaluation_diagnosis if indicator is not None else None
                ),
                candidate_fate_label=fate.label if fate is not None else None,
                candidate_fate_reason=fate.reason if fate is not None else None,
                seed_portfolio_hash=getattr(c, "seed_portfolio_hash", None),
                seed_direction_id=getattr(c, "seed_direction_id", None),
                seed_admission_lane=getattr(c, "seed_admission_lane", None),
                seed_admission_reason=getattr(c, "seed_admission_reason", None),
                extra={},
            )
        )

    if mode == "parent_chain":
        for c in commits:
            parent = (c.parent_commit_hash or "").strip()
            if parent and parent in commit_set:
                edges.append(CommitEdge(source=parent, target=c.commit_hash, kind="parent"))

    return CommitGraph(
        nodes=nodes,
        edges=edges,
        truncated=truncated,
        primary_metric_name=metric_name,
        primary_metric_higher_is_better=primary.higher_is_better,
    )
