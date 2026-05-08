"""Overview dashboard page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from loreley.api.pagination import MAX_PAGE_LIMIT
from loreley.ui.components.api import api_get_all_pages_or_stop, api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY, ISLAND_ID_KEY


def _build_overview_kpis(
    *,
    islands: object,
    jobs: object,
    island_id: object,
) -> dict[str, Any]:
    job_rows = _job_rows(jobs)
    island_rows = _island_rows(islands)
    status_counts = _job_status_counts(job_rows)
    metric_name, best_fitness = _overview_metric_summary(island_rows)
    selected_stats = _selected_island_stats(island_rows, island_id)

    return {
        "status_counts": status_counts,
        "total_jobs": len(job_rows),
        "succeeded": int(status_counts.get("succeeded", 0)),
        "failed": int(status_counts.get("failed", 0)),
        "running": int(status_counts.get("running", 0)),
        "metric_name": metric_name,
        "best_fitness": best_fitness,
        "coverage": selected_stats.get("coverage") if selected_stats else None,
        "qd_score": selected_stats.get("qd_score") if selected_stats else None,
        "norm_qd_score": selected_stats.get("norm_qd_score") if selected_stats else None,
        "occupied": selected_stats.get("occupied") if selected_stats else None,
        "cells": selected_stats.get("cells") if selected_stats else None,
    }


def _job_rows(jobs: object) -> list[dict[str, Any]]:
    if not isinstance(jobs, list):
        return []
    return [row for row in jobs if isinstance(row, dict)]


def _island_rows(islands: object) -> list[dict[str, Any]]:
    if not isinstance(islands, list):
        return []
    return [row for row in islands if isinstance(row, dict)]


def _job_status_counts(job_rows: list[dict[str, Any]]) -> dict[str, int]:
    status_counts: dict[str, int] = {}
    for row in job_rows:
        status = str(row.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
    return status_counts


def _overview_metric_summary(island_rows: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    metric_name = _first_island_metric_name(island_rows)
    higher_is_better = _first_higher_is_better(island_rows)
    try:
        values = [
            float(row.get("best_fitness", 0.0))
            for row in island_rows
            if row.get("best_fitness") is not None
        ]
    except Exception:
        return metric_name, None
    if not values:
        return metric_name, None
    return metric_name, max(values) if higher_is_better else min(values)


def _first_island_metric_name(island_rows: list[dict[str, Any]]) -> str | None:
    for row in island_rows:
        if row.get("metric_name"):
            return str(row.get("metric_name"))
    return None


def _first_higher_is_better(island_rows: list[dict[str, Any]]) -> bool:
    for row in island_rows:
        if row.get("higher_is_better") is not None:
            return bool(row.get("higher_is_better"))
    return True


def _selected_island_stats(
    island_rows: list[dict[str, Any]],
    island_id: object,
) -> dict[str, Any] | None:
    if not island_id:
        return None
    selected_id = str(island_id)
    for row in island_rows:
        if str(row.get("island_id") or "") == selected_id:
            return row
    return None


def render() -> None:
    """Render the overview page."""

    st.title("Overview")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    island_id = st.session_state.get(ISLAND_ID_KEY)

    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    with st.expander("Context", expanded=False):
        instance = api_get_or_stop(api_base_url, "/api/v1/instance") or {}
        st.write(
            {
                "experiment_id": instance.get("experiment_id_raw"),
                "root_commit_hash": instance.get("root_commit_hash"),
                "repository_slug": instance.get("repository_slug"),
                "island": island_id,
                "api_base_url": api_base_url,
            }
        )

    # Data pulls
    islands = api_get_or_stop(api_base_url, "/api/v1/archive/islands") or []
    jobs = api_get_all_pages_or_stop(
        api_base_url,
        "/api/v1/jobs/page",
        page_limit=MAX_PAGE_LIMIT,
        max_items=MAX_PAGE_LIMIT,
    )
    graph = api_get_or_stop(
        api_base_url,
        "/api/v1/graphs/commit_lineage",
        params={"max_nodes": 1000},
    ) or {}
    operator = api_get_or_stop(api_base_url, "/api/v1/operator/status") or {}

    # KPI cards
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    jobs_df: Any = pd.DataFrame(jobs)
    kpis = _build_overview_kpis(islands=islands, jobs=jobs, island_id=island_id)
    status_counts = kpis["status_counts"]
    total_jobs = kpis["total_jobs"]
    succeeded = kpis["succeeded"]
    failed = kpis["failed"]
    metric_name = kpis["metric_name"]
    best_fitness = kpis["best_fitness"]
    coverage = kpis["coverage"]
    qd_score = kpis["qd_score"]
    norm_qd_score = kpis["norm_qd_score"]
    occupied = kpis["occupied"]
    cells = kpis["cells"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs (loaded)", f"{total_jobs}")
    c2.metric("Succeeded", f"{succeeded}")
    c3.metric("Failed", f"{failed}")
    c4.metric(
        f"Best {metric_name or 'metric'}",
        f"{best_fitness:.6f}" if isinstance(best_fitness, (int, float)) else "n/a",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Coverage",
        f"{float(coverage) * 100:.2f}%" if isinstance(coverage, (int, float)) else "n/a",
    )
    c6.metric(
        "Objective norm QD",
        f"{float(norm_qd_score):.6f}" if isinstance(norm_qd_score, (int, float)) else "n/a",
    )
    c7.metric(
        "Objective QD",
        f"{float(qd_score):.6f}" if isinstance(qd_score, (int, float)) else "n/a",
    )
    c8.metric(
        "Occupied cells",
        f"{int(occupied)}/{int(cells)}"
        if isinstance(occupied, (int, float)) and isinstance(cells, (int, float))
        else "n/a",
    )

    _render_operator_status_band(operator)

    # Charts
    try:
        import plotly.express as px
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing plotly dependency: {exc}")
        return

    if status_counts:
        status_df = pd.DataFrame(
            [{"status": k, "count": int(v)} for k, v in status_counts.items()]
        ).sort_values("count", ascending=False)
        fig = px.bar(status_df, x="status", y="count", title="Job status (loaded)")
        st.plotly_chart(fig, width="stretch")

    # Job duration histogram
    if not jobs_df.empty and {"started_at", "completed_at"} <= set(jobs_df.columns):
        durations: Any = jobs_df[["started_at", "completed_at"]].dropna()
        if not durations.empty:
            durations = durations.copy()
            durations["started_at"] = pd.to_datetime(durations["started_at"], errors="coerce", utc=True)
            durations["completed_at"] = pd.to_datetime(durations["completed_at"], errors="coerce", utc=True)
            durations = durations.dropna()
            durations["duration_seconds"] = (durations["completed_at"] - durations["started_at"]).dt.total_seconds()
            durations = durations[durations["duration_seconds"] >= 0]
            if not durations.empty:
                fig = px.histogram(
                    durations,
                    x="duration_seconds",
                    nbins=40,
                    title="Job duration (seconds) - loaded jobs",
                )
                st.plotly_chart(fig, width="stretch")

    # Fitness over time (from graph nodes)
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if isinstance(nodes, list) and nodes:
        nodes_df: Any = pd.DataFrame([n for n in nodes if isinstance(n, dict)])
        value_column = "metric_value" if "metric_value" in nodes_df.columns else "fitness"
        if not nodes_df.empty and {"created_at", value_column} <= set(nodes_df.columns):
            nodes_df = nodes_df.copy()
            nodes_df["created_at"] = pd.to_datetime(nodes_df["created_at"], errors="coerce", utc=True)
            nodes_df[value_column] = pd.to_numeric(nodes_df[value_column], errors="coerce")
            nodes_df = nodes_df.dropna(subset=["created_at", value_column]).sort_values("created_at")
            if not nodes_df.empty:
                graph_higher_is_better = bool(graph.get("higher_is_better", True)) if isinstance(graph, dict) else True
                if graph_higher_is_better:
                    nodes_df["best_so_far"] = nodes_df[value_column].cummax()
                else:
                    nodes_df["best_so_far"] = nodes_df[value_column].cummin()
                fig = px.line(
                    nodes_df,
                    x="created_at",
                    y="best_so_far",
                    title=f"Best {metric_name or 'metric'} over time (loaded commits)",
                )
                st.plotly_chart(fig, width="stretch")

    st.subheader("Islands")
    st.dataframe(islands or [], width="stretch")


def _render_operator_status_band(operator: object) -> None:
    if not isinstance(operator, dict):
        return
    campaign = operator.get("campaign_program") if isinstance(operator.get("campaign_program"), dict) else {}
    scheduler = campaign.get("scheduler") if isinstance(campaign, dict) and isinstance(campaign.get("scheduler"), dict) else {}
    current = campaign.get("current_file") if isinstance(campaign, dict) and isinstance(campaign.get("current_file"), dict) else {}
    baseline = operator.get("baseline") if isinstance(operator.get("baseline"), dict) else {}
    repair = operator.get("repair_pool") if isinstance(operator.get("repair_pool"), dict) else {}
    health = operator.get("job_health") if isinstance(operator.get("job_health"), dict) else {}
    leases = health.get("job_leases") if isinstance(health, dict) and isinstance(health.get("job_leases"), dict) else {}

    st.subheader("Operator Status")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Baseline", str(baseline.get("root_baseline_status") or "missing"))
    baseline_value = baseline.get("root_baseline_value")
    o2.metric(
        "Baseline value",
        f"{float(baseline_value):.6f}" if isinstance(baseline_value, (int, float)) else "n/a",
    )
    o3.metric("Repair eligible", str(_count_value(repair, "by_repair_state", "eligible")))
    o4.metric("Stale running", str(leases.get("stale_running", 0)))

    o5, o6, o7, o8 = st.columns(4)
    o5.metric("Current program", _short_hash(current.get("hash")) or "missing")
    o6.metric("Active program", _short_hash(scheduler.get("active_hash")) or "n/a")
    o7.metric("Active repair jobs", str(repair.get("active_repair_jobs", 0)))
    o8.metric("Running missing lease", str(leases.get("running_without_lease", 0)))

    if scheduler.get("current_matches_active") is False:
        st.warning("Campaign program file hash differs from scheduler active hash.")
    if baseline.get("failure_kind"):
        st.warning(f"Baseline failure: {baseline.get('failure_kind')}")


def _count_value(data: dict[str, object], group_key: str, item_key: str) -> int:
    group = data.get(group_key)
    if not isinstance(group, dict):
        return 0
    try:
        return int(group.get(item_key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _short_hash(value: object) -> str | None:
    text = str(value or "").strip()
    return text[:12] if text else None
