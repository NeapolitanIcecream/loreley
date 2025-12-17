"""Overview dashboard page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import (
    API_BASE_URL_KEY,
    EXPERIMENT_ID_KEY,
    EXPERIMENT_LABEL_KEY,
    ISLAND_ID_KEY,
    REPOSITORY_SLUG_KEY,
)

def render() -> None:
    """Render the overview page."""

    st.title("Overview")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    experiment_id = st.session_state.get(EXPERIMENT_ID_KEY)
    experiment_label = st.session_state.get(EXPERIMENT_LABEL_KEY)
    repo_slug = st.session_state.get(REPOSITORY_SLUG_KEY)
    island_id = st.session_state.get(ISLAND_ID_KEY)

    if not api_base_url:
        st.error("API base URL is not configured.")
        return
    if not experiment_id:
        st.warning("No experiment selected.")
        return

    with st.expander("Context", expanded=False):
        st.write(
            {
                "repository": repo_slug,
                "experiment": experiment_label,
                "experiment_id": experiment_id,
                "island": island_id,
                "api_base_url": api_base_url,
            }
        )

    # Data pulls
    islands = api_get_or_stop(api_base_url, "/api/v1/archive/islands", params={"experiment_id": experiment_id}) or []
    jobs = api_get_or_stop(api_base_url, "/api/v1/jobs", params={"experiment_id": experiment_id, "limit": 2000}) or []
    graph = api_get_or_stop(
        api_base_url,
        "/api/v1/graphs/commit_lineage",
        params={"experiment_id": experiment_id, "max_nodes": 1000},
    ) or {}

    # KPI cards
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    jobs_df: Any = pd.DataFrame(jobs)
    status_counts = {}
    if not jobs_df.empty and "status" in jobs_df.columns:
        status_counts = jobs_df["status"].value_counts(dropna=False).to_dict()

    total_jobs = int(jobs_df.shape[0])
    succeeded = int(status_counts.get("succeeded", 0))
    failed = int(status_counts.get("failed", 0))
    running = int(status_counts.get("running", 0))

    best_fitness = None
    if isinstance(islands, list) and islands:
        try:
            best_fitness = max(float(i.get("best_fitness", 0.0)) for i in islands if isinstance(i, dict))
        except Exception:
            best_fitness = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs (loaded)", f"{total_jobs}")
    c2.metric("Succeeded", f"{succeeded}")
    c3.metric("Failed", f"{failed}")
    c4.metric("Best fitness", f"{best_fitness:.6f}" if isinstance(best_fitness, (int, float)) else "n/a")

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
        if not nodes_df.empty and {"created_at", "fitness"} <= set(nodes_df.columns):
            nodes_df = nodes_df.copy()
            nodes_df["created_at"] = pd.to_datetime(nodes_df["created_at"], errors="coerce", utc=True)
            nodes_df["fitness"] = pd.to_numeric(nodes_df["fitness"], errors="coerce")
            nodes_df = nodes_df.dropna(subset=["created_at", "fitness"]).sort_values("created_at")
            if not nodes_df.empty:
                nodes_df["best_so_far"] = nodes_df["fitness"].cummax()
                fig = px.line(nodes_df, x="created_at", y="best_so_far", title="Best fitness over time (loaded commits)")
                st.plotly_chart(fig, width="stretch")

    st.subheader("Islands")
    st.dataframe(islands or [], width="stretch")


