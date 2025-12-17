"""Commits page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY, EXPERIMENT_ID_KEY, ISLAND_ID_KEY


def render() -> None:
    st.title("Commits")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    experiment_id = st.session_state.get(EXPERIMENT_ID_KEY)
    island_id = st.session_state.get(ISLAND_ID_KEY)
    if not api_base_url:
        st.error("API base URL is not configured.")
        return
    if not experiment_id:
        st.warning("No experiment selected.")
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    params = {"experiment_id": experiment_id, "limit": 2000}
    if island_id:
        params["island_id"] = island_id
    rows = api_get_or_stop(api_base_url, "/api/v1/commits", params=params) or []
    df = pd.DataFrame(rows)

    st.subheader("Commits")
    if df.empty:
        st.info("No commits found.")
        return

    query = st.text_input("Search (commit hash / author / message)", value="").strip().lower()
    if query:
        for col in ["commit_hash", "author", "message"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
        mask = False
        for col in ["commit_hash", "author", "message"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(query, na=False)
        df = df[mask]

    grid = render_table(df, key="commits_grid", selection="single")
    sel = selected_rows(grid)

    st.divider()
    if not sel:
        st.info("Select a commit to see details.")
        return

    commit_hash = sel[0].get("commit_hash")
    if not commit_hash:
        st.warning("Selected row has no commit_hash.")
        return

    detail = api_get_or_stop(api_base_url, f"/api/v1/commits/{commit_hash}")
    st.subheader(f"Commit detail: {commit_hash}")

    if not isinstance(detail, dict):
        st.json(detail)
        return

    metrics = detail.get("metrics") if isinstance(detail.get("metrics"), list) else []

    left, right = st.columns([2, 1])
    with left:
        st.write(
            {
                "author": detail.get("author"),
                "island_id": detail.get("island_id"),
                "parent_commit_hash": detail.get("parent_commit_hash"),
                "created_at": detail.get("created_at"),
            }
        )
        st.text_area("Message", value=str(detail.get("message") or ""), height=120)

        with st.expander("Evaluation summary", expanded=False):
            st.write(detail.get("evaluation_summary") or "")

        with st.expander("Extra context", expanded=False):
            st.json(detail.get("extra_context") or {})

    with right:
        st.subheader("Metrics")
        if metrics:
            try:
                import pandas as pd  # already imported, but keep local clarity
                import plotly.express as px
            except Exception:
                st.json(metrics)
            else:
                mdf = pd.DataFrame(metrics)
                st.dataframe(mdf[["name", "value", "unit"]], use_container_width=True)
                fig = px.bar(mdf, x="name", y="value", title="Metrics", text="value")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics found for this commit.")


