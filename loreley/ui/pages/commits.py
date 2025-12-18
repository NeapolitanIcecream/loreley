"""Commits page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY, COMMIT_HASH_KEY, EXPERIMENT_ID_KEY, ISLAND_ID_KEY


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
    # Persist selection so users can recover if dataframe row-selection doesn't
    # trigger a rerun reliably in their environment.
    selected_commit_hash: str | None = None
    if sel:
        value = sel[0].get("commit_hash")
        selected_commit_hash = str(value).strip() if value else None

    commit_hashes: list[str] = []
    if "commit_hash" in df.columns:
        try:
            commit_hashes = [str(v) for v in df["commit_hash"].dropna().astype(str).tolist()]
        except Exception:
            commit_hashes = []
    # De-duplicate while preserving order.
    seen: set[str] = set()
    commit_hashes = [h for h in commit_hashes if not (h in seen or seen.add(h))]

    # Ensure state value remains valid even when the user filters the table.
    current = st.session_state.get(COMMIT_HASH_KEY)
    if isinstance(current, str) and current not in commit_hashes:
        st.session_state[COMMIT_HASH_KEY] = None

    if selected_commit_hash and selected_commit_hash in commit_hashes:
        st.session_state[COMMIT_HASH_KEY] = selected_commit_hash

    with st.expander("Selection", expanded=False):
        st.caption("If clicking a row doesn't update details, use this selector (it forces a rerun).")
        if commit_hashes:
            st.selectbox(
                "Selected commit",
                options=commit_hashes,
                index=None,
                key=COMMIT_HASH_KEY,
                placeholder="Select a commit hash…",
                format_func=lambda h: f"{h[:8]}…{h[-6:]}" if isinstance(h, str) and len(h) > 16 else str(h),
            )
        else:
            st.info("No commits available for selection.")

    commit_hash = st.session_state.get(COMMIT_HASH_KEY) or selected_commit_hash
    if not commit_hash:
        st.info("Select a commit to see details.")
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
                st.dataframe(mdf[["name", "value", "unit"]], width="stretch")
                fig = px.bar(mdf, x="name", y="value", title="Metrics", text="value")
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No metrics found for this commit.")


