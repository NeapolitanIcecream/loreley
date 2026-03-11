"""Commits page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_or_stop, api_get_page_or_stop, render_artifact_downloads
from loreley.ui.paging import (
    advance_cursor_pager,
    current_cursor,
    normalize_cursor_pager,
    pager_signature,
)
from loreley.ui.state import API_BASE_URL_KEY, COMMIT_HASH_KEY, ISLAND_ID_KEY

_COMMITS_CURSOR_KEY = "loreley_commits_cursor_stack"
_COMMITS_CURSOR_INDEX_KEY = "loreley_commits_cursor_index"
_COMMITS_CURSOR_SIGNATURE_KEY = "loreley_commits_cursor_signature"


def render() -> None:
    st.title("Commits")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    island_id = st.session_state.get(ISLAND_ID_KEY)
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    page_size = st.selectbox("Page size", [50, 100, 200, 500], index=1)
    query = st.text_input("Search (commit hash / author / subject)", value="").strip()

    params: dict[str, object] = {"limit": page_size}
    if island_id:
        params["island_id"] = island_id
    if query:
        params["query"] = query

    signature = pager_signature((api_base_url, island_id or "", query, page_size))
    state = normalize_cursor_pager(
        signature=signature,
        stored_signature=st.session_state.get(_COMMITS_CURSOR_SIGNATURE_KEY),
        cursors=st.session_state.get(_COMMITS_CURSOR_KEY),
        index=st.session_state.get(_COMMITS_CURSOR_INDEX_KEY),
    )
    st.session_state[_COMMITS_CURSOR_SIGNATURE_KEY] = state.signature
    st.session_state[_COMMITS_CURSOR_KEY] = list(state.cursors)
    st.session_state[_COMMITS_CURSOR_INDEX_KEY] = state.index

    cursor = current_cursor(state)
    if cursor:
        params["cursor"] = cursor
    page = api_get_page_or_stop(
        api_base_url,
        "/api/v1/commits/page",
        params=params,
    )
    rows = page.get("items") if isinstance(page, dict) else []
    df = pd.DataFrame(rows if isinstance(rows, list) else [])

    st.subheader("Commits")
    if df.empty:
        st.info("No commits found.")
        return

    st.caption(f"page={state.index + 1} items={len(df)}")
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("Previous page", disabled=state.index <= 0, key="commits_prev_page"):
            st.session_state[_COMMITS_CURSOR_INDEX_KEY] = state.index - 1
            st.session_state[COMMIT_HASH_KEY] = None
            _rerun()
    with next_col:
        if st.button(
            "Next page",
            disabled=not page.get("next_cursor"),
            key="commits_next_page",
        ):
            next_state = advance_cursor_pager(
                state,
                next_cursor=str(page.get("next_cursor") or ""),
            )
            st.session_state[_COMMITS_CURSOR_KEY] = list(next_state.cursors)
            st.session_state[_COMMITS_CURSOR_INDEX_KEY] = next_state.index
            st.session_state[COMMIT_HASH_KEY] = None
            _rerun()

    grid = render_table(df, key="commits_grid", selection="single")
    sel = selected_rows(grid)

    st.divider()
    # Persist table selection to session-state (used by the detail view below).
    if sel:
        value = sel[0].get("commit_hash")
        try:
            is_missing = value is None or pd.isna(value)
        except Exception:
            is_missing = value is None
        if not is_missing:
            selected_commit_hash = str(value).strip()
            if selected_commit_hash:
                st.session_state[COMMIT_HASH_KEY] = selected_commit_hash

    # Ensure state value remains valid even when the user filters the table.
    commit_hash = st.session_state.get(COMMIT_HASH_KEY)
    if isinstance(commit_hash, str):
        commit_hash = commit_hash.strip() or None
    if isinstance(commit_hash, str) and commit_hash and "commit_hash" in df.columns:
        visible_hashes: set[str] = set()
        try:
            values = df["commit_hash"].tolist()
        except Exception:
            values = []
        for v in values:
            try:
                is_missing = v is None or pd.isna(v)
            except Exception:
                is_missing = v is None
            if is_missing:
                continue
            s = str(v).strip()
            if s:
                visible_hashes.add(s)
        if commit_hash not in visible_hashes:
            st.session_state[COMMIT_HASH_KEY] = None
            commit_hash = None

    if not commit_hash:
        st.info("Select a commit to see details.")
        return

    detail = api_get_or_stop(
        api_base_url,
        f"/api/v1/commits/{commit_hash}",
    )
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
                "job_id": detail.get("job_id"),
                "created_at": detail.get("created_at"),
            }
        )
        st.text_input("Subject", value=str(detail.get("subject") or ""), disabled=True)
        st.text_area("Change summary", value=str(detail.get("change_summary") or ""), height=140, disabled=True)

        highlights = detail.get("highlights") if isinstance(detail.get("highlights"), list) else []
        key_files = detail.get("key_files") if isinstance(detail.get("key_files"), list) else []
        with st.expander("Highlights", expanded=True):
            if highlights:
                for item in highlights:
                    st.write(f"- {item}")
            else:
                st.write("No highlights.")

        with st.expander("Key files", expanded=False):
            if key_files:
                for item in key_files:
                    st.write(f"- {item}")
            else:
                st.write("No key files.")

        with st.expander("Evaluation summary", expanded=False):
            st.write(detail.get("evaluation_summary") or "")

        artifacts = detail.get("artifacts") if isinstance(detail.get("artifacts"), dict) else {}
        with st.expander("Artifacts", expanded=False):
            render_artifact_downloads(
                api_base_url=api_base_url,
                artifacts=artifacts,
                key_prefix=f"dl_commit_{commit_hash}",
                empty_message="No artifacts available for this commit.",
            )

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


def _rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
