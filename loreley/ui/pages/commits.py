"""Commits page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_bytes_or_stop, api_get_or_stop
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

    query = st.text_input("Search (commit hash / author / subject)", value="").strip().lower()
    if query:
        for col in ["commit_hash", "author", "subject", "change_summary"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
        mask = False
        for col in ["commit_hash", "author", "subject", "change_summary"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(query, na=False)
        df = df[mask]

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
        params={"experiment_id": experiment_id},
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
            if not artifacts:
                st.write("No artifacts available for this commit.")
            else:
                label_map = {
                    "planning_prompt_url": "Planning prompt",
                    "planning_raw_output_url": "Planning raw output",
                    "planning_plan_json_url": "Planning plan JSON",
                    "coding_prompt_url": "Coding prompt",
                    "coding_raw_output_url": "Coding raw output",
                    "coding_execution_json_url": "Coding execution JSON",
                    "evaluation_json_url": "Evaluation JSON",
                    "evaluation_logs_url": "Evaluation logs",
                }
                for key, label in label_map.items():
                    url = artifacts.get(key)
                    if not url:
                        continue
                    data, content_type = api_get_bytes_or_stop(api_base_url, str(url))
                    filename = str(url).rstrip("/").split("/")[-1] or f"{key}.txt"
                    st.download_button(
                        f"Download: {label}",
                        data=data,
                        file_name=filename,
                        mime=content_type or "application/octet-stream",
                        key=f"dl_{commit_hash}_{key}",
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


