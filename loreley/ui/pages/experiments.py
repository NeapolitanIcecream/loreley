"""Experiments page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.api import api_get_or_stop
from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.state import (
    API_BASE_URL_KEY,
    EXPERIMENT_ID_KEY,
    EXPERIMENT_LABEL_KEY,
    REPOSITORY_ID_KEY,
)


def render() -> None:
    st.title("Experiments")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    repo_id = st.session_state.get(REPOSITORY_ID_KEY)
    experiment_id = st.session_state.get(EXPERIMENT_ID_KEY)
    experiment_label = st.session_state.get(EXPERIMENT_LABEL_KEY)

    if not api_base_url:
        st.error("API base URL is not configured.")
        return
    if not repo_id:
        st.warning("No repository selected.")
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    experiments = api_get_or_stop(api_base_url, f"/api/v1/repositories/{repo_id}/experiments") or []
    df = pd.DataFrame(experiments)
    st.subheader("Experiments")
    if df.empty:
        st.info("No experiments found.")
        return

    grid = render_table(df, key="experiments_grid", selection="single")
    sel = selected_rows(grid)
    if sel:
        picked = sel[0]
        picked_id = picked.get("id")
        if picked_id:
            st.session_state[EXPERIMENT_ID_KEY] = str(picked_id)
            st.session_state[EXPERIMENT_LABEL_KEY] = str(picked.get("name") or picked.get("config_hash") or picked_id)

    st.divider()

    effective_experiment_id = st.session_state.get(EXPERIMENT_ID_KEY) or experiment_id
    if not effective_experiment_id:
        st.warning("Select an experiment to see details.")
        return

    detail = api_get_or_stop(api_base_url, f"/api/v1/experiments/{effective_experiment_id}")
    st.subheader(f"Selected experiment: {experiment_label or effective_experiment_id}")

    with st.expander("Config snapshot", expanded=True):
        st.json(detail.get("config_snapshot") if isinstance(detail, dict) else detail)

    islands = api_get_or_stop(
        api_base_url,
        "/api/v1/archive/islands",
        params={"experiment_id": effective_experiment_id},
    ) or []
    st.subheader("Islands")
    st.dataframe(islands, width="stretch")


