"""Archive explorer page."""

from __future__ import annotations

from typing import Any, cast

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY, EXPERIMENT_ID_KEY, ISLAND_ID_KEY


def render() -> None:
    st.title("Archive")

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
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing dependency: {exc}")
        return

    islands = api_get_or_stop(api_base_url, "/api/v1/archive/islands", params={"experiment_id": experiment_id}) or []
    st.subheader("Islands")
    st.dataframe(islands, width="stretch")

    if not island_id:
        st.info("Select an island in the sidebar to explore records.")
        return

    meta = api_get_or_stop(
        api_base_url,
        "/api/v1/archive/snapshot_meta",
        params={"experiment_id": experiment_id, "island_id": island_id},
    ) or {}

    dims = int(meta.get("dims", 0) or 0)
    cells_per_dim = int(meta.get("cells_per_dim", 0) or 0)
    st.caption(f"island={island_id} dims={dims} cells_per_dim={cells_per_dim} entries={meta.get('entry_count')}")

    records = api_get_or_stop(
        api_base_url,
        "/api/v1/archive/records",
        params={"experiment_id": experiment_id, "island_id": island_id},
    ) or []
    records_df = pd.DataFrame(records)
    if records_df.empty:
        st.info("No archive records yet.")
        return

    # Visualization
    st.subheader("Visualization")
    try:
        import plotly.express as px
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing plotly dependency: {exc}")
        return

    if dims == 2 and cells_per_dim > 0 and "cell_index" in records_df.columns:
        grid = np.full((cells_per_dim, cells_per_dim), np.nan, dtype=float)
        for _, row in records_df.iterrows():
            raw_idx = row.get("cell_index")
            if raw_idx is None:
                continue
            raw_fitness = row.get("fitness", 0.0)
            try:
                idx = int(cast(Any, raw_idx))
                coords = np.unravel_index(idx, (cells_per_dim, cells_per_dim))
                grid[coords] = float(cast(Any, raw_fitness))
            except Exception:
                continue
        fig = px.imshow(
            grid,
            title="Cell fitness heatmap (2D)",
            aspect="auto",
            origin="lower",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        # Scatter projection using selected dims.
        measures = records_df.get("measures")
        if measures is None:
            st.info("Records have no measures; cannot plot.")
        else:
            max_dim = 0
            try:
                max_dim = max(len(m) for m in measures if isinstance(m, list))
            except Exception:
                max_dim = 0
            if max_dim >= 2:
                dim_x = st.selectbox("X dimension", list(range(max_dim)), index=0)
                dim_y = st.selectbox("Y dimension", list(range(max_dim)), index=1)
                points = []
                for r in records:
                    if not isinstance(r, dict):
                        continue
                    vec = r.get("measures")
                    if not isinstance(vec, list) or len(vec) <= max(dim_x, dim_y):
                        continue
                    points.append(
                        {
                            "x": float(vec[dim_x]),
                            "y": float(vec[dim_y]),
                            "fitness": float(r.get("fitness", 0.0)),
                            "commit_hash": r.get("commit_hash"),
                            "cell_index": r.get("cell_index"),
                        }
                    )
                plot_df = pd.DataFrame(points)
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color="fitness",
                    hover_data=["commit_hash", "cell_index"],
                    title="Archive records scatter (selected measures)",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough measure dimensions to plot a scatter projection.")

    st.subheader("Records")
    grid_resp = render_table(records_df, key="archive_records_grid", selection="single")
    sel = selected_rows(grid_resp)
    if sel:
        commit_hash = sel[0].get("commit_hash")
        if commit_hash:
            detail = api_get_or_stop(api_base_url, f"/api/v1/commits/{commit_hash}")
            with st.expander("Selected commit detail", expanded=False):
                st.json(detail)


