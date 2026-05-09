"""Archive explorer page."""

from __future__ import annotations

from typing import Any, cast

import streamlit as st

from loreley.api.pagination import MAX_PAGE_LIMIT
from loreley.ui.archive_plotting import build_scatter_points
from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_all_pages_or_stop, api_get_or_stop, api_get_page_or_stop
from loreley.ui.paging import advance_cursor_pager, current_cursor, normalize_cursor_pager, pager_signature
from loreley.ui.state import API_BASE_URL_KEY, ISLAND_ID_KEY

_ARCHIVE_CURSOR_KEY = "loreley_archive_cursor_stack"
_ARCHIVE_CURSOR_INDEX_KEY = "loreley_archive_cursor_index"
_ARCHIVE_CURSOR_SIGNATURE_KEY = "loreley_archive_cursor_signature"


def render() -> None:
    st.title("Archive")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    island_id = st.session_state.get(ISLAND_ID_KEY)
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing dependency: {exc}")
        return

    islands = api_get_or_stop(api_base_url, "/api/v1/archive/islands") or []
    st.subheader("Islands")
    st.dataframe(islands, width="stretch")

    if not island_id:
        st.info("Select an island in the sidebar to explore records.")
        return

    meta = api_get_or_stop(
        api_base_url,
        "/api/v1/archive/snapshot_meta",
        params={"island_id": island_id},
    ) or {}

    dims = int(meta.get("dims", 0) or 0)
    cells_per_dim = int(meta.get("cells_per_dim", 0) or 0)
    entry_count = int(meta.get("entry_count", 0) or 0)
    page_size = st.selectbox("Page size", [100, 250, 500, 1000], index=1)

    signature = pager_signature((api_base_url, island_id, page_size))
    state = normalize_cursor_pager(
        signature=signature,
        stored_signature=st.session_state.get(_ARCHIVE_CURSOR_SIGNATURE_KEY),
        cursors=st.session_state.get(_ARCHIVE_CURSOR_KEY),
        index=st.session_state.get(_ARCHIVE_CURSOR_INDEX_KEY),
    )
    st.session_state[_ARCHIVE_CURSOR_SIGNATURE_KEY] = state.signature
    st.session_state[_ARCHIVE_CURSOR_KEY] = list(state.cursors)
    st.session_state[_ARCHIVE_CURSOR_INDEX_KEY] = state.index

    params = {"island_id": island_id, "limit": page_size}
    cursor = current_cursor(state)
    if cursor:
        params["cursor"] = cursor
    records_page = api_get_page_or_stop(
        api_base_url,
        "/api/v1/archive/records/page",
        params=params,
    )

    records = records_page.get("items") if isinstance(records_page, dict) else []
    visualization_records = _load_visualization_records(api_base_url, island_id)
    st.caption(
        f"island={island_id} dims={dims} cells_per_dim={cells_per_dim} "
        f"entries={entry_count} page={state.index + 1}"
    )
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("Previous page", disabled=state.index <= 0, key="archive_prev_page"):
            st.session_state[_ARCHIVE_CURSOR_INDEX_KEY] = state.index - 1
            _rerun()
    with next_col:
        if st.button(
            "Next page",
            disabled=not records_page.get("next_cursor"),
            key="archive_next_page",
        ):
            next_state = advance_cursor_pager(
                state,
                next_cursor=str(records_page.get("next_cursor") or ""),
            )
            st.session_state[_ARCHIVE_CURSOR_KEY] = list(next_state.cursors)
            st.session_state[_ARCHIVE_CURSOR_INDEX_KEY] = next_state.index
            _rerun()

    records_df = pd.DataFrame(records)
    if records_df.empty:
        st.info("No archive records yet.")
        return
    records_df = _decorate_archive_df(records_df)

    visualization_df = pd.DataFrame(visualization_records)

    metric_name = None
    higher_is_better = True
    if isinstance(islands, list):
        for entry in islands:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("island_id") or "") != str(island_id):
                continue
            metric_name = entry.get("metric_name")
            if entry.get("higher_is_better") is not None:
                higher_is_better = bool(entry.get("higher_is_better"))
            break

    value_key = "metric_value" if "metric_value" in records_df.columns and records_df["metric_value"].notna().any() else "fitness"
    value_label = metric_name or ("Metric" if value_key == "metric_value" else "Fitness")

    # Visualization
    st.subheader("Visualization")
    try:
        import plotly.express as px
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing plotly dependency: {exc}")
        return

    if dims == 2 and cells_per_dim > 0 and "cell_index" in visualization_df.columns:
        grid = np.full((cells_per_dim, cells_per_dim), np.nan, dtype=float)
        for _, row in visualization_df.iterrows():
            raw_idx = row.get("cell_index")
            if raw_idx is None:
                continue
            raw_fitness = row.get("fitness", 0.0)
            try:
                idx = int(cast(Any, raw_idx))
                coords = np.unravel_index(idx, (cells_per_dim, cells_per_dim))
                grid[coords] = float(cast(Any, row.get(value_key, raw_fitness)))
            except Exception:
                continue
        fig = px.imshow(
            grid,
            title=f"Cell {value_label.lower()} heatmap (2D)",
            aspect="auto",
            origin="lower",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        # Scatter projection using selected dims.
        measures = visualization_df.get("measures")
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
                points = build_scatter_points(
                    [r for r in visualization_records if isinstance(r, dict)],
                    dim_x=dim_x,
                    dim_y=dim_y,
                    value_key=value_key,
                )
                if not points:
                    st.info("No archive records with plottable values for the selected dimensions.")
                    return
                plot_df = pd.DataFrame(points)
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color="value",
                    hover_data=["commit_hash", "cell_index", "candidate_fate_label"],
                    title=f"Archive records scatter ({value_label.lower()})",
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
            detail = api_get_or_stop(
                api_base_url,
                f"/api/v1/commits/{commit_hash}",
            )
            with st.expander("Selected commit detail", expanded=False):
                st.json(detail)


def _load_visualization_records(api_base_url: str, island_id: str) -> list[dict[str, Any]]:
    payload = api_get_all_pages_or_stop(
        api_base_url,
        "/api/v1/archive/records/page",
        params={"island_id": island_id},
        page_limit=MAX_PAGE_LIMIT,
        max_items=None,
    )
    return [item for item in payload if isinstance(item, dict)]


def _decorate_archive_df(df):
    if df.empty:
        return df
    out = df.copy()
    if "candidate_fate_label" in out.columns:
        out["fate"] = out["candidate_fate_label"].fillna("unknown")
    if "delta_from_root_baseline" in out.columns:
        out["baseline_delta"] = out["delta_from_root_baseline"]
    if "agent_visible_evidence_count" in out.columns:
        out["evidence"] = out["agent_visible_evidence_count"].fillna(0).astype(int).map(
            lambda count: f"agent-visible:{count}" if count else "none"
        )
    return out


def _rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
