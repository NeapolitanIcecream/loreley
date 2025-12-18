"""Table helpers for the Streamlit UI.

We historically used Streamlit-AgGrid for interactive tables. However, row
selection + rerun behavior has proven to be fragile across versions and tends
to be the root cause of "selected but details not shown" bugs in pages.

This module provides a stable, uniform API for rendering a table and reading
the selected rows. It prefers Streamlit's native dataframe row selection when
available (Streamlit >= 1.35.0), and falls back to Streamlit-AgGrid when
necessary.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import streamlit as st


def _supports_streamlit_row_selection() -> bool:
    """Return True if st.dataframe supports row selection callbacks."""

    try:
        sig = inspect.signature(st.dataframe)
    except Exception:  # pragma: no cover - extremely unlikely
        return False
    return "on_select" in sig.parameters and "selection_mode" in sig.parameters


def _render_table_streamlit(df, *, key: str, selection: str) -> dict[str, Any]:
    """Render using Streamlit's native dataframe row selection."""

    # Streamlit's documented selection mode is currently "multi-row".
    # Implement "single" selection by truncating to the first selected row.
    selection_mode = "multi-row"
    table_key = f"{key}__streamlit_df"
    try:
        event = st.dataframe(
            df,
            # NOTE: `use_container_width` is deprecated and will be removed after 2025-12-31.
            # Keep this as `width="stretch"` for future Streamlit compatibility.
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode=selection_mode,
            key=table_key,
        )
    except TypeError:  # pragma: no cover - Streamlit signature/version dependent
        # Older Streamlit signatures may not accept a `key` argument here.
        event = st.dataframe(
            df,
            # NOTE: `use_container_width` is deprecated and will be removed after 2025-12-31.
            # Keep this as `width="stretch"` for future Streamlit compatibility.
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode=selection_mode,
        )

    selected_indices: list[int] = []
    try:
        selected_indices = list(getattr(getattr(event, "selection", None), "rows", []) or [])
    except Exception:
        selected_indices = []
    if selection == "single" and selected_indices:
        selected_indices = selected_indices[:1]

    selected: list[dict[str, Any]] = []
    if selected_indices:
        try:
            selected = df.iloc[selected_indices].to_dict(orient="records")
        except Exception:
            selected = []

    # Keep a dict-like contract compatible with downstream pages.
    return {
        "data": df,
        "selected_rows": selected,
        "selection_rows": selected_indices,
        "source": "streamlit",
    }


def _render_table_aggrid(df, *, key: str, selection: str) -> dict[str, Any]:
    """Render a dataframe with Streamlit-AgGrid and return the raw grid response."""

    try:
        from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode
    except Exception as exc:  # pragma: no cover - optional dependency
        st.error(f"Missing streamlit-aggrid dependency: {exc}")
        return {}

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True,
    )
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
    gb.configure_selection(selection, use_checkbox=(selection != "single"))
    grid_options = gb.build()

    resp = AgGrid(
        df,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        # NOTE: Keep update modes minimal to avoid version-specific behavior changes.
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        key=key,
    )

    # `AgGrid` may return a dict-like object depending on the package version.
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, Mapping):  # pragma: no cover - version dependent
        return dict(resp)
    return {"data": df, "selected_rows": [], "source": "aggrid"}


def render_table(df, *, key: str, selection: str = "single") -> dict[str, Any]:
    """Render a dataframe with a stable selection API.

    Parameters:
        df: A pandas DataFrame.
        key: Streamlit component key.
        selection: "single" or "multiple".
    """

    selection = selection if selection in {"single", "multiple"} else "single"
    if _supports_streamlit_row_selection():
        return _render_table_streamlit(df, key=key, selection=selection)
    return _render_table_aggrid(df, key=key, selection=selection)


def selected_rows(grid_response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract selected rows from a table response."""

    if not isinstance(grid_response, Mapping):
        return []

    # Native Streamlit tables: we return a uniform "selected_rows" list of dicts.
    value = grid_response.get("selected_rows")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]

    # Streamlit-AgGrid legacy shapes.
    value = grid_response.get("selected")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]

    # Some versions expose camelCase keys.
    value = grid_response.get("selectedRows")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]

    # Some versions nest selection under an object.
    value = grid_response.get("selection")
    if isinstance(value, dict):
        rows = value.get("rows") or value.get("selected_rows") or value.get("selectedRows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    return []


