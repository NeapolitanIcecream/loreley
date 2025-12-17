"""Thin wrappers around Streamlit-AgGrid for consistent tables."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_table(df, *, key: str, selection: str = "single") -> dict[str, Any]:
    """Render a dataframe with AgGrid and return the raw grid response.

    Parameters:
        df: A pandas DataFrame.
        key: Streamlit component key.
        selection: "single" or "multiple".
    """

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

    return AgGrid(
        df,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.SORTING_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        key=key,
    )


def selected_rows(grid_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract selected rows from an AgGrid response."""

    value = grid_response.get("selected_rows")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    value = grid_response.get("selected")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


