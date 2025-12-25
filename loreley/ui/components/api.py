"""Shared helpers for calling the UI API from Streamlit pages."""

from __future__ import annotations

from typing import Any

import streamlit as st

from loreley.ui.client import APIError, LoreleyAPIClient


def freeze_params(params: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Convert a params dict into a stable, cache-friendly tuple."""

    if not params:
        return ()
    items: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        items.append((str(key), str(value)))
    return tuple(sorted(items))


@st.cache_data(ttl=60, show_spinner=False)
def api_get(base_url: str, path: str, params: tuple[tuple[str, str], ...] = ()) -> Any:
    """Cached GET request returning JSON."""

    client = LoreleyAPIClient(base_url)
    return client.get_json(path, params=dict(params))


def api_get_or_stop(base_url: str, path: str, *, params: dict[str, Any] | None = None) -> Any:
    """GET JSON, showing an error and stopping the page on failures."""

    try:
        return api_get(base_url, path, freeze_params(params))
    except APIError as exc:
        st.error(f"API error: {exc}")
        st.stop()


def api_get_bytes_or_stop(
    base_url: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[bytes, str | None]:
    """GET raw bytes, showing an error and stopping the page on failures."""

    try:
        client = LoreleyAPIClient(base_url)
        return client.get_bytes(path, params=params)
    except APIError as exc:
        st.error(f"API error: {exc}")
        st.stop()


