"""Shared helpers for calling the UI API from Streamlit pages."""

from __future__ import annotations

from typing import Any

import streamlit as st

from loreley.api.artifacts import ARTIFACT_SPECS, artifact_filename
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


def get_api_client(base_url: str) -> LoreleyAPIClient:
    """Return an API client with connection reuse enabled.

    When available, this function is wrapped by `st.cache_resource` to keep a
    single client per base URL across reruns.
    """
    return LoreleyAPIClient(base_url, reuse_connections=True)


if hasattr(st, "cache_resource"):
    get_api_client = st.cache_resource(show_spinner=False)(get_api_client)  # type: ignore[assignment]


@st.cache_data(ttl=60, show_spinner=False)
def api_get(base_url: str, path: str, params: tuple[tuple[str, str], ...] = ()) -> Any:
    """Cached GET request returning JSON."""

    client = get_api_client(base_url)
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
        client = get_api_client(base_url)
        return client.get_bytes(path, params=params)
    except APIError as exc:
        st.error(f"API error: {exc}")
        st.stop()


def render_artifact_downloads(
    *,
    api_base_url: str,
    artifacts: dict[str, Any] | None,
    key_prefix: str,
    empty_message: str = "No artifacts available.",
) -> None:
    """Render artifact download buttons for an artifacts URL dict."""

    if not artifacts:
        st.write(empty_message)
        return

    rendered = False
    for artifact_key, label in ARTIFACT_SPECS:
        url = artifacts.get(f"{artifact_key}_url")
        if not url:
            continue
        data, content_type = api_get_bytes_or_stop(api_base_url, str(url))
        st.download_button(
            f"Download: {label}",
            data=data,
            file_name=artifact_filename(artifact_key),
            mime=content_type or "application/octet-stream",
            key=f"{key_prefix}_{artifact_key}",
        )
        rendered = True

    if not rendered:
        st.write(empty_message)


