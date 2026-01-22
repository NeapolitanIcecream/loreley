"""Streamlit entrypoint for the Loreley UI."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from loreley.ui.client import APIError, LoreleyAPIClient
from loreley.ui.components.api import api_get_or_stop, get_api_client
from loreley.ui.pages.archive import render as render_archive
from loreley.ui.pages.commits import render as render_commits
from loreley.ui.pages.graphs import render as render_graphs
from loreley.ui.pages.jobs import render as render_jobs
from loreley.ui.pages.logs import render as render_logs
from loreley.ui.pages.overview import render as render_overview
from loreley.ui.pages.settings import render as render_settings
from loreley.ui.state import API_BASE_URL_KEY, ISLAND_ID_KEY


# Streamlit infers the URL pathname for callable pages from the callable's name.
# Our page modules expose a same-named `render()` function, which would collide
# (e.g. multiple pages inferred as pathname "render"). Wrap them with uniquely
# named callables to ensure stable, unique routing.
def overview() -> None:
    render_overview()


def jobs() -> None:
    render_jobs()


def commits() -> None:
    render_commits()


def archive() -> None:
    render_archive()


def graphs() -> None:
    render_graphs()


def logs() -> None:
    render_logs()


def settings() -> None:
    render_settings()


def _init_session_defaults() -> None:
    api_base_url = os.getenv("LORELEY_UI_API_BASE_URL", "http://127.0.0.1:8000")
    st.session_state.setdefault(API_BASE_URL_KEY, api_base_url)
    st.session_state.setdefault(ISLAND_ID_KEY, None)


def _fetch_json(client: LoreleyAPIClient, path: str, *, params: dict[str, Any] | None = None) -> Any:
    """Fetch JSON from API with Streamlit-friendly errors."""

    try:
        return client.get_json(path, params=params)
    except APIError as exc:
        st.sidebar.error(f"API error: {exc}")
        st.stop()


def _render_sidebar() -> None:
    st.sidebar.header("Loreley")
    api_base_url = (
        st.sidebar.text_input(
        "API base URL",
        value=st.session_state[API_BASE_URL_KEY],
        help="Example: http://127.0.0.1:8000",
        )
        or ""
    ).strip()
    st.session_state[API_BASE_URL_KEY] = api_base_url

    if st.sidebar.button("Refresh data (clear cache)", key="clear_cache"):
        st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
        st.sidebar.success("Cache cleared")

    with st.sidebar.expander("Connection", expanded=False):
        if st.button("Ping API", key="ping_api"):
            try:
                client = get_api_client(api_base_url)
            except APIError as exc:
                st.sidebar.error(f"API error: {exc}")
                st.stop()
            payload = _fetch_json(client, "/api/v1/health")
            st.success("API reachable")
            st.json(payload)

    instance = api_get_or_stop(api_base_url, "/api/v1/instance") or {}
    if isinstance(instance, dict):
        exp_id = instance.get("experiment_id_raw")
        root_commit = instance.get("root_commit_hash")
        repo_slug = instance.get("repository_slug")
        if exp_id:
            st.sidebar.caption(f"Experiment: {exp_id}")
        if repo_slug:
            st.sidebar.caption(f"Repository: {repo_slug}")
        if root_commit:
            st.sidebar.caption(f"Root commit: {str(root_commit)[:12]}")

    islands = api_get_or_stop(api_base_url, "/api/v1/archive/islands") or []
    island_ids = [
        i.get("island_id")
        for i in islands
        if isinstance(i, dict) and i.get("island_id")
    ]
    island_ids = sorted({str(i) for i in island_ids})
    if island_ids:
        current_island = st.session_state.get(ISLAND_ID_KEY)
        if current_island not in island_ids:
            current_island = island_ids[0]
        selected_island = st.sidebar.selectbox("Island", island_ids, index=island_ids.index(current_island))
        st.session_state[ISLAND_ID_KEY] = selected_island


def main() -> None:
    """Main Streamlit entrypoint."""

    st.set_page_config(
        page_title="Loreley",
        layout="wide",
    )

    _init_session_defaults()
    _render_sidebar()

    # Use Streamlit's modern router when available; fallback to a simple single page.
    if hasattr(st, "Page") and hasattr(st, "navigation"):
        pages = [
            st.Page(overview, title="Overview"),
            st.Page(jobs, title="Jobs"),
            st.Page(commits, title="Commits"),
            st.Page(archive, title="Archive"),
            st.Page(graphs, title="Graphs"),
            st.Page(logs, title="Logs"),
            st.Page(settings, title="Settings"),
        ]
        nav = st.navigation(pages, position="sidebar")
        nav.run()
    else:  # pragma: no cover - version dependent
        overview()


if __name__ == "__main__":  # pragma: no cover
    main()


