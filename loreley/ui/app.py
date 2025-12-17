"""Streamlit entrypoint for the Loreley UI."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from loreley.ui.client import APIError, LoreleyAPIClient
from loreley.ui.components.api import api_get_or_stop
from loreley.ui.pages.archive import render as render_archive
from loreley.ui.pages.commits import render as render_commits
from loreley.ui.pages.experiments import render as render_experiments
from loreley.ui.pages.graphs import render as render_graphs
from loreley.ui.pages.jobs import render as render_jobs
from loreley.ui.pages.logs import render as render_logs
from loreley.ui.pages.overview import render as render_overview
from loreley.ui.pages.settings import render as render_settings
from loreley.ui.state import (
    API_BASE_URL_KEY,
    EXPERIMENT_ID_KEY,
    EXPERIMENT_LABEL_KEY,
    ISLAND_ID_KEY,
    REPOSITORY_ID_KEY,
    REPOSITORY_SLUG_KEY,
)


# Streamlit infers the URL pathname for callable pages from the callable's name.
# Our page modules expose a same-named `render()` function, which would collide
# (e.g. multiple pages inferred as pathname "render"). Wrap them with uniquely
# named callables to ensure stable, unique routing.
def overview() -> None:
    render_overview()


def experiments() -> None:
    render_experiments()


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
    st.session_state.setdefault(REPOSITORY_ID_KEY, None)
    st.session_state.setdefault(REPOSITORY_SLUG_KEY, None)
    st.session_state.setdefault(EXPERIMENT_ID_KEY, None)
    st.session_state.setdefault(EXPERIMENT_LABEL_KEY, None)
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
        st.sidebar.success("Cache cleared")

    client = LoreleyAPIClient(api_base_url)

    with st.sidebar.expander("Connection", expanded=False):
        if st.button("Ping API", key="ping_api"):
            payload = _fetch_json(client, "/api/v1/health")
            st.success("API reachable")
            st.json(payload)

    repos = api_get_or_stop(api_base_url, "/api/v1/repositories") or []
    if not repos:
        st.sidebar.warning("No repositories found.")
        st.stop()

    slug_to_repo: dict[str, dict[str, Any]] = {}
    for r in repos:
        if not isinstance(r, dict):
            continue
        slug = r.get("slug")
        repo_id = r.get("id")
        if not slug or not repo_id:
            continue
        slug_to_repo[str(slug)] = r

    repo_slugs = sorted(slug_to_repo)
    if not repo_slugs:
        st.sidebar.warning("No repositories found.")
        st.stop()

    current_slug = st.session_state.get(REPOSITORY_SLUG_KEY)
    if current_slug not in slug_to_repo:
        current_slug = repo_slugs[0]

    selected_slug = st.sidebar.selectbox("Repository", repo_slugs, index=repo_slugs.index(current_slug))
    repo = slug_to_repo[selected_slug]
    st.session_state[REPOSITORY_SLUG_KEY] = selected_slug
    st.session_state[REPOSITORY_ID_KEY] = repo.get("id")

    experiments = api_get_or_stop(api_base_url, f"/api/v1/repositories/{repo['id']}/experiments") or []
    exp_items: list[tuple[str, str]] = []
    for e in experiments:
        if not isinstance(e, dict):
            continue
        exp_id = e.get("id")
        if not exp_id:
            continue
        name = (e.get("name") or "").strip()
        config_hash = (e.get("config_hash") or "").strip()
        label = name or f"{config_hash[:8]}" if config_hash else str(exp_id)
        exp_items.append((label, str(exp_id)))

    if not exp_items:
        st.sidebar.warning("No experiments found for this repository.")
        st.stop()

    label_to_id = {label: exp_id for label, exp_id in exp_items}
    labels = [label for label, _ in exp_items]

    current_label = st.session_state.get(EXPERIMENT_LABEL_KEY)
    if current_label not in label_to_id:
        current_label = labels[0]

    selected_label = st.sidebar.selectbox("Experiment", labels, index=labels.index(current_label))
    st.session_state[EXPERIMENT_LABEL_KEY] = selected_label
    st.session_state[EXPERIMENT_ID_KEY] = label_to_id[selected_label]

    islands = (
        api_get_or_stop(
            api_base_url,
            "/api/v1/archive/islands",
            params={"experiment_id": label_to_id[selected_label]},
        )
        or []
    )
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
            st.Page(experiments, title="Experiments"),
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


