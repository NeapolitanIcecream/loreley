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


@st.cache_data(ttl=60, show_spinner=False)
def api_get_page(base_url: str, path: str, params: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    """Cached GET request returning a paginated JSON payload."""

    client = get_api_client(base_url)
    return client.get_json_page(path, params=dict(params))


def api_get_page_or_stop(base_url: str, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET a paginated JSON payload, showing an error and stopping on failures."""

    try:
        return api_get_page(base_url, path, freeze_params(params))
    except APIError as exc:
        st.error(f"API error: {exc}")
        st.stop()


def api_get_all_pages_or_stop(
    base_url: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    page_limit: int,
    max_items: int | None = None,
) -> list[Any]:
    """Fetch sequential cursor pages and flatten their `items` arrays."""

    items: list[Any] = []
    cursor: str | None = None
    while True:
        remaining = None if max_items is None else max(0, int(max_items) - len(items))
        if remaining == 0:
            break
        current_params = dict(params or {})
        current_params["limit"] = page_limit if remaining is None else min(page_limit, remaining)
        if cursor:
            current_params["cursor"] = cursor
        page = api_get_page_or_stop(base_url, path, params=current_params)
        page_items = page.get("items")
        if not isinstance(page_items, list):
            break
        items.extend(page_items)
        cursor = page.get("next_cursor")
        if not cursor or not page_items:
            break
    return items


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
        payload_key = f"{key_prefix}_{artifact_key}_payload"
        prepared = st.session_state.get(payload_key)
        if st.button(
            f"Prepare: {label}",
            key=f"{key_prefix}_{artifact_key}_prepare",
        ):
            data, content_type = api_get_bytes_or_stop(api_base_url, str(url))
            prepared = {
                "data": data,
                "content_type": content_type,
            }
            st.session_state[payload_key] = prepared

        if isinstance(prepared, dict) and "data" in prepared:
            st.download_button(
                f"Download: {label}",
                data=prepared["data"],
                file_name=artifact_filename(artifact_key),
                mime=str(prepared.get("content_type") or "application/octet-stream"),
                key=f"{key_prefix}_{artifact_key}_download",
            )
        rendered = True

    if not rendered:
        st.write(empty_message)


def render_evaluation_evidence(
    *,
    api_base_url: str,
    artifacts: list[dict[str, Any]] | None,
    key_prefix: str,
    empty_message: str = "No evaluation evidence available.",
) -> None:
    """Render evaluator-declared evidence separately from fixed worker artifacts."""

    if not artifacts:
        st.write(empty_message)
        return

    rendered = False
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        key = str(artifact.get("key") or f"artifact_{index}")
        label = str(artifact.get("label") or key)
        kind = str(artifact.get("kind") or "artifact")
        visibility = str(artifact.get("visibility") or "")
        st.markdown(f"**{label}**  `{key}`")
        st.caption(
            " | ".join(
                value
                for value in [
                    kind,
                    visibility,
                    _format_size(artifact.get("size_bytes")),
                    _format_hash(artifact.get("sha256")),
                ]
                if value
            )
        )
        summary = str(artifact.get("summary") or "").strip()
        if summary:
            st.write(summary)
        diagnostics = artifact.get("diagnostics")
        if isinstance(diagnostics, list) and diagnostics:
            for diagnostic in diagnostics[:5]:
                if not isinstance(diagnostic, dict):
                    continue
                message = str(diagnostic.get("message") or "").strip()
                if not message:
                    continue
                severity = str(diagnostic.get("severity") or "info")
                kind_value = str(diagnostic.get("kind") or "diagnostic")
                st.write(f"- {severity}/{kind_value}: {message}")
        download_url = artifact.get("download_url")
        if download_url:
            _render_evaluation_artifact_download(
                api_base_url=api_base_url,
                url=str(download_url),
                key_prefix=f"{key_prefix}_{key}",
                label=label,
                mime_type=str(artifact.get("mime_type") or "application/octet-stream"),
            )
        rendered = True

    if not rendered:
        st.write(empty_message)


def render_agent_feedback_preview(feedback: dict[str, Any] | None) -> None:
    """Render the exact bounded evidence text exposed to future agents."""

    if not feedback:
        st.write("No agent feedback preview available.")
        return
    st.caption(
        f"mode={feedback.get('mode')} budget_chars={feedback.get('budget_chars')} "
        f"included={len(feedback.get('included_artifact_keys') or [])} "
        f"omitted={feedback.get('omitted_artifact_count') or 0}"
    )
    text = str(feedback.get("text") or "")
    if text:
        st.text_area("Agent feedback preview", value=text, height=220, disabled=True)
    else:
        st.write("No agent-visible evaluation evidence.")


def _render_evaluation_artifact_download(
    *,
    api_base_url: str,
    url: str,
    key_prefix: str,
    label: str,
    mime_type: str,
) -> None:
    payload_key = f"{key_prefix}_payload"
    prepared = st.session_state.get(payload_key)
    if st.button(f"Prepare: {label}", key=f"{key_prefix}_prepare"):
        data, content_type = api_get_bytes_or_stop(api_base_url, url)
        prepared = {
            "data": data,
            "content_type": content_type,
        }
        st.session_state[payload_key] = prepared
    if isinstance(prepared, dict) and "data" in prepared:
        st.download_button(
            f"Download: {label}",
            data=prepared["data"],
            file_name=f"{label.replace(' ', '_')}{_extension_for_mime(mime_type)}",
            mime=str(prepared.get("content_type") or mime_type),
            key=f"{key_prefix}_download",
        )


def _format_size(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{int(value)} bytes"
    except (TypeError, ValueError):
        return ""


def _format_hash(value: object) -> str:
    text = str(value or "").strip()
    return f"sha256={text[:12]}" if text else ""


def _extension_for_mime(mime_type: str) -> str:
    if mime_type == "application/json":
        return ".json"
    if mime_type == "text/plain":
        return ".txt"
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/svg+xml":
        return ".svg"
    if mime_type == "text/html":
        return ".html"
    return ".bin"
