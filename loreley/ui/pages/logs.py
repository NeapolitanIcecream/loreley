"""Logs page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY


def render() -> None:
    st.title("Logs")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    role = st.selectbox("Role", ["scheduler", "worker", "ui_api"], index=0)
    files = api_get_or_stop(api_base_url, "/api/v1/logs", params={"role": role}) or []

    st.subheader("Log files")
    st.dataframe(files or [], width="stretch")

    if files:
        names = [f.get("name") for f in files if isinstance(f, dict) and f.get("name")]
        if names:
            name = st.selectbox("File", names, index=0)
            lines = st.slider("Tail lines", min_value=50, max_value=2000, value=200, step=50)
            tail = api_get_or_stop(
                api_base_url,
                "/api/v1/logs/tail",
                params={"role": role, "filename": name, "lines": lines},
            ) or {}
            st.subheader("Tail")
            st.code((tail or {}).get("content", ""), language="text")


