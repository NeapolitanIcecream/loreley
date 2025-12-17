"""Settings page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY


def render() -> None:
    st.title("Settings")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    health = api_get_or_stop(api_base_url, "/api/v1/health") or {}

    st.subheader("API health")
    st.json(health)

    if isinstance(health, dict):
        settings = health.get("settings")
        if settings is not None:
            st.subheader("Settings (safe)")
            st.json(settings)


