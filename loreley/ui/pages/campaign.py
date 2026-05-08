"""Campaign operator page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from loreley.ui.components.api import api_get_or_stop, api_post_or_stop
from loreley.ui.state import API_BASE_URL_KEY


def render() -> None:
    st.title("Campaign")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    status = api_get_or_stop(api_base_url, "/api/v1/operator/status") or {}
    if not isinstance(status, dict):
        st.json(status)
        return

    campaign = _dict(status.get("campaign_program"))
    current = _dict(campaign.get("current_file"))
    scheduler = _dict(campaign.get("scheduler"))
    baseline = _dict(status.get("baseline"))

    st.subheader("Program State")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current file", _short_hash(current.get("hash")) or "missing")
    c2.metric("Scheduler active", _short_hash(scheduler.get("active_hash")) or "n/a")
    c3.metric("Persisted", _short_hash(scheduler.get("persisted_hash")) or "n/a")
    c4.metric("Policy", str(scheduler.get("change_policy") or "n/a"))

    if scheduler.get("current_matches_active") is False:
        st.warning("The current file hash differs from the scheduler active campaign hash.")

    st.write(
        {
            "source_path": current.get("source_path"),
            "title": current.get("title"),
            "active_source": scheduler.get("active_source"),
            "current_hash": current.get("hash"),
            "active_hash": scheduler.get("active_hash"),
            "normalized_hash": current.get("normalized_hash"),
        }
    )

    st.subheader("Baseline")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Status", str(baseline.get("root_baseline_status") or "missing"))
    b2.metric("Metric", str(baseline.get("root_baseline_metric") or "n/a"))
    value = baseline.get("root_baseline_value")
    b3.metric("Value", f"{float(value):.6f}" if isinstance(value, (int, float)) else "n/a")
    b4.metric("Key", _short_hash(baseline.get("baseline_key_hash")) or "n/a")
    if baseline.get("failure_kind") or baseline.get("failure_summary"):
        st.warning(str(baseline.get("failure_kind") or baseline.get("failure_summary")))

    st.subheader("Sections")
    sections = _dict(current.get("sections"))
    if not sections:
        st.info("No campaign program sections found.")
    else:
        for label, value in sections.items():
            with st.expander(str(label), expanded=label == "goal"):
                if isinstance(value, list):
                    if value:
                        for item in value:
                            st.write(f"- {item}")
                    else:
                        st.write("None")
                else:
                    st.write(value if value not in (None, "") else "None")

    warnings = current.get("parse_warnings")
    if isinstance(warnings, list) and warnings:
        st.subheader("Warnings")
        st.dataframe(warnings, width="stretch")

    st.subheader("Baseline Ensure Task")
    col_start, col_refresh = st.columns(2)
    with col_start:
        if st.button("Create baseline ensure task", key="campaign_baseline_ensure"):
            task = api_post_or_stop(
                api_base_url,
                "/api/v1/operator/tasks/baseline-ensure",
                json_body={},
            )
            st.cache_data.clear()
            st.success(f"Task created: {task.get('id') if isinstance(task, dict) else 'unknown'}")
            _rerun()
    with col_refresh:
        if st.button("Refresh tasks", key="campaign_refresh_tasks"):
            st.cache_data.clear()
            _rerun()

    tasks = api_get_or_stop(api_base_url, "/api/v1/operator/tasks", params={"limit": 20}) or {}
    rows = tasks.get("items") if isinstance(tasks, dict) else []
    if isinstance(rows, list) and rows:
        st.dataframe(rows, width="stretch")
        latest = rows[0] if isinstance(rows[0], dict) else None
        if latest:
            with st.expander("Latest task", expanded=True):
                st.json(latest)
    else:
        st.info("No operator tasks yet.")


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _short_hash(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:12]


def _rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
