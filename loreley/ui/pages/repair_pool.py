"""Repair pool operator page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_page_or_stop, api_post_or_stop
from loreley.ui.paging import advance_cursor_pager, current_cursor, normalize_cursor_pager, pager_signature
from loreley.ui.state import API_BASE_URL_KEY

_REPAIR_CURSOR_KEY = "loreley_repair_cursor_stack"
_REPAIR_CURSOR_INDEX_KEY = "loreley_repair_cursor_index"
_REPAIR_CURSOR_SIGNATURE_KEY = "loreley_repair_cursor_signature"


def render() -> None:
    st.title("Repair Pool")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    _render_schedule_controls(api_base_url)
    filters = _render_filter_controls()
    params = _repair_pool_params(filters)
    state = _sync_repair_pager(api_base_url=api_base_url, filters=filters)

    cursor = current_cursor(state)
    if cursor:
        params["cursor"] = cursor
    page = api_get_page_or_stop(api_base_url, "/api/v1/repair/pool", params=params)
    _render_repair_summary(page.get("summary") if isinstance(page, dict) else {})

    rows = page.get("items") if isinstance(page, dict) else []
    df = pd.DataFrame(rows if isinstance(rows, list) else [])
    if df.empty:
        st.info("No failed candidates match these filters.")
        return

    df = _decorate_repair_df(df)
    st.caption(f"page={state.index + 1} items={len(df)}")
    _render_repair_pager(state=state, page=page if isinstance(page, dict) else {})

    grid = render_table(df, key="repair_pool_grid", selection="single")
    _render_selected_candidate(
        api_base_url=api_base_url,
        selected=selected_rows(grid),
    )


def _render_schedule_controls(api_base_url: str) -> None:
    schedule_col, refresh_col = st.columns(2)
    with schedule_col:
        if st.button("Schedule one repair", key="repair_schedule_one"):
            result = api_post_or_stop(api_base_url, "/api/v1/repair/schedule-one", json_body={})
            st.cache_data.clear()
            if isinstance(result, dict) and result.get("scheduled"):
                st.success(f"Scheduled repair job: {result.get('job_id')}")
            else:
                st.info(str(result.get("message") if isinstance(result, dict) else "No repair scheduled."))
            _rerun()
    with refresh_col:
        if st.button("Refresh repair pool", key="repair_refresh"):
            st.cache_data.clear()
            _rerun()


def _render_filter_controls() -> dict[str, object]:
    page_size = st.selectbox("Page size", [50, 100, 200, 500], index=1)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        repair_state = st.selectbox(
            "Repair state",
            ["all", "eligible", "scheduled", "repairing", "audit_only", "ineligible", "exhausted", "repaired", "quarantined", "discarded"],
            index=0,
        )
    with f2:
        lifecycle_status = st.selectbox(
            "Lifecycle",
            ["all", "active", "quarantined", "discarded"],
            index=0,
        )
    with f3:
        failure_kind = st.text_input("Failure kind", value="").strip()
    with f4:
        campaign_program_hash = st.text_input("Program hash", value="").strip()

    return {
        "page_size": page_size,
        "repair_state": repair_state,
        "lifecycle_status": lifecycle_status,
        "failure_kind": failure_kind,
        "campaign_program_hash": campaign_program_hash,
    }


def _repair_pool_params(filters: dict[str, object]) -> dict[str, object]:
    params: dict[str, object] = {"limit": filters["page_size"]}
    _add_param_if_selected(params, "repair_state", filters["repair_state"])
    _add_param_if_selected(params, "lifecycle_status", filters["lifecycle_status"])
    _add_param_if_present(params, "failure_kind", filters["failure_kind"])
    _add_param_if_present(params, "campaign_program_hash", filters["campaign_program_hash"])
    return params


def _add_param_if_selected(params: dict[str, object], key: str, value: object) -> None:
    if value != "all":
        params[key] = value


def _add_param_if_present(params: dict[str, object], key: str, value: object) -> None:
    if value:
        params[key] = value


def _sync_repair_pager(*, api_base_url: str, filters: dict[str, object]):
    signature = pager_signature(
        (
            api_base_url,
            filters["page_size"],
            filters["repair_state"],
            filters["lifecycle_status"],
            filters["failure_kind"],
            filters["campaign_program_hash"],
        )
    )
    state = normalize_cursor_pager(
        signature=signature,
        stored_signature=st.session_state.get(_REPAIR_CURSOR_SIGNATURE_KEY),
        cursors=st.session_state.get(_REPAIR_CURSOR_KEY),
        index=st.session_state.get(_REPAIR_CURSOR_INDEX_KEY),
    )
    st.session_state[_REPAIR_CURSOR_SIGNATURE_KEY] = state.signature
    st.session_state[_REPAIR_CURSOR_KEY] = list(state.cursors)
    st.session_state[_REPAIR_CURSOR_INDEX_KEY] = state.index
    return state


def _render_repair_summary(summary: object) -> None:
    if isinstance(summary, dict):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Failed candidates", int(summary.get("total_failed_candidates") or 0))
        c2.metric("Active repair jobs", int(summary.get("active_repair_jobs") or 0))
        c3.metric("Eligible", int(_counts(summary, "by_repair_state").get("eligible", 0)))
        c4.metric("Quarantined", int(_counts(summary, "by_lifecycle_status").get("quarantined", 0)))


def _render_repair_pager(*, state, page: dict[str, object]) -> None:
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("Previous page", disabled=state.index <= 0, key="repair_prev_page"):
            st.session_state[_REPAIR_CURSOR_INDEX_KEY] = state.index - 1
            _rerun()
    with next_col:
        if st.button("Next page", disabled=not page.get("next_cursor"), key="repair_next_page"):
            next_state = advance_cursor_pager(
                state,
                next_cursor=str(page.get("next_cursor") or ""),
            )
            st.session_state[_REPAIR_CURSOR_KEY] = list(next_state.cursors)
            st.session_state[_REPAIR_CURSOR_INDEX_KEY] = next_state.index
            _rerun()


def _render_selected_candidate(
    *,
    api_base_url: str,
    selected: list[dict[str, object]],
) -> None:
    if not selected:
        st.info("Select a failed candidate to inspect or change its operator state.")
        return

    candidate = selected[0]
    candidate_id = str(candidate.get("id") or "")
    st.subheader(f"Candidate: {candidate_id}")
    st.write(
        {
            "commit_hash": candidate.get("commit_hash"),
            "nearest_viable_ancestor_hash": candidate.get("nearest_viable_ancestor_hash"),
            "repair_state": candidate.get("repair_state"),
            "lifecycle_status": candidate.get("lifecycle_status"),
            "failure_kind": candidate.get("failure_kind"),
            "active_repair_job_id": candidate.get("active_repair_job_id"),
            "campaign_program_hash": candidate.get("campaign_program_hash"),
        }
    )
    st.text_area(
        "Diagnostic summary",
        value=str(candidate.get("diagnostic_summary") or candidate.get("failure_summary") or ""),
        height=140,
        disabled=True,
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Quarantine", disabled=not candidate_id, key="repair_quarantine"):
            _post_candidate_action(api_base_url, candidate_id, "quarantine")
    with a2:
        if st.button("Discard", disabled=not candidate_id, key="repair_discard"):
            _post_candidate_action(api_base_url, candidate_id, "discard")
    with a3:
        if st.button("Restore", disabled=not candidate_id, key="repair_restore"):
            _post_candidate_action(api_base_url, candidate_id, "restore")


def _decorate_repair_df(df):
    df = df.copy()
    if "commit_hash" in df.columns:
        df["commit"] = df["commit_hash"].astype(str).str.slice(0, 12)
    if "campaign_program_hash" in df.columns:
        df["program"] = df["campaign_program_hash"].fillna("").astype(str).str.slice(0, 12)
    if "active_repair_job_id" in df.columns:
        df["active_repair"] = df["active_repair_job_id"].fillna("").astype(str).str.slice(0, 12)
    if "diagnostic_policy_passed" in df.columns:
        df["diagnostic"] = df["diagnostic_policy_passed"].map(lambda v: "passed" if v is True else ("failed" if v is False else "n/a"))
    return df


def _counts(summary: dict[str, object], key: str) -> dict[str, int]:
    value = summary.get(key)
    return value if isinstance(value, dict) else {}


def _post_candidate_action(api_base_url: str, candidate_id: str, action: str) -> None:
    api_post_or_stop(
        api_base_url,
        f"/api/v1/repair/candidates/{candidate_id}/{action}",
        json_body={},
    )
    st.cache_data.clear()
    st.success(f"Candidate {action} applied.")
    _rerun()


def _rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()
