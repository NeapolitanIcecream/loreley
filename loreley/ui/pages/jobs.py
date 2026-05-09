"""Jobs page."""

from __future__ import annotations

import streamlit as st

from loreley.core.candidate_fate import CANDIDATE_FATE_LABELS
from loreley.db.models import JobStatus
from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import (
    api_get_or_stop,
    api_get_page_or_stop,
    api_post_or_stop,
    render_agent_feedback_preview,
    render_artifact_downloads,
    render_evaluation_evidence,
)
from loreley.ui.pages._confirmations import (
    expire_operator_confirmation,
    operator_confirmation_key,
)
from loreley.ui.paging import advance_cursor_pager, current_cursor, normalize_cursor_pager, pager_signature
from loreley.ui.state import API_BASE_URL_KEY

_JOBS_CURSOR_KEY = "loreley_jobs_cursor_stack"
_JOBS_CURSOR_INDEX_KEY = "loreley_jobs_cursor_index"
_JOBS_CURSOR_SIGNATURE_KEY = "loreley_jobs_cursor_signature"


def render() -> None:
    st.title("Jobs")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    if not api_base_url:
        st.error("API base URL is not configured.")
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    filters = _render_jobs_filters()
    _render_jobs_actions(api_base_url=api_base_url)
    params = _job_page_params(filters)
    state = _sync_jobs_pager(api_base_url=api_base_url, filters=filters)

    cursor = current_cursor(state)
    if cursor:
        params["cursor"] = cursor
    page = api_get_page_or_stop(
        api_base_url,
        "/api/v1/jobs/page",
        params=params,
    )
    rows = page.get("items") if isinstance(page, dict) else []
    df = pd.DataFrame(rows if isinstance(rows, list) else [])
    df = _filter_jobs_df(
        df,
        fate_filter=str(filters["fate_filter"]),
        evidence_filter=str(filters["evidence_filter"]),
    )
    df = _decorate_jobs_df(df)

    st.subheader("Jobs")
    st.caption(f"page={state.index + 1} items={len(df)}")
    _render_jobs_pager(state=state, page=page if isinstance(page, dict) else {})
    if df.empty:
        st.info("No jobs found.")
        return

    grid = render_table(df, key="jobs_grid", selection="single")
    _render_selected_job(
        api_base_url=api_base_url,
        selected=selected_rows(grid),
    )


def _render_jobs_filters() -> dict[str, object]:
    page_size = st.selectbox("Page size", [50, 100, 200, 500], index=1)
    status_options = ["all", *(status.value for status in JobStatus)]
    selected_status = st.selectbox("Status filter", options=status_options, index=0)
    job_kind = st.selectbox("Job kind", ["all", "evolution", "repair"], index=0)
    fate_filter = st.selectbox(
        "Fate",
        _fate_filter_options(),
        index=0,
    )
    evidence_filter = st.selectbox("Evidence", ["all", "has evidence", "agent-visible", "none"], index=0)
    return {
        "page_size": page_size,
        "selected_status": selected_status,
        "job_kind": job_kind,
        "fate_filter": fate_filter,
        "evidence_filter": evidence_filter,
    }


def _render_jobs_actions(*, api_base_url: str) -> None:
    retry_col, refresh_col = st.columns(2)
    with retry_col:
        retry_confirm_base_key = "jobs_retry_failed_stale_confirm"
        retry_confirmed = st.checkbox(
            "Confirm global failed-stale retry",
            help="This retries failed or stale jobs across the whole queue, ignoring the filters and cursor on this page.",
            key=operator_confirmation_key(retry_confirm_base_key),
        )
        if st.button(
            "Retry global failed-stale jobs",
            disabled=not retry_confirmed,
            key="jobs_retry_failed_stale",
        ):
            result = api_post_or_stop(
                api_base_url,
                "/api/v1/jobs/retry-failed-stale",
                json_body=_retry_failed_stale_payload(),
            )
            st.cache_data.clear()
            count = result.get("count") if isinstance(result, dict) else 0
            st.success(f"Retried global failed-stale jobs: {count}")
            expire_operator_confirmation(retry_confirm_base_key)
            _rerun()
    with refresh_col:
        if st.button("Refresh jobs", key="jobs_refresh"):
            st.cache_data.clear()
            _rerun()


def _retry_failed_stale_payload() -> dict[str, bool]:
    return {"all": True}


def _fate_filter_options() -> list[str]:
    return ["all", *sorted(CANDIDATE_FATE_LABELS)]


def _job_page_params(filters: dict[str, object]) -> dict[str, object]:
    params: dict[str, object] = {"limit": filters["page_size"]}
    if filters["selected_status"] != "all":
        params["status"] = filters["selected_status"]
    if filters["job_kind"] != "all":
        params["job_kind"] = filters["job_kind"]
    return params


def _sync_jobs_pager(*, api_base_url: str, filters: dict[str, object]):
    page_size = filters["page_size"]
    selected_status = filters["selected_status"]
    job_kind = filters["job_kind"]

    signature = pager_signature((api_base_url, selected_status, job_kind, page_size))
    state = normalize_cursor_pager(
        signature=signature,
        stored_signature=st.session_state.get(_JOBS_CURSOR_SIGNATURE_KEY),
        cursors=st.session_state.get(_JOBS_CURSOR_KEY),
        index=st.session_state.get(_JOBS_CURSOR_INDEX_KEY),
    )
    st.session_state[_JOBS_CURSOR_SIGNATURE_KEY] = state.signature
    st.session_state[_JOBS_CURSOR_KEY] = list(state.cursors)
    st.session_state[_JOBS_CURSOR_INDEX_KEY] = state.index
    return state


def _render_jobs_pager(*, state, page: dict[str, object]) -> None:
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("Previous page", disabled=state.index <= 0, key="jobs_prev_page"):
            st.session_state[_JOBS_CURSOR_INDEX_KEY] = state.index - 1
            _rerun()
    with next_col:
        if st.button("Next page", disabled=not page.get("next_cursor"), key="jobs_next_page"):
            next_state = advance_cursor_pager(
                state,
                next_cursor=str(page.get("next_cursor") or ""),
            )
            st.session_state[_JOBS_CURSOR_KEY] = list(next_state.cursors)
            st.session_state[_JOBS_CURSOR_INDEX_KEY] = next_state.index
            _rerun()


def _render_selected_job(
    *,
    api_base_url: str,
    selected: list[dict[str, object]],
) -> None:
    st.divider()
    if not selected:
        st.info("Select a job to see details.")
        return

    job_id = selected[0].get("id")
    if not job_id:
        st.warning("Selected row has no job id.")
        return

    detail = api_get_or_stop(api_base_url, f"/api/v1/jobs/{job_id}")
    st.subheader(f"Job detail: {job_id}")
    _render_job_detail(api_base_url=api_base_url, job_id=str(job_id), detail=detail)


def _rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def _render_job_detail(
    *,
    api_base_url: str,
    job_id: str,
    detail: object,
) -> None:
    if not isinstance(detail, dict):
        st.json(detail)
        return
    st.write(_job_top_fields(detail))
    retry_confirm_base_key = f"retry_job_{job_id}_confirm"
    retry_confirmed = st.checkbox(
        f"Confirm retry for job {job_id}",
        help="This requeues the selected job and clears stale candidate/result metadata through the existing retry API.",
        key=operator_confirmation_key(retry_confirm_base_key),
    )
    if st.button(
        "Retry this job",
        disabled=not retry_confirmed,
        key=f"retry_job_{job_id}",
    ):
        result = api_post_or_stop(
            api_base_url,
            f"/api/v1/jobs/{job_id}/retry",
            json_body={},
        )
        st.cache_data.clear()
        st.success(
            f"Retried {result.get('job_id') if isinstance(result, dict) else job_id}"
        )
        expire_operator_confirmation(retry_confirm_base_key)
        _rerun()
    st.text_area("Goal", value=str(detail.get("goal") or ""), height=100, disabled=True)
    if detail.get("iteration_hint"):
        st.caption(f"Iteration hint: {detail.get('iteration_hint')}")
    _render_detail_list_expander("Constraints", detail.get("constraints"))
    _render_detail_list_expander("Acceptance criteria", detail.get("acceptance_criteria"))
    _render_detail_list_expander("Inspirations", detail.get("inspiration_commit_hashes"))
    _render_evidence_sections(api_base_url=api_base_url, job_id=job_id, detail=detail)


def _job_top_fields(detail: dict[str, object]) -> dict[str, object]:
    return {
        key: detail.get(key)
        for key in [
            "status",
            "priority",
            "island_id",
            "scheduled_at",
            "started_at",
            "completed_at",
            "is_seed_job",
            "job_kind",
            "repair_source_candidate_id",
            "repair_mode",
            "candidate_commit_hash",
            "candidate_branch_name",
            "candidate_published_at",
            "candidate_fate_label",
            "candidate_fate_reason",
            "has_evaluation_evidence",
            "agent_visible_evidence_count",
            "top_evaluation_diagnosis",
            "result_commit_hash",
            "ingestion_status",
            "ingestion_attempts",
            "ingestion_cell_index",
            "ingestion_delta",
            "last_error",
        ]
    }


def _render_detail_list_expander(label: str, raw_items: object) -> None:
    with st.expander(label, expanded=False):
        items = raw_items if isinstance(raw_items, list) else []
        if not items:
            st.write("None")
            return
        for item in items:
            st.write(f"- {item}")


def _render_evidence_sections(
    *,
    api_base_url: str,
    job_id: str,
    detail: dict[str, object],
) -> None:
    with st.expander("Evaluation Evidence", expanded=True):
        render_evaluation_evidence(
            api_base_url=api_base_url,
            artifacts=_list_value(detail, "evaluation_artifacts"),
            key_prefix=f"evidence_job_{job_id}",
            empty_message="No evaluation evidence available for this job.",
        )
    with st.expander("Agent Feedback Preview", expanded=False):
        render_agent_feedback_preview(_dict_value(detail, "evaluation_agent_feedback"))
    with st.expander("Worker Artifacts", expanded=False):
        render_artifact_downloads(
            api_base_url=api_base_url,
            artifacts=_dict_value(detail, "artifacts"),
            key_prefix=f"dl_job_{job_id}",
            empty_message="No artifacts available for this job.",
        )


def _list_value(data: dict[str, object], key: str) -> list[dict[str, object]]:
    value = data.get(key)
    return value if isinstance(value, list) else []


def _dict_value(data: dict[str, object], key: str) -> dict[str, object] | None:
    value = data.get(key)
    return value if isinstance(value, dict) else None


def _filter_jobs_df(df, *, fate_filter: str, evidence_filter: str):
    if df.empty:
        return df
    out = df
    if fate_filter != "all" and "candidate_fate_label" in out.columns:
        out = out[_normalized_fate_labels(out["candidate_fate_label"]) == fate_filter]
    if evidence_filter != "all":
        if evidence_filter == "has evidence" and "has_evaluation_evidence" in out.columns:
            out = out[out["has_evaluation_evidence"].fillna(False).astype(bool)]
        elif evidence_filter == "agent-visible" and "agent_visible_evidence_count" in out.columns:
            out = out[out["agent_visible_evidence_count"].fillna(0).astype(int) > 0]
        elif evidence_filter == "none" and "has_evaluation_evidence" in out.columns:
            out = out[~out["has_evaluation_evidence"].fillna(False).astype(bool)]
    return out


def _decorate_jobs_df(df):
    if df.empty:
        return df
    out = df.copy()
    if "candidate_fate_label" in out.columns:
        out["fate"] = _normalized_fate_labels(out["candidate_fate_label"])
    if "agent_visible_evidence_count" in out.columns:
        out["evidence"] = out["agent_visible_evidence_count"].fillna(0).astype(int).map(
            lambda count: f"agent-visible:{count}" if count else "none"
        )
    if "campaign_program_hash" in out.columns:
        out["program"] = out["campaign_program_hash"].fillna("").astype(str).str.slice(0, 12)
    return out


def _normalized_fate_labels(series):
    return series.fillna("").astype(str).str.strip().replace("", "unknown")
