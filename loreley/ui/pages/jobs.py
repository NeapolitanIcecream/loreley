"""Jobs page."""

from __future__ import annotations

import streamlit as st

from loreley.db.models import JobStatus
from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import (
    api_get_or_stop,
    api_get_page_or_stop,
    render_agent_feedback_preview,
    render_artifact_downloads,
    render_evaluation_evidence,
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

    page_size = st.selectbox("Page size", [50, 100, 200, 500], index=1)
    status_options = ["all", *(status.value for status in JobStatus)]
    selected_status = st.selectbox("Status filter", options=status_options, index=0)

    params: dict[str, object] = {"limit": page_size}
    if selected_status != "all":
        params["status"] = selected_status

    signature = pager_signature((api_base_url, selected_status, page_size))
    state = normalize_cursor_pager(
        signature=signature,
        stored_signature=st.session_state.get(_JOBS_CURSOR_SIGNATURE_KEY),
        cursors=st.session_state.get(_JOBS_CURSOR_KEY),
        index=st.session_state.get(_JOBS_CURSOR_INDEX_KEY),
    )
    st.session_state[_JOBS_CURSOR_SIGNATURE_KEY] = state.signature
    st.session_state[_JOBS_CURSOR_KEY] = list(state.cursors)
    st.session_state[_JOBS_CURSOR_INDEX_KEY] = state.index

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

    st.subheader("Jobs")
    if df.empty:
        st.info("No jobs found.")
        return

    st.caption(f"page={state.index + 1} items={len(df)}")
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

    grid = render_table(df, key="jobs_grid", selection="single")
    sel = selected_rows(grid)

    st.divider()
    if not sel:
        st.info("Select a job to see details.")
        return

    job_id = sel[0].get("id")
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
            "candidate_commit_hash",
            "candidate_branch_name",
            "candidate_published_at",
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
