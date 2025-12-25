"""Jobs page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.aggrid import render_table, selected_rows
from loreley.ui.components.api import api_get_bytes_or_stop, api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY, EXPERIMENT_ID_KEY


def render() -> None:
    st.title("Jobs")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    experiment_id = st.session_state.get(EXPERIMENT_ID_KEY)
    if not api_base_url:
        st.error("API base URL is not configured.")
        return
    if not experiment_id:
        st.warning("No experiment selected.")
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        st.error(f"Missing pandas dependency: {exc}")
        return

    rows = api_get_or_stop(api_base_url, "/api/v1/jobs", params={"experiment_id": experiment_id, "limit": 2000}) or []
    df = pd.DataFrame(rows)

    st.subheader("Jobs")
    if df.empty:
        st.info("No jobs found.")
        return

    # Filters
    statuses: list[str] = []
    if "status" in df.columns:
        try:
            values = df["status"].tolist()
        except Exception:
            values = []
        normalized: set[str] = set()
        for v in values:
            try:
                is_missing = v is None or pd.isna(v)
            except Exception:
                is_missing = v is None
            if is_missing:
                continue
            s = str(v).strip()
            if s:
                normalized.add(s)
        statuses = sorted(normalized)
    selected_statuses = st.multiselect("Status filter", options=statuses, default=statuses)
    if selected_statuses and "status" in df.columns:
        df = df[df["status"].isin(selected_statuses)]

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
    if isinstance(detail, dict):
        top = {
            k: detail.get(k)
            for k in [
                "status",
                "priority",
                "island_id",
                "scheduled_at",
                "started_at",
                "completed_at",
                "is_seed_job",
                "result_commit_hash",
                "ingestion_status",
                "ingestion_attempts",
                "ingestion_cell_index",
                "ingestion_delta",
                "last_error",
            ]
        }
        st.write(top)
        st.text_area("Goal", value=str(detail.get("goal") or ""), height=100, disabled=True)
        if detail.get("iteration_hint"):
            st.caption(f"Iteration hint: {detail.get('iteration_hint')}")

        with st.expander("Constraints", expanded=False):
            items = detail.get("constraints") if isinstance(detail.get("constraints"), list) else []
            if items:
                for item in items:
                    st.write(f"- {item}")
            else:
                st.write("None")

        with st.expander("Acceptance criteria", expanded=False):
            items = detail.get("acceptance_criteria") if isinstance(detail.get("acceptance_criteria"), list) else []
            if items:
                for item in items:
                    st.write(f"- {item}")
            else:
                st.write("None")

        with st.expander("Inspirations", expanded=False):
            items = detail.get("inspiration_commit_hashes") if isinstance(detail.get("inspiration_commit_hashes"), list) else []
            if items:
                for item in items:
                    st.write(f"- {item}")
            else:
                st.write("None")

        artifacts = detail.get("artifacts") if isinstance(detail.get("artifacts"), dict) else {}
        with st.expander("Artifacts", expanded=False):
            if not artifacts:
                st.write("No artifacts available for this job.")
            else:
                label_map = {
                    "planning_prompt_url": "Planning prompt",
                    "planning_raw_output_url": "Planning raw output",
                    "planning_plan_json_url": "Planning plan JSON",
                    "coding_prompt_url": "Coding prompt",
                    "coding_raw_output_url": "Coding raw output",
                    "coding_execution_json_url": "Coding execution JSON",
                    "evaluation_json_url": "Evaluation JSON",
                    "evaluation_logs_url": "Evaluation logs",
                }
                for key, label in label_map.items():
                    url = artifacts.get(key)
                    if not url:
                        continue
                    data, content_type = api_get_bytes_or_stop(api_base_url, str(url))
                    filename = str(url).rstrip("/").split("/")[-1] or f"{key}.txt"
                    st.download_button(
                        f"Download: {label}",
                        data=data,
                        file_name=filename,
                        mime=content_type or "application/octet-stream",
                        key=f"dl_job_{job_id}_{key}",
                    )
    else:
        st.json(detail)


