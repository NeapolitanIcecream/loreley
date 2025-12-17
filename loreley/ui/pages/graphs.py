"""Graphs page."""

from __future__ import annotations

import streamlit as st

from loreley.ui.components.api import api_get_or_stop
from loreley.ui.state import API_BASE_URL_KEY, EXPERIMENT_ID_KEY


def render() -> None:
    st.title("Graphs")

    api_base_url = str(st.session_state.get(API_BASE_URL_KEY, "") or "")
    experiment_id = st.session_state.get(EXPERIMENT_ID_KEY)
    if not api_base_url:
        st.error("API base URL is not configured.")
        return
    if not experiment_id:
        st.warning("No experiment selected.")
        return

    mode = st.selectbox("Mode", ["parent_chain"], index=0)
    max_nodes = st.slider("Max nodes", min_value=50, max_value=2000, value=300, step=50)

    graph = api_get_or_stop(
        api_base_url,
        "/api/v1/graphs/commit_lineage",
        params={"experiment_id": experiment_id, "max_nodes": max_nodes, "mode": mode},
    ) or {}

    nodes = graph.get("nodes") if isinstance(graph, dict) else []
    edges = graph.get("edges") if isinstance(graph, dict) else []

    st.caption(f"nodes={len(nodes) if isinstance(nodes, list) else 0} edges={len(edges) if isinstance(edges, list) else 0}")

    # Simple plot: fitness vs time.
    try:
        import pandas as pd
        import plotly.express as px
    except Exception:
        st.subheader("Commit lineage (raw)")
        st.json(graph)
        return

    if isinstance(nodes, list) and nodes:
        df = pd.DataFrame([n for n in nodes if isinstance(n, dict)])
        if not df.empty and {"created_at", "fitness"} <= set(df.columns):
            df = df.copy()
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
            df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
            df = df.dropna(subset=["created_at", "fitness"])
            fig = px.scatter(df, x="created_at", y="fitness", color="island_id", title="Fitness vs time (loaded nodes)")
            st.plotly_chart(fig, width="stretch")

    st.subheader("Interactive network (PyVis)")
    render_pyvis = st.checkbox("Render interactive graph", value=True, help="May be slower for large graphs.")
    if render_pyvis and isinstance(nodes, list) and isinstance(edges, list) and nodes:
        try:
            import networkx as nx
            from pyvis.network import Network
            from streamlit.components.v1 import html as st_html
        except Exception as exc:  # pragma: no cover - optional dependency
            st.error(f"Missing graph dependencies: {exc}")
        else:
            # Build a NetworkX graph with attributes that pyvis understands.
            g = nx.DiGraph()

            island_ids = sorted({str(n.get("island_id") or "") for n in nodes if isinstance(n, dict)})
            island_ids = [i for i in island_ids if i]
            island_to_group = {island: idx + 1 for idx, island in enumerate(island_ids)}

            for n in nodes:
                if not isinstance(n, dict):
                    continue
                commit_hash = str(n.get("commit_hash") or "")
                if not commit_hash:
                    continue
                short = commit_hash[:8]
                island = str(n.get("island_id") or "")
                fitness = n.get("fitness")
                author = n.get("author")
                message = (n.get("message") or "")
                title = (
                    f"commit: {commit_hash}<br/>"
                    f"island: {island}<br/>"
                    f"fitness: {fitness}<br/>"
                    f"author: {author}<br/>"
                    f"message: {message}"
                )
                g.add_node(
                    commit_hash,
                    label=short,
                    title=title,
                    group=island_to_group.get(island, 0),
                    size=12,
                )

            for e in edges:
                if not isinstance(e, dict):
                    continue
                src = str(e.get("source") or "")
                dst = str(e.get("target") or "")
                if not src or not dst:
                    continue
                if src in g and dst in g:
                    g.add_edge(src, dst)

            nt = Network(height="700px", width="100%", directed=True, cdn_resources="in_line", select_menu=True)
            nt.from_nx(g)
            nt.toggle_physics(True)

            html_str: str
            if hasattr(nt, "generate_html"):
                html_str = nt.generate_html()  # type: ignore[attr-defined]
            else:
                # Fallback to writing and reading HTML.
                import tempfile
                import uuid
                from pathlib import Path

                tmp = Path(tempfile.gettempdir()) / f"loreley-graph-{uuid.uuid4().hex}.html"
                nt.show(str(tmp))
                html_str = tmp.read_text(encoding="utf-8", errors="replace")
                try:
                    tmp.unlink()
                except Exception:
                    pass

            st_html(html_str, height=740, scrolling=True)

    with st.expander("Raw graph JSON", expanded=False):
        st.json(graph)


