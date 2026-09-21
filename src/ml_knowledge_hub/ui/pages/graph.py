"""Knowledge graph relationship browser."""

from __future__ import annotations

import logging

import streamlit as st

from ml_knowledge_hub.ui.components import page_header, render_structured
from ml_knowledge_hub.ui.services import get_graph_service, get_registry


logger = logging.getLogger(__name__)


def _model_id(model: dict) -> str | None:
    return model.get("entity_id") or model.get("model_id") or model.get("id")


def render() -> None:
    page_header(
        "Knowledge Graph",
        "Explore structured relationships between projects, models, datasets, and metrics.",
        "Neo4j relationships",
    )

    try:
        projects = get_registry().list_projects()
    except Exception:
        st.error("The project registry could not be loaded.")
        return

    if not projects:
        st.info("No projects are available.")
        return

    selected = st.selectbox(
        "Select a project",
        projects,
        key="graph_project",
        format_func=lambda value: value.replace("_", " ").title(),
    )
    load_graph = st.button("Load relationships", type="primary")

    if load_graph:
        try:
            with st.spinner("Querying the knowledge graph…"):
                service = get_graph_service()
                models = service.get_project_models(selected)
                details = []
                for model in models:
                    model_id = _model_id(model)
                    details.append(
                        {
                            "model": model,
                            "datasets": service.get_model_datasets(model_id) if model_id else [],
                            "metrics": service.get_model_metrics(model_id) if model_id else [],
                        }
                    )
            st.session_state.graph_result = {
                "project_id": selected,
                "details": details,
            }
            st.session_state.graph_error = None
        except Exception:
            logger.exception("Knowledge graph query failed")
            st.session_state.graph_error = (
                "The knowledge graph could not be reached. Verify the Neo4j configuration and try again."
            )

    if st.session_state.get("graph_error"):
        st.error(st.session_state.graph_error)

    result = st.session_state.get("graph_result")
    if not result or result.get("project_id") != selected:
        st.info("Choose a project and load its relationships.")
        return

    details = result.get("details", [])
    st.subheader(selected.replace("_", " ").title())
    st.caption(f"{len(details)} connected model{'s' if len(details) != 1 else ''}")
    if not details:
        st.info("No model relationships were found for this project.")
        return

    for detail in details:
        model = detail["model"]
        name = model.get("name") or _model_id(model) or "Unnamed model"
        with st.container(border=True):
            st.markdown(f"### {name}")
            if model.get("description"):
                st.write(model["description"])
            model_id = _model_id(model)
            if model_id:
                st.caption(f"Entity ID · {model_id}")
            datasets_tab, metrics_tab, record_tab = st.tabs(
                ["Datasets", "Metrics", "Graph record"]
            )
            with datasets_tab:
                render_structured(detail.get("datasets", []))
            with metrics_tab:
                render_structured(detail.get("metrics", []))
            with record_tab:
                render_structured(model)
