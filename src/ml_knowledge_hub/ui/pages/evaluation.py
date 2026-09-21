"""Evaluation dashboard."""

from __future__ import annotations

import streamlit as st

from ml_knowledge_hub.ui.components import page_header
from ml_knowledge_hub.ui.services import load_evaluation_results


def _percent(value) -> str:
    return "—" if value is None else f"{float(value):.2f}%"


def render() -> None:
    page_header(
        "Evaluation",
        "Current benchmark results for agentic planning and end-to-end retrieval.",
        "Measured performance",
    )

    try:
        planning, retrieval = load_evaluation_results()
    except (OSError, ValueError):
        st.error("Evaluation result files could not be loaded.")
        return

    planning_summary = planning.get("summary", {})
    retrieval_summary = retrieval.get("summary", {})

    st.subheader("Agentic planning")
    st.caption(f"{planning_summary.get('total_queries', 0)} benchmark queries")
    columns = st.columns(4)
    metrics = [
        ("Routing accuracy", "final_routing_accuracy"),
        ("Execution-plan accuracy", "final_execution_plan_accuracy"),
        ("Full-plan accuracy", "final_full_plan_accuracy"),
        ("Revision success", "revision_success_rate"),
    ]
    for column, (label, key) in zip(columns, metrics):
        column.metric(label, _percent(planning_summary.get(key)))

    st.divider()
    st.subheader("End-to-end retrieval")
    st.caption(f"{retrieval_summary.get('total_queries', 0)} representative queries")
    first_row = st.columns(3)
    second_row = st.columns(2)
    retrieval_metrics = [
        ("Strict pass rate", "overall_pass_rate"),
        ("Project retrieval", "project_retrieval_accuracy"),
        ("Asset-type retrieval", "asset_type_retrieval_accuracy"),
        ("Required-fact accuracy", "required_fact_accuracy"),
        ("Source coverage", "source_coverage"),
    ]
    for column, (label, key) in zip(first_row + second_row, retrieval_metrics):
        column.metric(label, _percent(retrieval_summary.get(key)))

    st.info(
        "Current failure analysis indicates that project/entity resolution and "
        "project-filter propagation are the main remaining weaknesses."
    )

    with st.expander("Planning benchmark details"):
        st.dataframe(planning.get("results", []), width="stretch", hide_index=True)
    with st.expander("Retrieval benchmark details"):
        rows = []
        for item in retrieval.get("results", []):
            rows.append(
                {
                    "ID": item.get("id"),
                    "Query": item.get("query"),
                    "Result type": item.get("result_type"),
                    "Project match": item.get("project_match"),
                    "Asset type match": item.get("asset_type_match"),
                    "Facts match": item.get("facts_match"),
                    "Sources": item.get("has_sources"),
                    "Passed": item.get("passed"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
