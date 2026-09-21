"""Streamlit entry point for ML Knowledge Hub."""

from __future__ import annotations

import streamlit as st

from ml_knowledge_hub.ui.components import apply_styles
from ml_knowledge_hub.ui.pages import ask, evaluation, graph, projects


st.set_page_config(
    page_title="ML Knowledge Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

PAGES = {
    "Ask Knowledge Hub": ask.render,
    "Projects": projects.render,
    "Knowledge Graph": graph.render,
    "Evaluation": evaluation.render,
}

with st.sidebar:
    st.title("ML Knowledge Hub")
    st.caption("Agentic GraphRAG for ML project knowledge")
    st.divider()
    selected_page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.subheader("Configured components")
    st.caption("◦ Qdrant semantic index")
    st.caption("◦ Neo4j knowledge graph")
    st.caption("◦ OpenAI reasoning and generation")
    st.caption("◦ MCP interface")
    st.info("Run only one local process against the persistent Qdrant store.")

PAGES[selected_page]()
