"""Streamlit entry point for ML Knowledge Hub."""

from __future__ import annotations

import hmac
import os

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


def _configured_password() -> str | None:
    """Return APP_PASSWORD from Streamlit secrets or the environment, if set."""

    try:
        # Reading st.secrets also exports root-level secrets to os.environ,
        # which the backend services read for OpenAI and Neo4j credentials.
        password = st.secrets.get("APP_PASSWORD")
    except Exception:
        password = None
    return password or os.getenv("APP_PASSWORD")


def _require_password() -> None:
    """Block the app behind APP_PASSWORD when one is configured."""

    password = _configured_password()
    if not password or st.session_state.get("authenticated"):
        return

    st.title("ML Knowledge Hub")
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, password):
        st.session_state["authenticated"] = True
        st.rerun()
    elif entered:
        st.error("Incorrect password.")
    st.stop()


_require_password()

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

PAGES[selected_page]()
