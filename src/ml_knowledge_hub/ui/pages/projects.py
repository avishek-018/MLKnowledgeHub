"""Registered projects browser."""

from __future__ import annotations

import streamlit as st

from ml_knowledge_hub.ui.components import page_header
from ml_knowledge_hub.ui.services import get_registry


def render() -> None:
    page_header(
        "Projects",
        "Browse registered ML projects and the artifacts available for each one.",
        "Asset registry",
    )

    try:
        registry = get_registry()
        projects = registry.list_projects()
    except Exception:
        st.error("The project registry could not be loaded.")
        return

    total_assets = len(registry.list_assets())
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Projects", len(projects))
    metric_b.metric("Registered assets", total_assets)

    if not projects:
        st.info("No projects are currently registered.")
        return

    selected = st.selectbox(
        "Select a project",
        projects,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    assets = registry.get_assets_for_project(selected)

    st.subheader(selected.replace("_", " ").title())
    st.caption(f"{len(assets)} registered artifact{'s' if len(assets) != 1 else ''}")

    for asset in assets:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{asset.get('title') or 'Untitled artifact'}**")
            right.markdown(f"`{asset.get('asset_type', 'unknown')}`")
            if asset.get("status"):
                st.caption(f"Status · {asset['status']}")
            if asset.get("source_url"):
                st.link_button("Open source", asset["source_url"])
