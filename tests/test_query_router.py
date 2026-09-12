"""Tests for query routing utilities."""

from ml_knowledge_hub.query.router import route_query


def test_count_projects_route():
    plan = route_query(
        "How many projects do we have?"
    )

    assert plan.query_type == "metadata"
    assert plan.operation == "count_projects"


def test_list_projects_route():
    plan = route_query(
        "Please list all projects"
    )

    assert plan.query_type == "metadata"
    assert plan.operation == "list_projects"


def test_model_card_filter():
    plan = route_query(
        "Which model cards describe AI image detectors?"
    )

    assert plan.query_type == "filtered_semantic"
    assert plan.asset_types == ["model_card"]


def test_general_query():
    plan = route_query(
        "Which projects use diffusion-based methods?"
    )

    assert plan.query_type == "semantic"

def test_deployment_query_filter():
    plan = route_query(
        "What deployment issues were reported?"
    )

    assert plan.query_type == "filtered_semantic"

    assert "deployment_notes" in plan.asset_types
    assert "postmortem" in plan.asset_types