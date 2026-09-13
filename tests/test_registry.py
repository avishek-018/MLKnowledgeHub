"""Tests for registry services."""

import json

from ml_knowledge_hub.registry.service import AssetRegistry


def test_registry_add_and_list_asset(tmp_path):
    registry_path = tmp_path / "registry.json"

    registry = AssetRegistry(registry_path)

    registry.add_asset(
        {
            "project_id": "fraud_detection",
            "asset_type": "model_card",
            "title": "Fraud Detection Model",
        }
    )

    assets = registry.list_assets()

    assert len(assets) == 1

    assert assets[0]["project_id"] == "fraud_detection"


def test_registry_list_projects(tmp_path):
    registry_path = tmp_path / "registry.json"

    registry = AssetRegistry(registry_path)

    registry.add_asset(
        {
            "project_id": "project_a",
            "asset_type": "paper",
            "title": "Paper A",
        }
    )

    registry.add_asset(
        {
            "project_id": "project_b",
            "asset_type": "model_card",
            "title": "Model B",
        }
    )

    registry.add_asset(
        {
            "project_id": "project_a",
            "asset_type": "repository_readme",
            "title": "README A",
        }
    )

    assert registry.list_projects() == [
        "project_a",
        "project_b",
    ]

    assert registry.count_projects() == 2