"""Tests for knowledge graph schema."""

from ml_knowledge_hub.knowledge_graph.schema import MLAsset, AssetType


def test_create_model_asset():
    model = MLAsset(
        id="model_resnet50",
        name="ResNet50",
        asset_type=AssetType.MODEL,
        description="Residual convolutional neural network.",
    )

    assert model.name == "ResNet50"
    assert model.asset_type == AssetType.MODEL