from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    PAPER = "paper"
    MODEL = "model"
    DATASET = "dataset"
    TASK = "task"
    METRIC = "metric"
    REPOSITORY = "repository"


class MLAsset(BaseModel):
    id: str
    name: str
    asset_type: AssetType
    description: Optional[str] = None
    source: Optional[str] = None
    metadata: dict = Field(default_factory=dict)