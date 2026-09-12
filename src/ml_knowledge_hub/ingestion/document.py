"""Document models used by ingestion workflows."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    project_id: str
    asset_type: str
    title: str
    text: str

    source_url: str | None = None
    source_record: str | None = None
    license: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)