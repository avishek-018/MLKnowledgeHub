"""Corpus loading utilities for ingestion workflows."""

import json
from pathlib import Path

from ml_knowledge_hub.ingestion.document import Document
from ml_knowledge_hub.ingestion.parser import extract_text


def load_corpus(
    manifest_path: str | Path,
    corpus_root: str | Path,
) -> list[Document]:
    manifest_path = Path(manifest_path)
    corpus_root = Path(corpus_root)

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    documents = []

    for index, record in enumerate(
        manifest["records"]
    ):
        relative_path = Path(record["local_path"])

        # Manifest paths start with "corpus/"
        # while corpus_root points directly to data/raw/corpus.
        if relative_path.parts[0] == "corpus":
            relative_path = Path(*relative_path.parts[1:])

        full_path = corpus_root / relative_path

        if not full_path.exists():
            print(
                f"Warning: skipping missing file: "
                f"{full_path}"
            )
            continue

        text = extract_text(full_path)

        document_id = (
            f"{record['project_id']}__"
            f"{record['asset_type']}__"
            f"{index:03d}"
        )

        known_fields = {
            "project_id",
            "asset_type",
            "title",
            "local_path",
            "source_url",
            "source_record",
            "license",
        }

        extra_metadata = {
            key: value
            for key, value in record.items()
            if key not in known_fields
        }

        document = Document(
            document_id=document_id,
            project_id=record["project_id"],
            asset_type=record["asset_type"],
            title=record["title"],
            text=text,
            source_url=record.get("source_url"),
            source_record=record.get("source_record"),
            license=record.get("license"),
            metadata=extra_metadata,
        )

        documents.append(document)

    return documents