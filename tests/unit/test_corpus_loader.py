"""Smoke-test entry point for corpus loading workflows."""


from pathlib import Path

from ml_knowledge_hub.ingestion.corpus_loader import (
    load_corpus,
)


manifest_path = Path("data/raw/manifest.json")
corpus_root = Path("data/raw/corpus")

documents = load_corpus(
    manifest_path=manifest_path,
    corpus_root=corpus_root,
)

print(f"\nLoaded documents: {len(documents)}")

print("\nProjects:")

projects = sorted(
    {document.project_id for document in documents}
)

for project in projects:
    print(f"- {project}")

print(f"\nUnique projects: {len(projects)}")

print("\nAsset type counts:")

asset_types = {}

for document in documents:
    asset_types[document.asset_type] = (
        asset_types.get(document.asset_type, 0) + 1
    )

for asset_type, count in sorted(
    asset_types.items()
):
    print(f"{asset_type}: {count}")