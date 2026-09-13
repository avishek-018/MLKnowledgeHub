import argparse
from pathlib import Path

from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.ingestion.document import Document
from ml_knowledge_hub.ingestion.parser import extract_text
from ml_knowledge_hub.ingestion.chunker import chunk_document
from ml_knowledge_hub.registry.service import AssetRegistry
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore


def main():
    parser = argparse.ArgumentParser(
        description="Register and index a new ML Knowledge Hub asset."
    )

    parser.add_argument("--project-id", required=True)
    parser.add_argument("--asset-type", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--file", required=True)

    parser.add_argument("--source-url")
    parser.add_argument("--source-record")
    parser.add_argument("--license", default="NOASSERTION")

    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Asset file not found: {file_path}"
        )

    # --------------------------------------------------
    # 1. Register asset
    # --------------------------------------------------
    registry = AssetRegistry(
        "data/raw/manifest.json"
    )

    record = {
        "project_id": args.project_id,
        "asset_type": args.asset_type,
        "title": args.title,
        "local_path": str(file_path),
        "source_url": args.source_url,
        "source_record": args.source_record,
        "license": args.license,
        "status": "registered",
    }

    registry.add_asset(record)

    print("Asset registered.")

    # --------------------------------------------------
    # 2. Extract text
    # --------------------------------------------------
    text = extract_text(file_path)

    print(
        f"Extracted characters: {len(text)}"
    )

    # --------------------------------------------------
    # 3. Create normalized document
    # --------------------------------------------------
    document_id = (
        f"{args.project_id}__"
        f"{args.asset_type}__"
        f"{file_path.stem}"
    )

    document = Document(
        document_id=document_id,
        project_id=args.project_id,
        asset_type=args.asset_type,
        title=args.title,
        text=text,
        source_url=args.source_url,
        source_record=args.source_record,
        license=args.license,
    )

    # --------------------------------------------------
    # 4. Chunk document
    # --------------------------------------------------
    chunks = chunk_document(document)

    print(
        f"Generated chunks: {len(chunks)}"
    )

    # --------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------
    embedder = Embedder()

    embeddings = embedder.encode(
        [chunk.text for chunk in chunks]
    )

    # --------------------------------------------------
    # 6. Store in persistent Qdrant
    # --------------------------------------------------
    store = QdrantStore()

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        f"Indexed chunks: {len(chunks)}"
    )

    print("\nAsset ingestion complete.")
    print(f"Project: {args.project_id}")
    print(f"Type: {args.asset_type}")
    print(f"Title: {args.title}")
    print(f"Document ID: {document_id}")


if __name__ == "__main__":
    main()