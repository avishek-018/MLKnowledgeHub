"""Build the knowledge graph from the configured corpus."""


"""Build the knowledge graph from the full ML Knowledge Hub corpus."""

from dotenv import load_dotenv

from ml_knowledge_hub.ingestion.corpus_loader import (
    load_corpus,
)
from ml_knowledge_hub.knowledge_graph.entity_registry import (
    EntityRegistry,
)
from ml_knowledge_hub.knowledge_graph.entity_resolver import (
    EntityResolver,
)
from ml_knowledge_hub.knowledge_graph.extractor import (
    KGExtractor,
)
from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)
from ml_knowledge_hub.knowledge_graph.processing_pipeline import (
    KGProcessingPipeline,
)
from ml_knowledge_hub.knowledge_graph.resolution_pipeline import (
    EntityResolutionPipeline,
)

import argparse
import json
from pathlib import Path

# --------------------------------------------------
# Knowledge-graph build checkpoint utilities
# --------------------------------------------------

CHECKPOINT_PATH = Path(
    "data/knowledge_graph/build_checkpoint.json"
)


def load_checkpoint() -> set[str]:
    """
    Load document IDs that have already been processed
    successfully.

    Returning a set makes membership checks fast.
    """

    if not CHECKPOINT_PATH.exists():
        return set()

    with CHECKPOINT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return set(
        data.get("completed_document_ids", [])
    )


def save_checkpoint(
    completed_document_ids: set[str],
) -> None:
    """
    Persist successfully processed document IDs.

    The checkpoint is written after each successful
    Neo4j write so an interrupted build can resume.
    """

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "completed_document_ids": sorted(
            completed_document_ids
        )
    }

    with CHECKPOINT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

def main():

    # --------------------------------------------------
    # Command-line options
    # --------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Build the ML Knowledge Hub knowledge graph."
        )
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help=(
            "1-based document index to start from. "
            "Useful when recovering from an earlier run."
        ),
    )

    parser.add_argument(
        "--ignore-checkpoint",
        action="store_true",
        help=(
            "Process documents even if they are already "
            "listed in the checkpoint."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load credentials/configuration from .env.
    # --------------------------------------------------
    load_dotenv()

    # --------------------------------------------------
    # 1. Load all normalized corpus documents.
    #
    # This should load the same 24 documents that are
    # currently used by the RAG/indexing pipeline.
    # --------------------------------------------------

    # --------------------------------------------------
    # Load all corpus documents described by the manifest.
    #
    # Manifest paths begin with "corpus/", and load_corpus()
    # strips that prefix before joining with this root.
    # --------------------------------------------------
    documents = load_corpus(
        manifest_path="data/raw/manifest.json",
        corpus_root="data/raw/corpus",
        )



    print(
        f"Loaded documents: {len(documents)}"
    )


        # --------------------------------------------------
    # Load resume checkpoint.
    # --------------------------------------------------
    completed_document_ids = load_checkpoint()

    print(
        f"Checkpoint contains "
        f"{len(completed_document_ids)} completed documents."
    )

    # --------------------------------------------------
    # 2. Initialize KG components.
    # --------------------------------------------------
    extractor = KGExtractor()

    resolver = EntityResolver()

    entity_registry = EntityRegistry(
        "data/knowledge_graph/entities.json"
    )

    resolution_pipeline = EntityResolutionPipeline(
        registry=entity_registry,
        resolver=resolver,
    )

    processing_pipeline = KGProcessingPipeline(
        resolution_pipeline=resolution_pipeline,
        registry=entity_registry,
    )

    neo4j_store = Neo4jStore()

    # --------------------------------------------------
    # Counters for a final summary.
    # --------------------------------------------------
    processed_documents = 0
    skipped_documents = 0

    extracted_entities = 0
    extracted_relations = 0

    resolved_entities = 0
    resolved_relations = 0

    unresolved_entities = 0

    try:
        # --------------------------------------------------
        # 3. Process each corpus document independently.
        # --------------------------------------------------
        for index, document in enumerate(
            documents,
            start=1,
        ):

            # --------------------------------------------------
            # Skip documents before the requested starting index.
            #
            # For our current recovery run we will use:
            #   --start-index 7
            # --------------------------------------------------
            if index < args.start_index:
                continue


            # --------------------------------------------------
            # Skip documents already completed in an earlier run.
            # --------------------------------------------------
            if (
                not args.ignore_checkpoint
                and document.document_id
                in completed_document_ids
            ):
                print()
                print("=" * 70)

                print(
                    f"[{index}/{len(documents)}] "
                    f"{document.title}"
                )

                print(
                    "Skipped: already completed "
                    "according to checkpoint."
                )

                continue

            
            print()
            print("=" * 70)

            print(
                f"[{index}/{len(documents)}] "
                f"{document.title}"
            )

            print(
                f"Project: {document.project_id}"
            )

            print(
                f"Asset type: {document.asset_type}"
            )

            # --------------------------------------------------
            # Skip documents that contain no useful text.
            # --------------------------------------------------
            if not document.text.strip():
                print(
                    "Skipped: document contains no text."
                )

                skipped_documents += 1
                continue

            # --------------------------------------------------
            # 4. Extract raw KG entities and relations.
            # --------------------------------------------------
            raw_result = extractor.extract(
                text=document.text,
                project_id=document.project_id,
                asset_type=document.asset_type,
                title=document.title,
            )

            print(
                f"Raw entities: "
                f"{len(raw_result.entities)}"
            )

            print(
                f"Raw relations: "
                f"{len(raw_result.relations)}"
            )

            extracted_entities += len(
                raw_result.entities
            )

            extracted_relations += len(
                raw_result.relations
            )

            # --------------------------------------------------
            # 5. Resolve entity identities and rewrite
            #    relationships using canonical IDs.
            # --------------------------------------------------
            resolved_result = (
                processing_pipeline.process(
                    raw_result
                )
            )

            print(
                f"Canonical entities: "
                f"{len(resolved_result.entities)}"
            )

            print(
                f"Canonical relations: "
                f"{len(resolved_result.relations)}"
            )

            print(
                f"Unresolved entities: "
                f"{len(resolved_result.unresolved_entity_ids)}"
            )

            resolved_entities += len(
                resolved_result.entities
            )

            resolved_relations += len(
                resolved_result.relations
            )

            unresolved_entities += len(
                resolved_result.unresolved_entity_ids
            )

            # --------------------------------------------------
            # 6. Upsert the resolved graph into Neo4j.
            #
            # Existing canonical nodes/relations are merged,
            # so re-running the script should not duplicate them.
            # --------------------------------------------------
            neo4j_store.write_result(
                resolved_result
            )

            # --------------------------------------------------
            # Mark this document as complete only AFTER the
            # resolved KG has been successfully written to Neo4j.
            #
            # If extraction/resolution/write fails before here,
            # the document remains uncompleted and will be retried.
            # --------------------------------------------------
            completed_document_ids.add(
                document.document_id
            )

            save_checkpoint(
                completed_document_ids
            )

            processed_documents += 1

            print(
                "Neo4j write complete."
            )
            print(
                "Checkpoint updated."
            )

    finally:
        # --------------------------------------------------
        # Always close the Neo4j connection.
        # --------------------------------------------------
        neo4j_store.close()

    # --------------------------------------------------
    # 7. Print final build summary.
    # --------------------------------------------------
    print()
    print("=" * 70)
    print("KNOWLEDGE GRAPH BUILD SUMMARY")
    print("=" * 70)

    print(
        f"Documents processed: "
        f"{processed_documents}"
    )

    print(
        f"Documents skipped: "
        f"{skipped_documents}"
    )

    print(
        f"Raw entities extracted: "
        f"{extracted_entities}"
    )

    print(
        f"Raw relations extracted: "
        f"{extracted_relations}"
    )

    print(
        f"Resolved entity appearances: "
        f"{resolved_entities}"
    )

    print(
        f"Resolved relation appearances: "
        f"{resolved_relations}"
    )

    print(
        f"Unresolved entity appearances: "
        f"{unresolved_entities}"
    )

    print(
        f"Canonical registry size: "
        f"{len(entity_registry.list_entities())}"
    )




if __name__ == "__main__":
    main()
