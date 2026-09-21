"""Smoke-test knowledge graph query workflows."""


"""Test basic Neo4j knowledge-graph queries."""

from dotenv import load_dotenv

from ml_knowledge_hub.knowledge_graph.neo4j_store import (
    Neo4jStore,
)


def print_results(
    title: str,
    results: list[dict],
) -> None:
    """
    Pretty-print a list of Neo4j query results.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for item in results:
        print(item)


def main():
    # --------------------------------------------------
    # Load Neo4j credentials from .env.
    # --------------------------------------------------
    load_dotenv()

    store = Neo4jStore()

    try:
        # --------------------------------------------------
        # 1. Which models are used by the NYUAD project?
        # --------------------------------------------------
        models = store.get_models_for_project(
            "nyuad_ai_image_detector"
        )

        print_results(
            "Models for NYUAD project",
            models,
        )

        # --------------------------------------------------
        # 2. Which datasets are connected to its detector?
        #
        # Replace the model ID below if your canonical graph
        # uses a slightly different ID.
        # --------------------------------------------------
        datasets = store.get_datasets_for_model(
            "ai_generated_images_detector"
        )

        print_results(
            "Datasets for AI-generated image detector",
            datasets,
        )

        # --------------------------------------------------
        # 3. Which metrics does that model report?
        # --------------------------------------------------
        metrics = store.get_metrics_for_model(
            "ai_generated_images_detector"
        )

        print_results(
            "Metrics for AI-generated image detector",
            metrics,
        )

    finally:
        # Always close the driver.
        store.close()


if __name__ == "__main__":
    main()