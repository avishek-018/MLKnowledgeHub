"""
Evaluate end-to-end retrieval and answer quality for ML Knowledge Hub.

This script checks:
1. Whether retrieved sources come from the expected project.
2. Whether expected artifact types are retrieved.
3. Whether required facts appear in the generated answer.
4. Whether the system returned any supporting sources.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from ml_knowledge_hub.registry.service import AssetRegistry
from ml_knowledge_hub.metadata.service import MetadataService
from ml_knowledge_hub.embeddings.embedder import Embedder
from ml_knowledge_hub.vectorstore.qdrant_store import QdrantStore
from ml_knowledge_hub.rag.generator import RAGGenerator
from ml_knowledge_hub.knowledge_graph.neo4j_store import Neo4jStore
from ml_knowledge_hub.knowledge_graph.query_service import GraphQueryService
from ml_knowledge_hub.knowledge_graph.entity_registry import EntityRegistry
from ml_knowledge_hub.assistant.service import KnowledgeAssistant


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BENCHMARK_PATH = Path(
    "data/evaluation/answer_retrieval_benchmark.json"
)

OUTPUT_PATH = Path(
    "data/evaluation/answer_retrieval_results.json"
)


# ------------------------------------------------------------------
# Build assistant
# ------------------------------------------------------------------
def build_assistant() -> KnowledgeAssistant:
    """
    Construct the same KnowledgeAssistant stack used by the application.
    """

    load_dotenv()

    registry = AssetRegistry(
        registry_path="data/raw/manifest.json"
    )

    metadata_service = MetadataService(
        registry
    )

    embedder = Embedder()

    vector_store = QdrantStore()

    rag_generator = RAGGenerator()

    neo4j_store = Neo4jStore()

    graph_service = GraphQueryService(
        store=neo4j_store
    )

    entity_registry = EntityRegistry(
        "data/knowledge_graph/entities.json"
    )

    return KnowledgeAssistant(
        metadata_service=metadata_service,
        vector_store=vector_store,
        embedder=embedder,
        rag_generator=rag_generator,
        graph_service=graph_service,
        entity_registry=entity_registry,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def normalize_text(value) -> str:
    """
    Convert answer content to lowercase text for simple fact matching.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.lower()

    return json.dumps(
        value,
        ensure_ascii=False
    ).lower()


def get_sources(result: dict) -> list[dict]:
    """
    Read sources from the assistant result.

    Some execution paths may not include a sources field.
    """

    sources = result.get("sources", [])

    if not isinstance(sources, list):
        return []

    return sources


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_case(case: dict, result: dict) -> dict:
    """
    Evaluate one benchmark query against the assistant result.
    """

    expected_projects = set(
        case.get("expected_project_ids", [])
    )

    expected_asset_types = set(
        case.get("expected_asset_types", [])
    )

    required_facts = [
        fact.lower()
        for fact in case.get("required_facts", [])
    ]

    sources = get_sources(result)

    retrieved_projects = {
        source.get("project_id")
        for source in sources
        if source.get("project_id")
    }

    retrieved_asset_types = {
        source.get("asset_type")
        for source in sources
        if source.get("asset_type")
    }

    answer_text = normalize_text(
        result.get("answer")
    )

    # --------------------------------------------------------------
    # Project retrieval check
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # Project retrieval check
    #
    # Textual retrieval paths expose project IDs through sources.
    #
    # Graph-only answers may be correct but may not include textual
    # sources or project IDs in the response. In that case, project
    # retrieval is treated as not applicable rather than incorrect.
    # --------------------------------------------------------------
    result_type = result.get("type")

    if expected_projects:
        if retrieved_projects:
            project_match = bool(
                expected_projects & retrieved_projects
            )
        elif result_type == "graph":
            project_match = None
        else:
            project_match = False
    else:
        project_match = True

    # --------------------------------------------------------------
    # Artifact type retrieval check
    # --------------------------------------------------------------
    if expected_asset_types:
        asset_type_match = bool(
            expected_asset_types & retrieved_asset_types
        )
    else:
        asset_type_match = True

    # --------------------------------------------------------------
    # Required answer fact check
    # --------------------------------------------------------------
    fact_results = {
        fact: fact in answer_text
        for fact in required_facts
    }

    if required_facts:
        facts_match = all(
            fact_results.values()
        )
    else:
        facts_match = True

    # --------------------------------------------------------------
    # Source coverage
    #
    # Graph-only answers may legitimately have no textual sources.
    # For now we still record this separately instead of forcing
    # every query to fail when sources are absent.
    # --------------------------------------------------------------
    has_sources = len(sources) > 0

    # Main end-to-end correctness criterion
    passed = (
        project_match is not False
        and asset_type_match
        and facts_match
    )

    return {
        "id": case["id"],
        "query": case["query"],

        "result_type": result.get("type"),

        "expected_project_ids": sorted(expected_projects),
        "retrieved_project_ids": sorted(retrieved_projects),
        "project_match": project_match,

        "expected_asset_types": sorted(expected_asset_types),
        "retrieved_asset_types": sorted(retrieved_asset_types),
        "asset_type_match": asset_type_match,

        "required_facts": required_facts,
        "fact_results": fact_results,
        "facts_match": facts_match,

        "has_sources": has_sources,
        "passed": passed,
        # "project_evaluable_queries": project_evaluable,
        "answer_preview": normalize_text(
            result.get("answer")
        )[:500],
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        benchmark = json.load(f)

    assistant = build_assistant()

    results = []

    for case in benchmark:
        print(
            f"\n[{case['id']}] {case['query']}"
        )

        try:
            assistant_result = assistant.ask(
                case["query"]
            )

            evaluation = evaluate_case(
                case,
                assistant_result
            )

        except Exception as exc:
            evaluation = {
                "id": case["id"],
                "query": case["query"],
                "error": type(exc).__name__,
                "message": str(exc),
                "passed": False,
                "has_sources": False,
            }

        results.append(evaluation)

        print(
            "PASS"
            if evaluation["passed"]
            else "FAIL"
        )

    # --------------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------------
    total = len(results)

    passed = sum(
        1
        for item in results
        if item.get("passed")
    )

    project_matches = sum(
        1
        for item in results
        if item.get("project_match") is True
    )
    project_evaluable = sum(
        1
        for item in results
        if item.get("project_match") is not None
    )
    asset_type_matches = sum(
        1
        for item in results
        if item.get("asset_type_match") is True
    )

    fact_matches = sum(
        1
        for item in results
        if item.get("facts_match") is True
    )

    source_coverage = sum(
        1
        for item in results
        if item.get("has_sources")
    )

    summary = {
        "total_queries": total,
        "passed_queries": passed,

        "overall_pass_rate": round(
            100 * passed / total,
            2
        ),
        "project_evaluable_queries": project_evaluable,
        "project_retrieval_accuracy": round(
            100 * project_matches / project_evaluable,
            2
        ) if project_evaluable else None,

        "asset_type_retrieval_accuracy": round(
            100 * asset_type_matches / total,
            2
        ),

        "required_fact_accuracy": round(
            100 * fact_matches / total,
            2
        ),

        "source_coverage": round(
            100 * source_coverage / total,
            2
        ),
        
    }

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n==============================")
    print("Evaluation Summary")
    print("==============================")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(
        f"\nSaved detailed results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()