"""Extract knowledge graph entities and relationships from documents."""

"""LLM-based knowledge-graph extraction."""

import json
import os

from openai import OpenAI

from ml_knowledge_hub.knowledge_graph.extraction_schema import (
    EntityType,
    KGEntity,
    KGExtractionResult,
    KGRelation,
    RelationType,
)


class KGExtractor:
    def __init__(self, model: str | None = None):
        self.client = OpenAI()
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def extract(
        self,
        text: str,
        project_id: str,
        asset_type: str,
        title: str,
    ) -> KGExtractionResult:
        allowed_entity_types = [
            item.value for item in EntityType
        ]

        allowed_relation_types = [
            item.value for item in RelationType
        ]

        prompt = f"""
            You are extracting structured knowledge for an enterprise ML knowledge graph.

            Use ONLY the supplied document content.

            Project ID:
            {project_id}

            Asset type:
            {asset_type}

            Title:
            {title}

            Allowed entity types:
            {allowed_entity_types}

            Allowed relation types:
            {allowed_relation_types}

            Rules:
            1. Extract only entities explicitly supported by the document.
            2. Do not invent missing models, datasets, metrics, repositories, experiments, or deployments.
            3. Prefer canonical short names when clearly available.
            4. Entity IDs must be lowercase snake_case and reasonably stable.
            5. Every relation source_id and target_id must refer to an extracted entity.
            6. Include short evidence text for each relation.
            7. If the document does not support a relation, do not create it.
            8. Treat simulated/template documents cautiously. Do not convert proposed or hypothetical statements into factual deployments or incidents.
            # --------------------------------------------------
            # Relationship specificity rules
            # --------------------------------------------------
            9. Use RELATED_TO only as a last resort when no more specific allowed
            relationship correctly represents the evidence.

            10. When the supplied project clearly owns, contains, or uses an extracted
                model, represent the relationship as:
                    PROJECT --USES_MODEL--> MODEL

            11. When a project clearly uses an extracted dataset, use:
                    PROJECT --USES_DATASET--> DATASET

            12. Do not create RELATED_TO when USES_MODEL, USES_DATASET, STORED_IN,
                TRAINED_ON, EVALUATED_ON, SOLVES, REPORTS, TESTS, or DEPLOYED_AS
                accurately represents the relationship.

            13. Do not infer relationships merely because two entities occur in the
                same document. A relationship must be supported by the document or
                supplied project metadata.
            14. If you extract a PROJECT entity representing the supplied project,
                its entity_id MUST be exactly the supplied Project ID shown above.
                Do not derive or rename that project identifier from the title.
            Return ONLY valid JSON in this shape:

            {{
            "entities": [
                {{
                "entity_id": "example_id",
                "name": "Example",
                "entity_type": "model",
                "description": "Optional description",
                "properties": {{}}
                }}
            ],
            "relations": [
                {{
                "source_id": "example_id",
                "relation_type": "TRAINED_ON",
                "target_id": "dataset_id",
                "evidence": "Short supporting evidence"
                }}
            ]
            }}

            Document content:
            {text}
            """.strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw_output = response.output_text.strip()

        data = json.loads(raw_output)

        return KGExtractionResult.model_validate(data)