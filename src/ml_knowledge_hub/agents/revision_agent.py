"""Agent responsible for revising generated results."""

"""Revision agent for improving query plans using evaluator feedback."""

import json
import os

from openai import OpenAI

from ml_knowledge_hub.agents.schemas import PlanEvaluation
from ml_knowledge_hub.query.planner_schema import QueryPlanLLM


class RevisionAgent:
    """
    LLM agent responsible for revising a planner-generated query plan
    using structured feedback from the Evaluator Agent.

    The revision agent does not answer the user's question.
    It produces a new QueryPlanLLM.
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        self.client = OpenAI()

        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def revise(
        self,
        query: str,
        original_plan: QueryPlanLLM,
        evaluation: PlanEvaluation,
    ) -> QueryPlanLLM:
        """
        Revise the original plan in light of evaluator feedback.
        """

        original_plan_json = json.dumps(
            original_plan.model_dump(
                mode="json"
            ),
            indent=2,
        )

        evaluation_json = json.dumps(
            evaluation.model_dump(
                mode="json"
            ),
            indent=2,
        )

        prompt = f"""
You are the Revision Agent for an enterprise machine-learning
knowledge-management system.

You MUST NOT answer the user's question.

Your task is to revise a proposed retrieval plan after receiving
feedback from an Evaluator Agent.

You are given:

1. the original user question
2. the original planner-generated plan
3. the evaluator's critique and structured suggestions

Your responsibility is to produce the final revised query plan.


IMPORTANT BEHAVIOR

1. Carefully consider the evaluator's feedback.

2. Do NOT blindly copy every evaluator suggestion.

3. Preserve parts of the original plan that are already correct.

4. Change only the fields that need improvement.

5. Prefer the most precise available system capability that can
   fully answer the user's question.

6. Do not invent operations, entity types, or asset types.

7. Do not answer the user's question.

8. Return only a valid structured query plan.


SYSTEM CAPABILITIES


RETRIEVAL MODES

metadata:
Use authoritative catalog metadata for counts, inventories,
lists, and catalog-level information.

graph:
Use the knowledge graph for explicit relationships between
known entities.

hybrid:
Use graph relationships together with supporting document
evidence for broader summaries or explanations about entities.

semantic:
Use semantic document retrieval for document-content questions,
evidence discovery, and broad searches.


AVAILABLE OPERATIONS

Metadata:
- count_projects
- list_projects
- count_model_cards

Graph:
- project_models
- model_datasets
- model_metrics

Hybrid:
- model_context
- project_context

Semantic:
- semantic_search


GRAPH OPERATION SEMANTICS

project_models:
Return models associated with or used by a specific project.

model_datasets:
Return datasets associated with a specific model, including
training or evaluation datasets.

model_metrics:
Return metrics reported for a specific model.


HYBRID OPERATION SEMANTICS

model_context:
Combine structured graph context about a model with supporting
document evidence.

project_context:
Combine structured graph context about a project with supporting
document evidence.


SUPPORTED ENTITY TYPES

- project
- model
- dataset
- metric
- repository
- task
- experiment
- deployment


SUPPORTED ASSET TYPES

- paper
- repository_readme
- model_card
- dataset_card
- dataset_metadata
- reproducibility_report
- deployment_notes
- postmortem
- evaluation_report
- experiment_report
- project_brief


ORIGINAL USER QUESTION

{query}


ORIGINAL PLAN

{original_plan_json}


EVALUATOR FEEDBACK

{evaluation_json}


REVISION RULES

1. If evaluation.valid is true, normally preserve the original plan.

2. If evaluation.valid is false, revise the plan where the
   evaluator identified a meaningful problem.

3. suggested_query_type and suggested_operation are recommendations,
   not commands.

4. Preserve entity_mention unless the original extraction is clearly
   inconsistent with the user query.

5. Preserve entity_type and target_entity_type when they are already
   correct.

6. Remove or change asset_types only when the evaluator identifies
   over-filtering, under-filtering, or a mismatch with the user's
   information need.

7. The final plan must be executable using the listed capabilities.

8. Do not return commentary about the revision.


OUTPUT CONTRACT

Return ONLY one valid JSON object.

The JSON keys MUST be exactly:

- query_type
- operation
- entity_mention
- entity_type
- target_entity_type
- asset_types

The required structure is:

{{
  "query_type": "metadata | semantic | graph | hybrid",
  "operation": "one of the supported operations",
  "entity_mention": "entity mentioned by the user or null",
  "entity_type": "project | model | dataset | metric | repository | task | experiment | deployment | null",
  "target_entity_type": "project | model | dataset | metric | repository | task | experiment | deployment | null",
  "asset_types": ["supported asset types"] or null
}}

Return no markdown.
Return no explanation.
Return no text before or after the JSON.
""".strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw_output = response.output_text.strip()

        data = json.loads(
            raw_output
        )

        # --------------------------------------------------
        # Defensive normalization in case the LLM uses an
        # alternate name for query_type.
        # --------------------------------------------------
        if (
            "query_type" not in data
            and "retrieval_mode" in data
        ):
            data["query_type"] = data.pop(
                "retrieval_mode"
            )

        # --------------------------------------------------
        # Validate the final revised plan.
        # --------------------------------------------------
        return QueryPlanLLM.model_validate(
            data
        )