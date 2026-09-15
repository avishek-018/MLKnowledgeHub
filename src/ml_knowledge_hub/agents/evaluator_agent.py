"""Agent responsible for evaluating generated results."""

"""Evaluator agent for critiquing planner-generated query plans."""

import json
import os

from openai import OpenAI

from ml_knowledge_hub.agents.schemas import PlanEvaluation
from ml_knowledge_hub.query.planner_schema import QueryPlanLLM


class EvaluatorAgent:
    """
    LLM agent responsible for evaluating a proposed query plan.

    The evaluator does not answer the user's question.
    It critiques the planner's plan and suggests improvements
    when the plan does not match the system's capabilities.
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

    def evaluate(
        self,
        query: str,
        plan: QueryPlanLLM,
    ) -> PlanEvaluation:
        """
        Evaluate whether the proposed plan is suitable.

        Returns structured feedback only.
        """

        plan_json = json.dumps(
            plan.model_dump(
                mode="json"
            ),
            indent=2,
        )

        prompt = f"""
You are the Evaluator Agent for an enterprise machine-learning
knowledge-management system.

You MUST NOT answer the user's question.

Your task is to critically evaluate the Planner Agent's proposed
query plan.

You are given:

1. the original user question
2. the planner's proposed structured plan
3. the capabilities available in the system

Determine whether the proposed plan is appropriate and executable.

If the plan is correct, mark it valid.

If the plan is not correct, explain the problem and provide
structured suggestions for how the Planner Agent should revise it.

You do NOT directly rewrite or execute the plan.


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
evidence for summaries, explanations, or broader entity context.

semantic:
Use semantic document retrieval for evidence discovery,
qualitative questions, broad searches, and document-content
questions.


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
Returns models associated with or used by a specific project.

model_datasets:
Returns datasets associated with a specific model, including
training or evaluation datasets.

model_metrics:
Returns metrics reported for a specific model.


HYBRID OPERATION SEMANTICS

model_context:
Combines structured graph context about a model with supporting
document evidence.

project_context:
Combines structured graph context about a project with supporting
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


EVALUATION PRINCIPLES

1. Evaluate the plan based on the user's information need,
   not exact keywords.

2. Prefer the most precise available system capability that
   can fully answer the question.

3. A direct relationship question should use graph when an
   available graph operation represents that relationship.

4. Do not accept semantic search when an available graph
   operation directly answers the requested relationship.

5. Hybrid should be used when the user wants broader context
   or explanation about a specific project or model.

6. Semantic retrieval is appropriate when the answer primarily
   depends on documents or broad evidence discovery.

7. Metadata should be used for authoritative counts and lists
   when an available metadata operation directly answers the
   question.

8. Check that entity_type represents the entity the question
   starts from.

9. Check that target_entity_type represents the type of entity
   the user wants returned.

10. Check that the selected operation is consistent with both
    entity_type and target_entity_type.

11. asset_types should only restrict retrieval when the user's
    information need clearly benefits from a specific artifact
    category.

12. Do not suggest unnecessary asset filters.

13. Do not mark a plan invalid merely because another route
    could also work. Mark it invalid only when there is a
    meaningful planning problem.

14. Do not answer the user.

15. Do not invent system capabilities that are not listed above.
16. Reject a hybrid plan when the user's primary information need is
    clearly the content of a specific artifact type and semantic retrieval
    with that asset filter would answer the question directly.

17. A model or project mention does not automatically justify hybrid
    retrieval.

18. Broad entity-summary questions belong to hybrid retrieval.

19. Narrow evidence questions such as reproduction evidence, experiment
    findings, evaluation conclusions, implementation guidance, or
    incident details should generally use semantic retrieval when the
    relevant artifact type is known.

20. When evaluating a hybrid plan with asset_types set, check whether the
    current hybrid execution path actually needs graph context. If not,
    prefer semantic retrieval with the same asset filter.

ORIGINAL USER QUESTION

{query}


PROPOSED PLAN

{plan_json}


OUTPUT CONTRACT

Return ONLY one valid JSON object with exactly these keys:

- valid
- feedback
- issues
- suggested_query_type
- suggested_operation
- suggested_entity_type
- suggested_target_entity_type
- suggested_asset_types

Use null for any suggestion that is not necessary.

The required structure is:

{{
  "valid": true,
  "feedback": "short evaluation",
  "issues": [],
  "suggested_query_type": null,
  "suggested_operation": null,
  "suggested_entity_type": null,
  "suggested_target_entity_type": null,
  "suggested_asset_types": null
}}

If the plan is invalid, set valid to false and provide only the
changes that are actually needed.

Return no markdown.
Return no explanation outside the JSON.
""".strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw_output = response.output_text.strip()

        data = json.loads(
            raw_output
        )

        return PlanEvaluation.model_validate(
            data
        )