"""Agent responsible for planning knowledge hub tasks."""

"""Planner agent for converting user questions into structured query plans."""

import json
import os

from openai import OpenAI

from ml_knowledge_hub.query.planner_schema import QueryPlanLLM


class PlannerAgent:
    """
    LLM agent responsible for producing an initial structured query plan.

    The planner does not answer the user's question.
    It only decides how the ML Knowledge Hub should retrieve the answer.
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

    def plan(
        self,
        query: str,
    ) -> QueryPlanLLM:
        """
        Convert a natural-language question into a typed query plan.
        """

        prompt = f"""
You are the Planner Agent for an enterprise machine-learning
knowledge-management system.

You MUST NOT answer the user's question.

Your responsibility is to produce a structured retrieval plan
describing how the system should answer the question.


RETRIEVAL MODES

metadata:
Use authoritative catalog metadata when the question asks for
counts, inventories, lists, or catalog-level information.

graph:
Use the knowledge graph when the answer primarily depends on
explicit relationships between known entities.

hybrid:
Use both knowledge-graph structure and document evidence when the
user asks for an explanation, summary, or broader context about
a specific entity.

semantic:
Use semantic document retrieval when the answer depends primarily
on document content, evidence discovery, qualitative information,
or broad search.


ROUTING PRIORITY

Choose the most structured retrieval method that can fully answer
the user's information need.

1. metadata
   Use when an authoritative metadata operation directly answers
   the question.

2. graph
   Use when an available graph operation directly represents the
   relationship requested by the user.

3. hybrid
   Use when the user wants broader context or explanation about a
   specific entity and both graph structure and document evidence
   are useful.

4. semantic
   Use when metadata or graph operations cannot directly answer
   the question, or when the answer primarily depends on documents.

Do not use semantic retrieval when an available graph operation
directly represents the requested relationship.


AVAILABLE OPERATIONS

Metadata operations:
- count_projects
- list_projects
- count_model_cards

Graph operations:
- project_models
- model_datasets
- model_metrics

Hybrid operations:
- model_context
- project_context

Semantic operations:
- semantic_search


GRAPH OPERATION SEMANTICS

project_models:
Return models associated with or used by a specific project.

model_datasets:
Return datasets used to train, evaluate, or otherwise relate to a
specific model.

model_metrics:
Return metrics reported for a specific model.


HYBRID OPERATION SEMANTICS

model_context:
Retrieve structured graph relationships for a model and combine
them with supporting document evidence.

project_context:
Retrieve structured graph relationships for a project and combine
them with supporting document evidence.


AVAILABLE ASSET TYPES

paper:
Academic or research paper.

repository_readme:
Repository documentation describing software, implementation,
installation, usage, methods, or setup.

model_card:
Documentation describing a machine-learning model, including
intended use, evaluation, limitations, or metadata.

dataset_card:
Documentation describing a dataset and its intended use.

dataset_metadata:
Structured metadata describing a dataset.

reproducibility_report:
Documentation or evidence describing attempts to reproduce
research or experimental results.

deployment_notes:
Operational documentation describing production configuration,
deployment requirements, safeguards, rollback procedures, or
runtime behavior.

postmortem:
Documentation analyzing an incident, failure, or operational
problem after it occurred.

evaluation_report:
Documentation describing model or system evaluation results.

experiment_report:
Documentation describing experiments, configurations, or results.

project_brief:
High-level documentation describing project scope, objectives,
status, or goals.


ENTITY TYPES

Supported entity types are:

- project
- model
- dataset
- metric
- repository
- task
- experiment
- deployment


PLANNING RULES

1. Do not answer the user's question.

2. Select the retrieval mode based on the information required,
   not on exact phrases or keywords.

3. Prefer graph when an available graph operation directly matches
   the relationship the user is asking about.

4. Prefer hybrid when the user wants an overview, explanation,
   summary, or context about a known project or model.

5. Prefer semantic retrieval for questions whose answer is mainly
   contained in documents.

6. Use metadata for authoritative counts and catalog-level lists.

7. entity_mention should contain only the entity mentioned in the
   user's question.

8. Do not invent an entity mention.

9. entity_type represents the type of entity that the question
   starts from.

10. target_entity_type represents the type of entity that the user
    wants returned.

11. If the user asks for models related to a project:
    entity_type = project
    target_entity_type = model.

12. If the user asks for datasets related to a model:
    entity_type = model
    target_entity_type = dataset.

13. If the user asks for metrics related to a model:
    entity_type = model
    target_entity_type = metric.

14. If the query does not request a particular entity type,
    target_entity_type must be null.

15. Set asset_types only when restricting retrieval to particular
    document types would materially improve precision.

16. Infer asset types from the user's information need and the
    semantic meaning of the available artifact types.

17. Do not require the user to explicitly name the artifact type.

18. Use the minimum document-type filtering necessary.

19. Do not add an asset type merely because it might contain useful
    information.

20. For broad discovery questions across projects, models, methods,
    datasets, techniques, or topics, normally use asset_types = null.

21. Do not restrict broad project-discovery questions to
    project_brief.

22. When several artifact types could plausibly contain the answer
    and the user did not request a specific source category, prefer
    asset_types = null.

23. Return only values supported by the provided operations,
    entity types, and asset types.
24. If the user's primary information need is the CONTENT of a specific
    artifact category, prefer semantic retrieval even when a model or
    project is mentioned.

25. Questions asking for evidence of reproduction should normally use
    semantic retrieval with reproducibility_report when that artifact
    type directly matches the information need.

26. Questions asking for experimental findings, experimental results,
    or experiment details should normally use semantic retrieval with
    experiment_report when the answer is expected to come from that
    document type.

27. Questions asking for evaluation conclusions or evaluation findings
    should normally use semantic retrieval with evaluation_report.

28. Do not choose hybrid merely because a specific project or model is
    mentioned. Hybrid is for broad entity context or summaries, not for
    narrowly targeted document-content questions.

29. If asset_types is non-null and the user's request is primarily about
    the contents of those artifacts, semantic retrieval is generally
    preferred unless the graph is required to answer the question.


OUTPUT CONTRACT

Return ONLY one valid JSON object.

The JSON keys MUST be exactly:

- query_type
- operation
- entity_mention
- entity_type
- target_entity_type
- asset_types

Do NOT rename query_type to retrieval_mode, mode, route, or strategy.

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

User question:
{query}
""".strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw_output = response.output_text.strip()

        # --------------------------------------------------
        # Parse the planner's JSON response.
        # --------------------------------------------------
        data = json.loads(
            raw_output
        )

        # --------------------------------------------------
        # Defensive compatibility normalization.
        #
        # If the LLM returns "retrieval_mode" instead of the
        # required "query_type", normalize it before Pydantic
        # validation.
        # --------------------------------------------------
        if (
            "query_type" not in data
            and "retrieval_mode" in data
        ):
            data["query_type"] = data.pop(
                "retrieval_mode"
            )

        # --------------------------------------------------
        # Validate the planner output against the typed schema.
        # --------------------------------------------------
        return QueryPlanLLM.model_validate(
            data
        )