"""Grounded RAG answer generation."""

import os

from openai import OpenAI


class RAGGenerator:
    def __init__(self, model: str | None = None):
        self.client = OpenAI()
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

    def generate(
        self,
        question: str,
        retrieved_results,
    ) -> str:
        if not retrieved_results:
            return (
                "I could not find enough evidence in the knowledge base "
                "to answer this question."
            )

        context_parts = []

        for index, result in enumerate(
            retrieved_results,
            start=1,
        ):
            payload = result.payload
            source_label = f"S{index}"

            context_parts.append(
                f"""
            [{source_label}]
            Title: {payload.get("title")}
            Project: {payload.get("project_id")}
            Asset type: {payload.get("asset_type")}
            Document ID: {payload.get("document_id")}

            Content:
            {payload.get("text")}
            """.strip()
                    )

        context = "\n\n".join(context_parts)

        prompt = f"""
            You are an enterprise ML knowledge assistant.

            Answer the user's question using ONLY the supplied evidence.

            Rules:
            1. Do not use outside knowledge.
            2. Do not invent facts.
            3. If the evidence is insufficient, explicitly say so.
            4. Cite claims using source labels such as [S1] or [S2].
            5. If multiple sources support a statement, cite all relevant sources.
            6. Distinguish simulated enterprise documents from real public artifacts
            when that distinction matters.
            7. Keep the answer concise and directly responsive.
            8. Do not claim that a template describes an event that actually happened.

            User question:
            {question}

            Evidence:
            {context}
            """.strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text


    def generate_hybrid(
    self,
    question: str,
    graph_context: dict,
    retrieved_results,
) -> str:
        """
        Generate an answer using structured graph facts
        together with retrieved textual evidence.
        """

        import json

        # --------------------------------------------------
        # Convert retrieved vector results into numbered
        # evidence blocks that can be cited as [S1], [S2], ...
        # --------------------------------------------------
        evidence_blocks = []

        for index, result in enumerate(
            retrieved_results,
            start=1,
        ):
            payload = result.payload

            evidence_blocks.append(
                (
                    f"[S{index}]\n"
                    f"Title: {payload.get('title')}\n"
                    f"Project: {payload.get('project_id')}\n"
                    f"Asset type: {payload.get('asset_type')}\n"
                    f"Text:\n{payload.get('text')}"
                )
            )

        evidence_text = "\n\n".join(
            evidence_blocks
        )

        graph_text = json.dumps(
            graph_context,
            indent=2,
            ensure_ascii=False,
        )

        prompt = f"""
    You are answering a question about an enterprise machine-learning knowledge base.

    Use BOTH:
    1. structured knowledge-graph facts
    2. retrieved document evidence

    Question:
    {question}

    Structured graph context:
    {graph_text}

    Retrieved document evidence:
    {evidence_text}

    Rules:

    1. Use only the supplied graph context and document evidence.
    2. Do not invent facts.
    3. Graph facts may be stated directly when clearly represented.
    4. Use document evidence to explain or support graph facts.
    5. Cite textual evidence using [S1], [S2], etc.
    6. Do not create citations for graph-only facts unless supporting text exists.
    7. If graph and text evidence disagree, explicitly say so.
    8. Distinguish simulated/template enterprise documents from real public artifacts.
    9. If evidence is insufficient, say so clearly.
    10. Give a concise, readable answer rather than dumping raw graph data.
    """.strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text.strip()