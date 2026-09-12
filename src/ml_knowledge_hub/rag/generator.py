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