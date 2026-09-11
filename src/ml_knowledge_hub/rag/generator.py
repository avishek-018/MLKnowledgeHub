"""Response generation for retrieval-augmented workflows."""

from openai import OpenAI


class RAGGenerator:
    def __init__(self, model: str = "gpt-5.6-luna"):
        self.client = OpenAI()
        self.model = model

    def generate(
        self,
        question: str,
        retrieved_chunks,
    ) -> str:
        context_parts = []

        for result in retrieved_chunks:
            chunk_id = result.payload["chunk_id"]
            text = result.payload["text"]

            context_parts.append(
                f"[{chunk_id}]\n{text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an ML knowledge assistant.

Answer the question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- If the context is insufficient, say:
  "The retrieved evidence is insufficient to answer this question."
- Cite supporting chunk IDs using square brackets.
- Do not invent citations.
- Keep the answer concise but informative.

Question:
{question}

Context:
{context}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text