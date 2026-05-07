import os
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


SYSTEM_PROMPT = """You are a senior software architect helping users understand a codebase.
Use only the supplied code context. If information is missing, clearly say so.
Be precise, structured, and concise.
When relevant, include:
1) what the module/function does
2) how files relate
3) logic flow
4) notable limitations or assumptions."""


class GrokCodeAssistant:
    def __init__(self, model: str = "grok-3-mini"):
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY is not set. Add it to your environment.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.model = model

    @staticmethod
    def _build_context(retrieved_chunks: List[Dict], max_chars: int = 12000) -> str:
        """Build a token-efficient context block from retrieved code chunks."""
        context_parts: List[str] = []
        total_chars = 0

        for item in retrieved_chunks:
            snippet = item["text"][:1800]
            section = (
                f"File: {item['file_path']}\n"
                f"Chunk: {item['chunk_index']}\n"
                f"Relevance Score: {item['score']:.4f}\n"
                f"Code Snippet:\n{snippet}\n"
                f"{'-' * 60}\n"
            )
            if total_chars + len(section) > max_chars:
                break
            context_parts.append(section)
            total_chars += len(section)

        return "\n".join(context_parts)

    def answer_question(self, user_query: str, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "I could not find relevant indexed code for this question. Please re-index the project."

        context = self._build_context(retrieved_chunks)
        user_prompt = (
            "Answer the user's codebase question using the context below.\n\n"
            f"User Question:\n{user_query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Respond with clear sections and mention file paths when possible."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content or "No response generated."
