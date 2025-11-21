"""
Simple OpenAI chat client for generating final answers.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


class LLMClient:
    def __init__(self, provider: str = "openai", model: Optional[str] = None) -> None:
        self.provider = provider
        self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self._enabled = False
        self._client = None

        if provider != "openai":
            raise ValueError("Only OpenAI provider is implemented")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return

        self._client = OpenAI(api_key=api_key)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate_answer(
        self,
        query: str,
        contexts: List[str],
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self._enabled or not self._client:
            raise RuntimeError("LLM client is disabled (missing API key or dependency)")

        context_block = "\n\n".join(f"- {ctx.strip()}" for ctx in contexts if ctx.strip())
        system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions about CSE course materials. "
            "Use only the provided context snippets. If they do not contain the answer, "
            "respond with 'I have no info'. Make the sentences natural and concise."
        )

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {query}",
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

