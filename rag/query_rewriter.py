"""
Query rewriter for improving retrieval effectiveness.

Rewrites user queries into more specific, searchable forms that better match
document terminology and structure.
"""

from __future__ import annotations

from typing import Optional

import config
from rag.llm_client import LLMClient


class QueryRewriter:
    """
    Rewrite user queries for better semantic retrieval.
    
    Uses LLM to transform conversational or ambiguous queries into more
    specific forms that better match course document terminology.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        enabled: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 100,
    ) -> None:
        """
        Initialize query rewriter.
        
        Args:
            llm_client: LLM client for query rewriting
            enabled: Whether rewriting is enabled (can be disabled if LLM unavailable)
            temperature: Temperature for LLM generation (lower = more deterministic)
            max_tokens: Maximum tokens for rewritten query
        """
        self.llm_client = llm_client
        self.enabled = enabled and llm_client.enabled
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def is_available(self) -> bool:
        """Check if rewriting is available (LLM enabled and configured)."""
        return self.enabled and self.llm_client.enabled

    def rewrite(self, query: str) -> str:
        """
        Rewrite query to be more searchable.
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query, or original query if rewriting is disabled/fails
        """
        if not self.is_available:
            return query

        try:
            prompt = self._build_rewrite_prompt(query)
            rewritten = self._call_llm(prompt)
            
            # Validate rewritten query
            if rewritten and len(rewritten.strip()) > 0:
                return rewritten.strip()
            else:
                # Fallback to original if rewrite is empty
                return query
        except Exception as e:
            # Log error but don't fail - fallback to original query
            # In production, you might want to log this
            print(f"Query rewriting failed: {e}, using original query")
            return query

    def _build_rewrite_prompt(self, query: str) -> str:
        """Build the prompt for query rewriting."""
        return f"""Rewrite the following question about a CSE (Computer Science and Engineering) course to be more specific and searchable in course documents.

Extract key concepts, expand synonyms, and clarify intent. Focus on course-related terminology such as:
- Course topics, content, and learning objectives
- Prerequisites and required background knowledge
- Grading criteria, assessment methods, and evaluation
- Assignments, projects, and coursework requirements
- Course structure, schedule, and policies

Keep the rewritten question concise (1-2 sentences) and natural.

Original question: {query}

Rewritten question:"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM to rewrite the query."""
        if not self.llm_client.enabled:
            raise RuntimeError("LLM client is not available")

        system_prompt = (
            "You are a query rewriter for educational content search. "
            "Rewrite questions to be more specific and searchable while preserving the original intent."
        )

        return self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


def create_query_rewriter(
    llm_client: Optional[LLMClient] = None,
    enabled: Optional[bool] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> QueryRewriter:
    """
    Factory function to create QueryRewriter with environment variable support.
    
    Args:
        llm_client: LLM client (created if not provided)
        enabled: Whether rewriting is enabled (from env if not provided)
        temperature: Temperature for generation (from env if not provided)
        max_tokens: Max tokens for rewrite (from env if not provided)
        
    Returns:
        Configured QueryRewriter instance
    """
    if llm_client is None:
        # Get LLM config from centralized config module
        llm_config = config.get_llm_provider_config()
        provider = llm_config["provider"]
        
        if provider == "gemini":
            model, api_key = config.get_gemini_config()
            llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
        elif provider == "ollama":
            model, base_url = config.get_ollama_config()
            llm_client = LLMClient(provider=provider, model=model, base_url=base_url)
        else:
            llm_client = LLMClient(provider=provider)

    if enabled is None:
        enabled = config.get_query_rewriting_enabled()

    if temperature is None:
        temperature = config.get_query_rewriter_temperature()

    if max_tokens is None:
        max_tokens = config.get_query_rewriter_max_tokens()

    return QueryRewriter(
        llm_client=llm_client,
        enabled=enabled,
        temperature=temperature,
        max_tokens=max_tokens,
    )

