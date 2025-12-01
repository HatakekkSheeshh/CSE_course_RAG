"""
Multi-provider LLM client for generating answers and query rewriting.

Supports:
- Google Gemini (free tier available)
- Ollama (completely free, local deployment)
"""

from __future__ import annotations

from typing import List, Optional

import config
from rag.llm_provider import GeminiProvider, OllamaProvider, LLMProvider


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Usage:
        # Google Gemini 
        client = LLMClient(provider="gemini")
        
        # Ollama
        client = LLMClient(provider="ollama")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize LLM client.
        
        Args:
            provider: Provider name ("gemini", "ollama"). If None, uses config.
            model: Model name (provider-specific). If None, uses config.
            api_key: API key (for Gemini only - Ollama doesn't need it!). If None, uses config.
            base_url: Base URL of local Ollama service. If None, uses config.
        """
        # Get provider from config if not provided
        if provider is None:
            provider = config.get_llm_provider()
        
        self.provider_name = provider.lower()
        self._provider: Optional[LLMProvider] = None
        self._enabled = False

        # Get model from config if not provided
        if model is None:
            if provider == "gemini":
                model, _ = config.get_gemini_config()
            elif provider == "ollama":
                model, _ = config.get_ollama_config()
        
        # Get API key from config if not provided (for Gemini)
        if provider == "gemini" and api_key is None:
            _, api_key = config.get_gemini_config()
        
        # Get base_url from config if not provided (for Ollama)
        if provider == "ollama" and base_url is None:
            _, base_url = config.get_ollama_config()

        print(f"[LLMClient] Using model: {model}")

        # Initialize provider
        if self.provider_name == "gemini":
            self._provider = GeminiProvider(api_key=api_key, model=model)
        elif self.provider_name == "ollama":
            self._provider = OllamaProvider(base_url=base_url, model=model)
        else:
            # Provide clear error message
            raise ValueError(
                f"Unknown provider: '{provider}'. Supported providers: 'gemini', 'ollama'. "
                f"Please set LLM_PROVIDER environment variable to 'gemini' or 'ollama'."
            )

        self._enabled = self._provider.enabled if self._provider else False

    @property
    def enabled(self) -> bool:
        """Check if LLM client is enabled"""
        return self._enabled

    @property
    def model_name(self) -> str:
        """Get model name"""
        if self.provider_name == "gemini" and isinstance(self._provider, GeminiProvider):
            return self._provider.model
        elif self.provider_name == "ollama" and isinstance(self._provider, OllamaProvider):
            return self._provider.model
        return "unknown"

    @property
    def _client(self):
        """Backward compatibility - expose provider for direct access if needed"""
        return self._provider

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from prompt"""
        if not self._enabled or not self._provider:
            raise RuntimeError(f"{self.provider_name} client is disabled")

        return self._provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_answer(
        self,
        query: str,
        contexts: List[str],
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate answer from query and contexts (backward compatibility).
        
        This method maintains the original API for answer generation.
        """
        if not self._enabled or not self._provider:
            raise RuntimeError(f"{self.provider_name} client is disabled")

        context_block = "\n\n".join(f"- {ctx.strip()}" for ctx in contexts if ctx.strip())
        system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions about CSE course materials and syllabuses in Ho Chi Minh university of Technology. "
            "If they do not contain the answer, respond with your own creative answer."
            "Make the sentences natural and concise."
        )

        prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

        return self._provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )
