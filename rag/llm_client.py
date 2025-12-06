"""
Multi-provider LLM client for generating answers and query rewriting.

Supports:
- Google Gemini (free tier available)
- Ollama (completely free, local deployment)
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional

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
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream generate text from prompt"""
        if not self._enabled or not self._provider:
            raise RuntimeError(f"{self.provider_name} client is disabled")

        async for chunk in self._provider.stream_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk

    def generate_answer(
        self,
        query: str,
        contexts: List[str],
        *,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[str] = None,
    ) -> str:
        """
        Generate answer from query and contexts.
        
        Args:
            query: User question
            contexts: List of retrieved context chunks
            system_prompt: Optional custom system prompt
            conversation_history: Optional formatted conversation history
            
        Returns:
            Generated answer string
        """
        if not self._enabled or not self._provider:
            raise RuntimeError(f"{self.provider_name} client is disabled")

        # Use system prompt from config if not provided
        # Use history-aware prompt if history is provided
        if system_prompt is None:
            if conversation_history:
                system_prompt = config.get_rag_system_prompt_with_history()
            else:
                system_prompt = config.get_rag_system_prompt()

        # Format prompt using config helper (supports history)
        prompt = config.format_rag_prompt(
            query=query,
            contexts=contexts,
            conversation_history=conversation_history,
        )

        return self._provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )
    
    async def stream_generate_answer(
        self,
        query: str,
        contexts: List[str],
        *,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream generate answer from query and contexts.
        
        Args:
            query: User question
            contexts: List of retrieved context chunks
            system_prompt: Optional custom system prompt
            conversation_history: Optional formatted conversation history
            
        Yields:
            Answer tokens as they are generated
        """
        if not self._enabled or not self._provider:
            raise RuntimeError(f"{self.provider_name} client is disabled")

        # Use system prompt from config if not provided
        # Use history-aware prompt if history is provided
        if system_prompt is None:
            if conversation_history:
                system_prompt = config.get_rag_system_prompt_with_history()
            else:
                system_prompt = config.get_rag_system_prompt()

        # Format prompt using config helper (supports history)
        prompt = config.format_rag_prompt(
            query=query,
            contexts=contexts,
            conversation_history=conversation_history,
        )

        async for chunk in self._provider.stream_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        ):
            yield chunk
