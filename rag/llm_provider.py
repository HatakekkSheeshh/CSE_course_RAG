
"""
LLM Provider implementations for different LLM services.

This module contains provider classes for:
- Google Gemini (free tier available)
- Ollama (completely free, local deployment)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

# Provider imports (optional, fail gracefully if not installed)
try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from prompt"""
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if provider is enabled"""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._enabled = False

        if genai is None:
            return

        if not self.api_key:
            return

        try:
            genai.configure(api_key=self.api_key)
            self._enabled = True
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self._enabled:
            raise RuntimeError("Gemini client is disabled (missing API key or dependency)")

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            model = genai.GenerativeModel(self.model)
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(**generation_config),
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")


class OllamaProvider(LLMProvider):
    """
    Ollama provider (completely free, local deployment).
    
    Ollama runs as a local service - no API key needed!
    - Install Ollama: https://ollama.ai/download
    - Pull model: ollama pull llama2
    - Service runs at http://localhost:11434 by default
    - Models are stored locally on your machine
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: URL of local Ollama service (default: http://localhost:11434)
            model: Model name (must be pulled first with 'ollama pull <model>')
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
        self._enabled = False

        if ollama is None:
            return

        # Test connection to local Ollama service
        try:
            # Simple check - try to list models from local service
            ollama.list(base_url=self.base_url)
            self._enabled = True
        except Exception:
            # Ollama might not be running, but we'll try anyway
            # Will fail at generation if service is not available
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled and ollama is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.enabled or ollama is None:
            raise RuntimeError("Ollama client is disabled (not installed or not running)")

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 512,
                },
                base_url=self.base_url,
            )
            return response["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}. Make sure Ollama is running at {self.base_url}")