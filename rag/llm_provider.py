
"""
LLM Provider implementations for different LLM services.

This module contains provider classes for:
- Google Gemini (free tier available)
- Ollama (completely free, local deployment)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

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
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream generate text from prompt (default implementation uses generate).
        
        Subclasses should override this for true streaming support.
        """
        # Default: fallback to non-streaming generate
        result = self.generate(prompt, system_prompt, temperature, max_tokens)
        yield result


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
            error_msg = str(e)
            # Check for quota/rate limit errors
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    f"Gemini API quota exceeded. "
                    f"Free tier limit: 20 requests/day. "
                    f"Solutions: 1) Wait 24 hours for quota reset, 2) Switch to Ollama (local, unlimited), "
                    f"3) Upgrade to paid plan. Error: {e}"
                )
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream generate text from prompt using Gemini streaming API"""
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

            # Use stream=True for streaming
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(**generation_config),
                stream=True,
            )
            
            # Yield chunks as they arrive (convert sync iterator to async)
            # Gemini returns sync iterator, so we need to yield in async context
            import asyncio
            chunk_count = 0
            for chunk in response:
                if chunk.text:
                    chunk_count += 1
                    # Debug logging
                    print(f"[GEMINI] Chunk #{chunk_count}: {repr(chunk.text[:50])}")
                    # Yield immediately to allow other tasks to run
                    yield chunk.text
                    # Small delay to ensure proper async behavior
                    await asyncio.sleep(0)
            
            print(f"[GEMINI] Streaming completed: {chunk_count} chunks")
        except Exception as e:
            error_msg = str(e)
            # Check for quota/rate limit errors
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    f"Gemini API quota exceeded. "
                    f"Free tier limit: 20 requests/day. "
                    f"Solutions: 1) Wait 24 hours for quota reset, 2) Switch to Ollama (local, unlimited), "
                    f"3) Upgrade to paid plan. Error: {e}"
                )
            raise RuntimeError(f"Gemini streaming failed: {e}")


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
        self._client = None

        if ollama is None:
            return

        # Create Ollama client with base_url
        # Parse base_url to extract host and port
        # Handle formats like: http://localhost:11434, http://ollama:11434, localhost:11434
        url = self.base_url.replace("http://", "").replace("https://", "")
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host = url
                port = 11434
        else:
            host = url
            port = 11434
        
        # Store host and port for later use
        self._ollama_host = host
        self._ollama_port = port
        
        # Ollama Python library uses OLLAMA_HOST environment variable for connection
        # Format: host:port (no http://)
        # Set it permanently for this client instance
        ollama_host_value = f"{host}:{port}"
        old_ollama_host = os.environ.get("OLLAMA_HOST")
        
        # Set OLLAMA_HOST if it's different from current value
        if os.environ.get("OLLAMA_HOST") != ollama_host_value:
            os.environ["OLLAMA_HOST"] = ollama_host_value
        
        # Create client - Ollama library reads from OLLAMA_HOST env var
        self._client = None
        try:
            # Try different initialization methods
            try:
                # Method 1: Try Client(host=host, port=port) - newer API
                self._client = ollama.Client(host=host, port=port)
            except (TypeError, AttributeError):
                try:
                    # Method 2: Try Client(host=host) - port defaults to 11434
                    self._client = ollama.Client(host=host)
                except (TypeError, AttributeError):
                    # Method 3: Use default client (reads from OLLAMA_HOST env var)
                    self._client = ollama.Client()
            
            # Test connection by listing models
            try:
                self._client.list()
                self._enabled = True
            except Exception as conn_error:
                # Connection test failed
                print(f"Warning: Could not connect to Ollama at {self.base_url}: {conn_error}")
                self._enabled = False
                # Client is created but connection will fail at generation time
        except Exception as e:
            # Client creation failed
            print(f"Warning: Failed to create Ollama client: {e}")
            self._enabled = False
            # Try to create default client as fallback
            try:
                self._client = ollama.Client()
            except Exception:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled and ollama is not None and self._client is not None

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
            # Use client instance instead of passing base_url
            response = self._client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 512,
                },
            )
            return response["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}. Make sure Ollama is running at {self.base_url}")
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream generate text from prompt using Ollama streaming API"""
        if not self.enabled or ollama is None:
            raise RuntimeError("Ollama client is disabled (not installed or not running)")

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            # Use stream=True for streaming
            stream = self._client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 512,
                },
                stream=True,
            )
            
            # Yield chunks as they arrive (convert sync iterator to async)
            import asyncio
            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]
                    # Small delay to ensure proper async behavior
                    await asyncio.sleep(0)
        except Exception as e:
            raise RuntimeError(f"Ollama streaming failed: {e}. Make sure Ollama is running at {self.base_url}")


class QwenProvider(LLMProvider):
    """
    Qwen provider (Alibaba's Qwen models via Ollama backend).
    
    Qwen models are excellent for multilingual tasks including Vietnamese.
    - qwen2.5:0.5b - Very fast, ~400MB
    - qwen2.5:1.5b - Fast, ~1GB  
    - qwen2.5:3b - Balanced speed/quality, ~2GB (recommended)
    - qwen2.5:7b - High quality, ~4.5GB
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b"):
        """
        Initialize Qwen provider.
        
        Args:
            base_url: URL of Ollama service running Qwen models
            model: Qwen model name (default: qwen2.5:3b)
        """
        self.base_url = base_url or os.getenv("QWEN_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.model = model or os.getenv("QWEN_MODEL", "qwen2.5:3b")
        self._enabled = False
        self._client = None

        if ollama is None:
            print("[QwenProvider] Ollama library not installed")
            return

        # Parse base_url to extract host and port
        url = self.base_url.replace("http://", "").replace("https://", "")
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host = url
                port = 11434
        else:
            host = url
            port = 11434
        
        self._host = host
        self._port = port
        
        # Set OLLAMA_HOST environment variable
        ollama_host_value = f"{host}:{port}"
        if os.environ.get("OLLAMA_HOST") != ollama_host_value:
            os.environ["OLLAMA_HOST"] = ollama_host_value
        
        # Create client
        try:
            try:
                self._client = ollama.Client(host=host, port=port)
            except (TypeError, AttributeError):
                try:
                    self._client = ollama.Client(host=host)
                except (TypeError, AttributeError):
                    self._client = ollama.Client()
            
            # Test connection
            try:
                self._client.list()
                self._enabled = True
                print(f"[QwenProvider] Connected to {self.base_url}, model: {self.model}")
            except Exception as conn_error:
                print(f"[QwenProvider] Warning: Could not connect to {self.base_url}: {conn_error}")
                self._enabled = False
        except Exception as e:
            print(f"[QwenProvider] Warning: Failed to create client: {e}")
            self._enabled = False
            try:
                self._client = ollama.Client()
            except Exception:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled and ollama is not None and self._client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.enabled or ollama is None:
            raise RuntimeError("Qwen provider is disabled (Ollama not installed or not running)")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self._client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 512,
                },
            )
            return response["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Qwen generation failed: {e}. Make sure model '{self.model}' is pulled.")
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream generate text using Qwen model"""
        if not self.enabled or ollama is None:
            raise RuntimeError("Qwen provider is disabled (Ollama not installed or not running)")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            stream = self._client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 512,
                },
                stream=True,
            )
            
            import asyncio
            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]
                    await asyncio.sleep(0)
        except Exception as e:
            raise RuntimeError(f"Qwen streaming failed: {e}. Make sure model '{self.model}' is pulled.")