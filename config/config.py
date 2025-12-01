"""
Centralized configuration management.

Loads environment variables from .env file once at module import.
All other modules should import from this config instead of using os.getenv directly.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env file once at module import
try:
    from dotenv import load_dotenv
    
    # Load .env from project root
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    # Environment variables should be set manually or via system
    pass


# ============================================================================
# LLM Configuration
# ============================================================================

def get_llm_provider() -> str:
    """Get LLM provider (gemini or ollama)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    # Validate and fallback to gemini if invalid
    if provider not in ("gemini", "ollama"):
        return "gemini"
    return provider


def get_gemini_config() -> tuple[str, str]:
    """Get Gemini configuration: (model, api_key)."""
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GEMINI_API_KEY", "")
    return model, api_key


def get_ollama_config() -> tuple[str, str]:
    """Get Ollama configuration: (model, base_url)."""
    model = os.getenv("OLLAMA_MODEL", "llama2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Remove quotes if present (common mistake)
    if base_url.startswith('"') and base_url.endswith('"'):
        base_url = base_url[1:-1]
    elif base_url.startswith("'") and base_url.endswith("'"):
        base_url = base_url[1:-1]
    return model, base_url


# ============================================================================
# Query Rewriting Configuration
# ============================================================================

def get_query_rewriting_enabled() -> bool:
    """Check if query rewriting is enabled."""
    enabled_str = os.getenv("ENABLE_QUERY_REWRITING", "true").lower()
    return enabled_str in ("true", "1", "yes", "on")


def get_query_rewriter_temperature() -> float:
    """Get query rewriter temperature."""
    return float(os.getenv("QUERY_REWRITER_TEMPERATURE", "0.3"))


def get_query_rewriter_max_tokens() -> int:
    """Get query rewriter max tokens."""
    return int(os.getenv("QUERY_REWRITER_MAX_TOKENS", "100"))


# ============================================================================
# RAG Pipeline Configuration
# ============================================================================

def get_rag_data_dir() -> Path:
    """Get RAG data directory."""
    return Path(os.getenv("RAG_DATA_DIR", "data"))


def get_rag_index_dir() -> Path:
    """Get RAG index directory."""
    data_dir = get_rag_data_dir()
    index_dir = os.getenv("RAG_INDEX_DIR", str(data_dir / "indices"))
    return Path(index_dir)


def get_rag_retrieval_k() -> int:
    """Get retrieval K (number of chunks to retrieve)."""
    return int(os.getenv("RAG_RETRIEVAL_K", "8"))


def get_rag_rerank_k() -> int:
    """Get rerank K (number of chunks to rerank)."""
    return int(os.getenv("RAG_RERANK_K", "5"))


def get_rag_confidence_threshold() -> float:
    """Get confidence threshold for reranking."""
    return float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.1"))


def get_rag_only_course() -> str | None:
    """Get optional course filter."""
    course = os.getenv("RAG_ONLY_COURSE")
    return course if course else None


# ============================================================================
# API Configuration
# ============================================================================

def get_cors_allow_origins() -> list[str]:
    """Get CORS allowed origins."""
    origins_str = os.getenv("CORS_ALLOW_ORIGINS", "*")
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


# ============================================================================
# Convenience Functions for Common Configs
# ============================================================================

def get_llm_provider_config() -> dict:
    """
    Get complete LLM provider configuration.
    
    Returns:
        dict with keys: provider, model, api_key (for Gemini), base_url (for Ollama)
    """
    provider = get_llm_provider()
    config = {"provider": provider}
    
    if provider == "gemini":
        model, api_key = get_gemini_config()
        config.update({"model": model, "api_key": api_key})
    elif provider == "ollama":
        model, base_url = get_ollama_config()
        config.update({"model": model, "base_url": base_url})
    
    return config


def get_rag_pipeline_config() -> dict:
    """
    Get complete RAG pipeline configuration.
    
    Returns:
        dict with all RAG pipeline settings
    """
    return {
        "data_dir": get_rag_data_dir(),
        "index_dir": get_rag_index_dir(),
        "retrieval_k": get_rag_retrieval_k(),
        "rerank_k": get_rag_rerank_k(),
        "confidence_threshold": get_rag_confidence_threshold(),
        "only_course": get_rag_only_course(),
    }

