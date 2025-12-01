"""
Simple test script for query rewriting functionality.

Usage:
    python -m rag.test_query_rewriter "How do I pass?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config  # noqa: E402 - Load config first to initialize dotenv
from rag.llm_client import LLMClient  # noqa: E402
from rag.query_rewriter import QueryRewriter, create_query_rewriter  # noqa: E402


def test_rewriter() -> None:
    """Test query rewriter with example queries."""
    
    # Get provider from config
    provider = config.get_llm_provider()
    
    # Check if API key is set (for Gemini)
    if provider == "gemini":
        _, api_key = config.get_gemini_config()
        if not api_key:
            print("[WARNING] GEMINI_API_KEY not set. Query rewriting requires API key.")
            print("   Set it with: export GEMINI_API_KEY=...")
            print("   Or use Ollama (free, local): export LLM_PROVIDER=ollama")
            return
    
    # Create LLM client using config
    print(f"Initializing query rewriter with {provider}...")
    try:
        llm_config = config.get_llm_provider_config()
        if provider == "gemini":
            llm_client = LLMClient(
                provider=provider,
                model=llm_config["model"],
                api_key=llm_config["api_key"],
            )
        elif provider == "ollama":
            llm_client = LLMClient(
                provider=provider,
                model=llm_config["model"],
                base_url=llm_config["base_url"],
            )
        else:
            llm_client = LLMClient(provider=provider)
    except Exception as e:
        print(f"[ERROR] Failed to create LLM client: {e}")
        return
    
    if not llm_client.enabled:
        print("❌ LLM client is not enabled. Check your API key.")
        return
    
    rewriter = create_query_rewriter(llm_client=llm_client)
    
    if not rewriter.is_available:
        print("❌ Query rewriter is not available.")
        return
    
    print("✅ Query rewriter ready!\n")
    
    # Test queries
    test_queries = [
        "What do I need to know?",
        "How do I pass?",
        "What are the assignments?",
        "course info",
        "What topics are covered and how are they tested?",
    ]
    
    print("Testing query rewriting:\n")
    print("=" * 80)
    
    for original in test_queries:
        print(f"\nOriginal:  {original}")
        rewritten = rewriter.rewrite(original)
        print(f"Rewritten: {rewritten}")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided query
        query = " ".join(sys.argv[1:])
        
        provider = config.get_llm_provider()
        
        # Check API key
        if provider == "gemini":
            _, api_key = config.get_gemini_config()
            if not api_key:
                print("[ERROR] GEMINI_API_KEY not set.")
                sys.exit(1)
        
        # Create client using config
        try:
            llm_config = config.get_llm_provider_config()
            if provider == "gemini":
                llm_client = LLMClient(
                    provider=provider,
                    model=llm_config["model"],
                    api_key=llm_config["api_key"],
                )
            elif provider == "ollama":
                llm_client = LLMClient(
                    provider=provider,
                    model=llm_config["model"],
                    base_url=llm_config["base_url"],
                )
            else:
                llm_client = LLMClient(provider=provider)
        except Exception as e:
            print(f"[ERROR] Failed to create LLM client: {e}")
            sys.exit(1)
        
        if not llm_client.enabled:
            print(f"[ERROR] LLM client is not enabled for {provider}.")
            sys.exit(1)
        
        rewriter = create_query_rewriter(llm_client=llm_client)
        rewritten = rewriter.rewrite(query)
        
        print(f"Original:  {query}")
        print(f"Rewritten: {rewritten}")
    else:
        # Run test suite
        test_rewriter()

