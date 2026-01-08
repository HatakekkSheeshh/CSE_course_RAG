"""
Unit tests for the Embedding module.
"""
import pytest
import numpy as np
from models.embedding import Embedding


def test_embedding_initialization():
    """Test embedding model loads correctly."""
    embedding = Embedding()
    assert embedding.model is not None


def test_embed_single_text():
    """Test embedding generation for single text."""
    embedding = Embedding()
    text = "What is the grading policy?"
    
    result = embedding.embed(text)
    
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)
    
    # Check that embedding is not all zeros
    assert not np.allclose(result, 0)


def test_embed_batch():
    """Test batch embedding generation."""
    embedding = Embedding()
    texts = ["Query 1", "Query 2", "Query 3"]
    
    results = embedding.embed_batch(texts)
    
    assert len(results) == 3
    assert all(len(emb) == 384 for emb in results)
    assert not np.allclose(results[0], results[1])


def test_embed_empty_text():
    """Test embedding generation for empty text."""
    embedding = Embedding()
    text = ""
    
    result = embedding.embed(text)
    
    assert len(result) == 384


def test_embedding_consistency():
    """Test that same text produces same embedding."""
    embedding = Embedding()
    text = "Test consistency"
    
    result1 = embedding.embed(text)
    result2 = embedding.embed(text)
    
    assert np.allclose(result1, result2)
