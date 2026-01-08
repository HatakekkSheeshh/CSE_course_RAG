"""
Unit tests for the Reranker module.
"""
import pytest
from models.reranker import Reranker, RerankResult


@pytest.fixture
def reranker():
    """Create reranker instance for testing."""
    return Reranker()


@pytest.fixture
def sample_passages():
    """Sample passages for reranking."""
    return [
        ("chunk_1", "The final exam is worth 40% of the grade.", {"course": "CO3101"}),
        ("chunk_2", "Midterm exam accounts for 30% of the grade.", {"course": "CO3101"}),
        ("chunk_3", "Homework assignments are 20% of the grade.", {"course": "CO3101"}),
        ("chunk_4", "Class participation is 10% of the grade.", {"course": "CO3101"}),
    ]


def test_reranker_initialization(reranker):
    """Test reranker model loads correctly."""
    assert reranker.model is not None


def test_reranker_score(reranker, sample_passages):
    """Test reranking with score computation."""
    query = "What is the final exam percentage?"
    
    results = reranker.score(query, sample_passages)
    
    assert len(results) > 0
    assert all(isinstance(r, RerankResult) for r in results)
    # Results should be sorted by confidence (descending)
    assert results[0].confidence >= results[-1].confidence


def test_reranker_confidence_normalization(reranker, sample_passages):
    """Test that confidence scores are normalized (sum to 1)."""
    query = "What is the grading policy?"
    
    results = reranker.score(query, sample_passages)
    
    # Confidence scores should sum to approximately 1 (softmax normalization)
    total_confidence = sum(r.confidence for r in results)
    assert abs(total_confidence - 1.0) < 0.01


def test_reranker_relevance_ordering(reranker):
    """Test that more relevant passages get higher scores."""
    query = "final exam percentage"
    passages = [
        ("chunk_1", "The final exam is worth 40% of the grade.", {}),
        ("chunk_2", "The weather is nice today.", {}),
    ]
    
    results = reranker.score(query, passages)
    
    # First passage should be more relevant
    assert results[0].chunk_id == "chunk_1"
    assert results[0].confidence > results[1].confidence


def test_reranker_empty_passages(reranker):
    """Test reranker with empty passage list."""
    query = "test query"
    passages = []
    
    results = reranker.score(query, passages)
    
    assert len(results) == 0


def test_reranker_single_passage(reranker):
    """Test reranker with single passage."""
    query = "test query"
    passages = [("chunk_1", "Test passage", {})]
    
    results = reranker.score(query, passages)
    
    assert len(results) == 1
    # Single passage should have confidence close to 1
    assert results[0].confidence > 0.9
