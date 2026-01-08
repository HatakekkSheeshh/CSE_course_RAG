"""
Unit tests for the Query Pipeline module.
"""
import pytest
from pathlib import Path
from rag.query_pipeline import QueryPipeline


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing without actual indices."""
    data_dir = Path("data")
    index_dir = Path("data/indices")
    
    if not index_dir.exists():
        pytest.skip("Index directory not found")
    
    return QueryPipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        retrieval_k=5,
        rerank_k=3
    )


def test_pipeline_initialization(mock_pipeline):
    """Test pipeline initializes correctly."""
    assert mock_pipeline is not None
    assert mock_pipeline.retrieval_k == 5
    assert mock_pipeline.rerank_k == 3


def test_pipeline_has_required_components(mock_pipeline):
    """Test pipeline has all required components."""
    assert hasattr(mock_pipeline, 'embedding')
    assert hasattr(mock_pipeline, '_indices')
    assert hasattr(mock_pipeline, 'reranker')


def test_query_validation(mock_pipeline):
    """Test query input validation."""
    result = mock_pipeline.answer("")
    assert "status" in result
    
    result = mock_pipeline.answer("What is the grading policy?")
    assert "status" in result
    assert result["status"] in ["ok", "no_info"]
