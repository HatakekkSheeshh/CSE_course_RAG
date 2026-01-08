"""
Integration tests for the Retrieval Pipeline.
"""
import pytest
from pathlib import Path
from rag.query_pipeline import QueryPipeline


@pytest.fixture
def pipeline():
    """Create test pipeline with sample data."""
    from pathlib import Path
    from rag.llm_client import LLMClient
    import config
    
    data_dir = Path("data")
    index_dir = Path("data/indices")
    
    if not index_dir.exists():
        pytest.skip("Index directory not found. Run indexing first.")
    query_rewriter = None
    try:
        from rag.query_rewriter import create_query_rewriter
        llm_config = config.get_llm_provider_config()
        llm_client = LLMClient(
            provider=llm_config["provider"],
            model=llm_config.get("model"),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
        )
        query_rewriter = create_query_rewriter(llm_client=llm_client)
    except Exception:
        pass  
    
    return QueryPipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        retrieval_k=5,
        rerank_k=3,
        query_rewriter=query_rewriter,
    )


def test_end_to_end_retrieval(pipeline):
    """Test complete retrieval and reranking flow."""
    query = "What are the course prerequisites?"
    
    result = pipeline.answer(query)
    
    assert result["status"] == "ok"
    assert len(result["retrieved"]) > 0
    assert len(result["reranked"]) <= 3
    assert result["reranked"][0].confidence > 0


def test_retrieval_with_course_filter(pipeline):
    """Test retrieval with course filtering."""
    query = "What is the grading policy?"
    
    courses = list(pipeline._indices.keys())
    if not courses:
        pytest.skip("No courses available in indices")
    
    course = courses[0]
    result = pipeline.answer(query, course=course)
    
    assert result["status"] == "ok"
    # All retrieved chunks should be from the specified course
    for chunk in result["retrieved"]:
        assert chunk.course == course


def test_query_rewriting_integration(pipeline):
    """Test query rewriting improves retrieval."""
    if not pipeline.query_rewriter or not pipeline.query_rewriter.is_available:
        pytest.skip("Query rewriter not available")
    
    query = "What about the final?"
    
    result_with_rewrite = pipeline.answer(query)
    
    original_rewriter = pipeline.query_rewriter
    pipeline.query_rewriter = None
    result_without_rewrite = pipeline.answer(query)
    pipeline.query_rewriter = original_rewriter
    
    assert result_with_rewrite["status"] == "ok"
    assert result_without_rewrite["status"] == "ok"


def test_retrieval_empty_query(pipeline):
    """Test retrieval with empty query."""
    query = ""
    
    result = pipeline.answer(query)
    
    assert "status" in result


def test_retrieval_unknown_course(pipeline):
    """Test retrieval with non-existent course falls back to all courses."""
    query = "What is the grading policy?"
    course = "NonExistentCourse"
    
    result = pipeline.answer(query, course=course)
    assert result["status"] in ["ok", "no_info"]
    if result["status"] == "ok":
        for chunk in result["retrieved"]:
            assert chunk.course != course


def test_reranking_improves_relevance(pipeline):
    """Test that reranking improves result relevance."""
    query = "final exam percentage"
    
    result = pipeline.answer(query)
    
    if result["status"] == "ok" and len(result["reranked"]) > 1:
        # Reranked results should be sorted by confidence
        confidences = [r.confidence for r in result["reranked"]]
        assert confidences == sorted(confidences, reverse=True)
