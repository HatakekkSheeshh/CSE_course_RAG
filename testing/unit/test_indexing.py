"""
Unit tests for the Indexing module.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from models.indexing import (
    create_index,
    add_documents,
    search,
    save_index,
    load_index,
    chunks_to_metadata_map
)
from preprocessing.manifest import DocChunk, Metadata


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    return np.random.rand(10, 384).astype('float32')


@pytest.fixture
def sample_chunks():
    """Generate sample chunks for testing."""
    chunks = []
    for i in range(10):
        metadata = Metadata(
            doc_type="slide",
            course_id="test_course",
            source_file=f"test_{i}.pdf",
            page_index=i,
            language="en",
            ocr_engine="paddleocr",
            extractor_version="1.0",
            timestamp="2024-01-01T00:00:00"
        )
        chunk = DocChunk(
            id=f"chunk_{i}",
            text=f"This is test chunk {i}",
            metadata=metadata
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_create_index():
    """Test FAISS index creation."""
    index = create_index(embedding_dim=384)
    assert index is not None
    assert index.d == 384  # Dimension check


def test_add_documents(sample_embeddings):
    """Test adding documents to index."""
    index = create_index(embedding_dim=384)
    ids = [f"chunk_{i}" for i in range(10)]
    
    add_documents(index, sample_embeddings, ids)
    
    assert index.ntotal == 10  # 10 vectors added


def test_search(sample_embeddings):
    """Test similarity search."""
    index = create_index(embedding_dim=384)
    ids = [f"chunk_{i}" for i in range(10)]
    add_documents(index, sample_embeddings, ids)
    
    query_embedding = sample_embeddings[0]
    distances, indices = search(index, query_embedding, k=3)
    
    assert len(distances) == 3
    assert len(indices) == 3
    assert indices[0] == 0


def test_save_and_load_index(sample_embeddings, sample_chunks, temp_index_dir):
    """Test saving and loading index with metadata."""
    # Create and populate index
    index = create_index(embedding_dim=384)
    ids = [chunk.id for chunk in sample_chunks]
    add_documents(index, sample_embeddings, ids)
    
    # Create metadata map
    metadata_map = chunks_to_metadata_map(sample_chunks)
    
    # Save index
    save_index(index, temp_index_dir, metadata_map)
    
    # Check files exist
    assert (temp_index_dir / "index.faiss").exists()
    assert (temp_index_dir / "metadata.json").exists()
    
    # Load index
    loaded_index, loaded_metadata = load_index(temp_index_dir)
    
    assert loaded_index.ntotal == 10
    assert len(loaded_metadata) == 10
    assert "chunk_0" in loaded_metadata


def test_chunks_to_metadata_map(sample_chunks):
    """Test conversion of chunks to metadata map."""
    metadata_map = chunks_to_metadata_map(sample_chunks)
    
    assert len(metadata_map) == 10
    assert "chunk_0" in metadata_map
    assert metadata_map["chunk_0"]["text"] == "This is test chunk 0"
    assert metadata_map["chunk_0"]["metadata"]["course_id"] == "test_course"
