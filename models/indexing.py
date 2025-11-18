"""
FAISS index management for RAG system.

Provides functions to create, manage, and search FAISS indices
with metadata preservation.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import numpy as np
import faiss
from faiss import IndexFlatIP
from preprocessing.manifest import DocChunk


def create_index(embedding_dim: int) -> IndexFlatIP:
    """
    Create a new FAISS index.
    
    Args:
        embedding_dim: Dimension of embeddings (e.g., 384 for all-MiniLM-L6-v2)
        
    Returns:
        FAISS IndexFlatIP instance
    """
    return IndexFlatIP(embedding_dim)


def add_documents(
    index: IndexFlatIP,
    embeddings: np.ndarray,
    ids: List[str]
) -> None:
    """
    Add document embeddings to FAISS index.
    
    Args:
        index: FAISS index instance
        embeddings: Numpy array of shape (n_docs, embedding_dim)
        ids: List of chunk IDs corresponding to embeddings
    """
    # Normalize embeddings for cosine similarity (Inner Product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    
    index.add(normalized_embeddings.astype('float32'))


def search(
    index: IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the index for similar vectors.
    
    Args:
        index: FAISS index instance
        query_embedding: Query embedding vector of shape (embedding_dim,)
        k: Number of results to return
        
    Returns:
        Tuple of (distances, indices) arrays
    """
    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        query_norm = 1
    normalized_query = (query_embedding / query_norm).astype('float32').reshape(1, -1)
    
    distances, indices = index.search(normalized_query, k)
    return distances[0], indices[0]


def save_index(
    index: IndexFlatIP,
    path: Path,
    metadata_map: Dict[str, Dict]
) -> None:
    """
    Save FAISS index and metadata map to disk.
    
    Args:
        index: FAISS index instance
        path: Directory path to save index
        metadata_map: Dictionary mapping chunk IDs to DocChunk data
    """
    path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = path / "index.faiss"
    faiss.write_index(index, str(index_path))
    
    # Save metadata map
    metadata_path = path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_map, f, ensure_ascii=False, indent=2)


def load_index(path: Path) -> Tuple[IndexFlatIP, Dict[str, Dict]]:
    """
    Load FAISS index and metadata map from disk.
    
    Args:
        path: Directory path containing index.faiss and metadata.json
        
    Returns:
        Tuple of (index, metadata_map)
    """
    index_path = path / "index.faiss"
    metadata_path = path / "metadata.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load index
    index = faiss.read_index(str(index_path))
    
    # Validate index type (should be IndexFlatIP, not IndexFlatL2)
    reference_index = IndexFlatIP(index.d if hasattr(index, 'd') else 384)
    expected_metric_type = reference_index.metric_type
    
    if hasattr(index, 'metric_type'):
        metric_type = int(index.metric_type)
        if metric_type != expected_metric_type:
            print(f"⚠️  WARNING: Loaded index has metric_type={metric_type}")
            print(f"   Expected metric_type={expected_metric_type} (same as IndexFlatIP)")
            print(f"   This index may have been created with IndexFlatL2 instead of IndexFlatIP")
            print(f"   Search results may be incorrect for cosine similarity (normalized vectors)")
            print(f"   Recommendation: Delete this index and rebuild using --index command")
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_map = json.load(f)
    
    return index, metadata_map


def chunks_to_metadata_map(chunks: List[DocChunk]) -> Dict[str, Dict]:
    """
    Convert list of DocChunk objects to metadata map dictionary.
    
    Args:
        chunks: List of DocChunk objects
        
    Returns:
        Dictionary mapping chunk IDs to serialized DocChunk data
    """
    metadata_map = {}
    for chunk in chunks:
        metadata_map[chunk.id] = {
            "id": chunk.id,
            "text": chunk.text,
            "metadata": {
                "doc_type": chunk.metadata.doc_type,
                "course_id": chunk.metadata.course_id,
                "source_file": chunk.metadata.source_file,
                "page_index": chunk.metadata.page_index,
                "language": chunk.metadata.language,
                "ocr_engine": chunk.metadata.ocr_engine,
                "extractor_version": chunk.metadata.extractor_version,
                "timestamp": chunk.metadata.timestamp
            }
        }
    return metadata_map

