"""
Debug and visualization utilities for embeddings and FAISS indices.

Provides functions to inspect embeddings, visualize vectors, and debug FAISS indices.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
from faiss import IndexFlatIP
import faiss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models.indexing import load_index
from preprocessing.manifest import DocChunk


def inspect_embeddings(
    embeddings: np.ndarray,
    chunk_ids: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Inspect embedding statistics.
    
    Args:
        embeddings: Numpy array of embeddings (n_docs, embedding_dim)
        chunk_ids: Optional list of chunk IDs
        verbose: Whether to print statistics
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "shape": embeddings.shape,
        "n_vectors": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "dtype": str(embeddings.dtype),
        "mean": float(np.mean(embeddings)),
        "std": float(np.std(embeddings)),
        "min": float(np.min(embeddings)),
        "max": float(np.max(embeddings)),
        "norm_mean": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "norm_std": float(np.std(np.linalg.norm(embeddings, axis=1))),
    }
    
    if verbose:
        print("\n[DEBUG] Embedding Statistics:")
        print(f"  Shape: {stats['shape']} (n_vectors, embedding_dim)")
        print(f"  Number of vectors: {stats['n_vectors']} (chunks)")
        print(f"  Embedding dimension: {stats['embedding_dim']} (fixed by model)")
        print(f"  → Total elements: {stats['n_vectors']} × {stats['embedding_dim']} = {stats['n_vectors'] * stats['embedding_dim']:,}")
        print(f"  Mean value: {stats['mean']:.6f}")
        print(f"  Std value: {stats['std']:.6f}")
        print(f"  Min value: {stats['min']:.6f}")
        print(f"  Max value: {stats['max']:.6f}")
        print(f"  Mean norm: {stats['norm_mean']:.6f}")
        print(f"  Std norm: {stats['norm_std']:.6f}")
    
    return stats


def visualize_embedding_matrix(
    embeddings: np.ndarray,
    output_path: Optional[Path] = None,
    max_samples: int = 100,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize embedding matrix as heatmap.
    
    Args:
        embeddings: Numpy array of embeddings
        output_path: Path to save the plot (optional)
        max_samples: Maximum number of samples to visualize
        figsize: Figure size (width, height)
    """
    # Constrain the number of samples to visualize (<= 100)
    if embeddings.shape[0] > max_samples:
        indices = np.random.choice(embeddings.shape[0], max_samples, replace=False)
        sample_embeddings = embeddings[indices]
        title_suffix = f" (showing {max_samples}/{embeddings.shape[0]} samples)"
    else:
        sample_embeddings = embeddings
        title_suffix = ""
    
    plt.figure(figsize=figsize)
    plt.imshow(sample_embeddings, aspect='auto', cmap='viridis')
    plt.colorbar(label='Embedding Value')
    plt.title(f'Embedding Matrix Visualization{title_suffix}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Document Index')
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[DEBUG] Saved embedding matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_embeddings_2d(
    embeddings: np.ndarray,
    chunk_ids: Optional[List[str]] = None,
    method: str = "pca",
    output_path: Optional[Path] = None,
    max_samples: int = 1000,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualize embeddings in 2D using PCA or t-SNE.
    
    Args:
        embeddings: Numpy array of embeddings
        chunk_ids: Optional list of chunk IDs for labeling
        method: 'pca' or 'tsne'
        output_path: Path to save the plot (optional)
        max_samples: Maximum number of samples to visualize
        figsize: Figure size
    """
    # Constrain the number of samples to visualize (<= 1000)
    if embeddings.shape[0] > max_samples:
        indices = np.random.choice(embeddings.shape[0], max_samples, replace=False)
        sample_embeddings = embeddings[indices]
        sample_ids = [chunk_ids[i] for i in indices] if chunk_ids else None
        title_suffix = f" (showing {max_samples}/{embeddings.shape[0]} samples)"
    else:
        sample_embeddings = embeddings
        sample_ids = chunk_ids
        title_suffix = ""
    
    # Reduce dimensionality
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        title_prefix = "PCA"
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title_prefix = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'")
    
    print(f"[DEBUG] Reducing dimensions using {method.upper()}...")
    embeddings_2d = reducer.fit_transform(sample_embeddings)
    
    plt.figure(figsize=figsize)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
    plt.title(f'{title_prefix} Visualization of Embeddings{title_suffix}')
    plt.xlabel(f'{title_prefix} Component 1')
    plt.ylabel(f'{title_prefix} Component 2')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[DEBUG] Saved 2D visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def inspect_faiss_index(
    index: IndexFlatIP,
    metadata_map: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Inspect FAISS index statistics.
    
    Args:
        index: FAISS index instance
        metadata_map: Optional metadata map
        verbose: Whether to print statistics
        
    Returns:
        Dictionary with index statistics
    """
    stats = {
        "ntotal": index.ntotal,
        "d": index.d,
        "is_trained": index.is_trained,
        "metric_type": index.metric_type,
        "metadata_count": len(metadata_map) if metadata_map else 0,
    }
    
    if verbose:
        # Map metric type to name by comparing with reference IndexFlatIP
        metric_type_val = int(stats['metric_type'])
        reference_index = IndexFlatIP(stats['d'])
        expected_metric_type = int(reference_index.metric_type)
        
        # Compare with reference to determine type
        if metric_type_val == expected_metric_type:
            metric_name = "Inner Product (Cosine when normalized)"
            is_correct = True
        else:
            # Different metric type - likely L2
            metric_name = "L2 (Euclidean Distance) or other"
            is_correct = False
        
        print("\n[DEBUG] FAISS Index Statistics:")
        print(f"  Total vectors: {stats['ntotal']} (should match embedding n_vectors)")
        print(f"  Dimension: {stats['d']} (should match embedding_dim = 384)")
        print(f"  → Total capacity: {stats['ntotal']} × {stats['d']} = {stats['ntotal'] * stats['d']:,} values")
        print(f"  Is trained: {stats['is_trained']}")
        print(f"  Metric type: {metric_type_val} ({metric_name})")
        if not is_correct:
            print(f"  ⚠️  WARNING: Expected metric_type={expected_metric_type} (IndexFlatIP), but got {metric_type_val}")
            print(f"     This index may not work correctly with cosine similarity search!")
            print(f"     Recommendation: Rebuild index using IndexFlatIP")
        if metadata_map:
            print(f"  Metadata entries: {stats['metadata_count']} (should match n_vectors)")
    
    return stats


def test_search(
    index: IndexFlatIP,
    query_embedding: np.ndarray,
    metadata_map: Dict,
    k: int = 5,
    verbose: bool = True
) -> List[Dict]:
    """
    Test search functionality and return results with metadata.
    
    Args:
        index: FAISS index instance
        query_embedding: Query embedding vector
        metadata_map: Metadata map
        k: Number of results to return
        verbose: Whether to print results
        
    Returns:
        List of result dictionaries with metadata
    """
    from models.indexing import search
    
    distances, indices = search(index, query_embedding, k)
    
    results = []
    chunk_ids = list(metadata_map.keys())
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx < len(chunk_ids):
            chunk_id = chunk_ids[idx]
            chunk_data = metadata_map.get(chunk_id, {})
            result = {
                "rank": i + 1,
                "distance": float(dist),
                "chunk_id": chunk_id,
                "text_preview": chunk_data.get("text", "")[:100] + "..." if len(chunk_data.get("text", "")) > 100 else chunk_data.get("text", ""),
                "metadata": chunk_data.get("metadata", {})
            }
            results.append(result)
    
    if verbose:
        print(f"\n[DEBUG] Search Results (top {k}):")
        for result in results:
            print(f"\n  Rank {result['rank']}:")
            print(f"    Distance: {result['distance']:.4f}")
            print(f"    Chunk ID: {result['chunk_id']}")
            print(f"    Text preview: {result['text_preview']}")
            print(f"    Course: {result['metadata'].get('course_id', 'N/A')}")
            print(f"    Doc type: {result['metadata'].get('doc_type', 'N/A')}")
    
    return results


def save_debug_report(
    index_path: Path,
    embeddings_stats: Dict,
    index_stats: Dict,
    output_path: Optional[Path] = None
) -> None:
    """
    Save a debug report with statistics.
    
    Args:
        index_path: Path to the index directory
        embeddings_stats: Embedding statistics
        index_stats: Index statistics
        output_path: Path to save report (default: index_path/debug_report.json)
    """
    if output_path is None:
        output_path = index_path / "debug_report.json"
    
    report = {
        "index_path": str(index_path),
        "embeddings": embeddings_stats,
        "index": index_stats,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[DEBUG] Saved debug report to {output_path}")


def debug_index_from_path(
    index_path: Path,
    visualize: bool = False,
    output_dir: Optional[Path] = None
) -> None:
    """
    Debug an existing index by loading it and inspecting.
    
    Args:
        index_path: Path to index directory
        visualize: Whether to create visualizations
        output_dir: Directory to save visualizations (default: index_path)
    """
    print(f"\n[DEBUG] Inspecting index at {index_path}")
    
    # Load index
    index, metadata_map = load_index(index_path)
    
    # Inspect index
    index_stats = inspect_faiss_index(index, metadata_map, verbose=True)
    
    if visualize and output_dir is None:
        output_dir = index_path
    
    if visualize:
        # Reconstruct embeddings from index (for visualization)
        # Note: FAISS doesn't provide direct access to stored vectors for IndexFlatIP
        # We can only visualize during indexing, not after loading
        print("[DEBUG] Note: Visualization requires embeddings array. "
              "Use debug_index_during_build() during indexing for full visualization.")
    
    # Save debug report
    save_debug_report(index_path, {}, index_stats, output_dir / "debug_report.json" if output_dir else None)


def debug_index_during_build(
    embeddings: np.ndarray,
    chunk_ids: List[str],
    index: IndexFlatIP,
    metadata_map: Dict,
    output_dir: Path,
    visualize: bool = True
) -> None:
    """
    Debug index during building process (when embeddings are available).
    
    Args:
        embeddings: Embedding array
        chunk_ids: List of chunk IDs
        index: FAISS index instance
        metadata_map: Metadata map
        output_dir: Directory to save debug outputs
        visualize: Whether to create visualizations
    """
    print("\n[DEBUG] Creating debug outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inspect embeddings
    embeddings_stats = inspect_embeddings(embeddings, chunk_ids, verbose=True)
    
    # Inspect index
    index_stats = inspect_faiss_index(index, metadata_map, verbose=True)
    
    # Visualizations
    if visualize:
        print("[DEBUG] Creating visualizations...")
        
        # Matrix heatmap
        visualize_embedding_matrix(
            embeddings,
            output_dir / "embedding_matrix.png",
            max_samples=100
        )
        
        # 2D PCA visualization
        visualize_embeddings_2d(
            embeddings,
            chunk_ids,
            method="pca",
            output_path=output_dir / "embeddings_pca.png",
            max_samples=1000
        )
        
        # 2D t-SNE visualization (if not too many samples)
        if embeddings.shape[0] <= 1000:
            visualize_embeddings_2d(
                embeddings,
                chunk_ids,
                method="tsne",
                output_path=output_dir / "embeddings_tsne.png",
                max_samples=1000
            )
    
    # Save debug report
    save_debug_report(
        output_dir,
        embeddings_stats,
        index_stats,
        output_dir / "debug_report.json"
    )
    
    print(f"[DEBUG] Debug outputs saved to {output_dir}")

