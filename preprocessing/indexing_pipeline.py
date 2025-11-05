"""
Batch processing pipeline for chunking and indexing documents.

Processes all syllabus and material files, generates embeddings,
and creates FAISS indices.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
from tqdm import tqdm
from typing import Tuple

from .chunking import chunk_syllabus, chunk_material
from .manifest import DocChunk
from models.indexing import (
    create_index,
    add_documents,
    save_index,
    chunks_to_metadata_map
)
from models.embedding import Embedding
from models.debug_utils import (
    inspect_embeddings,
    inspect_faiss_index,
    debug_index_during_build
)


def load_syllabus_files(data_root: Path, course_name: str) -> List[Dict[str, Any]]:
    """
    Load all syllabus JSON files for a course.
    
    Args:
        data_root: Root data directory
        course_name: Course name (e.g., "Digital_Systems")
        
    Returns:
        List of parsed syllabus JSON dictionaries
    """
    syllabus_dir = data_root / course_name / "syllabus" / "parsed"
    if not syllabus_dir.exists():
        return []
    
    syllabus_files = list(syllabus_dir.glob("*.syllabus.json"))
    syllabus_data = []
    
    for file_path in sorted(syllabus_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                syllabus_data.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return syllabus_data


def load_material_file(data_root: Path, course_name: str) -> Optional[Dict[str, Any]]:
    """
    Load material JSON file for a course.
    
    Args:
        data_root: Root data directory
        course_name: Course name
        
    Returns:
        Parsed material JSON dictionary or None
    """
    material_file = data_root / course_name / "material" / "material.json"
    if not material_file.exists():
        return None
    
    try:
        with open(material_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {material_file}: {e}")
        return None


def process_course(
    data_root: Path,
    course_name: str,
    embedding_model: Embedding,
    chunk_size: int = 512,
    overlap: int = 50
) -> Tuple[List[DocChunk], int]:
    """
    Process a single course: chunk all documents.
    
    Args:
        data_root: Root data directory
        course_name: Course name
        embedding_model: Embedding model instance
        chunk_size: Chunk size in tokens
        overlap: Overlap size in tokens
        
    Returns:
        Tuple of (list of chunks, total documents processed)
    """
    all_chunks = []
    doc_count = 0
    
    # Process syllabus files
    syllabus_files = load_syllabus_files(data_root, course_name)
    for syllabus_json in syllabus_files:
        chunks = chunk_syllabus(syllabus_json)
        all_chunks.extend(chunks)
        doc_count += 1
    
    # Process material file
    material_json = load_material_file(data_root, course_name)
    if material_json:
        chunks = chunk_material(material_json)
        all_chunks.extend(chunks)
        doc_count += 1
    
    return all_chunks, doc_count


def build_index_for_course(
    data_root: Path,
    course_name: str,
    index_dir: Path,
    embedding_model: Embedding,
    embedding_dim: int = 384,
    chunk_size: int = 512,
    overlap: int = 50,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Build FAISS index for a single course.
    
    Args:
        data_root: Root data directory
        course_name: Course name
        index_dir: Directory to save index
        embedding_model: Embedding model instance
        embedding_dim: Embedding dimension
        chunk_size: Chunk size in tokens
        overlap: Overlap size in tokens
        batch_size: Batch size for embedding generation
        
    Returns:
        Dictionary with indexing statistics
    """
    print(f"\n[INDEX] Processing course: {course_name}")
    
    # Process course documents
    chunks, doc_count = process_course(
        data_root, course_name, embedding_model, chunk_size, overlap
    )
    
    if not chunks:
        print(f"[INDEX] No chunks found for {course_name}")
        return {
            "course": course_name,
            "chunks": 0,
            "documents": doc_count,
            "status": "no_chunks"
        }
    
    print(f"[INDEX] Generated {len(chunks)} chunks from {doc_count} documents")
    
    # Create index
    index = create_index(embedding_dim)
    
    # Generate embeddings in batches
    print(f"[INDEX] Generating embeddings...")
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.id for chunk in chunks]
    
    all_embeddings = []
    for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding"):
        batch = chunk_texts[i:i + batch_size]
        batch_embeddings = embedding_model.embed_batch(batch)
        # Convert list of lists to numpy array
        if isinstance(batch_embeddings, list):
            all_embeddings.extend(batch_embeddings)
        else:
            # If it's already a numpy array
            all_embeddings.extend(batch_embeddings.tolist())
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype='float32')
    
    # Debug: Inspect embeddings
    print(f"[INDEX] Inspecting embeddings...")
    inspect_embeddings(embeddings_array, chunk_ids, verbose=True)
    
    # Add to index
    print(f"[INDEX] Adding {len(chunks)} vectors to index...")
    add_documents(index, embeddings_array, chunk_ids)
    
    # Create metadata map
    metadata_map = chunks_to_metadata_map(chunks)
    
    # Debug: Inspect index
    inspect_faiss_index(index, metadata_map, verbose=True)
    
    # Debug: Create visualizations and report
    debug_dir = index_dir / "debug"
    try:
        debug_index_during_build(
            embeddings_array,
            chunk_ids,
            index,
            metadata_map,
            debug_dir,
            visualize=True
        )
    except Exception as e:
        print(f"[INDEX] Warning: Debug visualization failed: {e}")
        print(f"[INDEX] Continuing without visualizations...")
    
    # Save index
    print(f"[INDEX] Saving index to {index_dir}...")
    save_index(index, index_dir, metadata_map)
    
    print(f"[INDEX] ✓ Index created: {len(chunks)} chunks, {index.ntotal} vectors")
    
    return {
        "course": course_name,
        "chunks": len(chunks),
        "documents": doc_count,
        "vectors": index.ntotal,
        "status": "success"
    }


def build_indices_for_all_courses(
    data_root: Path,
    index_base_dir: Path,
    embedding_model: Embedding,
    embedding_dim: int = 384,
    chunk_size: int = 512,
    overlap: int = 50,
    batch_size: int = 32,
    only_course: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Build FAISS indices for all courses.
    
    Args:
        data_root: Root data directory
        index_base_dir: Base directory for indices
        embedding_model: Embedding model instance
        embedding_dim: Embedding dimension
        chunk_size: Chunk size in tokens
        overlap: Overlap size in tokens
        batch_size: Batch size for embedding generation
        only_course: If specified, only process this course
        
    Returns:
        List of indexing statistics for each course
    """
    results = []
    
    # Find all course directories
    if only_course:
        course_dirs = [data_root / only_course] if (data_root / only_course).exists() else []
    else:
        course_dirs = [
            d for d in data_root.iterdir()
            if d.is_dir() and d.name not in ["scratch", "raw", "converted", "processed", "indices"]
        ]
    
    if not course_dirs:
        print("[INDEX] No courses found to index")
        return results
    
    print(f"[INDEX] Found {len(course_dirs)} courses to index")
    
    for course_dir in sorted(course_dirs):
        course_name = course_dir.name
        index_dir = index_base_dir / course_name
        
        try:
            stats = build_index_for_course(
                data_root,
                course_name,
                index_dir,
                embedding_model,
                embedding_dim,
                chunk_size,
                overlap,
                batch_size
            )
            results.append(stats)
        except Exception as e:
            print(f"[INDEX] Error processing {course_name}: {e}")
            results.append({
                "course": course_name,
                "status": "error",
                "error": str(e)
            })
    
    return results

