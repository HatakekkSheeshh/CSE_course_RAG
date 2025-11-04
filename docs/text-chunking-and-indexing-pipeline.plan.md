# Text Chunking and Vector Indexing Pipeline

## Overview
Implement a complete pipeline to chunk extracted syllabus and material text, generate embeddings, and index them in FAISS (with optional Qdrant support) for RAG queries.

## Current State
- ✅ Data extraction complete: syllabus JSON files and material JSON files with metadata
- ✅ Embedding model setup: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- ✅ FAISS index structure: IndexFlatIP(384) initialized
- ✅ DocChunk dataclass defined in `preprocessing/manifest.py`

## Implementation Plan

### 1. Chunking Module (`preprocessing/chunking.py`)
- **Text chunking strategy:**
  - Recursive/semantic chunking for syllabus (structured sections)
  - Sliding window chunking for material slides (preserve context)
  - Configurable chunk size (default: 512 tokens) and overlap (default: 50 tokens)
  - Preserve metadata from source documents
  
- **Functions:**
  - `chunk_syllabus(syllabus_json: dict) -> List[DocChunk]`
  - `chunk_material(material_json: dict) -> List[DocChunk]`
  - `chunk_text(text: str, metadata: Metadata, chunk_size: int, overlap: int) -> List[DocChunk]`

### 2. Indexing Module (`models/indexing.py`)
- **FAISS Index Manager:**
  - `create_index(embedding_dim: int) -> IndexFlatIP`
  - `add_documents(index: IndexFlatIP, embeddings: np.ndarray, ids: List[str])`
  - `search(index: IndexFlatIP, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]`
  - `save_index(index: IndexFlatIP, path: Path, metadata_map: dict)`
  - `load_index(path: Path) -> Tuple[IndexFlatIP, dict]`

- **Metadata storage:**
  - Separate JSON file mapping chunk IDs to DocChunk objects
  - Enable retrieval of full chunk metadata during search

### 3. Batch Processing Pipeline (`preprocessing/indexing_pipeline.py`)
- **Main pipeline function:**
  - Load all syllabus JSON files from `data/<course>/syllabus/parsed/`
  - Load all material JSON files from `data/<course>/material/material.json`
  - Chunk all documents
  - Generate embeddings in batches
  - Build FAISS index
  - Save index and metadata map

### 4. CLI Integration (`run.py`)
- **New command:** `--index`
  - `pipeline_index(data_root: Path, index_dir: Path, chunk_size: int, overlap: int)`
  - Processes all courses or specific course
  - Output: `data/indices/<course>/index.faiss` and `metadata.json`

### 5. Optional: Qdrant Support (`models/qdrant_indexing.py`)
- **Qdrant client wrapper:**
  - Initialize Qdrant collection
  - Insert documents with metadata
  - Search functionality
  - Make it optional (only if qdrant-client installed)

### 6. Dependencies Update (`requirements.txt`)
- Add: `tiktoken` (for token counting)
- Optional: `qdrant-client` (if Qdrant support desired)

## Files to Create/Modify

**New Files:**
- `preprocessing/chunking.py` - Text chunking logic
- `models/indexing.py` - FAISS index management
- `preprocessing/indexing_pipeline.py` - Batch processing pipeline
- `models/qdrant_indexing.py` (optional) - Qdrant support

**Modify:**
- `run.py` - Add `--index` command and pipeline
- `requirements.txt` - Add tiktoken
- `models/embedding.py` - Fix import (IndexFlatIP from faiss)
- `preprocessing/manifest.py` - Enhance DocChunk if needed

## Data Flow
```
data/<course>/syllabus/parsed/*.syllabus.json
data/<course>/material/material.json
    ↓
[Chunking Module]
    ↓
List[DocChunk]
    ↓
[Embedding Module]
    ↓
List[Embeddings]
    ↓
[FAISS Index]
    ↓
data/indices/<course>/index.faiss
data/indices/<course>/metadata.json
```

## Testing Strategy
- Test chunking on sample syllabus and material files
- Verify metadata preservation
- Test FAISS search with sample queries
- Validate index persistence and loading

## Implementation Status

### Completed ✅
- ✅ Chunking Module (`preprocessing/chunking.py`)
- ✅ Indexing Module (`models/indexing.py`)
- ✅ Batch Processing Pipeline (`preprocessing/indexing_pipeline.py`)
- ✅ CLI Integration (`run.py` - `--index` command)
- ✅ Dependencies Updated (`requirements.txt`)
- ✅ Fixed `models/embedding.py` imports

### Usage
```bash
# Index all courses
python run.py --index

# Index a specific course
python run.py --index --only-course Digital_Systems

# Custom chunk size and overlap
python run.py --index --chunk-size 256 --chunk-overlap 25
```

The indices will be saved to `data/indices/<course>/index.faiss` with corresponding `metadata.json` files.

