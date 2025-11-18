# Pipeline: Material/Syllabus → Embedding

## Overview

This document explains the complete process from loading material/syllabus JSON files to generating embeddings and building FAISS indices.

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INPUT STAGE                                                  │
│    data/<course>/syllabus/parsed/*.syllabus.json               │
│    data/<course>/material/material.json                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. LOADING STAGE                                                │
│    load_syllabus_files()                                        │
│    load_material_file()                                         │
│    Location: preprocessing/indexing_pipeline.py                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CHUNKING STAGE                                               │
│    chunk_syllabus(syllabus_json) → List[DocChunk]              │
│    chunk_material(material_json) → List[DocChunk]              │
│    Location: preprocessing/chunking.py                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. EMBEDDING STAGE                                              │
│    embed_batch(chunk_texts) → np.ndarray(n_chunks, 384)        │
│    Location: models/embedding.py                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. NORMALIZATION & INDEXING                                     │
│    normalize embeddings → add to FAISS IndexFlatIP              │
│    Location: models/indexing.py                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT STAGE                                                 │
│    data/indices/<course>/index.faiss                            │
│    data/indices/<course>/metadata.json                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Process

### Step 1: Input Files

**Syllabus Files:**
- Location: `data/<course>/syllabus/parsed/*.syllabus.json`
- Format: JSON with structure:
  ```json
  {
    "metadata": {...},
    "course_info": {...},
    "assessments": [...],
    "course_des": {...},
    "raw_ocr_text": "..."
  }
  ```

**Material Files:**
- Location: `data/<course>/material/material.json`
- Format: JSON with structure:
  ```json
  {
    "course": "...",
    "course_id": "...",
    "schema_version": "material.v1",
    "slides": [
      {
        "page_index": 0,
        "chapter_num": 0,
        "source_file": "...",
        "raw_text": "...",
        "metadata": {...}
      }
    ]
  }
  ```

---

### Step 2: Loading Files

**Function:** `load_syllabus_files()` & `load_material_file()`
**Location:** `preprocessing/indexing_pipeline.py`

```python
# Load syllabus files
def load_syllabus_files(data_root: Path, course_name: str) -> List[Dict[str, Any]]:
    syllabus_dir = data_root / course_name / "syllabus" / "parsed"
    syllabus_files = list(syllabus_dir.glob("*.syllabus.json"))
    
    syllabus_data = []
    for file_path in sorted(syllabus_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            syllabus_data.append(data)
    
    return syllabus_data

# Load material file
def load_material_file(data_root: Path, course_name: str) -> Optional[Dict[str, Any]]:
    material_file = data_root / course_name / "material" / "material.json"
    if not material_file.exists():
        return None
    
    with open(material_file, 'r', encoding='utf-8') as f:
        return json.load(f)
```

---

### Step 3: Chunking Process

**Function:** `chunk_syllabus()` & `chunk_material()`
**Location:** `preprocessing/chunking.py`

#### 3.1 Syllabus Chunking

```python
def chunk_syllabus(syllabus_json: Dict[str, Any]) -> List[DocChunk]:
    chunks = []
    
    # Extract metadata
    metadata_dict = syllabus_json.get("metadata", {})
    course_info = syllabus_json.get("course_info", {})
    course_id = course_info.get("course_id", "")
    
    base_metadata = Metadata(
        doc_type="syllabus",
        course_id=course_id,
        source_file=metadata_dict.get("source_file"),
        page_index=metadata_dict.get("page_index", 0),
        ...
    )
    
    # 1. Chunk course info (256 tokens, overlap 25)
    if course_info:
        course_info_text = f"Course: {course_info.get('title', '')} (ID: {course_id})\n..."
        chunks.extend(chunk_text(course_info_text, base_metadata, 
                                 prefix=f"{course_id}-syllabus-info",
                                 chunk_size=256, overlap=25))
    
    # 2. Chunk assessments (512 tokens, overlap 50)
    if assessments:
        assessment_text = "..."
        chunks.extend(chunk_text(assessment_text, base_metadata,
                                 prefix=f"{course_id}-syllabus-assessments",
                                 chunk_size=512, overlap=50))
    
    # 3. Chunk course description (512 tokens, overlap 50)
    if course_des:
        desc_text = "..."
        chunks.extend(chunk_text(desc_text, base_metadata,
                                 prefix=f"{course_id}-syllabus-description",
                                 chunk_size=512, overlap=50))
    
    # 4. Chunk raw OCR text (512 tokens, overlap 50)
    if raw_text:
        chunks.extend(chunk_text(raw_text, base_metadata,
                                 prefix=f"{course_id}-syllabus-raw",
                                 chunk_size=512, overlap=50))
    
    return chunks
```

**Chunk Structure:**
```python
DocChunk(
    id="CO1023-syllabus-info-0000",
    text="Course: Digital Systems (ID: CO1023)\nCredits: 3...",
    metadata=Metadata(
        doc_type="syllabus",
        course_id="CO1023",
        source_file="...",
        page_index=0,
        ...
    )
)
```

#### 3.2 Material Chunking

```python
def chunk_material(material_json: Dict[str, Any]) -> List[DocChunk]:
    chunks = []
    course_id = material_json.get("course_id", "")
    slides = material_json.get("slides", [])
    
    for slide in slides:
        # Extract slide metadata
        base_metadata = Metadata(
            doc_type="slide",
            course_id=course_id,
            source_file=slide.get("source_file"),
            page_index=slide.get("page_index", 0),
            ...
        )
        
        # Get raw text from slide
        raw_text = slide.get("raw_text", "")
        
        # Create chunk prefix
        chapter_num = slide.get("chapter_num", 0)
        page_idx = slide.get("page_index", 0)
        prefix = f"{course_id}-chapter-{chapter_num}-slide-{page_idx:03d}"
        
        # Chunk the slide text (512 tokens, overlap 50)
        slide_chunks = chunk_text(raw_text, base_metadata, prefix,
                                  chunk_size=512, overlap=50)
        
        chunks.extend(slide_chunks)
    
    return chunks
```

**Chunk Structure:**
```python
DocChunk(
    id="CO1023-chapter-1-slide-005-0000",
    text="CO1023 - Course Introduction\nGrading Policy...",
    metadata=Metadata(
        doc_type="slide",
        course_id="CO1023",
        source_file=".../slide_005.png",
        page_index=5,
        ...
    )
)
```

#### 3.3 Core Chunking Function

```python
def chunk_text(text: str, metadata: Metadata, chunk_id_prefix: str,
               chunk_size: int = 512, overlap: int = 50) -> List[DocChunk]:
    """
    Chunks text into overlapping segments.
    
    Character-based approximation:
    - chunk_size tokens ≈ chunk_size * 4 characters
    - overlap tokens ≈ overlap * 4 characters
    """
    chunks = []
    char_chunk_size = chunk_size * 4  # 512 * 4 = 2048 chars
    char_overlap = overlap * 4         # 50 * 4 = 200 chars
    
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + char_chunk_size
        chunk_text_segment = text[start:end].strip()
        
        if chunk_text_segment:
            chunk_id = f"{chunk_id_prefix}-{chunk_idx:04d}"
            chunk = DocChunk(id=chunk_id, text=chunk_text_segment, 
                           metadata=metadata)
            chunks.append(chunk)
            chunk_idx += 1
        
        # Move with overlap
        start = end - char_overlap
    
    return chunks
```

**Output:** `List[DocChunk]` - Each chunk contains:
- `id`: Unique chunk identifier
- `text`: Text content (~512 tokens)
- `metadata`: Full metadata from source document

---

### Step 4: Embedding Generation

**Function:** `embed_batch()`
**Location:** `models/embedding.py`

#### 4.1 Embedding Model

```python
class Embedding():
    def __init__(self):
        # Load sentence-transformers model
        self.model = load_model("embed")  
        # Returns: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings (chunk texts)
            
        Returns:
            List of embedding vectors, each of dimension 384
        """
        return self.model.encode(texts).tolist()
```

**Model Details:**
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Type:** Dense vector embedding
- **Purpose:** Semantic similarity search

#### 4.2 Batch Processing

```python
# In build_index_for_course()
chunk_texts = [chunk.text for chunk in chunks]  # Extract texts
chunk_ids = [chunk.id for chunk in chunks]      # Extract IDs

all_embeddings = []
batch_size = 32  # Process 32 chunks at a time

for i in tqdm(range(0, len(chunk_texts), batch_size)):
    batch = chunk_texts[i:i + batch_size]
    batch_embeddings = embedding_model.embed_batch(batch)
    all_embeddings.extend(batch_embeddings)

# Convert to numpy array
embeddings_array = np.array(all_embeddings, dtype='float32')
# Shape: (n_chunks, 384)
```

**Input Example:**
```python
chunk_texts = [
    "Course: Digital Systems (ID: CO1023)...",
    "Grading Policy\nLab: 30%\nMidterm: 20%...",
    "CO1023 - Course Introduction\nLearning Outcome..."
]
```

**Output Example:**
```python
embeddings_array = np.array([
    [0.023, -0.145, 0.678, ..., 0.234],  # Chunk 1 embedding (384 dims)
    [-0.123, 0.456, -0.789, ..., 0.567],  # Chunk 2 embedding (384 dims)
    [0.345, -0.234, 0.123, ..., -0.345]   # Chunk 3 embedding (384 dims)
])
# Shape: (3, 384)
```

---

### Step 5: Normalization & FAISS Indexing

**Function:** `add_documents()`
**Location:** `models/indexing.py`

#### 5.1 Create FAISS Index

```python
# Create empty FAISS index
index = create_index(embedding_dim=384)
# Returns: IndexFlatIP(384) - Inner Product index for cosine similarity
```

#### 5.2 Normalize Embeddings

```python
def add_documents(index: IndexFlatIP, embeddings: np.ndarray, ids: List[str]):
    """
    Normalize embeddings and add to FAISS index.
    
    Normalization is required for cosine similarity:
    - After normalization, inner product = cosine similarity
    - All vectors become unit length (L2 norm = 1)
    """
    # Calculate L2 norm for each embedding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    
    # Normalize: divide each vector by its norm
    normalized_embeddings = embeddings / norms
    
    # Add to FAISS index
    index.add(normalized_embeddings.astype('float32'))
```

**Normalization Formula:**
```
For each embedding vector v:
    norm = ||v|| = sqrt(v₁² + v₂² + ... + v₃₈₄²)
    normalized_v = v / norm
```

**Result:**
- All embeddings become unit vectors (||normalized_v|| = 1)
- Inner product between normalized vectors = cosine similarity

#### 5.3 Save Metadata Map

```python
# Create metadata map
metadata_map = chunks_to_metadata_map(chunks)
# Structure: {chunk_id: {"id": "...", "text": "...", "metadata": {...}}}

# Save index and metadata
save_index(index, index_dir, metadata_map)
```

**Metadata Map Structure:**
```json
{
  "CO1023-syllabus-info-0000": {
    "id": "CO1023-syllabus-info-0000",
    "text": "Course: Digital Systems (ID: CO1023)...",
    "metadata": {
      "doc_type": "syllabus",
      "course_id": "CO1023",
      "source_file": "...",
      "page_index": 0
    }
  },
  "CO1023-chapter-1-slide-005-0000": {
    ...
  }
}
```

---

### Step 6: Output Files

**Index File:**
- Location: `data/indices/<course>/index.faiss`
- Format: FAISS binary index file
- Contains: All normalized embedding vectors

**Metadata File:**
- Location: `data/indices/<course>/metadata.json`
- Format: JSON
- Contains: Mapping from chunk IDs to full text and metadata

---

## Complete Pipeline Function

**Entry Point:** `pipeline_index()` in `run.py`

```python
def pipeline_index(
    data_root: Path,
    index_dir: Path,
    chunk_size: int = 512,
    overlap: int = 50,
    embedding_dim: int = 384,
    batch_size: int = 32,
    only_course: Optional[str] = None
):
    # 1. Initialize embedding model
    embedding_model = Embedding()
    
    # 2. Build indices for all courses
    results = build_indices_for_all_courses(
        data_root=data_root,
        index_base_dir=index_dir,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        only_course=only_course
    )
```

**Core Processing:** `build_index_for_course()` in `preprocessing/indexing_pipeline.py`

```python
def build_index_for_course(...):
    # Step 1: Load and chunk documents
    chunks, doc_count = process_course(...)
    # Returns: List[DocChunk] with all chunks from syllabus + material
    
    # Step 2: Create FAISS index
    index = create_index(embedding_dim)
    
    # Step 3: Extract texts and generate embeddings
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.id for chunk in chunks]
    
    all_embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_embeddings = embedding_model.embed_batch(batch)
        all_embeddings.extend(batch_embeddings)
    
    embeddings_array = np.array(all_embeddings, dtype='float32')
    
    # Step 4: Normalize and add to index
    add_documents(index, embeddings_array, chunk_ids)
    
    # Step 5: Create metadata map and save
    metadata_map = chunks_to_metadata_map(chunks)
    save_index(index, index_dir, metadata_map)
```

---

## Data Transformation Summary

| Stage | Input | Output | Size |
|-------|-------|--------|------|
| **Load** | JSON files | List of JSON dicts | Variable |
| **Chunk** | JSON dicts | List[DocChunk] | ~512 tokens/chunk |
| **Embed** | List[str] | np.ndarray | (n_chunks, 384) |
| **Normalize** | np.ndarray | np.ndarray | (n_chunks, 384), unit vectors |
| **Index** | np.ndarray | FAISS IndexFlatIP | n_chunks vectors |

---

## Example: Digital_Systems Course

**Input:**
- `data/Digital_Systems/syllabus/parsed/page_0.syllabus.json`
- `data/Digital_Systems/material/material.json`

**Processing:**
1. Load files → 2 documents
2. Chunk syllabus → 15 chunks
3. Chunk material → 349 chunks
4. **Total: 364 chunks**

**Embedding:**
- Generate 364 embeddings
- Each: 384-dimensional vector
- Shape: `(364, 384)`

**Normalization:**
- All vectors normalized to unit length
- Ready for cosine similarity search

**Output:**
- `data/indices/Digital_Systems/index.faiss` (364 vectors)
- `data/indices/Digital_Systems/metadata.json` (364 entries)

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 512 | Target chunk size in tokens |
| `overlap` | 50 | Overlap between chunks (tokens) |
| `embedding_dim` | 384 | Embedding vector dimension |
| `batch_size` | 32 | Number of texts to embed at once |

---

## Usage

```bash
# Build indices for all courses
python run.py --index

# Build index for specific course
python run.py --index --only-course Digital_Systems

# Custom chunk size and overlap
python run.py --index --chunk-size 256 --chunk-overlap 25

# Custom batch size
python run.py --index --batch-size 64
```

---

## Notes

1. **Character-based chunking:** Currently uses approximation (1 token ≈ 4 chars) instead of exact tiktoken counting
2. **Normalization:** All embeddings are normalized for cosine similarity via inner product
3. **Metadata preservation:** Full metadata is preserved in separate JSON file for retrieval
4. **Batch processing:** Embeddings generated in batches for efficiency
5. **Overlap:** 50-token overlap ensures context isn't lost at chunk boundaries
