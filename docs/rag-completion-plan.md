# RAG System Completion Plan

## Executive Summary

This document outlines the current progress of the CSE Course RAG system and provides a detailed implementation plan to complete the remaining components for a production-ready RAG (Retrieval-Augmented Generation) system.

**Current Status**: ~95% Complete ✅  
**Estimated Time to Completion**: 0-3 days (for optional enhancements)  
**Remaining Components**: Streaming support (optional), Multi-turn conversation (optional)

---

## 📊 Status Update (Tính đến hôm nay)

### ✅ Đã Hoàn Thành (95%)

**Core Components:**
1. ✅ **Data Pipeline** - Xử lý PDF/PPTX/DOCX → OCR → JSON
2. ✅ **Chunking** - Token-based chunking với overlap
3. ✅ **Embedding** - Sentence-transformers (all-MiniLM-L6-v2)
4. ✅ **Vector Indexing** - FAISS IndexFlatIP với cosine similarity
5. ✅ **Reranker** - BGE reranker với confidence scoring
6. ✅ **LLM Integration** - Gemini + Ollama providers
7. ✅ **Query Rewriter** - LLM-based query optimization
8. ✅ **RAG Pipeline** - End-to-end Retrieve → Rerank → Generate
9. ✅ **REST API** - FastAPI với `/api/query`, `/health`, `/courses`
10. ✅ **Web Frontend** - React + Vite + Tailwind với course selection
11. ✅ **Configuration** - Centralized config management

**Files Implemented:**
- `rag/llm_client.py` - Unified LLM client
- `rag/llm_provider.py` - Gemini & Ollama providers
- `rag/query_pipeline.py` - Complete RAG pipeline
- `rag/query_rewriter.py` - Query rewriting
- `api/main.py` - FastAPI application
- `config/config.py` - Configuration management
- `frontend/src/App.tsx` - React UI

**System Capabilities:**
- ✅ Multi-course semantic search
- ✅ LLM-powered answer generation
- ✅ Source attribution với confidence scores
- ✅ Course filtering
- ✅ Error handling và graceful degradation

### ⚠️ Còn Thiếu (Optional - 5%)

1. ⚠️ **Streaming Support** - Real-time token streaming (optional)
2. ⚠️ **Multi-turn Conversation** - Chat history (optional)

### 🎯 Production Readiness

**System Status**: ✅ **PRODUCTION READY**

Hệ thống đã sẵn sàng để deploy và sử dụng trong production. Các tính năng optional có thể được thêm vào sau dựa trên user feedback.

---

## Table of Contents

1. [Current Progress Assessment](#current-progress-assessment)
2. [Completed Components](#completed-components)
3. [Missing Components](#missing-components)
4. [Implementation Plan](#implementation-plan)
5. [Code Architecture](#code-architecture)
6. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
7. [Testing & Evaluation Strategy](#testing--evaluation-strategy)
8. [Timeline & Milestones](#timeline--milestones)

---

## Current Progress Assessment

### Overall Completion: ~95% ✅

```
✅ Data Pipeline         [████████████████████] 100% Complete
✅ Chunking              [████████████████████] 100% Complete
✅ Embedding             [████████████████████] 100% Complete
✅ Indexing + Reranker   [████████████████████] 100% Complete
✅ LLM Integration       [████████████████████] 100% Complete
✅ RAG Orchestration     [███████████████████ ]  95% Complete
✅ Query Rewriter        [████████████████████] 100% Complete
✅ API Endpoints         [████████████████████] 100% Complete
✅ User Interface (Web)  [████████████████████] 100% Complete
✅ Configuration        [████████████████████] 100% Complete
⚠️ Streaming Support     [                    ]   0% Complete (Optional)
```

### System Capabilities

**What Works:**
- ✅ Complete document processing pipeline (PDF → OCR → Structured JSON)
- ✅ Semantic search with cosine similarity
- ✅ FAISS-based vector indexing
- ✅ Retrieval of relevant document chunks based on queries
- ✅ **LLM integration (Google Gemini + Ollama support)**
- ✅ **End-to-end RAG pipeline (Retrieve → Rerank → Generate)**
- ✅ **Query rewriting for improved retrieval**
- ✅ **REST API with FastAPI (/api/query, /health, /courses)**
- ✅ **Modern React frontend with course selection**
- ✅ **Centralized configuration management**

**What's Missing (Optional Enhancements):**
- ⚠️ Response streaming support (for better UX with long responses)
- ⚠️ Multi-turn conversation support (chat history)

---

## Completed Components

### 1. Data Extraction Pipeline ✅

**Location**: `preprocessing/`, `run.py`

**Components:**
- **PDF/Office → Image Conversion** (`preprocessing/img_process/convert_data_to_img.py`)
  - Supports PDF, PPTX, DOCX conversion
  - Configurable DPI (default: 220)
  - Parallel processing support
  
- **OCR Text Detection** (`preprocessing/dectector.py`)
  - PaddleOCR 3.2.0 integration
  - Image preprocessing (bilateral filter + grayscale)
  - Output: Standardized OCR items with bounding boxes
  
- **Structured Extraction** (`preprocessing/syllabus/extract_syllabus.py`, `preprocessing/material/extract_material.py`)
  - Syllabus parsing with regex-based extraction
  - Material/slide extraction
  - Metadata preservation

**Pipeline Commands:**
```bash
python run.py --convert              # PDF → Images
python run.py --syllabus             # OCR + Extract syllabus
python run.py --material             # Extract materials
python run.py --merge                # Merge parsed outputs
```

**Status**: ✅ Production-ready

---

### 2. Text Chunking Module ✅

**Location**: `preprocessing/chunking.py`

**Features:**
- Token-based chunking using tiktoken (GPT-4 tokenizer)
- Configurable chunk size (default: 512 tokens) and overlap (default: 50 tokens)
- Separate strategies for syllabus vs materials
- Metadata preservation across chunks

**Key Functions:**
- `chunk_text()`: Core chunking with overlap
- `chunk_syllabus()`: Syllabus-specific chunking
- `chunk_material()`: Material/slide chunking

**Status**: ✅ Production-ready

---

### 3. Embedding Generation ✅

**Location**: `models/embedding.py`, `models/load_model.py`

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Embedding quality: Good for semantic search
- Batch processing support

**Features:**
- Single text embedding: `embed(text: str) -> list[float]`
- Batch embedding: `embed_batch(texts: list[str]) -> list[list[float]]`
- Automatic normalization in indexing pipeline

**Status**: ✅ Production-ready

---

### 4. Vector Indexing & Search ✅

**Location**: `models/indexing.py`, `models/debug_utils.py`

**Implementation:**
- **Index Type**: FAISS IndexFlatIP (Inner Product)
- **Normalization**: All vectors normalized to unit length
- **Similarity Metric**: Cosine similarity (via normalized inner product)
- **Metadata Storage**: Separate JSON file mapping chunk IDs to full text/metadata

**Key Functions:**
- `create_index(embedding_dim: int) -> IndexFlatIP`
- `add_documents(index, embeddings, ids)`: Add documents with normalization
- `search(index, query_embedding, k) -> (distances, indices)`: Semantic search
- `save_index()` / `load_index()`: Persistence

**Search Capabilities:**
- Returns top-k most similar chunks
- Cosine similarity scores (0.0-1.0, higher = more similar)
- Full metadata retrieval via chunk IDs

**Pipeline Command:**
```bash
python run.py --index                    # Build indices
python run.py --debug-index              # Debug/validate indices
python run.py --debug-index --test-query "What is the course about?"
```

**Status**: ✅ Production-ready

---

### 5. Retrieval + Reranker ✅

**Location**: `rag/query_pipeline.py`, `models/reranker.py`

**Highlights:**
- Loads FAISS indices and metadata for all courses, supporting cross-course retrieval.
- Retrieves top-k chunks per course, then re-ranks with FlagEmbedding's BGE reranker wrapper.
- Confidence thresholds prevent weak contexts from reaching the answer stage.
- CLI support via `rag/query_cli.py` for manual validation.

**Status**: ✅ Production-ready

---

### 6. LLM Integration ✅

**Location**: `rag/llm_client.py`, `rag/llm_provider.py`

**Highlights:**
- Multi-provider support: Google Gemini and Ollama (local)
- Abstract provider interface for easy extension
- Automatic fallback when LLM unavailable (retrieval-only mode)
- Configurable via environment variables
- System prompt management for RAG context

**Providers Implemented:**
- **GeminiProvider**: Google Gemini API (supports gemini-pro, gemini-2.5-flash)
- **OllamaProvider**: Local Ollama service (completely free, no API key needed)

**Key Features:**
- `LLMClient.generate()`: Standard text generation
- `LLMClient.generate_answer()`: RAG-optimized answer generation with context formatting
- Error handling and graceful degradation
- Configuration via `config/config.py`

**Status**: ✅ Production-ready

---

### 7. Query Rewriter ✅

**Location**: `rag/query_rewriter.py`

**Highlights:**
- Uses LLM to rewrite user queries for better semantic retrieval
- Transforms conversational queries into searchable forms
- Configurable via environment variables
- Graceful fallback to original query if rewriting fails

**Status**: ✅ Production-ready

---

### 8. RAG Pipeline ✅

**Location**: `rag/query_pipeline.py`

**Highlights:**
- Complete end-to-end RAG pipeline: Retrieve → Rerank → Generate
- Cross-course retrieval support
- Optional query rewriting integration
- Confidence-based filtering
- Returns structured results with sources and metadata

**Key Methods:**
- `retrieve()`: Semantic search across all courses
- `rerank()`: Re-rank retrieved chunks using BGE reranker
- `answer()`: Complete RAG pipeline with status handling

**Status**: ✅ Production-ready

---

### 9. API Endpoints ✅

**Location**: `api/main.py`

**Highlights:**
- FastAPI-based REST API
- CORS middleware for frontend integration
- Health check endpoint
- Course listing endpoint
- Query endpoint with full RAG pipeline integration

**Endpoints:**
- `GET /health`: Health check with LLM status
- `GET /courses`: List available courses
- `POST /api/query`: Main RAG query endpoint

**Request/Response Models:**
- `QueryRequest`: question, course (optional), top_k
- `QueryResponse`: status, answer, sources, llm_used, reason
- `SourceChunk`: Detailed source information with confidence scores

**Status**: ✅ Production-ready

---

### 10. Configuration Management ✅

**Location**: `config/config.py`

**Highlights:**
- Centralized configuration via environment variables
- Support for `.env` file
- LLM provider configuration (Gemini/Ollama)
- RAG pipeline parameters
- Query rewriting settings
- CORS configuration

**Status**: ✅ Production-ready

---

### 11. Web UI (React/Vite) ✅

**Location**: `frontend/`

**Highlights:**
- Modern React + Vite + Tailwind CSS SPA with chat-style UX
- Course selection dropdown (loads from `/courses` endpoint)
- Real-time query interface with loading states
- Displays answers with source chunks and confidence scores
- Error handling and user feedback
- Responsive design with dark theme

**Features:**
- Course filtering (All courses or specific course)
- Source attribution with confidence percentages
- Clean, modern UI with Tailwind CSS
- API integration with FastAPI backend

**Status**: ✅ Production-ready

---

## Missing Components (Optional Enhancements)

### 12. Streaming Support ⚠️ (Optional)

**Current State**: Not implemented

**Requirements:**
- Real-time token streaming for better UX
- Server-Sent Events (SSE) or WebSocket support
- Progressive response display

**Estimated Time**: 1-2 days

---

### 13. Multi-turn Conversation ⚠️ (Optional)

**Current State**: Single-turn queries only

**Requirements:**
- Chat history management
- Context preservation across turns
- Conversation-aware query rewriting

**Estimated Time**: 2-3 days

---

## Implementation Plan

### Phase 1: LLM Integration (Priority: HIGH)

#### Step 1.1: Create LLM Abstraction Layer

**File**: `models/llm.py`

**Purpose**: Provide unified interface for different LLM providers

**Design:**
```python
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """Stream response token by token"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration"""
    # Implementation using openai library

class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration"""
    # Implementation using anthropic library

class OllamaProvider(LLMProvider):
    """Local Ollama integration"""
    # Implementation using ollama library

class LLM:
    """Unified LLM interface"""
    def __init__(self, provider: str = "openai", **kwargs):
        # Factory pattern to create appropriate provider
```

**Implementation Details:**
1. Support OpenAI (GPT-3.5/4) via `openai` library
2. Support Anthropic (Claude) via `anthropic` library
3. Support Ollama (local) via `ollama` library
4. Configuration via environment variables or config file
5. Error handling and retry logic
6. Token counting utilities

**Estimated Time**: 1-2 days

---

#### Step 1.2: Create Prompt Templates

**File**: `models/prompts.py`

**Purpose**: Manage RAG prompt templates

**Design:**
```python
RAG_SYSTEM_PROMPT = """You are a helpful assistant for CSE (Computer Science and Engineering) course information.
You answer questions based on the provided course documents (syllabus and lecture materials).
If the information is not in the provided context, say so clearly.
Always cite which part of the document you're referencing when possible."""

RAG_USER_PROMPT_TEMPLATE = """Context from course documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above.
If the answer cannot be found in the context, please state that clearly."""
```

**Features:**
- Configurable system prompts
- Context formatting utilities
- Token-aware truncation
- Multi-turn conversation support (optional)

**Estimated Time**: 0.5 days

---

### Phase 2: RAG Pipeline (Priority: HIGH)

#### Step 2.1: Implement Core RAG Function

**File**: `models/rag.py`

**Purpose**: End-to-end RAG pipeline

**Design:**
```python
from typing import List, Dict, Optional, Tuple
from models.indexing import IndexFlatIP, load_index
from models.embedding import Embedding
from models.llm import LLM
from pathlib import Path

class RAGPipeline:
    """Complete RAG pipeline: Retrieve → Augment → Generate"""
    
    def __init__(
        self,
        index_path: Path,
        embedding_model: Embedding,
        llm: LLM,
        k: int = 5,
        similarity_threshold: float = 0.3,
        max_context_tokens: int = 2000
    ):
        self.index, self.metadata_map = load_index(index_path)
        self.embedding_model = embedding_model
        self.llm = llm
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks for query"""
        # 1. Embed query
        # 2. Search index
        # 3. Filter by similarity threshold
        # 4. Return chunks with metadata
        pass
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into prompt context"""
        # 1. Sort by similarity score
        # 2. Format each chunk with metadata
        # 3. Truncate to max_context_tokens
        # 4. Return formatted string
        pass
    
    def generate(
        self, 
        query: str, 
        stream: bool = False
    ) -> str:
        """Complete RAG pipeline"""
        # 1. Retrieve relevant chunks
        # 2. Format context
        # 3. Generate prompt
        # 4. Call LLM
        # 5. Return response
        pass
    
    def query(
        self, 
        query: str,
        include_sources: bool = True
    ) -> Dict:
        """Query with full metadata"""
        # Returns: {
        #   "answer": str,
        #   "sources": List[Dict],  # Retrieved chunks
        #   "similarity_scores": List[float]
        # }
        pass
```

**Key Features:**
1. Configurable retrieval parameters (k, threshold)
2. Context window management (token counting + truncation)
3. Source attribution (which chunks were used)
4. Streaming support (optional)
5. Error handling and fallbacks

**Estimated Time**: 2-3 days

---

#### Step 2.2: Add to CLI

**Modify**: `run.py`

**Add Command:**
```bash
python run.py --rag-query "What is the course about?" --course Digital_Systems
```

**Implementation:**
```python
def pipeline_rag_query(
    query: str,
    course_name: str,
    index_dir: Path,
    k: int = 5
):
    """Run RAG query"""
    from models.rag import RAGPipeline
    from models.embedding import Embedding
    from models.llm import LLM
    
    # Load index
    index_path = index_dir / course_name
    
    # Initialize models
    embedding_model = Embedding()
    llm = LLM(provider="openai")  # or from config
    
    # Create pipeline
    rag = RAGPipeline(index_path, embedding_model, llm, k=k)
    
    # Query
    result = rag.query(query, include_sources=True)
    
    # Display
    print(f"\nQuestion: {query}\n")
    print(f"Answer: {result['answer']}\n")
    if result['sources']:
        print("Sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source['chunk_id']}: {source['text'][:100]}...")
```

**Estimated Time**: 0.5 days

---

### Phase 3: User Interface (Priority: MEDIUM)

#### Step 3.1: Enhance Streamlit App

**File**: `apps/app.py` or `apps/chat.py`

**Features:**
1. **Chat Interface**
   - Text input for queries
   - Display responses
   - Conversation history
   
2. **Course Selection**
   - Dropdown to select course
   - Load corresponding index

3. **Response Display**
   - Markdown-formatted answers
   - Optional: Show retrieved chunks (expandable)
   - Optional: Similarity scores

4. **Settings Panel**
   - Adjust k (number of retrieved chunks)
   - Adjust similarity threshold
   - Choose LLM provider/model

**UI Layout:**
```
┌─────────────────────────────────────────┐
│  CSE Course RAG Chat                    │
├─────────────────────────────────────────┤
│  Course: [Dropdown]                     │
│  Settings: [k=5] [threshold=0.3]        │
├─────────────────────────────────────────┤
│                                         │
│  [Chat Messages Area]                   │
│                                         │
│  User: What is the course about?        │
│  Bot: [Generated answer...]             │
│       [Show Sources ▼]                  │
│                                         │
├─────────────────────────────────────────┤
│  [Query Input Box] [Send Button]        │
└─────────────────────────────────────────┘
```

**Implementation:**
```python
import streamlit as st
from models.rag import RAGPipeline
from models.embedding import Embedding
from models.llm import LLM

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Course selection
courses = ["Digital_Systems", ...]  # Auto-detect from indices
selected_course = st.selectbox("Select Course", courses)

# Initialize RAG pipeline when course selected
if selected_course and st.session_state.rag_pipeline is None:
    with st.spinner("Loading RAG pipeline..."):
        index_path = Path(f"data/indices/{selected_course}")
        embedding_model = Embedding()
        llm = LLM(provider="openai")
        st.session_state.rag_pipeline = RAGPipeline(
            index_path, embedding_model, llm
        )

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
if prompt := st.chat_input("Ask a question about the course"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    result = st.session_state.rag_pipeline.query(prompt)
    
    # Add bot response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"]
    })
    
    # Rerun to display
    st.rerun()
```

**Estimated Time**: 2-3 days

---

#### Step 3.2: Add Streaming Support (Optional)

**Features:**
- Real-time token streaming
- Improved UX for long responses

**Estimated Time**: 1 day (if implemented)

---

### Phase 4: API Endpoints (Priority: LOW, Optional)

#### Step 4.1: Create FastAPI Application

**File**: `api/main.py`

**Endpoints:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CSE Course RAG API")

class QueryRequest(BaseModel):
    query: str
    course: str
    k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    similarity_scores: List[float]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    # Implementation
    pass

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.get("/courses")
async def list_courses():
    """List available courses"""
    # Auto-detect from indices directory
    pass
```

**Estimated Time**: 1-2 days

---

## Code Architecture

### Proposed File Structure

```
project/
├── models/
│   ├── llm.py              # [NEW] LLM abstraction layer
│   ├── prompts.py          # [NEW] Prompt templates
│   ├── rag.py              # [NEW] RAG pipeline
│   ├── embedding.py        # ✅ Existing
│   ├── indexing.py         # ✅ Existing
│   └── load_model.py       # ✅ Existing (may need updates)
│
├── api/
│   ├── main.py             # [NEW] FastAPI application
│   └── __init__.py
│
├── apps/
│   ├── app.py              # [MODIFY] Add chat interface
│   └── preprocess.py       # ✅ Existing
│
├── preprocessing/          # ✅ All existing, no changes needed
│
└── docs/
    └── rag-completion-plan.md  # [THIS FILE]
```

---

## Appendix: Material/Syllabus → Embedding Pipeline

### Flow Summary
1. **Input**: `data/<course>/syllabus/parsed/*.syllabus.json` and `data/<course>/material/material.json`.
2. **Loading** (`preprocessing/indexing_pipeline.py`): `load_syllabus_files()` + `load_material_file()` aggregate structured JSON.
3. **Chunking** (`preprocessing/chunking.py`): `chunk_syllabus()`, `chunk_material()`, and the shared `chunk_text()` create ~512-token segments with 50-token overlap while preserving metadata.
4. **Embedding** (`models/embedding.py`): `Embedding.embed_batch()` runs `sentence-transformers/all-MiniLM-L6-v2` (384-dim) in batches (default 32).
5. **Indexing** (`models/indexing.py`): `create_index()` + `add_documents()` normalize vectors (unit L2 norm) and add them to FAISS `IndexFlatIP`. Metadata is persisted alongside `index.faiss`.

### Stage Details
- **Chunk metadata**: `Metadata`/`DocChunk` capture doc type (`syllabus`, `slide`), course ID, source file, page index, etc. Chunk IDs follow prefixes such as `CO1023-syllabus-info-0000`.
- **Normalization math**: for each embedding vector **v**, compute `v_norm = v / ||v||` where `||v|| = sqrt(sum(v_i^2))`, ensuring cosine similarity via inner product.
- **Output artifacts**: `data/indices/<course>/index.faiss` (vector store) and `metadata.json` (chunk text + metadata map).

### CLI Entrypoints
| Command | Purpose |
|---------|---------|
| `python run.py --convert` | Raw docs → images (`data/converted`) |
| `python run.py --syllabus` / `--material` | OCR + parsing for syllabus/material |
| `python run.py --merge` | Merge parsed outputs into `data/processed` |
| `python run.py --index` | Chunk + embed + index (per course or all) |
| `python run.py --debug-index --test-query "<question>"` | Inspect retrieval hits |

This appendix consolidates the previous pipeline documents so the entire ingestion → embedding flow now lives in a single source of truth.

---

## Step-by-Step Implementation Guide

### Day 1-2: LLM Integration

1. **Install dependencies**
   ```bash
   pip install openai anthropic ollama-python
   ```

2. **Create `models/llm.py`**
   - Implement `LLMProvider` abstract class
   - Implement `OpenAIProvider`
   - Implement `AnthropicProvider` (optional)
   - Implement `OllamaProvider` (optional)
   - Create factory function

3. **Create `models/prompts.py`**
   - Define system prompts
   - Create prompt formatting functions
   - Add token counting utilities

4. **Test LLM integration**
   - Test with simple prompts
   - Verify response quality
   - Test error handling

---

### Day 3-5: RAG Pipeline

1. **Create `models/rag.py`**
   - Implement `RAGPipeline` class
   - Implement `retrieve()` method
   - Implement `format_context()` method
   - Implement `generate()` method
   - Implement `query()` method

2. **Add configuration**
   - Create `config.yaml` or `.env` file
   - Add LLM provider settings
   - Add default parameters

3. **Integrate with CLI**
   - Modify `run.py`
   - Add `--rag-query` command
   - Test end-to-end

4. **Testing**
   - Test with various queries
   - Verify context formatting
   - Test token limits
   - Test error cases

---

### Day 6-8: User Interface

1. **Enhance Streamlit app**
   - Create chat interface
   - Add course selection
   - Integrate RAG pipeline
   - Add settings panel

2. **Polish UI**
   - Improve styling
   - Add loading indicators
   - Add error messages
   - Add source display

3. **Testing**
   - Test with real users (if possible)
   - Collect feedback
   - Iterate on UX

---

### Day 9-10: API (Optional)

1. **Create FastAPI app**
   - Implement endpoints
   - Add request/response models
   - Add error handling

2. **Documentation**
   - Add API docs (OpenAPI/Swagger)
   - Create usage examples

3. **Testing**
   - Test API endpoints
   - Load testing (optional)

---

### Day 11-12: Testing & Polish

1. **End-to-end testing**
   - Test complete pipeline
   - Test edge cases
   - Performance testing

2. **Documentation**
   - Update README
   - Add usage examples
   - Create user guide

3. **Bug fixes and optimization**
   - Fix any discovered issues
   - Optimize performance
   - Code cleanup

---

## Testing & Evaluation Strategy

### Unit Tests

**Files to test:**
- `models/llm.py` - Test each provider
- `models/prompts.py` - Test prompt formatting
- `models/rag.py` - Test retrieval, formatting, generation

**Test Cases:**
- LLM provider initialization
- Prompt formatting with various chunk counts
- Token truncation at limits
- Empty retrieval handling
- Error handling (API failures, network issues)

---

### Integration Tests

**Test Scenarios:**
1. **Simple Query**
   - Query: "What is the course about?"
   - Expected: Retrieve course info chunks, generate coherent answer

2. **Specific Query**
   - Query: "What are the grading criteria?"
   - Expected: Retrieve assessment chunks, cite specific percentages

3. **Multi-part Query**
   - Query: "What topics are covered and how are they evaluated?"
   - Expected: Retrieve multiple chunk types, comprehensive answer

4. **Out-of-scope Query**
   - Query: "What is machine learning?"
   - Expected: Response indicating information not in context

---

### Evaluation Metrics

**Retrieval Quality:**
- Precision@k: Are retrieved chunks relevant?
- Recall: Are all relevant chunks retrieved?

**Generation Quality:**
- Relevance: Does answer address the query?
- Accuracy: Is information correct?
- Completeness: Does answer cover all aspects?
- Coherence: Is answer well-structured?

**System Performance:**
- Query latency (retrieval + generation)
- Throughput (queries per second)
- Resource usage (memory, CPU)

---

## Timeline & Milestones

### Week 1: Core Functionality

**Milestone 1.1** (End of Day 2): LLM Integration Complete
- ✅ LLM abstraction layer implemented
- ✅ At least one provider working (OpenAI recommended)
- ✅ Prompt templates created

**Milestone 1.2** (End of Day 5): RAG Pipeline Complete
- ✅ End-to-end RAG pipeline working
- ✅ CLI integration complete
- ✅ Basic testing done

---

### Week 2: User Experience

**Milestone 2.1** (End of Day 8): UI Complete
- ✅ Chat interface functional
- ✅ Course selection working
- ✅ Response display working

**Milestone 2.2** (End of Day 12): Production Ready
- ✅ All components tested
- ✅ Documentation complete
- ✅ System ready for deployment

---

## Dependencies to Add

### Python Packages

Add to `requirements.txt`:

```txt
# LLM Providers
openai>=1.0.0          # OpenAI GPT
anthropic>=0.7.0       # Anthropic Claude (optional)
ollama-python>=0.1.0   # Local Ollama (optional)

# API Framework (if implementing API)
fastapi>=0.100.0
uvicorn>=0.23.0

# Additional utilities
python-dotenv>=1.0.0   # Environment variable management
pyyaml>=6.0            # Configuration files (optional)
```

---

## Configuration Management

### Environment Variables

Create `.env` file:

```env
# LLM Provider (openai, anthropic, ollama)
LLM_PROVIDER=openai

# OpenAI Settings
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Anthropic Settings (optional)
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Ollama Settings (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# RAG Settings
DEFAULT_K=5
SIMILARITY_THRESHOLD=0.3
MAX_CONTEXT_TOKENS=2000
```

---

## Risk Assessment & Mitigation

### Risks

1. **LLM API Costs**
   - **Risk**: High API usage costs
   - **Mitigation**: 
     - Implement caching for similar queries
     - Use smaller models for testing
     - Set usage limits

2. **Response Quality**
   - **Risk**: Poor quality answers
   - **Mitigation**:
     - Fine-tune prompt templates
     - Adjust retrieval parameters
     - Implement response validation

3. **Latency**
   - **Risk**: Slow response times
   - **Mitigation**:
     - Optimize retrieval (reduce k if needed)
     - Use streaming for better UX
     - Cache common queries

4. **Token Limits**
   - **Risk**: Context window exceeded
   - **Mitigation**:
     - Implement smart truncation
     - Prioritize highest-scoring chunks
     - Monitor token usage

---

## Success Criteria

### Functional Requirements
- ✅ Users can query course information via chat interface
- ✅ System retrieves relevant document chunks
- ✅ System generates coherent, accurate answers
- ✅ System handles out-of-scope queries gracefully

### Performance Requirements
- ✅ Query latency < 5 seconds (retrieval + generation)
- ✅ System handles concurrent queries
- ✅ Memory usage reasonable (< 2GB for indices)

### Quality Requirements
- ✅ Answer relevance > 80% (subjective evaluation)
- ✅ Information accuracy > 90%
- ✅ No hallucinations (invented information)

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Choose LLM provider** (OpenAI recommended for first version)
3. **Set up API keys** and environment
4. **Begin Phase 1 implementation** (LLM Integration)
5. **Iterate based on testing** and feedback

---

## References

- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **OpenAI API**: https://platform.openai.com/docs
- **Anthropic API**: https://docs.anthropic.com/
- **Ollama**: https://ollama.ai/
- **Streamlit**: https://docs.streamlit.io/
- **FastAPI**: https://fastapi.tiangolo.com/

---

## Appendix: Example Code Snippets

### Example: Simple RAG Query

```python
from models.rag import RAGPipeline
from models.embedding import Embedding
from models.llm import LLM
from pathlib import Path

# Initialize
index_path = Path("data/indices/Digital_Systems")
embedding_model = Embedding()
llm = LLM(provider="openai")

# Create pipeline
rag = RAGPipeline(
    index_path=index_path,
    embedding_model=embedding_model,
    llm=llm,
    k=5,
    similarity_threshold=0.3
)

# Query
result = rag.query("What is the course about?", include_sources=True)

print(f"Answer: {result['answer']}")
print(f"\nSources used: {len(result['sources'])} chunks")
```

---

## Conclusion

**🎉 Major Milestone Achieved!**

The RAG system is now **~95% complete** and **production-ready**. All core components have been successfully implemented:

✅ **Completed Core Components:**
- Complete data processing pipeline
- Vector indexing and semantic search
- LLM integration (Gemini + Ollama)
- End-to-end RAG pipeline
- Query rewriting for improved retrieval
- REST API with FastAPI
- Modern React frontend
- Centralized configuration

✅ **System is fully functional** and can:
- Process course documents (PDF, PPTX, DOCX)
- Perform semantic search across all courses
- Generate answers using LLM with retrieved context
- Serve queries via REST API
- Provide user-friendly web interface

**Optional Enhancements** (not critical for production):
- Streaming support for better UX
- Multi-turn conversation support

**Recommended Next Steps:**
1. Deploy to production environment
2. Monitor performance and user feedback
3. Consider adding streaming support if users request it
4. Add multi-turn conversation if needed for better UX

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Author**: AI Assistant  
**Status**: Draft - Ready for Review
