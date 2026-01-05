# CSE Course RAG

Retrieval-Augmented Generation system for CSE course materials. Documents are processed into FAISS indices, then exposed through a FastAPI backend and a React/Vite chat UI.

---

## System Pipeline

1. **Convert** – PDFs/Office docs ➜ page images (`preprocessing/img_process`).
2. **OCR & Parse** – PaddleOCR extracts text; syllabus/material parsers build structured JSON (`preprocessing/syllabus`, `preprocessing/material`).
3. **Chunk & Embed** – Text is chunked (`preprocessing/chunking.py`) and embedded with `sentence-transformers` (`models/embedding.py`).
4. **Index & Retrieve** – FAISS indices plus metadata maps live under `data/indices`.
5. **RAG Serving** – `rag/query_pipeline.py` retrieves + reranks chunks; `rag/llm_client.py` calls the LLM; `api/main.py` exposes `/api/query`.
6. **Chat UI** – `ui/` (React + Vite + Tailwind) calls the backend and surfaces answers + sources.

---

## Tech Stack

| Area            | Technologies |
|-----------------|--------------|
| Processing      | Python, PaddleOCR, pdf2image, LibreOffice, numpy |
| Retrieval       | sentence-transformers, FAISS, FlagEmbedding reranker |
| Backend API     | FastAPI, Pydantic, Uvicorn, Google Gemini / Ollama |
| Frontend        | React 19, Vite, TailwindCSS |
| Containerization| Docker, Docker Compose |

---

## Running the Project

### Prerequisites
- Docker Desktop (WSL2 recommended on Windows)
- LLM API key (see free options below)

### 0. Download Dataset

Before running the project, download the pre-built FAISS indices and processed data from HuggingFace:

```bash
python dataset.py
```

This will download the dataset from [`hatakekksheeshh/CSE_course_RAG`](https://huggingface.co/datasets/hatakekksheeshh/CSE_course_RAG) and set up the `data/` folder with:
- `indices/` – Pre-built FAISS indices
- `processed/` – Processed course data
- `raw/` – Raw PDF documents
- `converted/` – Converted images

### 1. Configure Environment

Create a `.env` file in the project root (copy from `.env.example`):

**Option A: Google Gemini (Free Tier - Recommended)**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here  # Get from https://makersuite.google.com/app/apikey
GEMINI_MODEL=gemini-pro
```

**Option B: Ollama (Completely Free, Local)**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2  # Install: ollama pull llama2
OLLAMA_BASE_URL=http://localhost:11434  # Use host.docker.internal:11434 when running in Docker
```

**Note for Docker users:** If running the backend in Docker and Ollama is running separately on Docker Desktop, use:
```env
OLLAMA_BASE_URL=http://host.docker.internal:11434  # Windows/Mac Docker Desktop
# OR
OLLAMA_BASE_URL=http://ollama:11434  # If Ollama is in the same docker-compose network
```

**Option C: Qwen (via Ollama - Recommended for Vietnamese/Multilingual)**
```env
LLM_PROVIDER=qwen
QWEN_MODEL=qwen2.5:3b  # Install: docker compose exec ollama ollama pull qwen2.5:3b
QWEN_BASE_URL=http://ollama:11434  # Use http://host.docker.internal:11434 if Ollama is on host
```

**Note:** Qwen models are excellent for multilingual tasks including Vietnamese. Available models:
- `qwen2.5:0.5b` - Very fast, ~400MB
- `qwen2.5:1.5b` - Fast, ~1GB
- `qwen2.5:3b` - Balanced speed/quality, ~2GB (recommended)
- `qwen2.5:7b` - High quality, ~4.5GB

**Query Rewriting** (optional but recommended):
```env
ENABLE_QUERY_REWRITING=true          # Enable query rewriting (default: true)
QUERY_REWRITER_TEMPERATURE=0.7       # LLM temperature for rewriting (default: 0.7)
QUERY_REWRITER_MAX_TOKENS=100        # Max tokens for rewritten query (default: 100)
```

### 2. Start Services
```bash
docker compose up --build
```
Services:
- `backend` – FastAPI on `http://localhost:8000`
- `frontend` – Vite dev server on `http://localhost:5173`

**Recreating Backend After Changing .env:**

If you modify the `.env` file, you need to recreate the backend container to apply the changes:

```bash
# Recreate only the backend service
docker compose up -d --force-recreate --build backend
```

Or recreate all services:
```bash
docker compose up -d --force-recreate --build
```

**Accessing Backend Container:**

To access the backend container and run CLI tasks (e.g., indexing, query testing):

```bash
docker compose exec backend bash
```

Once inside the container, you can run commands like:
```bash
# Run indexing
python run.py --index

# Test query
python -m rag.query_cli --question "What is the grading policy?"

# Run evaluation scripts
python evaluation/evaluate_rewriter_impact.py --queries evaluation/test_queries_extended.json
```

**Pulling Ollama Models:**

If you encounter "model not found" errors, you need to pull the model in the Ollama container:

```bash
# Access Ollama container
docker compose exec ollama bash

# List available models
ollama list

# Pull Qwen model (recommended: qwen2.5:3b)
ollama pull qwen2.5:3b

# Or pull other Qwen models
ollama pull qwen2.5:0.5b  # Very fast, ~400MB
ollama pull qwen2.5:1.5b  # Fast, ~1GB
ollama pull qwen2.5:7b    # High quality, ~4.5GB
```

Alternatively, pull from host machine:
```bash
docker compose exec ollama ollama pull qwen2.5:3b
```

### 3. Query
- Visit the UI at `http://localhost:5173` to chat.
- Direct API call:

  **Linux/Mac:**
  ```bash
  curl -X POST http://localhost:8000/api/query \
    -H "Content-Type: application/json" \
    -d '{"question":"What is the grading policy?"}'
  ```

  **Windows (PowerShell):**
  ```powershell
  $body = '{"question":"What is the grading policy?"}'
  Invoke-RestMethod -Uri http://localhost:8000/api/query -Method Post -ContentType "application/json" -Body $body
  ```

  See `docs/windows-api-testing.md` for more Windows options.

---

## Data / CLI Pipelines

All CLI steps are orchestrated via `run.py` and expect the `data/` workspace:

| Command | Purpose |
|---------|---------|
| `python run.py --convert` | Docs ➜ images (`data/converted`) |
| `python run.py --syllabus` / `--material` | OCR + parsing, writes per-course JSON |
| `python run.py --merge` | Merge parsed outputs into `data/processed/<course>` |
| `python run.py --index` | Chunk, embed, and build FAISS indices in `data/indices` |
| `python run.py --debug-index --test-query "<question>"` | Inspect retrieval quality |
| `python -m rag.query_cli --question "<question>"` | Query with reranking and optional rewriting |
| `python -m rag.test_query_rewriter "<question>"` | Test query rewriting functionality |
| `curl http://localhost:8000/health` | Check API health status |
| `curl http://localhost:8000/courses` | List available courses |

Populate `data/raw/<CourseName>/` with PDFs before running the pipeline.

---

## Evaluation

### Query Rewriter Impact Evaluation

Evaluate the impact of query rewriting on RAG system performance. Compares baseline (no rewriting) vs. with query rewriting:

**Metrics:**
- Retrieval Quality: Query-Chunk Semantic Similarity
- Answer Quality: Query-Answer Semantic Similarity
- Answer Faithfulness: Answer-Chunk Overlap
- Latency: Response time comparison

**Basic usage:**
```bash
python3 evaluation/evaluate_rewriter_impact.py --queries evaluation/test_queries_extended.json
```

**With custom output:**
```bash
python3 evaluation/evaluate_rewriter_impact.py \
  --queries evaluation/test_queries_extended.json \
  --output evaluation/rewriter_evaluation_results.json
```

**Arguments:**
- `--queries`: Path to test queries JSON file (default: `evaluation/test_queries_extended.json`)
- `--output`: Path to output results JSON file (default: `evaluation/rewriter_evaluation_results.json`)

### Conversation Context Memory Evaluation

Evaluate conversation context memory effectiveness with three metrics:
1. Context Utilization Rate: Percentage of follow-up queries that successfully reference previous context
2. Context Window Effectiveness: Answer quality across different conversation lengths
3. Follow-up Query Quality: Comparison of answer quality with/without context

**Basic usage:**
```bash
python3 evaluation/evaluate_conversation_context.py --conversations evaluation/test_conversations.json
```

**With custom output and threshold:**
```bash
python evaluation/evaluate_conversation_context.py \
  --conversations evaluation/test_conversations.json \
  --output evaluation/conversation_context_results.json \
  --threshold 0.3
```

**Arguments:**
- `--conversations`: Path to test conversations JSON file (default: `evaluation/test_conversations.json`)
- `--output`: Path to output results JSON file (default: `evaluation/conversation_context_results.json`)
- `--threshold`: Context utilization similarity threshold (default: `0.3`)

### Dataset Statistics

Collect dataset statistics (courses, documents, chunks, tokens):
```bash
python evaluation/collect_dataset_stats.py
```

**With custom paths:**
```bash
python evaluation/collect_dataset_stats.py \
  --data-root ./data \
  --index-dir ./data/indices
```

---

## Repository Layout

- `preprocessing/` – converters, OCR, parsing helpers.
- `models/` – embedding loader, FAISS utilities, reranker wrapper.
- `rag/` – retrieval + LLM orchestration.
- `api/` – FastAPI app exposing `/health`, `/courses`, `/api/query`.
- `ui/` – Vite/React chat frontend (Dockerfile included).
- `docs/` – design references (e.g., `docs/rag-completion-plan.md`).

---

## Additional Notes

- Backend dependencies are listed in `requirements.txt`.
- Frontend dependencies are managed via `ui/package.json`.
- Set `GEMINI_API_KEY` (for Gemini) or configure Ollama before bringing up the stack; the API will fall back to retrieval-only answers if no LLM is available.
- The legacy Streamlit console in `apps/app.py` remains for pipeline inspection but is not part of the default Docker workflow.

---

## License

MIT License © 2025 Nguyen Quoc Hieu, Ho Chi Minh City University of Technology
See the `LICENSE` file for full text.
