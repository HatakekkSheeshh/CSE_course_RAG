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
OLLAMA_BASE_URL=http://localhost:11434
```

**Query Rewriting** (optional but recommended):
```env
ENABLE_QUERY_REWRITING=true          # Enable query rewriting (default: true)
QUERY_REWRITER_TEMPERATURE=0.3       # LLM temperature for rewriting (default: 0.3)
QUERY_REWRITER_MAX_TOKENS=100        # Max tokens for rewritten query (default: 100)
```

**Free LLM Options**: See `docs/free-llm-providers-guide.md` for setup instructions.

(Optional) override `RAG_DATA_DIR`, `RAG_INDEX_DIR`, etc.

### 2. Start Services
```bash
docker compose up --build
```
Services:
- `backend` – FastAPI on `http://localhost:8000`
- `frontend` – Vite dev server on `http://localhost:5173`

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