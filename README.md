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
| Backend API     | FastAPI, Pydantic, Uvicorn, OpenAI SDK |
| Frontend        | React 19, Vite, TailwindCSS |
| Containerization| Docker, Docker Compose |

---

## Running the Project

### Prerequisites
- Docker Desktop (WSL2 recommended on Windows)
- OpenAI API key (or compatible LLM provider)

### 1. Configure Environment
Create a `.env` (or export variables) with at least:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
LLM_PROVIDER=openai
```
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
  ```bash
  curl -X POST http://localhost:8000/api/query \
    -H "Content-Type: application/json" \
    -d '{"question":"What is the grading policy?"}'
  ```

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
- Set `OPENAI_API_KEY` before bringing up the stack; the API will fall back to retrieval-only answers if no LLM is available.
- The legacy Streamlit console in `apps/app.py` remains for pipeline inspection but is not part of the default Docker workflow.

---

## License

MIT License © 2025 Nguyen Quoc Hieu, Ho Chi Minh City University of Technology

See the `LICENSE` file for full text.