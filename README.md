# CSE Course RAG

Convert course documents to images, extract syllabus content with OCR, and produce structured artifacts ready for downstream RAG (Retrieval-Augmented Generation) pipelines. The project ships with a containerized environment and a Streamlit demo console to run the end‑to‑end processing steps.

---

## Highlights

- **[Convert]** PDFs/Office slides ➜ images (via `pdf2image` + headless LibreOffice).
- **[OCR + Parse]** PaddleOCR ➜ normalized items and syllabus JSON per page.
- **[Merge]** Course-level aggregation of parsed JSON for analysis or indexing.
- **[Containerized]** Reproducible env with `Dockerfile` and `docker-compose.yml`.
- **[Demo UI]** Streamlit console in `apps/` to run each step and view logs.

---

## Architecture

```mermaid
flowchart LR
  A[Raw Docs (PDF/Office)] -->|Convert| B[Images per Course]
  B -->|OCR| C[Page OCR JSON]
  C -->|Parse| D[Page Syllabus JSON]
  D -->|Merge| E[Course Outputs (processed/)]
  E -->|Explore| F[Streamlit Demo]
```

---

## Project Structure

- `Dockerfile` – Base image with Python, LibreOffice, system deps, and Python libs.
- `docker-compose.yml` – One service `app` exposing port `8000` and mounting the repo.
- `requirements.txt` – Python dependencies (OCR, CV, RAG-related libs, Streamlit).
- `run.py` – CLI entrypoint for data pipelines:
  - `pipeline_convert()` – Convert raw docs to images.
  - `pipeline_ocr_and_extract()` – OCR each image and save page-level JSON.
  - `pipeline_merge_all()` – Merge parsed JSON into course-level outputs.
- `preprocessing/` – Helpers for conversion, OCR, path organization, and merging.
- `models/` – Model loaders (WIP; `load_model.py`).
- `apps/app.py` – Streamlit demo console to run pipelines and stream logs.
- `data/` – Workspace for inputs/outputs (ignored in git).
- `docs/` – Project documents (timelines/specs).

---

## Prerequisites

- Docker Desktop (Windows with WSL2 recommended)
- Optional: NVIDIA/CUDA not required (CPU stack by default)

---

## Quick Start (Docker)

1) Build the image:
```powershell
docker compose build app
```

2) Start the dev container (detached):
```powershell
docker compose up -d app
```

3) Open a shell inside the container:
```powershell
docker compose exec app bash
```

4) Run the Streamlit demo on port 8000:
```bash
streamlit run apps/app.py --server.address 0.0.0.0 --server.port 8000
```
Then open: http://localhost:8000

---

## Data Pipelines (CLI)

The CLI lives in `run.py` and expects a `data/` workspace:

- `data/raw/` – raw inputs (PDFs, Office docs, etc.)
- `data/converted/` – generated images grouped per course
- `data/scratch/` – intermediate artifacts (annotations, temp files)
- `data/processed/` – merged outputs per course

Run any combination of steps:

```bash
# 1) Convert raw docs to images (png) into data/converted
python run.py --convert --dpi 220 --data-raw ./data/raw --data-cvt-root ./data/converted

# 2) OCR + Extract syllabus items from images into per-page JSON
python run.py --ocr --data-root ./data --data-cvt-root ./data/converted

# 3) Merge parsed syllabus JSON across pages per course
python run.py --merge --data-root ./data --out-root ./data/processed
```

Notes:
- The code detects syllabus images under `converted/<COURSE>/Syllabus/**`.
- OCR engine: PaddleOCR (see `preprocessing/dectector.py`).

Outputs are saved under `data/` by default:

- `data/converted/<COURSE>/Syllabus/` – images produced by the converter.
- `data/scratch/` – temporary artifacts and annotated images.
- `data/<course>/syllabus/parsed/*.syllabus.json` – page‑level syllabus JSON.
- `data/processed/<course>/` – merged course outputs (from the merge step).

---

## Development Workflow

- The repo is bind-mounted into the container at `/workspace` (`.:/workspace:rw`).
- Edit code on the host; changes are immediately visible in the container.
- Only rebuild the image when you change dependencies in `requirements.txt`:
  ```powershell
  docker compose build app
  docker compose up -d app
  ```
- For interactive sessions, keep STDIN/TTY enabled (already set in compose).

### Running Streamlit during development
```bash
streamlit run apps/app.py --server.address 0.0.0.0 --server.port 8000
```
If you prefer a dev-specific stack, add a `docker-compose.dev.yml` with a `command`
to start Streamlit automatically, then run with `-f docker-compose.dev.yml`.

---

## Common Commands

- Show services:
  ```powershell
  docker compose ps
  ```
- Get a shell in the app container:
  ```powershell
  docker compose exec app bash
  ```
- Install a new Python lib temporarily for testing:
  ```powershell
  docker compose exec app pip install <package>
  ```
  For reproducibility, also add it to `requirements.txt` and rebuild.

---

## Troubleshooting

- **Cannot open in browser**: Ensure the app listens on `0.0.0.0:8000` inside container
  and that `ports: ["8000:8000"]` is present. Open `http://localhost:8000` on host.
- **Slow rebuilds**: Dependencies are layer-cached by copying `requirements.txt` first
  in the `Dockerfile`. Changing only source files should not invalidate dependency layers.
- **Missing system libs for OCR/PDF**: The base image installs `poppler-utils`,
  LibreOffice, and OpenMP runtime (`libgomp1`). Check the `Dockerfile` if you extend it.

---

## Data Conventions

- **Input layout**
  - Place raw documents under `data/raw/<CourseName>/...`
  - Converter writes images to `data/converted/<CourseName>/Syllabus/`.
- **OCR item schema (simplified)**
  - Each page JSON contains a list of detected items (text, bbox, score, line order).
- **Syllabus JSON (per page)**
  - Captures page metadata: `source_file`, `page_index`, `language`, `ocr_engine`, `timestamp`, `raw_ocr_text`, `course_name`.
- **Merged outputs**
  - Aggregated per course into `data/processed/<course>/...` using `preprocessing/path_process/merge_parsed.py`.

---

## Dummy steps (scaffold)

- **[Step 0: prepare folders]**
  ```bash
  mkdir -p data/raw data/converted data/processed
  ```

- **[Step 1: add some raw inputs]**
  - Drop a couple of PDF/Office files into `data/raw/SomeCourse/`.

- **[Step 2: convert to images]**
  ```bash
  python run.py --convert --dpi 220 --data-raw ./data/raw --data-cvt-root ./data/converted
  ```

- **[Step 3: OCR + extract syllabus]**
  ```bash
  python run.py --ocr --data-root ./data --data-cvt-root ./data/converted
  ```

- **[Step 4: merge parsed outputs]**
  ```bash
  python run.py --merge --data-root ./data --out-root ./data/processed
  # or only one course
  # python run.py --merge --data-root ./data --out-root ./data/processed --only-course SomeCourse
  ```

- **[Step 5: quick model sanity-check]**
  ```python
  from models.load_model import load_model
  embed, index, ocr = load_model()
  print('embed dim:', embed.get_sentence_embedding_dimension())
  print('faiss d:', index.d)
  ```

- **[Step 6: run demo UI (Streamlit)]**
  ```bash
  streamlit run apps/app.py --server.address 0.0.0.0 --server.port 8000
  # open http://localhost:8000
  ```

- **[Docker one-liners]**
  ```powershell
  docker compose up -d app
  docker compose exec app bash
  ```

---

## Notes

- Model loader in `models/load_model.py` is currently a WIP placeholder.
- The `docs/` folder includes project documents; integrate key requirements into the
  app/pipelines as needed.

---

## Roadmap

- **[Short‑term]** Sample data bundle and example outputs for quick validation.
- **[Short‑term]** Streamlit panels to preview OCR overlays and parsed syllabus per page.
- **[Mid‑term]** Optional embedding + FAISS indexing pipeline for RAG queries.
- **[Mid‑term]** REST endpoints to trigger pipelines and fetch artifacts.

---

## License

Specify your license here (e.g., MIT) if applicable.
