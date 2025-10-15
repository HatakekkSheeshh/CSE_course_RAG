from __future__ import annotations
import argparse, json, re, os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# -------------------- build corpus (giữ nguyên kiểu test.py) -------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def build_corpus(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    ci = d.get("course_info", {}) or {}
    parts.append(f"Course title: {ci.get('title') or ''}")
    parts.append(f"Course ID: {ci.get('course_id') or ''}")
    parts.append(f"Credits: {ci.get('credits') or ''}")
    parts.append(f"Applied semester: {ci.get('applied_semester') or ''}")

    for comp in (d.get("assessments") or []):
        line = [
            f"Assessment: {comp.get('name') or ''}",
            f"hours={comp.get('hours')}",
            f"credits={comp.get('credits')}",
        ]
        if comp.get("ratio") is not None:
            line.append(f"ratio={comp['ratio']}%")
        et_parts = []
        for et in (comp.get("evaluation_type") or []):
            frag = et.get("name") or ""
            if et.get("ratio") is not None:
                frag += f" ({et['ratio']}%)"
            if et.get("duration_min") is not None:
                frag += f", duration={et['duration_min']} minutes"
            et_parts.append(frag)
        if et_parts:
            line.append("eval=[" + "; ".join(et_parts) + "]")
        parts.append(" | ".join(filter(None, line)))

    parts.append(d.get("raw_ocr_text") or "")
    return _norm("\n".join(parts))

# ---------------------------- chunk with overlap -------------------------
def chunk_text(text: str, chunk_words: int = 350, overlap_words: int = 60) -> List[Dict[str, Any]]:
    words = re.findall(r"\S+", text)
    chunks, i, idx = [], 0, 0
    while i < len(words):
        j = min(len(words), i + chunk_words)
        chunk = " ".join(words[i:j])
        chunks.append({"id": f"chunk_{idx}", "start": i, "end": j, "text": chunk})
        idx += 1
        if j == len(words):
            break
        i = max(j - overlap_words, i + 1)
    return chunks

# ----------------------- Sentence-Transformers + FAISS -------------------
def embed_texts(texts: List[str], model_name: str, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    # encode returns np.ndarray [N, D]
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=False)
    # L2-normalize để dùng dot-product ~ cosine
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms
    return emb.astype("float32")

def build_faiss_index(emb: np.ndarray):
    import faiss
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product ~ cosine (vì đã L2-normalize)
    index.add(emb)                  # add all vectors
    return index

def search_faiss(index, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_vec, k)  # distances (cosine sim), indices
    return D, I

# ------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chunk + Overlap + Sentence-Transformers + FAISS + Query")
    ap.add_argument("--src", required=True, help="Path to <course>.syllabus.merged.json")
    ap.add_argument("--out-dir", required=True, help="Output directory for artifacts")
    ap.add_argument("--chunk-words", type=int, default=350)
    ap.add_argument("--overlap-words", type=int, default=60)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model name (default: all-MiniLM-L6-v2)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--queries", nargs="*", default=[
        "final exam ratio",
        "midterm exam duration minutes",
        "applied semester HK202",
        "course credits",
    ])
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + build corpus + chunk
    data = json.loads(src.read_text(encoding="utf-8"))
    full_text = build_corpus(data)
    chunks = chunk_text(full_text, args.chunk_words, args.overlap_words)

    # Save chunks jsonl (tiện debug)
    chunks_jsonl = out_dir / f"{src.stem.replace('.syllabus.merged','')}.syllabus.chunks.jsonl"
    with chunks_jsonl.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print("[JSONL]", chunks_jsonl)

    # 2) Embed all chunks
    chunk_texts = [c["text"] for c in chunks]
    emb = embed_texts(chunk_texts, model_name=args.model, device=args.device, batch_size=args.batch_size)

    # 3) Build FAISS index
    index = build_faiss_index(emb)

    # 4) Persist index + mapping
    #    - .index: faiss binary
    #    - .npy  : numpy embeddings (optional)
    #    - .meta.json: chunk metadata (ids, start, end) để map kết quả -> snippet
    try:
        import faiss
        idx_path = out_dir / f"{src.stem.replace('.syllabus.merged','')}.syllabus.faiss.index"
        faiss.write_index(index, str(idx_path))
        print("[FAISS] index ->", idx_path)
    except Exception as e:
        print("[FAISS] write_index skipped:", e)

    npy_path = out_dir / f"{src.stem.replace('.syllabus.merged','')}.syllabus.embeddings.npy"
    np.save(npy_path, emb)
    print("[NPY] embeddings ->", npy_path)

    meta = [{"id": c["id"], "start": c["start"], "end": c["end"]} for c in chunks]
    meta_path = out_dir / f"{src.stem.replace('.syllabus.merged','')}.syllabus.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[META] ->", meta_path)

    # (optional) parquet cho chunks
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(chunks),
                       str(out_dir / f"{src.stem.replace('.syllabus.merged','')}.syllabus.chunks.parquet"))
    except Exception:
        pass

    # 5) Interactive queries (demo)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)

    def embed_query(q: str) -> np.ndarray:
        qv = model.encode([q], normalize_embeddings=True)  # normalize ở đây luôn
        return qv.astype("float32")

    for q in args.queries:
        qv = embed_query(q)
        D, I = search_faiss(index, qv, k=args.top_k)
        print(f"\nQ: {q}")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            ch = chunks[int(idx)]
            snip = ch["text"][:200].replace("\n", " ")
            print(f"  {rank}. score={score:.4f} | {ch['id']} | {snip}...")

if __name__ == "__main__":
    main()


"""
python3 test.py \
  --src data_processed/advanced_programming/advanced_programming.syllabus.merged.json \
  --out-dir data_processed/advanced_programming \
  --chunk-words 350 --overlap-words 60 \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu \
  --top-k 5 \
  --queries "final exam ratio" "midterm exam duration minutes" "applied semester HK202" "course credits"

"""