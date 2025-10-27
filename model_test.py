from __future__ import annotations
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path
import json

json_path = Path("test/Introduction_to_Computing/syllabus/parsed/slide_001.syllabus.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def canonical_course_card(d: Dict) -> str:
    """ course_info """
    ci = d["course_info"]
    fmt = ci["course_format"]
    return (
        f"Course: {ci['title']} ({ci['course_id']}). "
        f"Credits: {ci['credits']}. Applied semester: {ci['applied_semester']}. "
        "Format: "
        f"lectures {fmt['lectures']}h; "
        f"labs/practices {fmt['labs_practices']}h; projects {fmt['projects']}h; "
        f"self-study {fmt['self_study']}h; "
        f"total {fmt['total_hours']}h."
    )

def canonical_assessment(a: Dict, course_id: str) -> str:
    """ assessments """
    ratio = (a.get("ratio") and f"{a['ratio']}%") or "N/A"
    duration = (a.get("duration_min") and f"{a['duration_min']} min") or "N/A"
    fmt = a.get("format") or "N/A"
    # Thêm course_id để ngữ cảnh rõ khi index nhiều học phần
    return (
        f"[{course_id}] Assessment: {a['name']}. "
        f"Ratio: {ratio}. Duration: {duration}. Format: {fmt}."
    )

def to_passages(d: Dict) -> List[Dict]:
    course_id = d["course_info"]["course_id"]
    passages = []
    passages.append({
        "id": f"{course_id}::card",
        "type": "course_card",
        "text": canonical_course_card(d),
        "metadata": d["course_info"]
    })
    for i, a in enumerate(d["assessments"]):
        passages.append({
            "id": f"{course_id}::assessment::{i}",
            "type": "assessment",
            "text": canonical_assessment(a, course_id),
            "metadata": {"course_id": course_id, **a}
        })
    return passages

# 1) Chuẩn hoá thành các passage
passages = to_passages(data)
texts = [p["text"] for p in passages]

# 2) Tạo embedding bằng SBERT
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, normalize_embeddings=True)  # shape: (N, 384)

# Kết quả (ví dụ) để lưu kèm FAISS / DB:
records = [
    {"id": p["id"], "text": p["text"], "embedding": embeddings[i], "metadata": p["metadata"]}
    for i, p in enumerate(passages)
]

print(f"Records: {records}")

import faiss
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))
# Lúc truy vấn:
q = "What is the duration and format of the final exam of CO1027?"
q_emb = model.encode([q], normalize_embeddings=True).astype(np.float32)
D, I = index.search(q_emb, k=3)
hits = [records[i] for i in I[0]]
print(hits[0]["text"])
