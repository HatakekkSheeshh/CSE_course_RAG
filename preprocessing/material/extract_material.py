"""
Extract material information from OCR text items on one page.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import List, Dict, Any, Optional

from preprocessing.material.material import Material, MaterialItem

def _bbox_height(polygon: List[List[float]]) -> float:
    ys = [p[1] for p in polygon]
    return max(ys) - min(ys)


def extract_titles_from_items(
    items: List[Dict[str, Any]],
    *,
    height_quantile: float = 0.9,
    min_len: int = 3,
    page_index: Optional[int] = None,
) -> List[MaterialItem]:
    """Pick probable titles by text height heuristic.

    - Compute height for each OCR item from its polygon.
    - Use the top quantile (default 90%) height as threshold.
    - Keep items with text length >= min_len to reduce noise.
    """
    if not items:
        return []

    heights = []
    for it in items:
        poly = it.get("polygon") or []
        if isinstance(poly, list) and len(poly) >= 2:
            try:
                h = float(_bbox_height(poly))
            except Exception:
                h = 0.0
        else:
            h = 0.0
        heights.append(h)

    # robust threshold: choose a value near the upper quantile using median of top-k
    k = max(1, int(len(heights) * (1 - height_quantile)))
    top_heights = sorted(heights, reverse=True)[:k]
    thr = median(top_heights) if top_heights else 0.0

    out: List[MaterialItem] = []
    for it, h in zip(items, heights):
        text = str(it.get("text") or "").strip()
        if not text or len(text) < min_len:
            continue
        if h >= thr and thr > 0:
            out.append(
                MaterialItem(
                    title=text,
                    chapter_num=0,
                    page_index=-1 if page_index is None else page_index,
                )
            )
    return out


def save_material(course: str, items: List[MaterialItem], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = Material(course=course, items=items, raw_text=None)
    payload = {
        "schema_version": mat.schema_version,
        "metadata": mat.metadata.__dict__,
        "course": mat.course,
        "items": [
            {"title": it.title, "chapter_num": it.chapter_num, "page_index": it.page_index}
            for it in items
        ],
        "raw_text": None,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
