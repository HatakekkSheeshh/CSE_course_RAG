from __future__ import annotations
from pathlib import Path
import re
import json
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from preprocessing.manifest import Metadata
from preprocessing.material.material import Material


# Detect chapter number from folder names like Chapter_0, Chapter-7, etc.
_CHAPTER_RE = re.compile(r"chapter[_\-\s]*(\d+)", re.IGNORECASE)

def chapter_num_from_path(source_file: str | Path) -> int:
    """
    Infer chapter number from a path like
    .../Chapter_7/slide_012.png -> 7.
    Returns 0 if not found.
    """
    p = Path(source_file)
    for part in reversed(p.parts):
        m = _CHAPTER_RE.search(part)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                break
    return 0

def _is_noise_title(text: str) -> bool:
    """
    Decide if this text block is boilerplate / not a real slide title.

    We drop stuff like:
    - University / faculty headers
    - Department lines
    - Lecturer / Credits blocks
    - Very short junk tokens ("BK", "TP.HCM", etc.)
    """
    norm = text.strip()
    # strip non-alnum for length check
    alnum_only = re.sub(r"[^0-9A-Za-z]+", "", norm)

    # super short tokens like "BK", "TP.HCM", etc.
    if len(alnum_only) <= 2:
        return True

    lower = norm.lower()

    noise_keywords = [
        "university",
        "faculty",
        "department",
        "hochiminh city university of technology",
        "ho chi minh city university of technology",
        "hcmut",
        "bk",
        "tp.hcm",
        "tp hcm",
        "school of",
        "institute",
        "office of",
        "lecturer",    # new: drop lecturer info
        "credit",      # new: drop 'Credits: 3'
        "credits",
    ]

    for kw in noise_keywords:
        if kw in lower:
            return True

    return False

def extract_titles_from_items(
    items: List[Dict[str, Any]],
    page_index: int,
) -> List[Dict[str, Any]]:
    """
    Extract the *chapter title / course title* from the FIRST slide of a chapter.

    Rules:
    - Only run on page_index == 0. If not slide 0, return [].
    - On slide 0:
        * Find max 'size' among OCR boxes.
        * Keep only boxes with size >= 0.9 * max_size
          (i.e. "largest text on the slide", not just "kinda big").
        * Drop boilerplate via _is_noise_title.
        * Sort by index_num (visual order).
        * Deduplicate text.
    - Return a list of dicts: { "title": "...", "order_hint": int }
    """

    if page_index != 0:
        # We only consider the first slide in a chapter as having the "title".
        return []

    # 1. collect numeric sizes to understand scale
    sizes: List[float] = []
    for it in items:
        size_val = it.get("size", 0)
        if isinstance(size_val, (int, float)):
            sizes.append(size_val)

    if not sizes:
        return []

    max_size = max(sizes)

    # 2. only keep text boxes that are almost as large as the biggest text
    #    0.9 means "must be extremely large", so we skip lecturer/credits/etc.
    big_threshold = max_size * 0.9

    candidates: List[Dict[str, Any]] = []

    for it in items:
        size_val = it.get("size", 0)
        if not isinstance(size_val, (int, float)):
            continue

        if size_val >= big_threshold:
            raw_text = (it.get("text") or "").strip()
            if not raw_text:
                continue

            # remove obvious boilerplate
            if _is_noise_title(raw_text):
                continue

            order_hint = it.get("index_num", 10**9)

            candidates.append(
                {
                    "title": raw_text,
                    "order_hint": order_hint,
                }
            )

    # 3. sort by visual/read order so we keep "Course" then "Introduction"
    candidates.sort(key=lambda c: c["order_hint"])

    # 4. dedupe by the actual string
    seen = set()
    picked: List[Dict[str, Any]] = []
    for c in candidates:
        if c["title"] not in seen:
            seen.add(c["title"])
            picked.append(c)

    return picked

def material(
    items: List[Dict[str, Any]],
    *,
    source_file: str,
    page_index: int,
    course_name: str,
    course_id: str,
    language: str = "en",
    ocr_engine: Optional[str] = None,
    extractor_version: Optional[str] = None,
    timestamp: Optional[str] = None,
    raw_ocr_text: Optional[str] = None,
) -> Material:
    """
    Parse ONE SLIDE (one image) worth of OCR 'items' into a Material object.
    - Only slide 0 can produce titles (extract_titles_from_items enforces that).
    - Slides >0 will still produce a Material with empty items[], so we
      preserve metadata and raw_text for debugging / later use.
    """

    # picked = extract_titles_from_items(items, page_index=page_index)

    # ch = chapter_num_from_path(source_file)

    # Convert picked candidates into MaterialItem objects
    """
    page_items: List[MaterialItem] = [
        MaterialItem(
            title=p["title"],
            chapter_num=ch,
            page_index=page_index,
        )
        for p in picked
    ]
    """

    meta = Metadata(
        doc_type="slide",
        course_id=course_id or "",
        source_file=str(source_file),
        page_index=page_index,
        language=language,
        ocr_engine=ocr_engine,
        extractor_version=extractor_version,
        timestamp=timestamp,
    )

    return Material(
        course=course_name,
        metadata=meta,
        raw_text=raw_ocr_text,
    )


def save_material(
    course_name: str,
    course_id: str,
    mats: List[Material],
    out_path: Path,
) -> None:
    """
    Write debug-friendly JSON with per-slide info.

    Structure:
    {
      "course": "<normalized course name>",
      "course_id": "<course code>",
      "schema_version": "material.v1",
      "slides": [
        {
          "page_index": 0,
          "chapter_num": 0,
          "source_file": ".../slide_001.png",
          "metadata": { ... Metadata ... },
          "raw_text": "... full OCR text of this slide ...",
          "items": [
            { "title": "Programming Fundamentals", "chapter_num": 0, "page_index": 0 },
            { "title": "Chapter 0: Introduction", "chapter_num": 0, "page_index": 0 }
          ]
        },
        {
          "page_index": 1,
          "chapter_num": 0,
          "source_file": ".../slide_002.png",
          "metadata": { ... },
          "raw_text": "...",
          "items": []
        }
      ]
    }
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    slides_payload = []
    for m in mats:
        ch_num = chapter_num_from_path(m.metadata.source_file)

        slide_entry = {
            "page_index": m.metadata.page_index,
            "chapter_num": ch_num,
            "source_file": m.metadata.source_file,
            "metadata": asdict(m.metadata),
            "raw_text": m.raw_text,
        }
        slides_payload.append(slide_entry)

    payload = {
        "course": course_name,
        "course_id": course_id,
        "schema_version": "material.v1",
        "slides": slides_payload,
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
