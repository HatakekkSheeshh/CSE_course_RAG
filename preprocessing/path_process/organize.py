from __future__ import annotations
from pathlib import Path
import shutil
import json
from typing import Optional, Dict, Any
from .pathing import ensure_layout_for_source, update_manifest

"""
Store OCR outputs into the normalized syllabus layout:

    data/<course>/syllabus/
    - ocr_json/<stem>.ocr.json
    - text/<stem>.txt                  (if plain_text provided)
    - images/<original-name>           (if copy_source_image=True)
    - annotated/<name>                 (if annotated_image is provided)

Returns a dict with the saved paths (as strings).
"""

def _unique_path(target_dir: Path, filename: str) -> Path:
    """
    Return a non-conflicting path inside target_dir.
    If filename exists, suffixes like __1, __2, ... are added.
    """
    p = target_dir / filename
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    i = 1
    while True:
        cand = target_dir / f"{stem}__{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def save_ocr_result(
    *,
    src_file: Path,
    items: Any,                                 # Typically a list[dict] from your OCR
    plain_text: Optional[str] = None,           # Optional: concatenated text
    annotated_image: Optional[Path] = None,     # Optional: a temp annotated image to store
    copy_source_image: bool = False,            # Copy original image into images/ if True
    data_root: Path = Path("data"),
    data_cvt_root: Path = Path("data_cvt"),
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:

    src_file = Path(src_file)
    layout = ensure_layout_for_source(
        src_file, data_root=data_root, data_cvt_root=data_cvt_root
    )

    base = src_file.stem

    # Optionally copy the original image to images/
    saved_image = None
    if copy_source_image and src_file.exists():
        dst_img = _unique_path(layout.images, src_file.name)
        shutil.copy2(src_file, dst_img)
        saved_image = str(dst_img)

    # Save the OCR JSON payload
    ocr_payload = {
        "meta": {
            "source_path": str(src_file),
            "course": layout.course_name,
            **(extra_meta or {}),
        },
        "items": items,
    }
    dst_json = _unique_path(layout.ocr_json, f"{base}.ocr.json")
    Path(dst_json).write_text(
        json.dumps(ocr_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save flattened text (optional)
    saved_txt = None
    if plain_text is not None:
        dst_txt = _unique_path(layout.text, f"{base}.txt")
        Path(dst_txt).write_text(plain_text, encoding="utf-8")
        saved_txt = str(dst_txt)

    # Save annotated image (optional)
    saved_anno = None
    if annotated_image and Path(annotated_image).exists():
        anno = Path(annotated_image)
        dst_anno = _unique_path(layout.annotated, anno.name)
        shutil.copy2(anno, dst_anno)
        saved_anno = str(dst_anno)

    # Update manifest for traceability
    update_manifest(
        {
            "course": layout.course_name,
            "source_path": str(src_file),
            "dest_json": str(dst_json),
            "dest_text": saved_txt,
            "dest_image": saved_image,
            "dest_annotated": saved_anno,
        },
        layout.manifest_file,
    )

    return {
        "json": str(dst_json),
        "text": saved_txt,
        "image": saved_image,
        "annotated": saved_anno,
        "course": layout.course_name,
        "syllabus_root": str(layout.syllabus_root),
    }, layout
