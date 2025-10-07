from __future__ import annotations
from pathlib import Path
import json
from dataclasses import asdict
from typing import Dict, List, Optional

from preprocessing.convert_data_to_img import convert_chapters_and_syllabus_to_images_parallel
from preprocessing.dectector import OCRTextDetector
from preprocessing.extract_syllabus import extract_syllabus

from preprocessing.organize import save_ocr_result
from preprocessing.pathing import extract_course_name

from preprocessing.overlay import write_svg_overlay

ROOT = Path(__file__).resolve().parent


# ---------- Function Helpers ----------
def _find_child_dir_casefold(parent: Path, name_cf: str) -> Optional[Path]:
    for child in parent.iterdir():
        if child.is_dir() and child.name.casefold() == name_cf:
            return child
    return None


def collect_syllabus_images_by_course(root: Path) -> Dict[Path, List[Path]]:
    """
    Return a mapping of:
        { <course_dir_path>: [list of image paths under its Syllabus/**] }
    Courses without a Syllabus/ folder are omitted.
    """
    data_cvt = root / "data_cvt"
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    out: Dict[Path, List[Path]] = {}

    if not data_cvt.is_dir():
        return out

    for course_dir in sorted([p for p in data_cvt.iterdir() if p.is_dir()]):
        syllabus_dir = _find_child_dir_casefold(course_dir, "syllabus")
        if syllabus_dir is None:
            continue

        imgs = [p for p in syllabus_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in image_exts]
        if imgs:
            out[course_dir] = sorted(imgs)
    return out


# ---------- Pipeline ----------
def main(do_convert: bool = False):
    """
    End-to-end pipeline (Syllabus-only, grouped by course):
      1) (optional) Convert PDFs -> images into data_cvt/
      2) Build {course_dir: [Syllabus images]}
      3) For each course:
           - OCR each image -> items
           - Extract syllabus from items
           - Persist into data/<course>/syllabus/{ocr_json,text,annotated?,images?}
    """
    # data_root = ROOT / "data"
    # data_cvt_root = ROOT / "data_cvt"

    # 1) Convert (optional). Converter may generate multiple outputs,
    if do_convert:
        print("Converting (may generate both Chapters and Syllabus outputs)...")
        _ = convert_chapters_and_syllabus_to_images_parallel(
            data_root=ROOT / "data_raw",
            out_root=ROOT / "data_cvt",
            dpi=220,
            fmt="png",
            overwrite=False,
            keep_temp_pdf=False,
            max_workers=3,
        )

    print("Detecting (OCR)...")
    out_dir = ROOT / "test"  # For debug outputs (JSON, annotated.png)
    # out_dir = None
    detector = OCRTextDetector(out_dir=str(out_dir))  

    img_path = ROOT / "data_cvt" / "CO1005_Introduction_to_Computing" / "Syllabus" / "slide_001.png"

    print(f"--- Processing image: {img_path.name}")
    items = detector.run(str(img_path))
    plain_text = " ".join([it.get("text", "") for it in items if it.get("text")])
    syllabus = extract_syllabus(items)
    out_paths, layout = save_ocr_result(
        src_file=img_path,
        items=items,
        plain_text=plain_text,
        annotated_image=ROOT / "scratch" / "annotated.png",                   # pass a path or PIL image if rendering bboxes is necessary
        copy_source_image=True,                 # set True to copy originals into images/
        data_root=ROOT / "test",
        data_cvt_root=ROOT / "data_cvt",
        extra_meta={"engine": "paddleocr"},
    )
    overlay_svg = write_svg_overlay(img_path, items, layout.annotated)
    print(f"     ANNOTATED: {overlay_svg}")

    parsed_dir = layout.syllabus_root / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_json = parsed_dir / f"{img_path.stem}.syllabus.json"
    parsed_json.write_text(json.dumps(asdict(syllabus), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {img_path.name}")
    print(f"     OCR JSON  : {out_paths['json']}")
    if out_paths['text']:
        print(f"     TEXT      : {out_paths['text']}")
    print(f"     PARSED    : {parsed_json}")
    print("\n")

    print(f"\nAll done.")


if __name__ == "__main__":
    main(do_convert=False)
