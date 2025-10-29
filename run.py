from __future__ import annotations
from pathlib import Path
import json
import argparse
from dataclasses import asdict
from typing import Dict, List, Optional, Any
import re   
from preprocessing.img_process.convert_data_to_img import convert_chapters_and_syllabus_to_images_parallel
from preprocessing.dectector import OCRTextDetector
from preprocessing.syllabus.extract_syllabus import syllabus

from preprocessing.path_process.organize import save_ocr_result
from preprocessing.path_process.pathing import extract_course_name
from preprocessing.material.extract_material import (
    extract_titles_from_items,
    save_material,
)
from preprocessing.path_process.merge_parsed import merge_folder, save_outputs

from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo 
except ImportError:
    ZoneInfo = None

# ---------------------------------------------------------------------
# Time helper
def now_iso(tz_name: str | None = "Asia/Ho_Chi_Minh") -> str:
    if tz_name and ZoneInfo is not None:
        tz = ZoneInfo(tz_name)
    else:
        tz = timezone.utc
    return datetime.now(tz).isoformat(timespec="seconds")

ROOT = Path(__file__).resolve().parent
ROOT_data = ROOT / "data"

# ---------------------------------------------------------------------
# Small helper to find child dir case-insensitively
def _find_child_dir_casefold(parent: Path, name_cf: str) -> Optional[Path]:
    for child in parent.iterdir():
        if child.is_dir() and child.name.casefold() == name_cf:
            return child
    return None

# ---------------------------------------------------------------------

# Collect syllabus images grouped by course (under data_cvt/<COURSE>/Syllabus/**)
def collect_images_by_course(root: Path, kind: str = "syllabus") -> Dict[Path, List[Path]]:
    """
    Scan the data/converted directory and collect slide images grouped by course.

    Behavior depends on `kind`:

    - kind="syllabus":
        returns { <course_dir>: [all images under Syllabus/**] }

    - kind="material":
        returns { <course_dir>: [all images under every Chapter_*/** for that course] }
    """

    data_cvt = root / "converted"

    # Allowed image extensions
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    # Final result:
    # { course_dir: [list of image Paths in order] }
    out: Dict[Path, List[Path]] = {}

    if not data_cvt.is_dir():
        return out

    def _chapter_sort_key(p: Path) -> int:
        """
        Extract numeric chapter index from folder name like 'Chapter_7' -> 7.
        If it can't be parsed, return a very large number so it goes last.
        This prevents Chapter_10 from appearing before Chapter_2.
        """
        m = re.search(r"chapter[_\-\s]*(\d+)", p.name.casefold())
        return int(m.group(1)) if m else 10**9

    # Iterate each course directory under converted/
    for course_dir in sorted([p for p in data_cvt.iterdir() if p.is_dir()]):
        if kind.casefold() == "syllabus":
            # Find "Syllabus" directory inside the course directory (case-insensitive)
            syllabus_dir = _find_child_dir_casefold(course_dir, "syllabus")
            if syllabus_dir is None:
                continue

            imgs = [
                p for p in syllabus_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in image_exts
            ]

            if imgs:
                # Sort images by filename for stable order
                imgs_sorted = sorted(imgs, key=lambda p: p.name.casefold())
                out[course_dir] = imgs_sorted

        elif kind.casefold() == "material":
            # Find all chapter folders: Chapter_0, Chapter_1, ...
            chapter_dirs = [
                d for d in course_dir.iterdir()
                if d.is_dir() and d.name.casefold().startswith("chapter")
            ]

            if not chapter_dirs:
                continue

            chapter_dirs_sorted = sorted(chapter_dirs, key=_chapter_sort_key)

            imgs_all: List[Path] = []

            for ch in chapter_dirs_sorted:
                # Collect all images inside each Chapter_X folder
                ch_imgs = [
                    p for p in ch.rglob("*")
                    if p.is_file() and p.suffix.lower() in image_exts
                ]

                ch_imgs_sorted = sorted(ch_imgs, key=lambda p: p.name.casefold())
                imgs_all.extend(ch_imgs_sorted)

            if imgs_all:
                out[course_dir] = imgs_all

        else:
            # Invalid mode -> ignore
            pass

    return out

# =========================== PIPELINES =================================

def pipeline_convert(data_raw: Path, data_cvt: Path, dpi: int = 220):
    print("Converting (may generate both Chapters and Syllabus outputs)...")
    _ = convert_chapters_and_syllabus_to_images_parallel(
        data_root=data_raw, out_root=data_cvt, dpi=dpi, fmt="png",
        overwrite=False, keep_temp_pdf=False, max_workers=3,
    )

def pipeline_ocr_and_extract(data_root: Path, data_cvt_root: Path):
    images_by_course = collect_syllabus_images_by_course(ROOT_data)  
    if not images_by_course:
        print("[DETECT] No Syllabus images found under data_cvt/<COURSE_DIR>/Syllabus/. Nothing to do.")
        return

    print("[DETECT] Detecting (OCR)")
    out_dir = ROOT_data / "scratch" 
    detector = OCRTextDetector(out_dir=str(out_dir))

    total_images = 0
    for course_dir, images in images_by_course.items():
        course_name = extract_course_name(course_dir.name)
        print(f"=== Course: {course_name}  ({course_dir}) ===")
        processed = 0
        for img_path in images:
            print(f"Processing image: {img_path.name}")
            if not img_path.exists():
                continue

            items = detector.predict(str(img_path))
            plain_text = " ".join([it.get("text", "") for it in items if it.get("text")])

            out_paths, layout = save_ocr_result(
                src_file=img_path,
                items=items,
                plain_text=plain_text,
                annotated_image=ROOT_data / "scratch" / "annotated.png",
                copy_source_image=True,
                data_root=data_root,
                data_cvt_root=data_cvt_root,
                extra_meta={"engine": "paddleocr"},
            )

            syl = syllabus(
                items,
                source_file=str(img_path),
                page_index=processed,
                language="en",
                ocr_engine="PaddleOCR 3.2",
                extractor_version="1.0.0",
                timestamp=now_iso(),
                raw_ocr_text=plain_text,
                course_name=course_name,
            )

            parsed_dir = layout.syllabus_root / "parsed"
            parsed_dir.mkdir(parents=True, exist_ok=True)
            parsed_json = parsed_dir / f"{img_path.stem}.syllabus.json"
            parsed_json.write_text(json.dumps(asdict(syl), ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[OK] {img_path.name}")
            print(f"     OCR JSON  : {out_paths['json']}")
            if out_paths['text']:
                print(f"     TEXT      : {out_paths['text']}")
            print(f"     PARSED    : {parsed_json}\n")

            processed += 1
            total_images += 1

        print(f"Done course: {course_name} | images processed: {processed}\n")

    print(f"All done. Total images processed: {total_images}")

def pipeline_merge_all(data_root: Path, out_root: Path, only_course: Optional[str] = None):
    for course_dir in sorted(p for p in (data_root).iterdir() if p.is_dir()):
        if only_course and course_dir.name != only_course:
            continue
        syllabus_root = course_dir / "syllabus"
        parsed_dirs = [d for d in [syllabus_root / "parsed", syllabus_root / "parse"] if d.exists()]
        if not parsed_dirs:
            continue

        parsed_dir = parsed_dirs[0]
        print(f"[MERGE] {course_dir.name}  <-  {parsed_dir}")

        merged = merge_folder(parsed_dir)
        if not merged:
            print(f"[SKIP] No JSON in {parsed_dir}")
            continue

        course_slug = course_dir.name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        out_dir = out_root / course_slug
        save_outputs(merged, out_dir, name=f"{course_slug}")

def pipeline_extract_material(data_root: Path):
    """
    Aggregate titles per course from syllabus OCR JSONs.

    Reads: data/<course>/syllabus/ocr_json/*.ocr.json
    Writes: data/<course>/material/material.json
    """
    images_by_course = collect_images_by_course(ROOT_data, kind="material")  
    if not images_by_course:
        print("[DETECT] No Material images found under data_cvt/<COURSE_DIR>/Material/. Nothing to do.")
        return
    """
    for course_dir in sorted(p for p in (data_root).iterdir() if p.is_dir()):
        material_dir = course_dir / "material"
        if not material_dir.exists():
            continue
        course_name = course_dir.name
        print(f"[MATERIAL] Course: {course_name}")

        all_items = []
        json_files = sorted(syllabus_ocr.glob("*.ocr.json"))
        for idx, jf in enumerate(json_files):
            try:
                payload = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            items = payload.get("items") or []
            titles = extract_titles_from_items(items, page_index=idx)
            all_items.extend(titles)

        material_dir = course_dir / "material"
        out_path = material_dir / "material.json"
        save_material(course_name, all_items, out_path)
        print(f"[MATERIAL] Saved {len(all_items)} items -> {out_path}")
    """

# ============================== CLI ====================================

def main():
    ap = argparse.ArgumentParser(description="Syllabus pipeline: Convert -> OCR/Extract -> Merge")
    ap.add_argument("--convert", action="store_true", help="Run PDF-to-image conversion into data/converted")
    ap.add_argument("--ocr", action="store_true", help="Run OCR & extraction -> data/<course>/syllabus/parsed")
    ap.add_argument("--merge", action="store_true", help="Merge parsed/*.syllabus.json -> data/processed")
    ap.add_argument("--material", action="store_true", help="Extract chapter titles -> data/<course>/material/material.json")
    ap.add_argument("--only-course", default=None, help="Process only one course (folder name under data/)")
    ap.add_argument("--data-root", default=str(ROOT_data / "data"), help="Root directory for parsed outputs (default: ./data)")
    ap.add_argument("--data-cvt-root", default=str(ROOT_data / "converted"), help="Root directory for converted images (default: ./converted)")
    ap.add_argument("--data-raw", default=str(ROOT_data / "raw"), help="Root directory for raw inputs (PDFs, etc.) (default: ./raw)")
    ap.add_argument("--out-root", default=str(ROOT_data / "processed"), help="Output directory for merged artifacts (default: ./processed)")
    ap.add_argument("--dpi", type=int, default=220, help="DPI used for PDF-to-image conversion (default: 220)")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    data_cvt_root = Path(args.data_cvt_root)
    data_raw = Path(args.data_raw)
    out_root = Path(args.out_root)

    # Ensure required directories exist; create them if missing
    for d in [data_root, data_cvt_root, data_raw, out_root]:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                print(f"[INIT] Created missing directory: {d}")
            except Exception as e:
                print(f"Error: failed to create directory {d}: {e}")
                return

    if args.convert:
        pipeline_convert(data_raw=data_raw, data_cvt=data_cvt_root, dpi=args.dpi)

    if args.ocr:
        pipeline_ocr_and_extract(data_root=data_root, data_cvt_root=data_cvt_root)

    if args.merge:
        pipeline_merge_all(data_root=data_root, out_root=out_root, only_course=args.only_course)

    if args.material:
        pipeline_extract_material(data_root=data_root)

if __name__ == "__main__":
    main()
