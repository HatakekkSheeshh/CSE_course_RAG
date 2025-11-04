from __future__ import annotations
from pathlib import Path
import json
import argparse
from dataclasses import asdict
from typing import Dict, List, Optional, Any
import re
import numpy as np   
from preprocessing.img_process.convert_data_to_img import convert_chapters_and_syllabus_to_images_parallel
from preprocessing.dectector import OCRTextDetector
from preprocessing.syllabus.extract_syllabus import syllabus

from preprocessing.path_process.organize import save_ocr_result
from preprocessing.path_process.pathing import extract_course_name, extract_course_id 
from preprocessing.material.extract_material import material, save_material
from preprocessing.path_process.merge_parsed import merge_folder, save_outputs
from preprocessing.indexing_pipeline import build_indices_for_all_courses
from models.embedding import Embedding
from models.debug_utils import debug_index_from_path, test_search
from models.indexing import load_index

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
    images_by_course = collect_images_by_course(ROOT_data, kind="syllabus")  
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
    Aggregate chapter titles per course from slide images (Chapter_*).
    Writes: data/<course>/material/material.json
    """
    images_by_course = collect_images_by_course(ROOT_data, kind="material")
    if not images_by_course:
        print("[MATERIAL] No Chapter_* images found. Nothing to do.")
        return

    detector = OCRTextDetector(out_dir=str(data_root / "scratch"))

    for course_dir, images in images_by_course.items():
        course_name = extract_course_name(course_dir.name)
        course_id = extract_course_id(course_dir.name)

        print(f"[MATERIAL] Course: {course_name} ({course_id})")

        mats: List[Material] = []

        if Path(data_root / course_name / "material").exists():
            print("[MATERIAL] Material already exists. Skipping.")
            continue

        for idx, img_path in enumerate(images):
            if not img_path.exists():
                print("Not exist")
                continue
            print("[INDEX] {}".format(idx))

            ocr_items = detector.predict(str(img_path))

            raw_text = " ".join(
                [it.get("text", "") for it in ocr_items if it.get("text")]
            )

            mat = material(
                ocr_items,
                source_file=str(img_path),
                page_index=idx,          # index of this slide within the chapter
                course_name=course_name,
                course_id=course_id,
                language="en",
                ocr_engine="PaddleOCR 3.2",
                extractor_version="1.0.0",
                timestamp=now_iso(),
                raw_ocr_text=raw_text,
            )

            mats.append(mat)

        out_dir = data_root / course_name / "material"
        out_path = out_dir / "material.json"

        save_material(
            course_name=course_name,
            course_id=course_id,
            mats=mats,
            out_path=out_path,
        )

        print(f"[MATERIAL] Saved {len(mats)} slides -> {out_path}\n")


def pipeline_index(
    data_root: Path,
    index_dir: Path,
    chunk_size: int = 512,
    overlap: int = 50,
    embedding_dim: int = 384,
    batch_size: int = 32,
    only_course: Optional[str] = None
):
    """
    Build FAISS indices for all courses.
    
    Args:
        data_root: Root directory containing course data
        index_dir: Directory to save indices
        chunk_size: Chunk size in tokens (default: 512)
        overlap: Overlap size in tokens (default: 50)
        embedding_dim: Embedding dimension (default: 384)
        batch_size: Batch size for embedding generation (default: 32)
        only_course: If specified, only index this course
    """
    print("[INDEX] Initializing embedding model...")
    embedding_model = Embedding()
    
    print("[INDEX] Building indices...")
    results = build_indices_for_all_courses(
        data_root=data_root,
        index_base_dir=index_dir,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        only_course=only_course
    )
    
    # Print summary
    print("\n[INDEX] Summary:")
    total_chunks = sum(r.get("chunks", 0) for r in results)
    total_vectors = sum(r.get("vectors", 0) for r in results)
    successful = sum(1 for r in results if r.get("status") == "success")
    
    print(f"  Courses processed: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Total vectors: {total_vectors}")
    
    for result in results:
        status = result.get("status", "unknown")
        course = result.get("course", "unknown")
        if status == "success":
            chunks = result.get("chunks", 0)
            vectors = result.get("vectors", 0)
            print(f"  ✓ {course}: {chunks} chunks, {vectors} vectors")
        else:
            print(f"  ✗ {course}: {status}")


def pipeline_debug_index(
    index_dir: Path,
    course_name: Optional[str] = None,
    query_text: Optional[str] = None,
    k: int = 5
):
    """
    Debug existing FAISS indices.
    
    Args:
        index_dir: Directory containing indices
        course_name: Specific course to debug (if None, debug all)
        query_text: Optional query text to test search
        k: Number of search results
    """
    if course_name:
        course_dirs = [index_dir / course_name] if (index_dir / course_name).exists() else []
    else:
        course_dirs = [d for d in index_dir.iterdir() if d.is_dir() and (d / "index.faiss").exists()]
    
    if not course_dirs:
        print(f"[DEBUG] No indices found in {index_dir}")
        return
    
    for course_dir in sorted(course_dirs):
        print(f"\n[DEBUG] Debugging index: {course_dir.name}")
        try:
            debug_index_from_path(course_dir, visualize=False)
            
            # Test search if query provided
            if query_text:
                print(f"\n[DEBUG] Testing search with query: '{query_text}'")
                index, metadata_map = load_index(course_dir)
                embedding_model = Embedding()
                query_embedding = np.array(embedding_model.embed(query_text), dtype='float32')
                test_search(index, query_embedding, metadata_map, k=k, verbose=True)
        except Exception as e:
            print(f"[DEBUG] Error debugging {course_dir.name}: {e}")


# ============================== CLI ====================================

def main():
    ap = argparse.ArgumentParser(description="Syllabus pipeline: Convert -> OCR/Extract -> Merge")
    ap.add_argument("--convert", action="store_true", help="Run PDF-to-image conversion into data/converted")
    ap.add_argument("--syllabus", action="store_true", help="Run OCR & extraction -> data/<course>/syllabus/parsed")
    ap.add_argument("--merge", action="store_true", help="Merge parsed/*.syllabus.json -> data/processed")
    ap.add_argument("--material", action="store_true", help="Extract chapter titles -> data/<course>/material/material.json")
    ap.add_argument("--index", action="store_true", help="Build FAISS indices for chunked documents -> data/indices")
    ap.add_argument("--debug-index", action="store_true", help="Debug existing FAISS indices")
    ap.add_argument("--test-query", default=None, help="Test search with query text (use with --debug-index)")
    ap.add_argument("--only-course", default=None, help="Process only one course (folder name under data/)")
    ap.add_argument("--data-root", default=str(ROOT_data / "data"), help="Root directory for parsed outputs (default: ./data)")
    ap.add_argument("--data-cvt-root", default=str(ROOT_data / "converted"), help="Root directory for converted images (default: ./converted)")
    ap.add_argument("--data-raw", default=str(ROOT_data / "raw"), help="Root directory for raw inputs (PDFs, etc.) (default: ./raw)")
    ap.add_argument("--out-root", default=str(ROOT_data / "processed"), help="Output directory for merged artifacts (default: ./processed)")
    ap.add_argument("--index-dir", default=str(ROOT_data / "indices"), help="Output directory for FAISS indices (default: ./indices)")
    ap.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens for indexing (default: 512)")
    ap.add_argument("--chunk-overlap", type=int, default=50, help="Overlap size in tokens between chunks (default: 50)")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation (default: 32)")
    ap.add_argument("--dpi", type=int, default=220, help="DPI used for PDF-to-image conversion (default: 220)")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    data_cvt_root = Path(args.data_cvt_root)
    data_raw = Path(args.data_raw)
    out_root = Path(args.out_root)
    index_dir = Path(args.index_dir)

    # Ensure required directories exist; create them if missing
    dirs_to_check = [data_root, data_cvt_root, data_raw, out_root]
    if args.index:
        dirs_to_check.append(index_dir)
    
    for d in dirs_to_check:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                print(f"[INIT] Created missing directory: {d}")
            except Exception as e:
                print(f"Error: failed to create directory {d}: {e}")
                return

    if args.convert:
        pipeline_convert(data_raw=data_raw, data_cvt=data_cvt_root, dpi=args.dpi)

    if args.syllabus:
        pipeline_ocr_and_extract(data_root=data_root, data_cvt_root=data_cvt_root)

    if args.merge:
        pipeline_merge_all(data_root=data_root, out_root=out_root, only_course=args.only_course)

    if args.material:
        pipeline_extract_material(data_root=data_root)

    if args.index:
        pipeline_index(
            data_root=data_root,
            index_dir=index_dir,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            only_course=args.only_course
        )
    
    if args.debug_index:
        pipeline_debug_index(
            index_dir=index_dir,
            course_name=args.only_course,
            query_text=args.test_query,
            k=5
        )

if __name__ == "__main__":
    main()
