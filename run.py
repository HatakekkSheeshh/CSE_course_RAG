import argparse, csv, json, math, re, subprocess, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from pdf2image import convert_from_path
from PIL import Image
from preprocessing.utils import *

# ---------- OCR (DBNet + SVTR, PaddleOCR) ----------
@dataclass
class OCRLine:
    text: str
    conf: float
    bbox: List[Tuple[float,float]]  # 4 điểm (x,y) theo PaddleOCR

def init_ocr(det_model_dir: str, rec_model_dir: str):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=False, lang='en',
        det_model_dir=".../dbnet",     
        rec_model_dir=".../svtr",     
        rec_char_dict_path=".../svtr/en_dict.txt",  
        det_limit_side_len=1920,
        rec_batch_num=32,
        use_gpu=False,                
    )
    return ocr

def run_ocr(ocr, image: Image.Image) -> List[OCRLine]:
    import numpy as np
    img = np.array(image.convert("RGB"))
    result = ocr.ocr(img, cls=False)
    lines = []
    for page in result:
        if not page: continue
        for (bbox, (txt, conf)) in page:
            lines.append(OCRLine(text=txt.strip(), conf=float(conf), bbox=bbox))
    return lines

# ---------- Hậu xử lý layout ----------
def bbox_center(b):
    xs = [p[0] for p in b] 
    ys = [p[1] for p in b]
    return (sum(xs)/4.0, sum(ys)/4.0)

def bbox_height(b):
    ys = [p[1] for p in b]
    return max(ys) - min(ys)

def sort_lines(lines: List[OCRLine]) -> List[OCRLine]:
    return sorted(lines, key=lambda l: (bbox_center(l.bbox)[1], bbox_center(l.bbox)[0]))

def group_paragraphs(lines: List[OCRLine], y_gap_ratio=0.9):
    # Group into paragraphs based on vertical distance
    if not lines: return []
    lines = sort_lines(lines)
    heights = [bbox_height(l.bbox) for l in lines]
    avg_h = (sum(heights)/len(heights)) if heights else 18.0
    paras = []
    cur = [lines[0]]
    for prev, cur_line in zip(lines, lines[1:]):
        y_prev = bbox_center(prev.bbox)[1]
        y_cur  = bbox_center(cur_line.bbox)[1]
        if (y_cur - y_prev) > y_gap_ratio * avg_h:
            paras.append(cur)
            cur = [cur_line]
        else:
            cur.append(cur_line)
    paras.append(cur)
    # Join text per paragraph
    texts = []
    for grp in paras:
        t = " ".join(l.text for l in sorted(grp, key=lambda l: bbox_center(l.bbox)[0]))
        texts.append(t)
    return texts

def detect_headings(lines: List[OCRLine], factor=1.5):
    # Heuristic: bbox with large height > factor * median -> heading.
    if not lines: return []
    hs = [bbox_height(l.bbox) for l in lines]
    hs_sorted = sorted(hs)
    med = hs_sorted[len(hs)//2]
    headings = [l for l in lines if bbox_height(l.bbox) >= factor*med and len(l.text) <= 100]
    return headings

def sectionize(lines: List[OCRLine]):
    """
    Tạo (section_path, text) đơn giản:
    - heading: coi như h1 theo thứ tự xuất hiện
    - các đoạn tiếp theo gán vào section gần nhất
    """
    sec = ("Untitled",)
    buckets = []  # [(sec_path, text)]
    heads = detect_headings(lines)
    head_y = {id(h): bbox_center(h.bbox)[1] for h in heads}
    lines_sorted = sort_lines(lines)
    buf = []
    for l in lines_sorted:
        if l in heads:
            # flush
            if buf:
                buckets.append((sec, " ".join(buf)))
                buf = []
            sec = (l.text.strip(),)
        else:
            buf.append(l.text.strip())
    if buf:
        buckets.append((sec, " ".join(buf)))
    return buckets  # [(("Title",), "para ...")]

# ---------- Chunking ----------
def chunk_pairs(pairs, chunk=420, overlap=80):
    # pairs: [(section_path, text_str)]
    win = int(chunk * 1.5)
    step = max(50, int((chunk - overlap) * 1.5))
    for sec_path, text in pairs:
        words = text.split()
        i = 0
        while i < len(words):
            piece = " ".join(words[i:i+win]).strip()
            if not piece: break
            yield sec_path, piece
            i += step

# ---------- Orchestrator ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="processed")
    ap.add_argument("--chunk", type=int, default=420)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--det_model_dir", required=True)
    ap.add_argument("--rec_model_dir", required=True)
    args = ap.parse_args()

    ROOT = Path(args.root); OUT = Path(args.out)
    INDEX = OUT / "index"; INDEX.mkdir(parents=True, exist_ok=True)

    # Init OCR
    ocr = init_ocr(args.det_model_dir, args.rec_model_dir)

    # 1) courses.csv (Syllabus cũng OCR để đồng nhất pipeline)
    rows = []
    for course_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        code, slug = course_from_dir(course_dir)
        if not code: continue
        # OCR syllabus
        meta = {"course_name": slug, "instructor": None, "ects": None, "semester": None}
        syl = find_one_syllabus(course_dir)
        if syl:
            images = file_to_images(syl, dpi=args.dpi)
            all_text = []
            for im in images:
                lines = run_ocr(ocr, im)
                all_text.extend([l.text for l in sort_lines(lines)])
            big = "\n".join(all_text)
            # bắt key-value đơn giản
            def kv(keys):
                for k in keys:
                    m = re.search(rf"{k}\s*[:\-]\s*(.+)", big, re.I)
                    if m: return m.group(1).strip()
                return None
            meta["course_name"] = kv(["Course Name","Course Title","Title"]) or slug
            meta["instructor"]  = kv(["Instructor","Lecturer","Teacher"])
            meta["ects"]        = kv(["ECTS","Credits"])
            meta["semester"]    = kv(["Semester","Term"])

        rows.append({
            "course_code": code,
            "course_slug": slug,
            "course_name": meta["course_name"],
            "instructor": meta["instructor"],
            "ects": meta["ects"],
            "semester": meta["semester"],
            "path": str(course_dir.resolve())
        })

    if rows:
        with (INDEX / "courses.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # 2) Chapter → OCR → chunks.jsonl
    for r in rows:
        code = r["course_code"]; course_dir = Path(r["path"])
        course_out = OUT / code; course_out.mkdir(parents=True, exist_ok=True)
        out_jsonl = course_out / "chunks.jsonl"

        records = []
        for ch in sorted(course_dir.glob("**/Chapter_*.*")):
            if ch.suffix.lower() not in (".pdf",".pptx"): continue
            m = re.search(r"Chapter[_\-](\d+)", ch.name, re.I)
            chapter_no = int(m.group(1)) if m else None

            images = file_to_images(ch, dpi=args.dpi)
            for idx, im in enumerate(images, start=1):
                lines = run_ocr(ocr, im)
                pairs = sectionize(lines)  # [(("Heading",), "para...")]
                for i, (sec_path, piece) in enumerate(chunk_pairs(pairs, args.chunk, args.overlap)):
                    rec = {
                        "id": f"{code}-Ch{chapter_no}-p{idx:03d}-{i:04d}",
                        "text": piece,
                        "meta": {
                            "course_code": code,
                            "course_name": r["course_name"],
                            "instructor": r["instructor"],
                            "ects": r["ects"],
                            "semester": r["semester"],
                            "chapter": chapter_no,
                            "section_path": list(sec_path),
                            "source_file": ch.name,
                            "source_path": str(ch),
                            "loc": {"pages": [idx] if ch.suffix.lower()==".pdf" else None,
                                    "slides": [idx] if ch.suffix.lower()==".pptx" else None},
                            "token_count": approx_tokens(piece)
                        }
                    }
                    records.append(rec)

        write_jsonl(out_jsonl, records)
        print(f"{course_dir.name}: {len(records)} chunks → {out_jsonl}")

if __name__ == "__main__":
    main()