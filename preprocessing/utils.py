from pathlib import Path
import argparse, csv, json, math, re, subprocess, tempfile, math, json
from pdf2image import convert_from_path
from PIL import Image
from typing import List, Tuple

# ---------- Utils ----------
def approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))  # ~4 chars/token

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def course_from_dir(p: Path):
    m = re.match(r"^(CO\d{4})_(.+)$", p.name)
    return (m.group(1), m.group(2).replace("_"," ").strip()) if m else (None, None)

def find_one_syllabus(d: Path):
    for p in sorted(d.glob("**/Syllabus.*")):
        if p.suffix.lower() in (".pdf",".pptx"):
            return p
    return None

# ---------- PPTX → PDF ----------
def pptx_to_pdf(pptx_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "soffice","--headless","--convert-to","pdf",
        "--outdir", str(out_dir), str(pptx_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_dir / (pptx_path.stem + ".pdf")

# ---------- PDF/PPTX → list[Image] ----------
def file_to_images(path: Path, dpi=220) -> List[Image.Image]:
    if path.suffix.lower()==".pdf":
        pages = convert_from_path(str(path), dpi=dpi)
        return pages
    elif path.suffix.lower()==".pptx":
        with tempfile.TemporaryDirectory() as td:
            pdf_path = pptx_to_pdf(path, Path(td))
            pages = convert_from_path(str(pdf_path), dpi=dpi)
            return pages
    else:
        return []