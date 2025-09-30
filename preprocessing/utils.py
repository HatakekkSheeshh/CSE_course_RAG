from pathlib import Path
import argparse, csv, json, math, re, subprocess, tempfile, math, json
from pdf2image import convert_from_path
from PIL import Image
import re
from typing import Literal, Tuple, Union, Optional, List

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


NumKind = Literal["int", "float"]

_INT_RX   = re.compile(r'^[+-]?\d+$')
_FLOAT_RX = re.compile(
    r'^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$'  # 12, 12.3, .5, 1e-3, 1.2E+5
)

def _normalize_separators(s: str, prefer_decimal: Literal[".", ","] = ".") -> str:
    s = s.strip().replace('_', '')
    s = re.sub(r'\s+', '', s)

    m = re.match(r'^([+-]?.*?)([eE][+-]?\d+)?$', s)
    if not m:
        return s
    mantissa, exponent = m.group(1), (m.group(2) or "")

    has_dot   = '.' in mantissa
    has_comma = ',' in mantissa

    def drop_thousands(text: str, sep: str) -> str:
        parts = text.split(sep)
        if len(parts) <= 1:
            return text
        if all(len(p) == 3 and p.isdigit() for p in parts[1:-1]) and parts[0].lstrip('+-').isdigit():
            return ''.join(parts)
        return text

    if has_dot and has_comma:
        # Dấu nào xuất hiện sau cùng thì coi là *thập phân*
        last_sep = mantissa.rfind('.')
        last_com = mantissa.rfind(',')
        decimal_is = '.' if last_sep > last_com else ','
        thousands_is = ',' if decimal_is == '.' else '.'
        mantissa = drop_thousands(mantissa, thousands_is)
        if decimal_is == ',':
            mantissa = mantissa.replace(',', '.')
        else:
            mantissa = mantissa.replace(',', '')  
    elif has_comma and not has_dot:
        cnt = mantissa.count(',')
        if cnt == 1:
            left, right = mantissa.split(',')
            if len(right) == 3 and right.isdigit() and left.lstrip('+-').isdigit():
                mantissa = mantissa.replace(',', '')
            else:
                mantissa = mantissa.replace(',', '.')
        else:
            mantissa = drop_thousands(mantissa, ',')
            mantissa = mantissa.replace(',', '.') if ',' in mantissa else mantissa
    elif has_dot and not has_comma:
        cnt = mantissa.count('.')
        if cnt == 1:
            left, right = mantissa.split('.')
            if len(right) == 3 and right.isdigit() and left.lstrip('+-').isdigit():
                mantissa = mantissa.replace('.', '')
        else:
            mantissa = drop_thousands(mantissa, '.')
    else:
        pass

    return mantissa + exponent

def parse_number(
    text: Union[str, int, float],
    *,
    prefer_decimal: Literal[".", ","] = ".",
    accept_scientific: bool = True,
) -> Optional[Tuple[NumKind, Union[int, float]]]:
    if text is None:
        return None

    # Trường hợp đã là số
    if isinstance(text, bool):
        return None
    if isinstance(text, int):
        return ("int", text)
    if isinstance(text, float):
        return ("float", float(text))

    if not isinstance(text, str):
        return None

    s = _normalize_separators(text, prefer_decimal=prefer_decimal)

    if _INT_RX.fullmatch(s):
        try:
            return ("int", int(s))
        except ValueError:
            return None

    if accept_scientific and _FLOAT_RX.fullmatch(s):
        try:
            v = float(s)
            if v == float("inf") or v == float("-inf") or (v != v):
                return None
            return ("float", v)
        except ValueError:
            return None

    return None
