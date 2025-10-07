from __future__ import annotations
from pathlib import Path
import re
import json
from dataclasses import dataclass
from typing import Optional, Dict

# Patterns to extract the <course name> from a course folder under data_cvt/
# Examples matched:
#   "CO2017_Operating_Systems"      -> name = "Operating_Systems"
#   "Operating Systems (CO2017)"    -> name = "Operating_Systems"
COURSE_DIR_PATTERNS = [
    re.compile(r"^(?P<code>[A-Z]{2}\d{4,5})[ ._\-]+(?P<name>.+)$"),
    re.compile(r"^(?P<name>.+)[ _\-]*\((?P<code>[A-Z]{2}\d{4,5})\)$"),
]

def sanitize_folder_name(name: str) -> str:
    """
    Make a safe folder name:
    - remove illegal characters
    - collapse whitespace
    - replace spaces with underscores
    """
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def extract_course_name(course_dir_name: str) -> str:
    """
    Try to infer <course name> from the course directory name.
    If no pattern matches, return a sanitized version of the original folder name.
    """
    for pat in COURSE_DIR_PATTERNS:
        m = pat.match(course_dir_name)
        if m:
            return sanitize_folder_name(m.group("name"))
    return sanitize_folder_name(course_dir_name)

def find_course_root_in_datacvt(src: Path, data_cvt_root: Path) -> Optional[Path]:
    """
    Given a source path inside data_cvt/<COURSE_DIR>/..., return that <COURSE_DIR> folder.
    Returns None if src is not inside data_cvt.
    """
    src = src.resolve()
    data_cvt_root = data_cvt_root.resolve()
    try:
        rel = src.relative_to(data_cvt_root)
    except Exception:
        return None
    parts = list(rel.parts)
    return (data_cvt_root / parts[0]).resolve() if parts else None

@dataclass
class DestLayout:
    course_name: str
    root: Path            # data/<course_name>
    syllabus_root: Path   # data/<course_name>/syllabus
    images: Path          # .../images
    ocr_json: Path        # .../ocr_json
    text: Path            # .../text
    annotated: Path       # .../annotated
    manifest_file: Path   # data/_manifest_syllabus.json

def ensure_layout_for_source(
    src_path: Path,
    data_root: Path = Path("data"),
    data_cvt_root: Path = Path("data_cvt"),
) -> DestLayout:
    """
    Setup objects' paths for class DestLayout:

    - Build the target folder layout for a source file located under:
        data_cvt/<COURSE_DIR>/...
    - Creates (if missing):
        data/<course>/syllabus/{images, ocr_json, text, annotated}
    """
    course_root = find_course_root_in_datacvt(src_path, data_cvt_root)
    course_dir_name = course_root.name if course_root else src_path.parent.name
    course_name = extract_course_name(course_dir_name)

    course_root_out = data_root / course_name
    syllabus_root = course_root_out / "syllabus"

    images = syllabus_root / "images"
    ocr_json = syllabus_root / "ocr_json"
    text = syllabus_root / "text"
    annotated = syllabus_root / "annotated"

    for p in (images, ocr_json, text, annotated):
        p.mkdir(parents=True, exist_ok=True)

    manifest_file = data_root / "_manifest_syllabus.json"
    return DestLayout(
        course_name=course_name,
        root=course_root_out,
        syllabus_root=syllabus_root,
        images=images,
        ocr_json=ocr_json,
        text=text,
        annotated=annotated,
        manifest_file=manifest_file,
    )

def update_manifest(entry: Dict, manifest_file: Path) -> None:
    """
    Append a record to data/_manifest_syllabus.json (create file if not present).
    """
    items = []
    if manifest_file.exists():
        try:
            items = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            items = []
    items.append(entry)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
