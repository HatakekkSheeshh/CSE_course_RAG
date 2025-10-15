# preprocessing/syllabus/extract_syllabus.py
"""
Extract structured syllabus information from OCR text items on one page.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
import unicodedata

from preprocessing.syllabus.regex import RX

from preprocessing.syllabus.syllabus import (
        EvaluationType, AssessmentComponent, CourseInfo, CourseDescription, SyllabusV1
    )

try:
    from preprocessing.manifest import Metadata
except Exception:
    class Metadata:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# ---------------------------------------------------------------------------
def records(items: List[Dict[str, Any]]):
    recs = []
    for it in items:
        xs = [p[0] for p in it["polygon"]]
        ys = [p[1] for p in it["polygon"]]
        recs.append({
            "text": it["text"] or "",
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        })
    return recs

def find_groups(recs: List[Dict[str, Any]], pattern: str) -> Optional[Tuple[str, ...]]:
    rx = re.compile(pattern, re.I)
    for r in recs:
        m = rx.search(r["text"])
        if m:
            return m.groups()
    return None

def find_rec(
    recs: List[Dict[str, Any]],
    pattern: str,
    *,
    nth: int = 0,             
    order: str = "keep",       
    per_match: bool = False,   
    flags: int = re.I,
) -> Optional[Dict[str, Any]]:
    rx = re.compile(pattern, flags)

    if order == "yx":
        iterable = sorted(recs, key=lambda r: (r.get("y", 0), r.get("x", 0)))
    else:
        iterable = recs

    if nth < 0:
        hits = []
        if per_match:
            for r in iterable:
                for m in rx.finditer(r.get("text", "") or ""):
                    hits.append(r)
        else:
            for r in iterable:
                if rx.search(r.get("text", "") or ""):
                    hits.append(r)
        return hits[nth] if hits else None

    # nth >= 0
    idx = 0
    if per_match:
        for r in iterable:
            for _ in rx.finditer(r.get("text", "") or ""):
                if idx == nth:
                    return r
                idx += 1
    else:
        for r in iterable:
            if rx.search(r.get("text", "") or ""):
                if idx == nth:
                    return r
                idx += 1
    return None

def find_text(recs, pattern: str) -> Optional[str]:
    rx = re.compile(pattern, re.I)
    for r in recs:
        m = rx.search(r["text"])
        if m:
            # ưu tiên group đặt tên 'en', rồi group(1), cuối cùng group(0)
            if hasattr(m.re, "groupindex") and "en" in m.re.groupindex:
                return m.group("en")
            return m.group(1) if m.groups() else m.group(0)
    return None

def list_matches(recs: List[Dict[str, Any]], pattern: str) -> int:
    rx = re.compile(pattern, re.I)
    hits = [(r["y"], r["x"], r["text"]) for r in recs if rx.search(r.get("text", "") or "")]
    return len(hits)

def _match_text(pattern, text, flags=re.I) -> Optional[str]:
    rx = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
    m = rx.search(text or "")
    if not m:
        return None
    # Ưu tiên nhóm đặt tên 'en', rồi group(1), rồi full match
    if hasattr(m.re, "groupindex") and "en" in m.re.groupindex:
        return m.group("en")
    return m.group(1) if m.groups() else m.group(0)

def _in_box(rec: Dict[str, Any], x_min: float, x_max: float, y_min: float, y_max: float, mode: str) -> bool:
    rxmin, rxmax = rec["x_min"], rec["x_max"]
    rymin, rymax = rec["y_min"], rec["y_max"]
    cx, cy = rec["x"], rec["y"]

    if mode == "center":
        return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)
    elif mode == "overlap":
        return not (rxmax < x_min or rxmin > x_max or rymax < y_min or rymin > y_max)
    elif mode == "inside":
        return (rxmin >= x_min and rxmax <= x_max and rymin >= y_min and rymax <= y_max)
    else:
        raise ValueError("mode must be one of: 'center', 'overlap', 'inside'")

def find_rec_in_box(
    recs: List[Dict[str, Any]],
    pattern: str,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    mode: str = "center",     # 'center' | 'overlap' | 'inside'
    flags: int = re.I,
) -> Optional[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for r in recs:
        if not _in_box(r, x_min, x_max, y_min, y_max, mode):
            continue
        mt = _match_text(pattern, r.get("text",""), flags=flags)
        if mt is None:
            continue
        """
        rr = dict(r)
        rr["_match"] = mt
        cands.append(rr)
        """
        return mt

    if not cands:
        return None

    # ưu tiên theo vị trí (trên→dưới, trái→phải)
    cands.sort(key=lambda z: (z["y"], z["x"]))
    return cands[0]

# Assessment-related regexes and helpers removed — user will reimplement assessment extraction
def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None: return None
    try:
        if isinstance(x, (int, float)): return int(round(x))
        m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", "."))
        return int(round(float(m.group(0)))) if m else None
    except Exception:
        return None

def size_of_block(block: Dict[str, Any]) -> Optional[int | float]:
    try:
        width = block.get("x_max") - block.get("x_min")
        height = block.get("y_max") - block.get("y_min")
        if width is not None and height is not None:
            return height, width
        return None
    except Exception:
        return None


def extract(items: List[Dict[str, Any]]) -> Tuple[CourseInfo, List]:
    recs = records(items)

    """ 
    1st slide
    """
    # Init
    ci = CourseInfo()
    asses: List = []

    # ---- Header ----
    title       = (find_groups(recs, RX["title"]) or (None,))[0]
    course_id   = (find_groups(recs, RX["course_id"]) or (None,))[0]
    credits_g   = find_groups(recs, RX["credits"])
    credits     = _to_int(credits_g[0]) if credits_g else None
    applied_sem = (find_groups(recs, RX["applied_sem"]) or (None,))[0]


    # print(title)
    # print(course_id)
    # print(credits)
    # print(applied_sem)
    
    if title or course_id or credits or applied_sem:
        ci = CourseInfo(
            title=title,
            course_id=course_id,
            credits=credits,
            applied_semester=applied_sem,
        )

        # ---- Assessments ----
        rows = ["lectures", "tutorial", "labs", "projects", "self_study", "others"]
        matched = list_matches(recs, RX["projects"])

        if matched < 2:
            # print("1 tables")
            lessons_rec = find_rec(recs, RX["col_lessons"])
            credits_rec = find_rec(recs, RX["col_credits"], nth=-1)
            ratio_rec = find_rec(recs, RX["col_ratio"])
            evaluation_rec = find_rec(recs, RX["col_evaluate"])
            duration_rec = find_rec(recs, RX["col_duration"])

            if lessons_rec and credits_rec and ratio_rec and evaluation_rec and duration_rec: 
                padding = 70
                x_lession_thresshold = (lessons_rec["x_min"] - padding, lessons_rec["x_max"] + padding)
                x_credits_thresshold = (credits_rec["x_min"] - padding, credits_rec["x_max"] + padding)
                x_ratio_threasshold = (ratio_rec["x_min"] - padding, ratio_rec["x_max"] + padding)
                x_evaluation_thresshold = (evaluation_rec["x_min"] - padding, evaluation_rec["x_max"] + padding)
                x_duration_thresshold = (duration_rec["x_min"] - padding, duration_rec["x_max"] + padding)

                print("Column X thresholds:")
                print("  Lessons:", x_lession_thresshold)       
                print("  Credits:", x_credits_thresshold)
                print("  Evaluation:", x_evaluation_thresshold)
                print("  Duration:", x_duration_thresshold)

                for i in range(len(rows)):
                    hours = credits = ratio = None
                    value = find_text(recs, RX[rows[i]])
                    if value and isinstance(value, str):
                        value = value.replace("(", "").replace(")", "").strip() 
                        value_rec = find_rec(recs, RX[rows[i]])
                        y_threashold = (value_rec["y_min"] - padding, value_rec["y_max"] + padding)


                        ass_key = ["ass_lessons", "ass_credits", "ass_ratio", "ass_format", "ass_duration"]
                        ass_val = [x_lession_thresshold, x_credits_thresshold, x_ratio_threasshold, x_evaluation_thresshold, x_duration_thresshold]
                        debug = ["col_lessons", "col_credits", "col_ratio", "col_evaluate", "col_duration"]
                        ass_assign = []

                        for j in range(len(ass_val)):
                            # Lessons: 6 elements
                            # Credits: 6 elements
                            # Ratio: 7 elements
                            # Evaluation: 7 elements
                            # Duration: 7 elements
                            dg = find_text(recs, RX[debug[j]]).replace("(", "").replace(")","").strip().lower()
                            # print(dg)
                            if value == "Lectures" and dg in {"ratio", "evaluation type", "duration"}:
                                # print("Lectures case")
                                
                                y_threashold_lectures = [(value_rec["y"] - 150, value_rec["y"]), (value_rec["y"] - 30, value_rec["y"] + 150)]

                                for k in range(2):
                                    all_cells = find_rec_in_box(
                                        recs,
                                        RX[ass_key[j]],
                                        x_min=ass_val[j][0], x_max=ass_val[j][1],
                                        y_min=y_threashold_lectures[k][0], y_max=y_threashold_lectures[k][1],
                                        mode="inside"
                                    )
                                    if not all_cells:
                                        all_cells = ""
                                    ass_assign.append(all_cells)

                            else:
                                all_cells = find_rec_in_box(
                                    recs,
                                    RX[ass_key[j]],
                                    x_min=ass_val[j][0], x_max=ass_val[j][1],
                                    y_min=y_threashold[0], y_max=y_threashold[1],
                                    mode="inside"
                                )
                                ass_assign.append(all_cells)

                        # print("\nAssessment:")
                        # print(ass_assign)
                        # print("\n")
                        
                        
                        hours = ass_assign[0]
                        credits = ass_assign[1]
                        eva: List[EvaluationType] = []
                        if value.lower() in {"lectures"}:
                            eva.append(EvaluationType(
                                name=str(ass_assign[4]),
                                ratio=_to_int(ass_assign[2]),
                                duration_min=_to_int(ass_assign[6]),
                            ))

                            eva.append(EvaluationType(
                                name=str(ass_assign[5]),
                                ratio=_to_int(ass_assign[3]),
                                duration_min=_to_int(ass_assign[7]),
                            ))

                        else:
                            ratio = ass_assign[2]

                    asses.append(AssessmentComponent(
                        name=str(value),
                        hours=_to_int(hours),
                        credits=_to_int(credits),
                        ratio=_to_int(ratio),
                        evaluation_type=eva
                    ))
        else:
            # print("2 tables")
            # Syllabus with 2 tables
            lessons_rec = find_rec(recs, RX["col_lessons"])
            credits_rec = find_rec(recs, RX["col_credits"], nth=1)
            ratio_rec = find_rec(recs, RX["col_ratio"], nth=-1)
            # print(ratio_rec["x_min"])
            duration_rec = find_rec(recs, RX["col_duration"])

            ass_assign = []

            if lessons_rec and credits_rec and ratio_rec and duration_rec:
                padding_x = 80
                padding_y = 50
                x_lession_thresshold = (lessons_rec["x_min"] - padding_x, lessons_rec["x_max"] + padding_x)
                x_credits_thresshold = (credits_rec["x_min"] - padding_x, credits_rec["x_max"] + padding_x)
                x_ratio_threasshold = (ratio_rec["x_min"] - padding_x, ratio_rec["x_max"] + padding_x)
                x_duration_thresshold = (duration_rec["x_min"] - padding_x, duration_rec["x_max"] + padding_x)

                # print("Column X thresholds:")
                # print("  Hours:", x_lession_thresshold)       
                # print("  Credits:", x_credits_thresshold)
                # print("  Ratio:", x_ratio_threasshold)
                # print("  Duration:", x_duration_thresshold)

                # 1st table
                rows = ["lectures", "tutorial", "labs", "projects", "self_study", "others"]
                # print("1st table")
                for i in range(len(rows)):
                    hours = credits = None
                    value = find_text(recs, RX[rows[i]])
                    # print(f"Value: {value}")
                    if value and isinstance(value, str):
                        value = value.replace("(", "").replace(")", "").strip() 
                        value_rec = find_rec(recs, RX[rows[i]])
                        y_threashold = (value_rec["y_min"] - padding_y, value_rec["y_max"] + padding_y)
                        ass_val = [x_lession_thresshold, x_credits_thresshold]
                        ass_key = ["ass_lessons", "ass_credits"]
                        ass_assign = []
                        # print(f"Y threashold: {y_threashold}")
                        for j in range(len(ass_val)):
                            all_cells = find_rec_in_box(
                                recs,
                                RX[ass_key[j]],
                                x_min=ass_val[j][0], x_max=ass_val[j][1],
                                y_min=y_threashold[0], y_max=y_threashold[1],
                                mode="inside"
                            )
                            ass_assign.append(all_cells)
                            

                    # print(f"ASS: {ass_assign}") 
                    hours = ass_assign[0]
                    credits = ass_assign[1] if len(ass_assign) > 1 else None
                    asses.append(AssessmentComponent(
                        name=str(value),
                        credits=_to_int(credits),
                        hours=_to_int(hours),
                    ))  
                # print("\n\n")
                # 2nd table
                rows = ["tutorial", "labs", "projects", "mid", "final"]
                # print("2nd table")
                for i in range(len(rows)):
                    ratio = duration = None
                    value = find_text(recs, RX[rows[i]])
                    if value and isinstance(value, str):
                        value = value.replace("(", "").replace(")", "").strip() 
                        value_rec = find_rec(recs, RX[rows[i]], nth=-1)
                        ass_key = ["ass_ratio", "ass_duration"]
                        ass_val = [x_ratio_threasshold, x_duration_thresshold]
                        y_threashold = (value_rec["y_min"] - padding_y, value_rec["y_max"] + padding_y)
                        ass_assign = []
                        for j in range(len(ass_val)):
                            all_cells = find_rec_in_box(
                                recs,
                                RX[ass_key[j]],
                                x_min=ass_val[j][0], x_max=ass_val[j][1],
                                y_min=y_threashold[0], y_max=y_threashold[1],
                                mode="inside"
                            )
                            ass_assign.append(all_cells)                      
                    # print(ass_assign) 
                    ratio = ass_assign[0]
                    duration = ass_assign[1]

                    for ass in asses:
                        if rows[i] in {"mid", "final"}:
                            if ass.name.lower() == "lectures":
                                ass.evaluation_type.append(
                                    EvaluationType(
                                        name=str(value),
                                        ratio=_to_int(ratio),
                                        duration_min=_to_int(duration),
                                    )
                                )

                        if rows[i] == ass.name.lower():
                            ass.ratio = _to_int(ratio)



    return ci, asses

# Public wrapper
def syllabus(
    items: List[Dict[str, Any]],
    source_file: str,
    page_index: int,
    language: str = "en",
    ocr_engine: str = "ppocrv3",
    extractor_version: str = "1.2.0",
    timestamp: Optional[str] = None,
    raw_ocr_text: Optional[str] = None,
    **kwargs,
) -> SyllabusV1:

    ci, asses = extract(items)

    if ci.title is None:
        course_name = kwargs.get("course_name")
        if isinstance(course_name, str) and course_name.strip():
            ci.title = course_name.strip()

    meta = Metadata(
        doc_type="syllabus",
        course_id=ci.course_id or "",
        source_file=str(source_file),
        page_index=int(page_index),
        language=language,
        ocr_engine=ocr_engine,
        extractor_version=extractor_version,
        timestamp=timestamp,
    )

    return SyllabusV1(
        schema_version="syllabus.v1",
        metadata=meta,
        course_info=ci,
        assessments=asses,
        course_des=CourseDescription(),
        raw_ocr_text=raw_ocr_text,
    )
