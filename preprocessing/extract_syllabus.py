import re
from typing import List, Dict, Any, Optional, Tuple
from preprocessing.regex import RX 
from preprocessing.syllabus import *
from preprocessing.utils import parse_number

def centerize(poly: List[List[float]]) -> Tuple[float,float]:
    xs = [p[0] for p in poly] 
    ys = [p[1] for p in poly]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def records(items: List[Dict[str, Any]]):
    recs = []
    for it in items:
        xs = [p[0] for p in it["polygon"]]
        ys = [p[1] for p in it["polygon"]]
        recs.append({
            "text": it["text"],
            "x": sum(xs)/len(xs),
            "y": sum(ys)/len(ys),
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        })
    return recs

def find_groups(recs, pattern):
    rx = re.compile(pattern, re.I)
    for r in recs:
        m = rx.search(r["text"])
        if m:
            return m.groups()
    return None

def y_of_label(recs, pattern, matching=0) -> Optional[float]:
    rx = re.compile(pattern, re.I)
    hits = [r for r in recs if rx.search(r["text"])]
    if not hits:
        return None
   
    hits.sort(key=lambda r: (r["y"], r["x"]))
    # print(f"Hits: \n{hits}")
    matching = max(0, min(matching, len(hits)-1))
    # print(f"Matching: {matching}")
    y = hits[matching]["y"]
    # print(f"Y: {y}")
    return y

def value_in_col(recs, y, tol=40, x_min=640, x_max=760, regex=r"\d+(?:\.\d+)?"):
    rx = re.compile(regex, re.I)
    cand = [r for r in recs if x_min <= r["x"] <= x_max and abs(r["y"] - y) <= tol]
    if not cand:
        return None
    cand.sort(key=lambda r: abs(r["y"] - y))    
    base_y = cand[0]["y"]
    band   = [r for r in cand if abs(r["y"] - base_y) <= 10] 
    band.sort(key=lambda r: r["x_min"])
    joined = " ".join(" ".join(r["text"].split()) for r in band)
    m = rx.search(joined)
    if not m:
        return None
    if "en" in rx.groupindex:
        return m.group("en").strip()
    return (m.group(1) if rx.groups else m.group(0)).strip()

def extract_syllabus(items) -> Syllabus:
    recs = records(items)

    # Header
    title           = (find_groups(recs, RX["title"]) or [None])[0]
    course_id       = (find_groups(recs, RX["course_id"]) or [None])[0]
    ce              = (find_groups(recs, RX["credits"]) or [None])[0] 
    applied_sem     = (find_groups(recs, RX["applied_sem"]) or [None])[0]

    # Course format 
    fmt = {}
    for key, pat in [
        ("lectures", RX["lectures"]),
        ("tutorial", RX["tutorial"]),
        ("labs_practices", RX["labs"]),
        ("projects", RX["projects"]),
        ("self_study", RX["self_study"]),
        ("others", RX["others"]),
        ("total_hours", RX["total_hours"]),
    ]:
        y = y_of_label(recs, pat)
        if y is not None:
            v = value_in_col(recs, y)
            if v is not None:
                fmt[key] = v

    # Assessments
    asses: List[AssessmentComponent] = []
    rows = [
        ("Tutorial",        RX["tutorial"],             1),
        ("labs_practices",  RX["labs"],                 1),
        ("Projects",        RX["ass_project"],          1),
        ("Midterm Exam",    RX["ass_midterm"],          0),
        ("Final Exam",      RX["ass_final"],            0),
    ]

    for name, lab_pat, matching in rows:
        ratio: Optional[int] = None
        fmt_cell: Optional[str] = None
        duration: Optional[int] = None
            
        y = y_of_label(recs, lab_pat, matching=matching)

        if y is not None:
            # Ratio
            ratio    = value_in_col(recs, y, tol=40, x_min=630, x_max=720,  regex=RX["ass_ratio"])
            # print(f"Ratio: {ratio}")
            # Format
            fmt_cell = value_in_col(recs, y, tol=40, x_min=730, x_max=1120, regex=RX["ass_format"])
            # print(f"Format: {fmt_cell}")
            # Duration
            duration = value_in_col(recs, y, tol=40, x_min=1120, x_max=1360, regex=RX["ass_duration"])
            # print(f"Duration: {duration}")
        
        asses.append(AssessmentComponent(
            name=name, ratio=ratio, format=fmt_cell, duration_min=duration
        ))

    ci = CourseInfo(
        title=title,
        course_id=course_id,
        credits=ce,                        
        applied_semester=applied_sem,
        course_format=fmt,
    )
    
    return Syllabus(course_info=ci, assessments=asses)


