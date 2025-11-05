# preprocessing/syllabus/extract_syllabus.py
"""
Extract structured syllabus information from OCR text items.

This module processes OCR detection results from syllabus images and extracts
structured information including course details and assessment components.
It handles different syllabus table layouts and converts free-form OCR text
into structured data objects.

Main Features:
    - Course information extraction (title, course_id, credits, semester)
    - Assessment component extraction (hours, credits, ratios, evaluation types)
    - Handles single-table and dual-table syllabus layouts
    - Spatial text matching using bounding box coordinates
    - Regex-based field extraction with fallback patterns

Data Flow:
    OCR Items (List[Dict]) → Records (with spatial info) → Structured Objects
    
Output Format:
    - CourseInfo: Basic course metadata
    - List[AssessmentComponent]: Assessment breakdown with evaluation details
    - Syllabus: Complete structured syllabus object with metadata

Author:
    Syllabus extraction module for RAG system
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple
from preprocessing.syllabus.regex import RX
from preprocessing.manifest import Metadata
from preprocessing.syllabus.syllabus import (
        EvaluationType, AssessmentComponent, CourseInfo, Syllabus
    )


def records(items: List[Dict[str, Any]]):
    """
    Convert OCR detection items to records with spatial information.
    
    Transforms OCR polygon-based detections into a format with centroid coordinates
    and bounding box boundaries for spatial matching operations.
    
    Args:
        items: List of OCR detection items, each containing:
               - "text": Detected text string
               - "polygon": List of 4 [x, y] coordinate pairs
    
    Returns:
        List of record dictionaries, each containing:
            - text: Original text string
            - x: Centroid X coordinate (average of polygon X coordinates)
            - y: Centroid Y coordinate (average of polygon Y coordinates)
            - x_min, x_max: Minimum and maximum X coordinates (bounding box)
            - y_min, y_max: Minimum and maximum Y coordinates (bounding box)
    
    Note:
        Centroid coordinates are used for spatial matching and ordering.
        Bounding box coordinates enable overlap/containment checks.
    """
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
    """
    Find first matching record and extract regex capture groups.
    
    Searches through records and returns captured groups from the first match.
    Useful for extracting structured data from text (e.g., course ID, credits).
    
    Args:
        recs: List of record dictionaries with "text" field
        pattern: Regex pattern with capture groups
    
    Returns:
        Tuple of captured groups from first match, or None if no match
    
    Example:
        >>> find_groups(recs, r"Course\s+ID:\s*(\w+)")
        # Returns: ("CS101",) if found
    """
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
    """
    Find record matching pattern, with ordering and counting options.
    
    Searches for records matching a regex pattern with various ordering and
    matching strategies. Supports finding nth occurrence or negative indexing.
    
    Args:
        recs: List of record dictionaries to search
        pattern: Regex pattern to match against record text
        nth: Index of match to return (0=first, -1=last, etc.)
             If negative, collects all matches then returns nth from end
        order: Ordering strategy:
               - "keep": Preserve original record order
               - "yx": Sort by Y then X coordinate (top-to-bottom, left-to-right)
        per_match: If True, count each regex match separately (for multiple
                   matches in same text). If False, count each record once
        flags: Regex flags (default: re.I for case-insensitive)
    
    Returns:
        Matching record dictionary, or None if not found
    
    Example:
        >>> find_rec(recs, r"Lecture", nth=0, order="yx")
        # Returns first "Lecture" record when sorted top-to-bottom
    
    Note:
        Negative nth values collect all matches first, then index from end.
        Useful for finding "last occurrence" or "second-to-last" matches.
    """
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
    """
    Extract text from first matching record using regex groups.
    
    Searches records and extracts text using prioritized group selection:
    1. Named group 'en' (English text)
    2. First capture group
    3. Full match
    
    Args:
        recs: List of record dictionaries
        pattern: Regex pattern, optionally with named group 'en' or capture groups
    
    Returns:
        Extracted text string, or None if no match
    
    Example:
        >>> find_text(recs, r"(?:EN|English):\s*(?P<en>\w+)")
        # Returns value of 'en' group if present
    """
    rx = re.compile(pattern, re.I)
    for r in recs:
        m = rx.search(r["text"])
        if m:
            if hasattr(m.re, "groupindex") and "en" in m.re.groupindex:
                return m.group("en")
            return m.group(1) if m.groups() else m.group(0)
    return None


def list_matches(recs: List[Dict[str, Any]], pattern: str) -> int:
    """
    Count number of records matching pattern.
    
    Args:
        recs: List of record dictionaries
        pattern: Regex pattern to match
    
    Returns:
        Count of matching records
    """
    rx = re.compile(pattern, re.I)
    hits = [(r["y"], r["x"], r["text"]) for r in recs if rx.search(r.get("text", "") or "")]
    return len(hits)


def _match_text(pattern, text, flags=re.I) -> Optional[str]:
    """
    Internal helper: Extract text from regex match with group priority.
    
    Args:
        pattern: Regex pattern (string or compiled pattern)
        text: Text to search
        flags: Regex flags
    
    Returns:
        Extracted text using priority: named group 'en' > group(1) > group(0),
        or None if no match
    """
    rx = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
    m = rx.search(text or "")
    if not m:
        return None
    if hasattr(m.re, "groupindex") and "en" in m.re.groupindex:
        return m.group("en")
    return m.group(1) if m.groups() else m.group(0)


def _in_box(rec: Dict[str, Any], x_min: float, x_max: float, y_min: float, y_max: float, mode: str) -> bool:
    """
    Check if record is within or intersects specified bounding box.
    
    Args:
        rec: Record dictionary with x, y, x_min, x_max, y_min, y_max
        x_min, x_max, y_min, y_max: Bounding box boundaries
        mode: Matching mode:
              - "center": Record centroid must be inside box
              - "overlap": Record bounding box overlaps with specified box
              - "inside": Record bounding box completely inside specified box
    
    Returns:
        True if record matches the spatial criterion, False otherwise
    
    Raises:
        ValueError: If mode is not one of the valid options
    """
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
    mode: str = "center",
    flags: int = re.I,
) -> Optional[Dict[str, Any]]:
    """
    Find record matching pattern within specified spatial bounding box.
    
    Combines spatial filtering with regex matching to locate text in specific
    regions of the document (e.g., table cells, specific sections).
    
    Args:
        recs: List of record dictionaries
        pattern: Regex pattern to match
        x_min, x_max, y_min, y_max: Bounding box boundaries
        mode: Spatial matching mode ('center', 'overlap', 'inside')
        flags: Regex flags
    
    Returns:
        First matching record found in box (sorted by Y then X), or None
    
    Note:
        Returns matched text string directly, not the record object.
        This differs from other find_rec variants for convenience.
    """
    cands: List[Dict[str, Any]] = []
    
    for r in recs:
        if not _in_box(r, x_min, x_max, y_min, y_max, mode):
            continue
        mt = _match_text(pattern, r.get("text",""), flags=flags)
        if mt is None:
            continue
        return mt

    if not cands:
        return None

    cands.sort(key=lambda z: (z["y"], z["x"]))
    return cands[0]


def _to_int(x: Optional[str]) -> Optional[int]:
    """
    Convert string or numeric value to integer with robust parsing.
    
    Handles various formats: integers, floats, strings with commas/decimals.
    Extracts first numeric value found in string.
    
    Args:
        x: Value to convert (string, int, float, or None)
    
    Returns:
        Integer value, or None if conversion fails
    
    Example:
        >>> _to_int("3.5 credits")
        4  # Rounds to nearest integer
        >>> _to_int("1,234")
        1234
        >>> _to_int("abc")
        None
    """
    if x is None: 
        return None
    try:
        if isinstance(x, (int, float)): 
            return int(round(x))
        
        m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", "."))
        return int(round(float(m.group(0)))) if m else None
    except Exception:
        return None


def size_of_block(block: Dict[str, Any]) -> Optional[int | float]:
    """
    Calculate dimensions of a bounding block.
    
    Args:
        block: Dictionary with x_min, x_max, y_min, y_max keys
    
    Returns:
        Tuple of (height, width), or None if calculation fails
    """
    try:
        width = block.get("x_max") - block.get("x_min")
        height = block.get("y_max") - block.get("y_min")
        if width is not None and height is not None:
            return height, width
        return None
    except Exception:
        return None


def extract(items: List[Dict[str, Any]]) -> Tuple[CourseInfo, List]:
    """
    Extract structured course information and assessment components from OCR items.
    
    Main extraction function that processes OCR detection results to extract:
    - Course metadata (title, ID, credits, semester)
    - Assessment components with hours, credits, ratios, and evaluation types
    
    Handles two syllabus layouts:
    1. Single table: All assessment data in one table
    2. Dual tables: Assessment data split across two tables
    
    Args:
        items: List of OCR detection items from syllabus page, each containing:
               - "text": Detected text string
               - "polygon": List of 4 [x, y] coordinate pairs
    
    Returns:
        Tuple of:
            - CourseInfo: Extracted course metadata (may be empty)
            - List[AssessmentComponent]: List of assessment components with details
    
    Algorithm:
        1. Convert OCR items to records with spatial information
        2. Extract course info from header section using regex patterns
        3. Detect table layout (1 or 2 tables) based on "projects" keyword count
        4. For single table:
           - Locate column headers (Lessons, Credits, Ratio, Evaluation, Duration)
           - Extract row data using spatial matching (bounding box intersections)
           - Handle special case: Lectures has multiple evaluation types
        5. For dual tables:
           - First table: Extract hours and credits for each component
           - Second table: Extract ratios and durations, match to components
    
    Note:
        Uses spatial matching with padding to handle OCR alignment issues.
        Special handling for "Lectures" row which may have multiple evaluation types.
    """
    recs = records(items)

    ci = CourseInfo()
    asses: List = []

    title       = (find_groups(recs, RX["title"]) or (None,))[0]
    course_id   = (find_groups(recs, RX["course_id"]) or (None,))[0]
    credits_g   = find_groups(recs, RX["credits"])
    credits     = _to_int(credits_g[0]) if credits_g else None
    applied_sem = (find_groups(recs, RX["applied_sem"]) or (None,))[0]

    if title or course_id or credits or applied_sem:
        ci = CourseInfo(
            title=title,
            course_id=course_id,
            credits=credits,
            applied_semester=applied_sem,
        )

        rows = ["lectures", "tutorial", "labs", "projects", "self_study", "others"]
        matched = list_matches(recs, RX["projects"])

        if matched < 2:
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
                    eva: List[EvaluationType] = []
                    
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
                            dg = find_text(recs, RX[debug[j]]).replace("(", "").replace(")","").strip().lower()
                            
                            if value == "Lectures" and dg in {"ratio", "evaluation type", "duration"}:
                                y_threashold_lectures = [
                                    (value_rec["y"] - 150, value_rec["y"]),
                                    (value_rec["y"] - 30, value_rec["y"] + 150)
                                ]

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

                        hours = ass_assign[0]
                        credits = ass_assign[1]
                        
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
            lessons_rec = find_rec(recs, RX["col_lessons"])
            credits_rec = find_rec(recs, RX["col_credits"], nth=1)
            ratio_rec = find_rec(recs, RX["col_ratio"], nth=-1)
            duration_rec = find_rec(recs, RX["col_duration"])

            ass_assign = []

            if lessons_rec and credits_rec and ratio_rec and duration_rec:
                padding_x = 80
                padding_y = 50
                x_lession_thresshold = (lessons_rec["x_min"] - padding_x, lessons_rec["x_max"] + padding_x)
                x_credits_thresshold = (credits_rec["x_min"] - padding_x, credits_rec["x_max"] + padding_x)
                x_ratio_threasshold = (ratio_rec["x_min"] - padding_x, ratio_rec["x_max"] + padding_x)
                x_duration_thresshold = (duration_rec["x_min"] - padding_x, duration_rec["x_max"] + padding_x)

                rows = ["lectures", "tutorial", "labs", "projects", "self_study", "others"]
                
                for i in range(len(rows)):
                    hours = credits = None
                    value = find_text(recs, RX[rows[i]])
                    if value and isinstance(value, str):
                        value = value.replace("(", "").replace(")", "").strip() 
                        value_rec = find_rec(recs, RX[rows[i]])
                        
                        y_threashold = (value_rec["y_min"] - padding_y, value_rec["y_max"] + padding_y)
                        
                        ass_val = [x_lession_thresshold, x_credits_thresshold]
                        ass_key = ["ass_lessons", "ass_credits"]
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
                            

                    hours = ass_assign[0]
                    credits = ass_assign[1] if len(ass_assign) > 1 else None
                    
                    asses.append(AssessmentComponent(
                        name=str(value),
                        credits=_to_int(credits),
                        hours=_to_int(hours),
                    ))  

                rows = ["tutorial", "labs", "projects", "mid", "final"]
                
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


def syllabus(
    items: List[Dict[str, Any]],
    source_file: str,
    page_index: int,
    language: str = "en",
    ocr_engine: str = "ppocrv3",
    extractor_version: str = "1.2.0",
    timestamp: Optional[str] = None,
    raw_text: Optional[str] = None,
    **kwargs,
) -> Syllabus:
    """
    Public API: Extract complete syllabus object from OCR items.
    
    Main entry point for syllabus extraction. Processes OCR detection results
    and creates a structured Syllabus object with metadata and course information.
    
    Args:
        items: List of OCR detection items from syllabus page
        source_file: Path/filename of source image file
        page_index: Page number (0-indexed) in source document
        language: Document language code (default: "en")
        ocr_engine: OCR engine identifier (default: "ppocrv3")
        extractor_version: Version of extraction logic (default: "1.2.0")
        timestamp: Optional timestamp string for when extraction occurred
        raw_text: Optional raw text content (if available separately)
        **kwargs: Additional arguments:
                 - course_name: Fallback course name if not extracted from image
    
    Returns:
        Syllabus object containing:
            - schema_version: Schema version identifier
            - metadata: Document metadata (source, page, OCR info, etc.)
            - course_info: Extracted course information
            - assessments: List of assessment components
            - raw_text: Optional raw text content
    
    Example:
        >>> items = detector.predict("syllabus.png")  # OCR results
        >>> syllabus_obj = syllabus(
        ...     items=items,
        ...     source_file="syllabus.png",
        ...     page_index=0,
        ...     course_name="Digital Systems"
        ... )
        >>> print(syllabus_obj.course_info.title)
        "Digital Systems"
    
    Note:
        If course title cannot be extracted from image, attempts to use
        course_name from kwargs as fallback.
    """
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

    return Syllabus(
        schema_version="syllabus.v1",
        metadata=meta,
        course_info=ci,
        assessments=asses,
        raw_text=raw_text,
    )
