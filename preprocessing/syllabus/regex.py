# preprocessing/syllabus/regex.py
RX = {
    # Header fields
    "title":            r"Course title\s*:\s*(.+)",
    "course_id":        r"Course ID\)\s*:\s*([A-Z]+\d+)",
    "credits":          r"Credits\)\s*:\s*(\d+)(?:[^\n]*?(?:ECTS|ETCS)\s*:\s*(\d+|[-–—]*))?",
    "applied_sem":      r"Applied from semester\)\s*:\s*((?:HK\s*\d{2,4})|\d{4,5})",

    # Teaching/study type row anchors
    "lectures":         r"\(?Lectures\)?",
    "tutorial":         r"\(?Tutorial\)?",
    "labs":             r"\(?Labs/Practices\)?",
    "projects":         r"\(?Projects\)?",
    "self_study":       r"Self-?study",
    "others":           r"\(?Others\)?",
    "total_hours":      r"\(?Total\)?",
    "mid":              r"\(?Midterm Exam\)?",
    "final":            r"\(?Final Exam\)?",

    # Column headers (more tolerant)
    "col_lessons":      r"\(Lessons\)|\bLessons?\b|\bHours?\b",
    "col_credits":      r"\(Credits\)|\bCredits?\b",
    "col_notes":        r"\(Notes\)|\bNotes?\b",
    "col_ratio":        r"\(Ratio\)",
    "col_format":       r"\(Format\)",
    "col_evaluate":     r"\(Evaluation type\)",
    "col_duration":     r"\(Duration\)|\b(?:Duration|Time)\b",

    # Assessments (English-only anchors)
    "ass_midterm":      r"\bMid\s*[- ]?\s*term(?:\s*(?:Exam|Test|Examination))?\b",
    "ass_final":        r"\bFinal(?:\s*(?:Exam|Test|Examination))?\b",

    # Value pickers
    "ass_lessons":      r"(?<!\d)(\d{1,3})",
    "ass_credits":      r"^\s*(\d{1,3}(?:[.,]\d+)?)\s*$",
    "ass_ratio":        r"(?<!\d)(\d{1,3})(?=\s*%)",
    "ass_format":       r"[\(\[]\s*(?P<en>[A-Za-z][A-Za-z0-9&/,\- ]{2,})\s*[\)\]]",
    "ass_format_alt":   r"\b(Constructed response|MCQ|Multiple[- ]Choice(?: Questions)?|Essay|Oral|Practical|Written)\b",
    "ass_duration":     r"(?<!\d)(\d{1,3})(?=\s*(?:phút|phut|min(?:ute)?s?|\(\s*minutes\s*\)))",
}
