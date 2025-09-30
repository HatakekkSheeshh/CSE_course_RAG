RX = {
    # Header fields
    "title":            r"Course title\s*:\s*(.+)",
    "course_id":        r"Course ID\)\s*:\s*([A-Z]+\d+)",
    "credits":          r"Credits\)\s*:\s*(\d+).*(?:ETCS)\s*:\s*(\d+)",
    "applied_sem":      r"Applied from semester\)\s*:\s*(\d+)",

    # Course format rows (Teaching/study type)
    "lectures":         r"\(Lectures\)",
    "tutorial":         r"\(Tutorial\)",
    "labs":             r"\(Labs/Practices\)",
    "projects":         r"\(Projects\)",
    "self_study":       r"Self-?study",
    "others":           r"\(Others\)",
    "total_hours":      r"\(Total\)",

    # Assessments block
    "ass_midterm":      r"\bMidterm Exam\b",
    "ass_final":        r"\bFinal Exam\b",
    "ass_project":      r"\(Projects\)",
    "ass_ratio":        r"(?<!\d)(\d{1,3})(?=\s*%)",    # → 20/40/40
    "ass_duration":     r"(?<!\d)(\d{1,3})(?=\s*(?:phút|phut|ph??t|min(?:ute)?s?|\(\s*minutes\s*\)))",
    "ass_format":       r"[\(\[]\s*(?P<en>[A-Za-z][A-Za-z0-9&/,\- ]{2,})\s*[\)\]]",
}
