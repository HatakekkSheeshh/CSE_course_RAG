# --- Data Domain (Syllabus) - v1 ---

from dataclasses import dataclass, field
from typing import List, Optional, Literal

# Metadata đi kèm để embed/tracking
@dataclass
class Metadata:
    doc_type: Literal["syllabus"] = "syllabus"
    course_id: str = ""                       # "CO1005"
    source_file: Optional[str] = None         # "path/to/image_or_pdf#page=1"
    page_index: int = 0
    language: str = "en"
    ocr_engine: Optional[str] = None          # "PaddleOCR 2.7"
    extractor_version: Optional[str] = None   # "1.0.0"
    timestamp: Optional[str] = None           # ISO8601

@dataclass
class CourseInfo:
    title: Optional[str] = None               # "Introduction to Computing"
    course_id: Optional[str] = None           # "CO1005"
    credits: Optional[int] = None             # 3
    applied_semester: Optional[str] = None    # "20231", "HK202", ...

    @property
    def ects(self) -> Optional[int]:
        return self.credits * 2 if self.credits is not None else None

@dataclass
class AssessmentComponent:
    name: str                                 # "labs_practices" | "projects" | "midterm_exam" | "final_exam" | "total" | ...
    ratio: Optional[int] = None               # 0..100
    format: Optional[str] = None              # "MCQ", "Constructed response", "--", ...
    duration_min: Optional[int] = None        # minutes | None

@dataclass
class Prerequisites:
    recommended: List[str] = field(default_factory=list)   # HT/KN
    prereq: List[str] = field(default_factory=list)        # TQ
    coreq: List[str] = field(default_factory=list)         # SH

@dataclass
class SyllabusV1:
    schema_version: Literal["syllabus.v1"] = "syllabus.v1"
    metadata: Metadata = field(default_factory=Metadata)
    course_info: CourseInfo = field(default_factory=CourseInfo)
    assessments: List[AssessmentComponent] = field(default_factory=list)
    prerequisites: Prerequisites = field(default_factory=Prerequisites)
    raw_ocr_text: Optional[str] = None
