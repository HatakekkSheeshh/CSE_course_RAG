from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Tuple

# --- OCR/Layout ---
BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2 (or your 4-pt polygon)
Polygon = List[Tuple[float, float]]

@dataclass
class OCRItem:
    text: str
    score: float
    polygon: Polygon
    page: int

@dataclass
class Block:
    kind: Literal["title","heading","kv","paragraph","table","list","footer"]
    items: List[OCRItem]
    bbox: BBox
    page: int

@dataclass
class Page:
    index: int
    blocks: List[Block] = field(default_factory=list)
    image_path: Optional[str] = None

@dataclass
class Document:
    doc_id: str
    file_name: str
    pages: List[Page]
    source_meta: Dict[str,str] = field(default_factory=dict)  # course, faculty,...

# --- Data Domain (Syllabus) ---
@dataclass
class CourseInfo:
    name_vi: Optional[str]
    name_en: Optional[str]
    course_id: Optional[str]
    ects: Optional[float]
    credits: Optional[float]
    applied_semester: Optional[str]
    course_format: Dict[str, float]  # {"Lecture_hours":30, "Lab_hours":20, ...}

@dataclass
class Prerequisite:
    course_id: Optional[str]
    course_title: Optional[str]
    kind: Literal["Prereq","Coreq","Recommended","None"] = "None"

@dataclass
class UnitInCharge:
    department: Optional[str]
    office: Optional[str]
    phone: Optional[str]
    lecturer: Optional[str]
    email: Optional[str]

@dataclass
class Material:
    title: str
    authors: Optional[str] = None
    edition: Optional[str] = None
    year: Optional[str] = None
    type: Literal["textbook","reference"] = "textbook"

@dataclass
class LearningOutcomeItem:
    code: str                                                   # e.g., "LO.2.3"
    vi: Optional[str]
    en: Optional[str]
    mapped_assessment: List[str]                                # e.g., ["Midterm","Final","Large assignment"]

@dataclass
class AssessmentComponent:
    name: str                                                   # Midterm, Final, Project, Lab...
    ratio: float                                                # 0.1 ~ 1.0
    format: Optional[str]                                       # MCQ, written, practice...
    duration_min: Optional[int]

@dataclass
class TeachingMethod:
    name: str                                                   # blended, lecture, practice...
    notes: Optional[str]=None

@dataclass
class SessionPlan:
    session_no: int
    topics_vi: List[str]
    topics_en: List[str]
    learning_outcomes: List[str]                                # references to LO codes
    activities: List[str]                                       # lecturer/student activities

@dataclass
class StudyGuideline:
    notes_vi: Optional[str]
    notes_en: Optional[str]

@dataclass
class EditingInfo:
    edited_semester: Optional[str]
    version: Optional[str]
    last_change: Optional[str]

@dataclass
class Syllabus:
    course_info: CourseInfo
    prerequisites: List[Prerequisite]
    knowledge_block: Optional[str]                              # Foundation/Major/...
    unit_in_charge: Optional[UnitInCharge]
    materials: List[Material]
    learning_outcomes: List[LearningOutcomeItem]
    assessments: List[AssessmentComponent]
    teaching_methods: List[TeachingMethod]
    session_plan: List[SessionPlan]
    study_guidelines: Optional[StudyGuideline]
    other_requirements: Optional[str]
    editing_info: Optional[EditingInfo]
    anchors: Dict[str, Polygon] = field(default_factory=dict)   # heading polygons for traceability
