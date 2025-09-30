from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Tuple

# --- OCR/Layout ---
BBox = Tuple[float, float, float, float]    # x1,y1,x2,y2
Polygon = List[Tuple[float, float]]         # (x1,y1) (x1, y2) (x2, y1) (x2, y2)

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
    source_meta: Dict[str,str] = field(default_factory=dict)    # course, faculty,...

""" --- Data Domain (Syllabus) --- """
@dataclass
# Slide 1st
class CourseInfo:
    title: Optional[str]                                        # Course title: Mathematical Modeling
    course_id: Optional[str]                                    # CO2011    
    credits: Optional[int]                                      # 3
    applied_semester: Optional[str]                             # 20211
    course_format: Dict[str, float]                             # {"Lecture_hours":30, "Lab_hours":20, ...}

    @property
    def etcs(self):
        return credits * 2

@dataclass
class AssessmentComponent:
    name: str                                                   # Midterm, Final, Project, Lab...
    ratio: float                                                # 20%, 40%...
    format: Optional[str]                                       # MCQ, written, practice...
    duration_min: Optional[int]                                 # 70, 80 minutes

# Slide 2nd
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
class TeachingMethod:
    name: str                                                   # blended, lecture, practice...
    notes: Optional[str]=None

@dataclass
class SessionPlan:
    session_no: int
    topics: List[str]
    learning_outcomes: List[str]                                # references to LO codes
    activities: List[str]                                       # lecturer/student activities

@dataclass
class StudyGuideline:
    notes: Optional[str]

@dataclass
class EditingInfo:
    edited_semester: Optional[str]
    version: Optional[str]
    last_change: Optional[str]

@dataclass
class Syllabus:
    # Slide 1st
    course_info: CourseInfo
    assessments: List[AssessmentComponent]

    """
    prerequisites: List[Prerequisite]
    knowledge_block: Optional[str]                              # Foundation/Major/...
    unit_in_charge: Optional[UnitInCharge]
    materials: List[Material]
    learning_outcomes: List[LearningOutcomeItem]
    teaching_methods: List[TeachingMethod]
    session_plan: List[SessionPlan]
    study_guidelines: Optional[StudyGuideline]
    other_requirements: Optional[str]
    editing_info: Optional[EditingInfo]
    anchors: Dict[str, Polygon] = field(default_factory=dict)   # heading polygons for traceability
    """