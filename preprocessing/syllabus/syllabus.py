from dataclasses import dataclass, field
from typing import List, Optional, Literal
from preprocessing.manifest import Metadata

@dataclass
class EvaluationType:
    name: str                           # Midterm
    ratio: Optional[int] = None         # 30 (%)
    duration_min: Optional[int] = None  # 60 (minutes)

@dataclass
class CourseInfo:
    title: Optional[str] = None
    course_id: Optional[str] = None
    credits: Optional[int] = None
    applied_semester: Optional[str] = None

    @property
    def ects(self) -> Optional[int]:
        return self.credits * 2 if self.credits is not None else None

@dataclass
class AssessmentComponent:
    name: str
    hours: Optional[int] = None
    credits: Optional[int] = None
    ratio: Optional[int] = None
    evaluation_type: List[EvaluationType] = field(default_factory=list)

@dataclass
class Syllabus:
    schema_version: Literal["syllabus.v1"] = "syllabus.v1"
    metadata: Metadata = field(default_factory=Metadata)
    course_info: CourseInfo = field(default_factory=CourseInfo)
    assessments: List[AssessmentComponent] = field(default_factory=list)
    raw_text: Optional[str] = None
