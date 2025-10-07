from dataclasses import dataclass
from typing import Optional

@dataclass
class Meta:
    doc_type: str           # "slide" | "syllabus" | 
    course_id: str
    source_file: str
    slide_index: int
    heading: Optional[str] = None
    language: Optional[str] = None

@dataclass
class DocChunk:
    id: str
    text: str
    metadata: Meta

"""
chunk = DocChunk(
    id="CO1005-slide-001-0001",
    text="…",
    metadata=Meta(
        doc_type="slide",
        course_id="CO1005",
        source_file="slide_001.txt",
        slide_index=1,
        heading="Introduction",
        language="en"
    )
)
"""