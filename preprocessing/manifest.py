from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Metadata:
    doc_type: Literal["syllabus", "slide"] = "syllabus"
    course_id: str = ""                                     # "CO1005"
    source_file: Optional[str] = None                       # "path/to/image_or_pdf#page=1"
    page_index: int = 0
    language: str = "en"
    ocr_engine: Optional[str] = None                        # "PaddleOCR 2.7"
    extractor_version: Optional[str] = None                 # "1.0.0"
    timestamp: Optional[str] = None                         # ISO8601

@dataclass
class DocChunk:
    id: str
    text: str
    metadata: Metadata

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