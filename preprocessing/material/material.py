from dataclasses import dataclass, field
from typing import List, Optional, Literal
from preprocessing.manifest import Metadata

@dataclass
class MaterialItem:
    title: str
    chapter_num: int
    page_index: int

@dataclass
class Material:
    course: str
    items: List[MaterialItem]
    schema_version: Literal["material.v1"] = "material.v1"
    metadata: Metadata = field(default_factory=Metadata)
    raw_text: Optional[str] = None
