from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
from paddleocr import PaddleOCR
from typing import Optional

"""
Load model for different purposes
"""
def load_model(kind: str = "ocr", embedding_dim: Optional[int] = None):
    if kind == "ocr":
        return PaddleOCR(
            lang="en",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            ocr_version="PP-OCRv3",
        )
    elif kind == "embed":
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown kind: {kind}")