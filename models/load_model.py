from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
from FlagEmbedding import FlagReranker
from typing import Optional

"""
Load model for different purposes
"""
def load_model(kind: str = "ocr", *, model_name: Optional[str] = None):
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
    elif kind == "reranker":
        # Use large bge reranker for better scoring; can run on CPU (fp32) by default
        name = model_name or "BAAI/bge-reranker-large"
        return FlagReranker(name, use_fp16=False)
    else:
        raise ValueError(f"Unknown kind: {kind}")