from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
from paddleocr import PaddleOCR

"""
Demo: In development process
Warning: Not in use
"""
def load_model(kind: str = "ocr"):
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
    elif kind == "index":
        return IndexFlatIP(384) 
    else:
        raise ValueError(f"Unknown kind: {kind}")