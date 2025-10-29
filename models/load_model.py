from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
from paddleocr import PaddleOCR

"""
Demo: In development process
Warning:Not in use
"""
def load_model():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index_model = IndexFlatIP(384)
    ocr_model = PaddleOCR(
        lang="en",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        ocr_version="PP-OCRv3",
    )
    return embed_model, index_model, ocr_model