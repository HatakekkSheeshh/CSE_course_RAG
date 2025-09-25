from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
models = ROOT / "models"
det_model_path = str(models / "dbnet")
rec_model_path = str(models / "svtr")


ocr = PaddleOCR(
    text_detection_model_dir = det_model_path,
    text_recognition_model_dir = rec_model_path,
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation = False,                          
)

img = convert_from_path(
    str(ROOT / "data" / "CO1005_Introduction_to_Computing" / "Chapter_0.pdf"),
    dpi=220
)[0]

res = ocr.ocr(np.array(img.convert("RGB")))  
for i, line in enumerate(res[0], 1):
    box, (text, conf) = line
    print(f"#{i}")
    print("  box :", box)
    print("  text:", text)
    print(f"  conf: {conf:.4f}")
