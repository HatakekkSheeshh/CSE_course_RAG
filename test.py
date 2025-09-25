from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
pdf = ROOT / "data" / "CO1005_Introduction_to_Computing" / "Chapter_0.pdf"
img = np.array(convert_from_path(str(pdf), dpi=220)[0].convert("RGB"))

ocr = PaddleOCR(
    ocr_version="PP-OCRv5",
    det=True,   
    rec=True, 

    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,

)

res = ocr.predict(img)   

for i, line in enumerate(res[0], 1):
    box, (text, conf) = line
    print(f"#{i}\n  box : {box}\n  text: {text}\n  conf: {conf:.4f}")
