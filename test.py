# ocr_basic_v320_fixed2.py
from pathlib import Path
import json, sys
import numpy as np
import cv2, tempfile
from typing import List, Dict

IMG_PATH = Path("/mnt/d/Project/AI_project/data_cvt/CO1005_Introduction_to_Computing/Syllabus/slide_001.png")
OUT_DIR  = Path("./ocr_out_v320")
SCORE_THR = 0.5
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_txt_json(items):
    (OUT_DIR / "result.txt").write_text(
        "\n".join([it["text"] for it in items if it["score"] >= SCORE_THR]),
        encoding="utf-8"
    )
    (OUT_DIR / "result.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def draw_polys(items, img_bgr):
    img = img_bgr.copy()
    for it in items:
        if it["score"] < SCORE_THR:
            continue
        pts = np.array(it["polygon"], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        x, y = pts[0]
        cv2.putText(img, f'{it["text"]} ({it["score"]:.2f})',
                    (int(x), int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,36,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / "annotated.png"), img)

def light_preproc(img_path: Path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def pick(d, keys):
    """ Return first satisfied key in d """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def xyxy_to_quad(box):
    x1, y1, x2, y2 = map(float, box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def result_to_items(result) -> List[Dict]:
    """
    Standardize OCRResult (dict-like):
      [{ "text": str, "score": float, "polygon": [[x,y],...4 points) }]
    """
    items: List[Dict] = []

    texts  = pick(result, ["texts", "rec_texts"])
    scores = pick(result, ["scores", "rec_scores"])

    boxes = pick(result, ["boxes", "dt_polys", "rec_polys", "det_polys"])
    xyxy  = None
    if boxes is None:
        xyxy = pick(result, ["rec_boxes"])  # [x1,y1,x2,y2]

   
    if texts is None or scores is None or (boxes is None and xyxy is None):
        return items

    # Standardize number of elements
    n = min(len(texts), len(scores),
            len(boxes) if boxes is not None else len(xyxy))

    for i in range(n):
        if boxes is not None:
            poly = np.asarray(boxes[i]).tolist()    # [[x,y],...4]
        else:
            poly = xyxy_to_quad(xyxy[i])            # xyxy -> quad

        items.append({
            "text":  str(texts[i]),
            "score": float(scores[i]),
            "polygon": [[float(x), float(y)] for x, y in poly],
        })

    return items

def run_with_paddleocr():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang = "en",
        use_textline_orientation=False,   
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False             
    )

    BASENAME = IMG_PATH.stem
    TMP_DIR = Path(tempfile.gettempdir())
    tmp_path = TMP_DIR / f"{BASENAME}_prep.png"

    img_bgr = light_preproc(IMG_PATH)
    cv2.imwrite(str(tmp_path), img_bgr)
    ite = iter(ocr.predict(input = str(tmp_path)))
    result = next(ite) 

    result.save_to_img(str(OUT_DIR))
    result.save_to_json(str(OUT_DIR))

    items = result_to_items(result)
    write_txt_json(items)
    draw_polys(items, img_bgr)

if __name__ == "__main__":
    run_with_paddleocr()
    print(f"Saved outputs in: {OUT_DIR.resolve()}")
