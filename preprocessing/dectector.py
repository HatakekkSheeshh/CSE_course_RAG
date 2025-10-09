"""
OCR wrapper for PaddleOCR 3.2.0
- predict -> OCRResult dict-like
- standardize to List[Dict]: [{ "text": str, "score": float, "polygon": [[x,y],... (4 points)] }]
- save to json/txt/annotated image
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import cv2, tempfile

class OCRTextDetector:
    def __init__(
        self,
        lang: str = "en",
        score_thr: float = 0.5,
        out_dir: Optional[Path | str] = None,
        use_preproc: bool = True,
        **ocr_kwargs,
    ):
        """
        Wrapper for PaddleOCR 3.2.0 (predict -> OCRResult dict-like).
        - lang: "en"
        - score_thr: threshold for filtering results
        - out_dir: export folder for json/txt/ảnh annotate
        - use_preproc: preprocessor (bilateral + gray)
        - **ocr_kwargs: valid parameters of PaddleOCR
        """
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            lang = lang,
            use_textline_orientation = False,
            use_doc_orientation_classify = False,
            use_doc_unwarping = False,
            ocr_version="PP-OCRv3",
            **ocr_kwargs
        )
        self.score_thr = float(score_thr)
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_preproc = use_preproc

    # ---------- utils ----------
    @staticmethod
    def pick(d: dict, keys: List[str]):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    @staticmethod
    def xyxy_to_quad(box):
        x1, y1, x2, y2 = map(float, box)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    @staticmethod
    def light_preproc(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def result_to_items(self, result) -> List[Dict]:
        """
        Standardize OCRResult (dict-like):
        [{ "text": str, "score": float, "polygon": [[x,y],... (4 points)] }]
        """
        items: List[Dict] = []

        texts  = self.pick(result, ["texts", "rec_texts"])
        scores = self.pick(result, ["scores", "rec_scores"])
        boxes  = self.pick(result, ["boxes", "dt_polys", "rec_polys", "det_polys"])
        xyxy   = None
        if boxes is None:
            xyxy = self.pick(result, ["rec_boxes"])         # [x1,y1,x2,y2]

        if texts is None or scores is None or (boxes is None and xyxy is None):
            return items

        n = min(len(texts), len(scores),
                len(boxes) if boxes is not None else len(xyxy))

        for i in range(n):
            if boxes is not None:
                poly = np.asarray(boxes[i]).tolist()            # [[x,y],...4]
            else:
                poly = self.xyxy_to_quad(xyxy[i])               # xyxy -> quad
            items.append({
                "text":  str(texts[i]),
                "score": float(scores[i]),
                "polygon": [[float(x), float(y)] for x, y in poly],
            })

        return items

    # ---------- APIs ----------
    def predict_items(self, img_path: Path | str) -> List[Dict]:
        """
        Run OCR and return items predicted
        """
        BASENAME = Path(img_path).stem
        TMP_DIR = Path(tempfile.gettempdir())
        tmp_path = TMP_DIR / f"{BASENAME}_prep.png"

        img_path = Path(img_path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        if self.use_preproc:
            img_bgr = self.light_preproc(img_bgr)

        cv2.imwrite(str(tmp_path), img_bgr)
        result = next(iter(self.ocr.predict(input=str(tmp_path))))

        if self.out_dir:
            result.save_to_json(str(self.out_dir))
            result.save_to_img(str(self.out_dir))

        items = self.result_to_items(result)
        return [it for it in items if it["score"] >= self.score_thr], img_bgr

    def annotate(self, img_bgr, items: List[Dict], out_name: str = "annotated.png") -> Optional[Path]:
        """
        Draw polygon + text on image
        """
        img = img_bgr.copy()
        for it in items:
            pts = np.array(it["polygon"], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            cv2.putText(img, f'{it["text"]} ({it["score"]:.2f})',
                        (int(x), int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,36,255), 1, cv2.LINE_AA)

        if self.out_dir:
            out_path = self.out_dir / out_name
            cv2.imwrite(str(out_path), img)
            return out_path
        return None

    def save_items(self, items: List[Dict], txt_name="result.txt", json_name="result.json"):
        if not self.out_dir:
            return
        (self.out_dir / txt_name).write_text(
            "\n".join([it["text"] for it in items]), encoding="utf-8"
        )
        (self.out_dir / json_name).write_text(
            json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def run(self, img_path: Path | str):
        print("Start extracting...")
        items, img_bgr = self.predict_items(img_path)
        self.save_items(items)
        self.annotate(img_bgr, items)

        print("--- Extracting from Images is completed ---")
        return items
