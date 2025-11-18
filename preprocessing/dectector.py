"""
OCR wrapper for PaddleOCR 3.2.0

This module provides a high-level interface for text detection and recognition
using PaddleOCR. It standardizes OCR results to a consistent format and provides
utilities for visualization and export.

Features:
    - Text detection and recognition from images
    - Standardized output format: List[Dict] with text, score, and polygon
    - Image preprocessing (bilateral filter + grayscale)
    - Annotation visualization with bounding boxes
    - Export to JSON, TXT, and annotated images

Author:
    OCRTextDetector class for PaddleOCR 3.2.0 integration

Standard Output Format:
    [
        {
            "text": str,           # Detected text string
            "score": float,        # Confidence score (0.0-1.0)
            "polygon": [[x,y], ...] # 4 corner points of bounding polygon
        },
        ...
    ]
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import cv2, tempfile


class OCRTextDetector:
    """
    Wrapper class for PaddleOCR text detection and recognition.
    
    Provides a simplified interface to PaddleOCR with standardized output format,
    preprocessing options, and result visualization/export capabilities.
    
    Attributes:
        ocr: PaddleOCR instance for text detection/recognition
        score_thr: Minimum confidence threshold for filtering results
        out_dir: Output directory for saving results (None if not specified)
        use_preproc: Whether to apply image preprocessing before OCR
    
    Example:
        >>> detector = OCRTextDetector(lang="en", score_thr=0.5, out_dir="./output")
        >>> items = detector.predict("image.png")
        >>> # Returns: List[Dict] with text, score, polygon for each detection
    """
    
    def __init__(
        self,
        lang: str = "vi",
        score_thr: float = 0.5,
        out_dir: Optional[Path | str] = None,
        use_preproc: bool = True,
        **ocr_kwargs,
    ):
        """
        Initialize OCR detector with PaddleOCR backend.
        
        Args:
            lang: Language code for OCR model ("en", "vi", etc.)
            score_thr: Confidence threshold (0.0-1.0) for filtering results.
                      Only detections with score >= score_thr are returned
            out_dir: Optional directory path for exporting results.
                    If provided, creates directory and saves JSON/TXT/annotated images
            use_preproc: Whether to apply image preprocessing (bilateral filter + grayscale)
                        before OCR. Improves text detection in some cases
            **ocr_kwargs: Additional keyword arguments passed directly to PaddleOCR
                        constructor (e.g., use_angle_cls, det_model_dir, etc.)
        
        Note:
            PaddleOCR is configured with:
            - use_textline_orientation = False
            - use_doc_orientation_classify = False
            - use_doc_unwarping = False
            - ocr_version = "PP-OCRv3"
        """
        from paddleocr import PaddleOCR  # pyright: ignore[reportMissingImports]
        
        # Initialize PaddleOCR with specified configuration
        self.ocr = PaddleOCR(
            lang = lang,
            use_textline_orientation = False,
            use_doc_orientation_classify = False,
            use_doc_unwarping = False,
            ocr_version="PP-OCRv3",
            **ocr_kwargs
        )
        
        # Store configuration
        self.score_thr = float(score_thr)
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_preproc = use_preproc


    # ==================== UTILITY METHODS ====================

    @staticmethod
    def pick(d: dict, keys: List[str]):
        """
        Extract the first non-None value from dictionary using a list of possible keys.
        
        Useful for handling different OCR result formats where field names may vary.
        
        Args:
            d: Dictionary to search in
            keys: List of key names to try in order
        
        Returns:
            First non-None value found, or None if all keys are missing/None
        
        Example:
            >>> pick(result, ["texts", "rec_texts"])
            # Returns result["texts"] if exists, else result["rec_texts"], else None
        """
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    @staticmethod
    def xyxy_to_quad(box):
        """
        Convert axis-aligned bounding box (x1,y1,x2,y2) to 4-point polygon.
        
        Transforms from [x1, y1, x2, y2] format to quadrilateral coordinates:
        [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        
        Args:
            box: List or array of 4 values [x1, y1, x2, y2]
        
        Returns:
            List of 4 coordinate pairs representing the polygon corners
        
        Example:
            >>> xyxy_to_quad([10, 20, 100, 150])
            [[10.0, 20.0], [100.0, 20.0], [100.0, 150.0], [10.0, 150.0]]
        """
        x1, y1, x2, y2 = map(float, box)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    @staticmethod
    def light_preproc(img_bgr):
        """
        Apply light preprocessing to improve OCR accuracy.
        
        Steps:
            1. Convert BGR to grayscale
            2. Apply bilateral filter (reduces noise while preserving edges)
            3. Convert back to BGR (3-channel) for PaddleOCR compatibility
        
        Args:
            img_bgr: Input image in BGR format (OpenCV format)
        
        Returns:
            Preprocessed image in BGR format
        
        Note:
            Bilateral filter parameters (d=7, sigmaColor=50, sigmaSpace=50) are
            tuned for text preservation while reducing image noise.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def result_to_items(self, result) -> List[Dict]:
        """
        Convert PaddleOCR result to standardized format.
        
        Handles different PaddleOCR output formats and converts to consistent structure:
        [{ "text": str, "score": float, "polygon": [[x,y],... (4 points)] }]
        
        Args:
            result: PaddleOCR result object (dict-like) containing:
                   - texts/rec_texts: List of recognized text strings
                   - scores/rec_scores: List of confidence scores
                   - boxes/dt_polys/rec_polys/det_polys: List of polygon coordinates
                   - rec_boxes: List of [x1,y1,x2,y2] bounding boxes (if polygons not available)
        
        Returns:
            List of dictionaries, each containing:
                - text: Detected text string
                - score: Confidence score (0.0-1.0)
                - polygon: List of 4 [x, y] coordinate pairs forming the bounding polygon
        
        Note:
            Returns empty list if required fields (texts, scores, boxes) are missing.
            Converts xyxy format to polygon if only bounding boxes are available.
        """
        items: List[Dict] = []

        # Extract fields with fallback to alternative key names
        texts  = self.pick(result, ["texts", "rec_texts"])
        scores = self.pick(result, ["scores", "rec_scores"])
        boxes  = self.pick(result, ["boxes", "dt_polys", "rec_polys", "det_polys"])
        xyxy   = None
        
        # Fallback: try to get bounding boxes in xyxy format
        if boxes is None:
            xyxy = self.pick(result, ["rec_boxes"])  # [x1,y1,x2,y2]

        # Validate that we have all required data
        if texts is None or scores is None or (boxes is None and xyxy is None):
            return items

        # Ensure all lists have the same length (take minimum to avoid index errors)
        n = min(len(texts), len(scores),
                len(boxes) if boxes is not None else len(xyxy))

        # Convert each detection to standardized format
        for i in range(n):
            # Convert polygon format: use existing polygon or convert from xyxy
            if boxes is not None:
                poly = np.asarray(boxes[i]).tolist()  # [[x,y],...4]
            else:
                poly = self.xyxy_to_quad(xyxy[i])     # xyxy -> quad
            
            # Create standardized item dictionary
            items.append({
                "text":  str(texts[i]),
                "score": float(scores[i]),
                "polygon": [[float(x), float(y)] for x, y in poly],
            })

        return items


    # ==================== PUBLIC API METHODS ====================

    def predict_items(self, img_path: Path | str) -> List[Dict]:
        """
        Run OCR on image and return detected text items.

        Args:
            img_path: Path to input image file
        
        Returns:
            Tuple of (items, img_bgr):
                - items: List of detected text items (filtered by score_thr)
                - img_bgr: Loaded image in BGR format (for annotation/visualization)
        
        Raises:
            FileNotFoundError: If image file cannot be read
        
        Note:
            Creates temporary preprocessed image file for PaddleOCR input.
            Temporary file is cleaned up automatically by the OS.
        """
        BASENAME = Path(img_path).stem
        TMP_DIR = Path(tempfile.gettempdir())
        tmp_path = TMP_DIR / f"{BASENAME}_prep.png"

        # Load image
        img_path = Path(img_path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Apply preprocessing if enabled
        if self.use_preproc:
            img_bgr = self.light_preproc(img_bgr)

        # Save preprocessed image temporarily (PaddleOCR requires file path)
        cv2.imwrite(str(tmp_path), img_bgr)
        
        # Run OCR: predict returns iterator, get first (and only) result
        result = next(iter(self.ocr.predict(input=str(tmp_path))))

        # Save raw OCR results if output directory is configured
        if self.out_dir:
            result.save_to_json(str(self.out_dir))
            result.save_to_img(str(self.out_dir))

        # Convert to standardized format and filter by threshold
        items = self.result_to_items(result)
        filtered_items = [it for it in items if it["score"] >= self.score_thr]
        
        return filtered_items, img_bgr

    def annotate(self, img_bgr, items: List[Dict], out_name: str = "annotated.png") -> Optional[Path]:
        """
        Draw bounding polygons and text labels on image for visualization.
        
        Annotates the image with:
            - Green polylines showing text bounding polygons
            - Text labels with confidence scores above each detection
        
        Args:
            img_bgr: Input image in BGR format (OpenCV format)
            items: List of detection items with "polygon", "text", and "score" keys
            out_name: Output filename for annotated image
        
        Returns:
            Path to saved annotated image if out_dir is configured, else None
        
        Note:
            Creates a copy of the input image, so original is not modified.
            Uses green color (0, 255, 0) for polygons and red (36, 36, 255) for text.
        """
        img = img_bgr.copy()
        
        # Draw each detection
        for it in items:
            # Draw polygon (green, 2px thick, closed)
            pts = np.array(it["polygon"], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            # Draw text label above polygon (top-left corner)
            x, y = pts[0]
            label_text = f'{it["text"]} ({it["score"]:.2f})'
            cv2.putText(img, label_text,
                        (int(x), int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,36,255), 1, cv2.LINE_AA)

        # Save annotated image if output directory is configured
        if self.out_dir:
            out_path = self.out_dir / out_name
            cv2.imwrite(str(out_path), img)
            return out_path
        return None

    def save_items(self, items: List[Dict], txt_name="result.txt", json_name="result.json"):
        """
        Save detection results to text and JSON files.
        
        Exports OCR results in two formats:
            - TXT file: Plain text, one line per detection (text only)
            - JSON file: Complete data with text, score, and polygon coordinates
        
        Args:
            items: List of detection items to save
            txt_name: Filename for text output (default: "result.txt")
            json_name: Filename for JSON output (default: "result.json")
        
        Note:
            Only saves if out_dir was specified during initialization.
            Files are saved with UTF-8 encoding to support multilingual text.
        """
        if not self.out_dir:
            return
        
        # Save as plain text (one line per detection)
        (self.out_dir / txt_name).write_text(
            "\n".join([it["text"] for it in items]), encoding="utf-8"
        )
        
        # Save as JSON (complete data with scores and polygons)
        (self.out_dir / json_name).write_text(
            json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def predict(self, img_path: Path | str):
        """
        Complete OCR pipeline: detect, save, and annotate.
        
        Main entry point that performs full OCR workflow:
            1. Run OCR detection/recognition
            2. Save results to files (TXT and JSON)
            3. Create annotated visualization image
        
        Args:
            img_path: Path to input image file
        
        Returns:
            List of detected text items (filtered by score_thr)
        
        Example:
            >>> detector = OCRTextDetector(out_dir="./output")
            >>> items = detector.predict("document.png")
            >>> # Results saved to ./output/result.txt, result.json, annotated.png
        """
        print("Start predicting...")
        
        # Run OCR and get results
        items, img_bgr = self.predict_items(img_path)
        
        # Save results to files
        self.save_items(items)
        
        # Create annotated visualization
        self.annotate(img_bgr, items)

        print("--- Extracting from Images is completed ---")
        return items
