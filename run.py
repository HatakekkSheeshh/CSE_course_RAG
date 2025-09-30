from preprocessing.convert_data_to_img import *
from preprocessing.dectector import OCRTextDetector
from preprocessing.extract_syllabus import extract_syllabus
from dataclasses import asdict
import json 
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def data_cvt():
    data_root = ROOT / "data"
    out_root  = ROOT / "data_cvt"
    manifest = convert_chapters_and_syllabus_to_images_parallel(
        data_root=data_root,
        out_root=out_root,
        dpi=220,
        fmt="png",
        overwrite=False,                    
        # poppler_path=r"C:\Program Files\poppler-24.02.0\Library\bin",  # Windows 
        keep_temp_pdf=False,
        max_workers=3
    )

    for src, imgs in manifest.items():
        print(f"{src} -> {len(imgs)} image, Example: {imgs[:2]}")

def main():
    """ Pipeline """ 
    # Initialize
    detector = OCRTextDetector(out_dir="./test1 ")

    # Converting to PNG
    print("Converting...")
    # data_cvt()

    # Detecting & Recognizing text 
    print("Detecting...")
    img_path = ROOT / "data_cvt" / "CO1027_Programming_Fudamentals" / "Syllabus" / "slide_001.png"
    items = detector.run(str(img_path))

    # Filter [key: value]
    print("Extracting...")
    syllabus = extract_syllabus(items)
    out_dir = ROOT / "test2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "syllabus_extracted.json"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(syllabus), f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved JSON to {out_json}")

if __name__ == "__main__":
    main()