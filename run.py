from preprocessing.convert_data_to_img import *
from preprocessing.dectector import OCRTextDetector

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
    detector = OCRTextDetector(out_dir="./test")

    # Converting to PNG
    data_cvt()

    # Detecting & Recognizing text 
    img_path = ROOT / "data_cvt" / "CO2011_MaThematical Modeling" / "Syllabus" / "slide_001.png"
    detector.run(str(img_path))

if __name__ == "__main__":
    main()