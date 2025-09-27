from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf2image import convert_from_path
import subprocess, shutil, os, tempfile, uuid

# ---------- helpers ----------

def render_pdf_to_images(
    pdf_path: Path, out_dir: Path, dpi: int, fmt: str,
    poppler_path: Optional[str], overwrite: bool
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Skip nếu đã có ảnh
    if not overwrite:
        existing = sorted(out_dir.glob(f"slide_*.{fmt}"))
        if existing:
            return [str(p) for p in existing]

    pages = convert_from_path(
        str(pdf_path), dpi=dpi, fmt=fmt, poppler_path=poppler_path
    )
    num_w = max(3, len(str(len(pages))))  # zero-pad: 001...
    outs: List[str] = []
    for i, img in enumerate(pages, 1):
        fp = out_dir / f"slide_{i:0{num_w}d}.{fmt}"
        img.save(fp)
        outs.append(str(fp))
    return outs


def pptx_to_pdf(pptx_path: Path, out_pdf_dir: Path, timeout: int = 300) -> Path:
    pptx_path = pptx_path.resolve()
    out_pdf_dir = out_pdf_dir.resolve()

    if not pptx_path.exists():
        raise FileNotFoundError(f"Input not found: {pptx_path}")

    soffice = shutil.which("soffice")
    if not soffice:
        raise RuntimeError("LibreOffice 'soffice' not found in PATH. Install libreoffice-impress.")

    is_snap = "/snap/" in Path(soffice).as_posix()
    under_mnt = pptx_path.as_posix().startswith("/mnt/")
    needs_copy = is_snap and under_mnt

    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    dst_pdf = out_pdf_dir / (pptx_path.stem + ".pdf")

    profile_dir = Path(tempfile.gettempdir()) / f"lo_profile_{uuid.uuid4().hex}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_url = f"file://{profile_dir.as_posix()}"

    work_dir = None
    src_for_lo = pptx_path
    try:
        if needs_copy:
            work_dir = Path(tempfile.mkdtemp(prefix="lo_work_"))
            src_for_lo = work_dir / pptx_path.name
            shutil.copy2(pptx_path, src_for_lo)

        # 6) Chạy convert
        cmd = [
            soffice,
            "--headless", "--norestore", "--nodefault", "--nolockcheck",
            f"-env:UserInstallation={profile_url}",
            "--convert-to", "pdf:impress_pdf_Export",
            "--outdir", str(out_pdf_dir),
            str(src_for_lo),
        ]

        proc = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, text=True
        )

        # 7) Kiểm tra file đích
        if not dst_pdf.exists():
            # fallback: tìm mọi pdf bắt đầu bằng stem (một số bản LO đổi tên)
            cands = sorted(out_pdf_dir.glob(pptx_path.stem + "*.pdf"))
            if cands:
                return cands[0]

            # Nếu vẫn không thấy, trả về thông tin debug hữu ích
            raise RuntimeError(
                "PDF not found after conversion.\n"
                f"Expected: {dst_pdf}\n"
                f"SOFFICE: {soffice}\n"
                f"SNAP: {is_snap} | UNDER_MNT: {under_mnt}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        return dst_pdf

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"LibreOffice failed (exit {e.returncode}).\n"
            f"SOFFICE: {soffice}\n"
            f"SNAP: {is_snap} | UNDER_MNT: {under_mnt}\n"
            f"CMD: {' '.join(e.cmd) if hasattr(e, 'cmd') else 'n/a'}\n"
            f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice timed out during conversion.")
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def rel_to_out(src: Path, data_root: Path, out_root: Path) -> Path:
    rel = src.relative_to(data_root)                        # ex: CO1005/.../Chapter_0.pdf
    out_dir = out_root / rel.with_suffix("")                # ex: data_cvt/CO1005/.../Chapter_0
    return out_dir


def process_one(
    src: Path, data_root: Path, out_root: Path, dpi: int, fmt: str,
    overwrite: bool, poppler_path: Optional[str], keep_temp_pdf: bool
) -> Tuple[str, List[str]]:
    out_dir = rel_to_out(src, data_root, out_root)

    if not overwrite and any(out_dir.glob(f"slide_*.{fmt}")):
        return str(src), [str(p) for p in sorted(out_dir.glob(f"slide_*.{fmt}"))]

    if src.suffix.lower() == ".pdf":
        outs = render_pdf_to_images(src, out_dir, dpi, fmt, poppler_path, overwrite)
        return str(src), outs

    # PPTX
    tmp_pdf_dir = out_dir.parent / "tmp_pptx_pdf"
    pdf = pptx_to_pdf(src, tmp_pdf_dir)
    outs = render_pdf_to_images(pdf, out_dir, dpi, fmt, poppler_path, overwrite)
    if not keep_temp_pdf:
        try:
            pdf.unlink(missing_ok=True)
            if tmp_pdf_dir.exists() and not any(tmp_pdf_dir.iterdir()):
                tmp_pdf_dir.rmdir()
        except Exception:
            pass
    return str(src), outs

# ---------- public API ----------
def convert_chapters_and_syllabus_to_images_parallel(
    data_root: Path,
    out_root: Path,                
    dpi: int = 220,
    fmt: str = "png",
    overwrite: bool = False,
    poppler_path: Optional[str] = None, 
    keep_temp_pdf: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Loop data_root/** for:
      - Chapter_*.pdf / Chapter_*.pptx
      - Syllabus.pdf / Syllabus.pptx
    → render image into out_root.
    """
    fmt = fmt.lower()
    assert fmt in {"png", "jpg", "jpeg"}
    data_root = data_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Target
    def is_target(p: Path) -> bool:
        if not p.is_file():
            return False
        name = p.name
        suf = p.suffix.lower()
        if suf not in {".pdf", ".pptx"}:
            return False
        if name.lower().startswith("chapter_") or name.lower().startswith("syllabus"):
            return True
        return False

    targets: List[Path] = [p for p in sorted(data_root.rglob("*")) if is_target(p)]
    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(4, cpu)

    results: Dict[str, List[str]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                process_one, src, data_root, out_root, dpi, fmt,
                overwrite, poppler_path, keep_temp_pdf
            )
            for src in targets
        ]
        for fut in as_completed(futs):
            try:
                k, v = fut.result()
                results[k] = v
            except Exception as e:
                print(f"[ERROR] {e}")

    return results


