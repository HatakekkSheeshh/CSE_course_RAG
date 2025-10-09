"""
Build SVG overlays for OCR results on syllabus images.
This module provides functionality to create SVG overlays that visualize OCR results
on syllabus images, including bounding boxes and text annotations.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import base64, html

def _normalize_bbox(bbox: Any, W: int, H: int) -> Tuple[str, List[Tuple[float, float]]]:
    def sx(x): return x*W if 0.0 <= x <= 1.0 else x
    def sy(y): return y*H if 0.0 <= y <= 1.0 else y
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox); x1, y1, x2, y2 = sx(x1), sy(y1), sx(x2), sy(y2)
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return "rect", [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
        pts = [(sx(float(x)), sy(float(y))) for x, y in bbox[:4]]
        return "poly", pts
    return "unknown", []

def write_svg_overlay(img_path: Path, items: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        W, H = im.size

    # embed raster as base64 (self-contained)
    b64 = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n']
    svg += [
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>\n',
        f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" fill="none" stroke="blue" stroke-width="1"/>\n',
        f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="{W}" height="{H}"/>\n'
    ]

    drawn = 0
    for it in items:
        bbox = it.get("bbox") or it.get("box") or it.get("poly")
        kind, pts = _normalize_bbox(bbox, W, H)
        txt = (it.get("text") or "")[:40]
        if kind == "rect":
            x1,y1 = pts[0]; x2,y2 = pts[2]
            svg.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{(x2-x1):.1f}" height="{(y2-y1):.1f}" '
                       f'fill="none" stroke="red" stroke-width="2"/>\n')
            if txt: svg.append(f'<text x="{x1:.1f}" y="{max(0,y1-4):.1f}" font-size="14" fill="red">{html.escape(txt)}</text>\n')
            drawn += 1
        elif kind == "poly":
            points = " ".join(f'{x:.1f},{y:.1f}' for x,y in pts)
            svg.append(f'<polygon points="{points}" fill="none" stroke="red" stroke-width="2"/>\n')
            if txt:
                x1,y1 = pts[0]
                svg.append(f'<text x="{x1:.1f}" y="{max(0,y1-4):.1f}" font-size="14" fill="red">{html.escape(txt)}</text>\n')
            drawn += 1

    svg.append("</svg>\n")
    out_svg = out_dir / f"{img_path.stem}.overlay.svg"
    out_svg.write_text("".join(svg), encoding="utf-8")
    print(f"[overlay] W={W} H={H} drawn={drawn} -> {out_svg}")
    return out_svg
