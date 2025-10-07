from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import html

def _normalize_bbox(bbox: Any) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Accept either [x1,y1,x2,y2] or polygon [[x,y], ...].
    Return ("rect"|"poly", list_of_points)
    """
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = map(float, bbox)
        return "rect", [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
        pts = [(float(x), float(y)) for x, y in bbox[:4]]
        return "poly", pts
    return "unknown", []

def _svg_header(w: int, h: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'

def _svg_footer() -> str:
    return "</svg>\n"

def _svg_rect(x1: float, y1: float, x2: float, y2: float) -> str:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="red" stroke-width="2" />\n'

def _svg_poly(pts: List[Tuple[float, float]]) -> str:
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polygon points="{p}" fill="none" stroke="red" stroke-width="2" />\n'

def _svg_label(x: float, y: float, text: str) -> str:
    t = html.escape(text[:40]) if text else ""
    return f'<text x="{x:.1f}" y="{max(0.0, y - 4):.1f}" font-size="14" fill="red">{t}</text>\n'

def write_svg_overlay(img_path: Path, items: List[Dict[str, Any]], out_dir: Path) -> Path:
    """
    Build an SVG overlay (red boxes + short labels) for a given image and items.
    The SVG is independent (no embedded raster), sized to the page for easy stacking.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        w, h = im.size

    svg = [_svg_header(w, h)]
    for it in items:
        kind, pts = _normalize_bbox(it.get("bbox"))
        if kind == "rect" and len(pts) == 4:
            x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[2][0], pts[2][1]
            svg.append(_svg_rect(x1, y1, x2, y2))
            if it.get("text"):
                svg.append(_svg_label(x1, y1, it["text"]))
        elif kind == "poly" and len(pts) >= 4:
            svg.append(_svg_poly(pts))
            if it.get("text"):
                x1, y1 = pts[0]
                svg.append(_svg_label(x1, y1, it["text"]))

    svg.append(_svg_footer())
    out_svg = out_dir / f"{img_path.stem}.overlay.svg"
    out_svg.write_text("".join(svg), encoding="utf-8")
    return out_svg
