from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

# ========================= MERGE UTILITIES =============================

def _is_nullish(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "--"))

def _set_if_nullish(target: Dict, key: str, src_val: Any):
    if _is_nullish(src_val):
        return
    if key not in target or _is_nullish(target[key]):
        target[key] = src_val

def _merge_evaltype(dst_et: Dict[str, Any], src_et: Dict[str, Any]) -> Dict[str, Any]:
    for fld in ("notes", "ratio", "format", "duration_min"):
        _set_if_nullish(dst_et, fld, src_et.get(fld))
    return dst_et

def _merge_assessment(dst_comp: Dict[str, Any], src_comp: Dict[str, Any]) -> Dict[str, Any]:
    for fld in ("hours", "credits", "ratio", "format", "duration_min", "notes"):
        _set_if_nullish(dst_comp, fld, src_comp.get(fld))

    dst_list = dst_comp.get("evaluation_type") or []
    src_list = src_comp.get("evaluation_type") or []
    idx = {(et.get("name") or "").strip().lower(): et for et in dst_list if et.get("name")}
    for et in src_list:
        nm = (et.get("name") or "").strip()
        if not nm:
            continue
        key = nm.lower()
        if key in idx:
            _merge_evaltype(idx[key], et)
        else:
            dst_list.append(et)
            idx[key] = et
    dst_comp["evaluation_type"] = dst_list
    return dst_comp

def _merge_assessments(dst: List[Dict[str, Any]], src: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not dst:
        return src[:] if src else []
    if not src:
        return dst
    idx = {(c.get("name") or "").strip().lower(): c for c in dst if c.get("name")}
    for comp in src:
        nm = (comp.get("name") or "").strip()
        if not nm:
            continue
        key = nm.lower()
        if key in idx:
            _merge_assessment(idx[key], comp)
        else:
            dst.append(comp)
            idx[key] = comp
    return dst

def deep_merge_syllabus(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    if "course_info" in src:
        dst.setdefault("course_info", {})
        for k, v in src["course_info"].items():
            _set_if_nullish(dst["course_info"], k, v)

    if "prerequisites" in src:
        dst.setdefault("prerequisites", {})
        for key in ("recommended", "prereq", "coreq"):
            a = dst["prerequisites"].get(key) or []
            b = src["prerequisites"].get(key) or []
            seen = set(x.strip().lower() for x in a if isinstance(x, str))
            for x in b:
                if isinstance(x, str) and x.strip().lower() not in seen:
                    a.append(x)
                    seen.add(x.strip().lower())
            dst["prerequisites"][key] = a

    if "assessments" in src:
        dst["assessments"] = _merge_assessments(dst.get("assessments") or [], src["assessments"] or [])

    if "metadata" in src:
        dst.setdefault("metadata", {})
        for k, v in src["metadata"].items():
            _set_if_nullish(dst["metadata"], k, v)

    _set_if_nullish(dst, "raw_ocr_text", src.get("raw_ocr_text"))

    for k, v in src.items():
        if k in ("course_info", "prerequisites", "assessments", "metadata", "raw_ocr_text", "schema_version"):
            continue
        if isinstance(v, dict):
            dst.setdefault(k, {})
            for kk, vv in v.items():
                _set_if_nullish(dst[k], kk, vv)
        elif isinstance(v, list):
            if not dst.get(k):
                dst[k] = v
        else:
            _set_if_nullish(dst, k, v)

    if "schema_version" in src and _is_nullish(dst.get("schema_version")):
        dst["schema_version"] = src["schema_version"]
    return dst

def merge_folder(parsed_dir: Path) -> Dict[str, Any]:
    files = sorted(parsed_dir.glob("*.syllabus.json")) or sorted(parsed_dir.glob("*.json"))
    merged: Dict[str, Any] = {}
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skip {fp.name}: {e}")
            continue
        merged = deep_merge_syllabus(merged, data)
    return merged

def save_outputs(merged: Dict[str, Any], out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.syllabus.merged.json"
    json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[JSON] ->", json_path)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = pa.Table.from_pylist([merged])  # 1 record, giữ nested
        pq.write_table(tbl, str(out_dir / f"{name}.syllabus.merged.parquet"))
        print("[PARQUET] ->", out_dir / f"{name}.syllabus.merged.parquet")
    except Exception as e_pa:
        try:
            import pandas as pd
            from pandas import json_normalize
            df = json_normalize(merged, sep=".")
            df.to_parquet(out_dir / f"{name}.syllabus.merged.parquet", index=False)
            print("[PARQUET/pandas] ->", out_dir / f"{name}.syllabus.merged.parquet")
        except Exception as e_pd:
            print(f"[WARN] Parquet not written: {e_pa or e_pd}")



