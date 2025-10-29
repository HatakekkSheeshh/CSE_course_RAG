import streamlit as st
from typing import Callable

def render(
    tab,
    *,
    python_bin: Callable[[], str],
    run_cmd: Callable[[list[str]], tuple[int, str]],
    data_raw: str,
    data_cvt: str,
    data_root: str,
    out_root: str,
    dpi: int,
) -> None:
    """Render the Preprocessing tab (convert → ocr → merge)."""
    with tab:
        st.subheader("Preprocessing")

        t1, t2, t3 = st.tabs(["Convert", "OCR for Extracting", "Merge"]) 

        with t1:
            st.subheader("Convert raw documents to images")
            if st.button("Run Convert", type="primary"):
                st.info("Running convert…")
                cmd = [
                    python_bin(),
                    "run.py",
                    "--convert",
                    "--dpi",
                    str(dpi),
                ]
                code, _ = run_cmd(cmd)
                if code == 0:
                    st.success("Convert done")
                else:
                    st.error(f"Convert failed with code {code}")

        with t2:
            st.subheader("OCR for Extracting syllabus")
            if st.button("Run OCR/Extract", type="primary"):
                st.info("Running OCR/Extract…")
                cmd = [
                    python_bin(),
                    "run.py",
                    "--ocr",
                ]
                code, _ = run_cmd(cmd)
                if code == 0:
                    st.success("OCR/Extract done")
                else:
                    st.error(f"OCR/Extract failed with code {code}")

        with t3:
            st.subheader("Merge parsed syllabus JSON into course-level outputs")
            only_course = st.text_input("Only course (optional)", value="")
            if st.button("Run Merge", type="primary"):
                st.info("Running Merge…")
                cmd = [
                    python_bin(),
                    "run.py",
                    "--merge",
                ]
                if only_course.strip():
                    cmd += ["--only-course", only_course.strip()]
                code, _ = run_cmd(cmd)
                if code == 0:
                    st.success("Merge done")
                else:
                    st.error(f"Merge failed with code {code}")