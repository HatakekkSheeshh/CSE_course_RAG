import os
import subprocess
import sys
import textwrap
from pathlib import Path
import streamlit as st

with tab_preprocessing:
    st.subheader("Preprocessing")
    
    tab1, tab2, tab3 = st.tabs(["1) Convert", "2) OCR/Extract", "3) Merge", ""])

    with tab1:
        st.subheader("Convert raw documents to images")
        if st.button("Run Convert", type="primary"):
            st.info("Running convert…")
            cmd = [
                _python_bin(),
                "run.py",
                "--convert",
                "--dpi",
                str(dpi),
                "--data-raw",
                data_raw,
                "--data-cvt-root",
                data_cvt,
            ]
            code, _ = run_cmd(cmd)
            if code == 0:
                st.success("Convert done")
            else:
                st.error(f"Convert failed with code {code}")

    with tab2:
        st.subheader("OCR & Extract syllabus")
        if st.button("Run OCR/Extract", type="primary"):
            st.info("Running OCR/Extract…")
            cmd = [
                _python_bin(),
                "run.py",
                "--ocr",
                "--data-root",
                data_root,
                "--data-cvt-root",
                data_cvt,
            ]
            code, _ = run_cmd(cmd)
            if code == 0:
                st.success("OCR/Extract done")
            else:
                st.error(f"OCR/Extract failed with code {code}")

    with tab3:
        st.subheader("Merge parsed syllabus JSON into course-level outputs")
        only_course = st.text_input("Only course (optional)", value="")
        if st.button("Run Merge", type="primary"):
            st.info("Running Merge…")
            cmd = [
                _python_bin(),
                "run.py",
                "--merge",
                "--data-root",
                data_root,
                "--out-root",
                out_root,
            ]
            if only_course.strip():
                cmd += ["--only-course", only_course.strip()]
            code, _ = run_cmd(cmd)
            if code == 0:
                st.success("Merge done")
            else:
                st.error(f"Merge failed with code {code}")