"""
Framework: Streamlit
    For deploying model as demo
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
import streamlit as st
import sys

# ensure we can import sibling modules when run via streamlit
sys.path.append(str(Path(__file__).parent))
from preprocess import render as render_preprocess  # noqa: E402
try:
    from query import render as render_query  # noqa: E402
except Exception:
    render_query = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _python_bin() -> str:
    # Prefer python3 inside the container
    return "python3" if sys.platform != "win32" else "python"


def run_cmd(args: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    logs: list[str] = []
    for line in proc.stdout or []:
        logs.append(line.rstrip())
        # stream to the UI progressively
        st.write(line.rstrip())
    code = proc.wait()
    return code, "\n".join(logs)


st.set_page_config(page_title="CSE Course RAG", layout="wide")
st.title("CSE Course RAG")

st.markdown(
    textwrap.dedent(
        f"""
        - **Project root**: `{ROOT}`
        - **Data workspace**: `{DATA_DIR}`
        - Pipelines are executed via `run.py` in the container.
        """
    )
)

with st.expander("Workspace Paths", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        data_raw = st.text_input("data/raw", value=str(DATA_DIR / "raw"))
        data_cvt = st.text_input("data/converted", value=str(DATA_DIR / "converted"))
    with col2:
        data_root = st.text_input("data root", value=str(DATA_DIR))
        out_root = st.text_input("out root (processed)", value=str(DATA_DIR / "processed"))
    dpi = st.number_input("DPI (convert)", min_value=72, max_value=600, value=220, step=1)

# Tabs
tab_preprocessing, tab_query = st.tabs(["Preprocessing", "Query"])

# Serving preprocessing tab
render_preprocess(
    tab_preprocessing,
    python_bin=_python_bin,
    run_cmd=run_cmd,
    data_raw=data_raw,
    data_cvt=data_cvt,
    data_root=data_root,
    out_root=out_root,
    dpi=int(dpi),
)

# Serving query tab
if render_query is not None:
    render_query(tab_query)
else:
    with tab_query:
        st.subheader("Query")
        st.info("Coming soon.")
