from __future__ import annotations

import streamlit as st
from pathlib import Path

from rag.query_pipeline import QueryPipeline, NO_INFO_MESSAGE
from rag.llm_client import LLMClient


WELCOME_TEXT = "Welcome the user, I am a chatbot specifically used for finding CSE course in short time."


@st.cache_resource(show_spinner=False)
def get_pipeline() -> QueryPipeline:
    return QueryPipeline(
        data_dir=Path("data"),
        index_dir=Path("data") / "indices",
    )


@st.cache_resource(show_spinner=False)
def get_llm_client() -> LLMClient:
    return LLMClient()


def render(tab) -> None:
    with tab:
        st.subheader("Query")
        st.info(WELCOME_TEXT)

        question = st.text_area("Ask about a CSE course", height=120)
        course = st.text_input("Specific course folder (optional)")
        run = st.button("Run Query", type="primary")

        if not run:
            return

        if not question.strip():
            st.warning("Please enter a question first.")
            return

        try:
            pipeline = get_pipeline()
        except Exception as exc:
            st.error(f"Failed to initialize pipeline: {exc}")
            return

        with st.spinner("Searching and reranking..."):
            result = pipeline.answer(question.strip(), course=course.strip() or None)

        if result.get("status") != "ok":
            st.warning(NO_INFO_MESSAGE)
            return

        best = result["best_chunk"]
        st.success(f"Confidence: {result.get('confidence', 0.0):.0%}")
        with st.expander("Top context chunk", expanded=True):
            st.write(best.text)
            if best.metadata:
                st.json(best.metadata)

        llm_client = get_llm_client()
        if llm_client.enabled:
            with st.spinner("Generating answer..."):
                contexts = [best.text] + [
                    chunk.text for chunk in result.get("reranked", [])[1:3]
                ]
                try:
                    answer = llm_client.generate_answer(question, contexts)
                    st.write(answer)
                except Exception as exc:
                    st.error(f"LLM generation failed: {exc}")
        else:
            st.info("LLM disabled (set GEMINI_API_KEY or configure Ollama to enable answers).")

