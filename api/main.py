"""
FastAPI application exposing the CSE Course RAG pipeline via HTTP.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.llm_client import LLMClient
from rag.query_pipeline import NO_INFO_MESSAGE, QueryPipeline, RetrievedChunk

app = FastAPI(title="CSE Course RAG API", version="0.1.0")

allowed_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    confidence: float
    course: Optional[str] = None
    metadata: Optional[Dict] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=4, description="End-user question")
    course: Optional[str] = Field(
        default=None,
        description="Optional course folder name to scope retrieval (matches data/indices/<course>)",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Number of top reranked chunks to surface in the resSponse",
    )


class QueryResponse(BaseModel):
    status: str
    question: str
    answer: str
    sources: List[SourceChunk]
    llm_used: bool
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    indices_loaded: bool
    llm_enabled: bool


class CoursesResponse(BaseModel):
    courses: List[str]


@lru_cache(maxsize=1)
def get_pipeline() -> QueryPipeline:
    data_dir = Path(os.getenv("RAG_DATA_DIR", "data"))
    index_dir = Path(os.getenv("RAG_INDEX_DIR", data_dir / "indices"))
    retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "8"))
    rerank_k = int(os.getenv("RAG_RERANK_K", "5"))
    confidence_threshold = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.1"))
    only_course = os.getenv("RAG_ONLY_COURSE")

    return QueryPipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        confidence_threshold=confidence_threshold,
        only_course=only_course,
    )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("OPENAI_MODEL")
    try:
        return LLMClient(provider=provider, model=model)
    except ValueError as exc:  # pragma: no cover - provider validation
        raise HTTPException(status_code=400, detail=str(exc))


def build_source_chunks(
    retrieved: List[RetrievedChunk],
    reranked: Sequence,
    limit: int,
) -> List[SourceChunk]:
    retrieved_map = {chunk.chunk_id: chunk for chunk in retrieved}
    sources: List[SourceChunk] = []
    for result in reranked[:limit]:
        retrieved_chunk = retrieved_map.get(result.chunk_id)
        sources.append(
            SourceChunk(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                confidence=result.confidence,
                course=getattr(retrieved_chunk, "course", None),
                metadata=(retrieved_chunk.metadata if retrieved_chunk else result.metadata),
            )
        )
    return sources


@app.on_event("startup")
def preload_components() -> None:
    # Trigger lazy singletons so first request is fast.
    get_pipeline()
    try:
        get_llm_client()
    except HTTPException:
        # LLM is optional; keep serving retrieval-only answers.
        pass


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    pipeline = get_pipeline()
    llm_client = get_llm_client()
    return HealthResponse(
        status="ok",
        indices_loaded=bool(pipeline),
        llm_enabled=llm_client.enabled,
    )


@app.get("/courses", response_model=CoursesResponse)
def list_courses() -> CoursesResponse:
    pipeline = get_pipeline()
    pipeline._ensure_loaded()
    return CoursesResponse(courses=sorted(pipeline._indices.keys()))


@app.post("/api/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    result = pipeline.answer(payload.question, course=payload.course)

    if result.get("status") != "ok":
        return QueryResponse(
            status=result.get("status", "no_info"),
            question=payload.question,
            answer=result.get("message", NO_INFO_MESSAGE),
            sources=[],
            reason=result.get("reason"),
            llm_used=False,
        )

    reranked = result.get("reranked", [])
    retrieved = result.get("retrieved", [])
    sources = build_source_chunks(retrieved, reranked, payload.top_k)

    llm_client = get_llm_client()
    llm_used = llm_client.enabled if llm_client else False

    if llm_used:
        contexts = [chunk.text for chunk in sources if chunk.text.strip()]
        answer = llm_client.generate_answer(payload.question, contexts)
    else:
        answer = sources[0].text if sources else NO_INFO_MESSAGE

    return QueryResponse(
        status="ok",
        question=payload.question,
        answer=answer,
        sources=sources,
        llm_used=llm_used,
    )

