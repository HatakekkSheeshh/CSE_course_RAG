"""
FastAPI application exposing the CSE Course RAG pipeline via HTTP.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from rag.conversation import ConversationManager
from rag.llm_client import LLMClient
from rag.query_pipeline import NO_INFO_MESSAGE, QueryPipeline, RetrievedChunk

# Optional import for query rewriting
try:
    from rag.query_rewriter import create_query_rewriter
except ImportError:
    create_query_rewriter = None  # type: ignore

app = FastAPI(title="CSE Course RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_cors_allow_origins() or ["*"],
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
    question: str = Field(..., min_length=1, description="End-user question")
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
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation history (optional, auto-generated if not provided)",
    )
    start_new_conversation: bool = Field(
        default=False,
        description="If True, start a new conversation (clear history for this session or create new session)",
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


# Global conversation manager
conversation_manager = ConversationManager()

@lru_cache(maxsize=1)
def get_pipeline() -> QueryPipeline:

    rag_config = config.get_rag_pipeline_config()

    query_rewriter = None
    if create_query_rewriter is not None:
        try:
            llm_client = get_llm_client()
            query_rewriter = create_query_rewriter(llm_client=llm_client)
        except Exception:
            pass

    return QueryPipeline(
        data_dir=rag_config["data_dir"],
        index_dir=rag_config["index_dir"],
        retrieval_k=rag_config["retrieval_k"],
        rerank_k=rag_config["rerank_k"],
        confidence_threshold=rag_config["confidence_threshold"],
        only_course=rag_config["only_course"],
        query_rewriter=query_rewriter,
    )

@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    # Get configuration from centralized config module
    llm_config = config.get_llm_provider_config()
    provider = llm_config["provider"]
    
    if provider == "gemini":
        return LLMClient(
            provider=provider,
            model=llm_config["model"],
            api_key=llm_config["api_key"],
        )
    elif provider == "ollama":
        return LLMClient(
            provider=provider,
            model=llm_config["model"],
            base_url=llm_config["base_url"],
        )
    elif provider == "qwen":
        # Qwen uses Ollama as backend
        return LLMClient(
            provider=provider,
            model=llm_config["model"],
            base_url=llm_config["base_url"],
        )
    else:
        # This should never happen due to validation in config, but just in case
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported: 'gemini', 'ollama', 'qwen'"
        )

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
    # Generate session_id if not provided
    import uuid

    if payload.start_new_conversation:
        if payload.session_id:
            conversation_manager.clear_history(payload.session_id)
        session_id = str(uuid.uuid4())  # Always create new session
    else:
        session_id = payload.session_id or str(uuid.uuid4())
    
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
        
        # Get conversation history if available
        conversation_history = None
        if session_id:
            history = conversation_manager.format_history_for_llm(
                session_id,
                max_messages=10,
            )
            if history:
                conversation_history = history
        
        # Use generate_answer with history support
        answer = llm_client.generate_answer(
            query=payload.question,
            contexts=contexts,
            conversation_history=conversation_history,
        )
        
        # Save to conversation history
        conversation_manager.add_message(session_id, "user", payload.question)
        conversation_manager.add_message(session_id, "assistant", answer)
    else:
        answer = sources[0].text if sources else NO_INFO_MESSAGE

    return QueryResponse(
        status="ok",
        question=payload.question,
        answer=answer,
        sources=sources,
        llm_used=llm_used,
    )


@app.post("/api/query/stream")
async def query_rag_stream(payload: QueryRequest):
    """
    Streaming endpoint for RAG queries with conversation history support.
    
    Returns Server-Sent Events (SSE) stream with tokens as they are generated.
    """
    import uuid
    
    # Generate session_id if not provided
    import uuid
    
    # If start_new_conversation is True, create new session or clear existing one
    if payload.start_new_conversation:
        if payload.session_id:
            # Clear history for existing session
            conversation_manager.clear_history(payload.session_id)
        session_id = str(uuid.uuid4())  # Always create new session
    else:
        session_id = payload.session_id or str(uuid.uuid4())
    
    pipeline = get_pipeline()
    result = pipeline.answer(payload.question, course=payload.course)

    async def generate_stream():
        if result.get("status") != "ok":
            # Send error as single event
            error_data = {
                "type": "error",
                "status": result.get("status", "no_info"),
                "message": result.get("message", NO_INFO_MESSAGE),
                "reason": result.get("reason"),
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        reranked = result.get("reranked", [])
        retrieved = result.get("retrieved", [])
        sources = build_source_chunks(retrieved, reranked, payload.top_k)

        # Send sources first
        sources_data = {
            "type": "sources",
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "text": s.text,
                    "score": s.score,
                    "confidence": s.confidence,
                    "course": s.course,
                }
                for s in sources
            ],
        }
        yield f"data: {json.dumps(sources_data)}\n\n"

        llm_client = get_llm_client()
        llm_used = llm_client.enabled if llm_client else False

        if llm_used:
            contexts = [chunk.text for chunk in sources if chunk.text.strip()]
            
            # Get conversation history if available
            conversation_history = None
            if session_id:
                history = conversation_manager.format_history_for_llm(
                    session_id,
                    max_messages=10,
                )
                if history:
                    conversation_history = history
            
            # Stream answer using stream_generate_answer with history support
            full_answer = ""
            token_count = 0
            async for token in llm_client.stream_generate_answer(
                query=payload.question,
                contexts=contexts,
                conversation_history=conversation_history,
            ):
                if token:  # Only send non-empty tokens
                    token_count += 1
                    full_answer += token
                    token_data = {
                        "type": "token",
                        "token": token,
                    }

                    print(f"[STREAM] Token #{token_count}: {repr(token[:50])}")  # Log first 50 chars

                    yield f"data: {json.dumps(token_data)}\n\n"
            
            print(f"[STREAM] Completed: {token_count} tokens, total length: {len(full_answer)}")
            
            # Send completion event
            completion_data = {
                "type": "done",
                "answer": full_answer,
                "llm_used": True,
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
            # Save to conversation history
            conversation_manager.add_message(session_id, "user", payload.question)
            conversation_manager.add_message(session_id, "assistant", full_answer)
        else:
            # Fallback if no LLM is used
            answer = sources[0].text if sources else NO_INFO_MESSAGE
            answer_data = {
                "type": "done",
                "answer": answer,
                "llm_used": False,
            }
            yield f"data: {json.dumps(answer_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )

