"""
End-to-end retrieval plus reranking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from models.embedding import Embedding
from models.indexing import load_index, search
from models.reranker import Reranker, RerankResult

# Optional import for query rewriting
try:
    from rag.query_rewriter import QueryRewriter
except ImportError:
    QueryRewriter = None  # type: ignore


NO_INFO_MESSAGE = "I have no info"


@dataclass
class CourseIndex:
    name: str
    path: Path
    chunk_ids: List[str]
    metadata: Dict[str, Dict]
    index: any  # FAISS IndexFlatIP, but keep loose to avoid circular type


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict
    score: float
    course: str


class QueryPipeline:
    """
    Load FAISS indices, retrieve candidate chunks, rerank, and expose results.
    """

    def __init__(
        self,
        *,
        data_dir: Path | str = Path("data"),
        index_dir: Path | str = Path("data") / "indices",
        retrieval_k: int = 8,
        rerank_k: int = 5,
        confidence_threshold: float = 0.1,
        preload: bool = True,
        only_course: Optional[str] = None,
        query_rewriter: Optional[any] = None,  # QueryRewriter type, but avoid circular import
    ) -> None:
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.confidence_threshold = confidence_threshold
        self.only_course = only_course

        self.embedding = Embedding()
        self.reranker = Reranker()
        self.query_rewriter = query_rewriter

        self._indices: Dict[str, CourseIndex] = {}
        if preload:
            self._load_indices()

    # ------------------------------------------------------------------ #
    def _load_indices(self) -> None:
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")

        courses = [
            p
            for p in self.index_dir.iterdir()
            if p.is_dir() and (p / "index.faiss").exists()
        ]

        if self.only_course:
            courses = [p for p in courses if p.name == self.only_course]

        if not courses:
            raise RuntimeError(f"No FAISS indices found under {self.index_dir}")

        for course_path in courses:
            index, metadata_map = load_index(course_path)
            chunk_ids = list(metadata_map.keys())

            self._indices[course_path.name] = CourseIndex(
                name=course_path.name,
                path=course_path,
                chunk_ids=chunk_ids,
                metadata=metadata_map,
                index=index,
            )

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if not self._indices:
            self._load_indices()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, course: Optional[str] = None) -> List[RetrievedChunk]:
        self._ensure_loaded()

        # Rewrite query if rewriter is available
        search_query = query
        if self.query_rewriter and self.query_rewriter.is_available:
            search_query = self.query_rewriter.rewrite(query, course=course)

        query_embedding = np.array(self.embedding.embed(search_query), dtype="float32")
        all_results: List[RetrievedChunk] = []

        course_items = (
            [self._indices[course]]
            if course and course in self._indices
            else self._indices.values()
        )

        for course_idx in course_items:
            distances, indices = search(course_idx.index, query_embedding, k=self.retrieval_k)
            for score, idx in zip(distances, indices):
                if idx < 0 or idx >= len(course_idx.chunk_ids):
                    continue
                chunk_id = course_idx.chunk_ids[idx]
                chunk_data = course_idx.metadata.get(chunk_id, {})
                text = chunk_data.get("text", "").strip()
                metadata = chunk_data.get("metadata", {})
                all_results.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=metadata,
                        score=float(score),
                        course=course_idx.name,
                    )
                )

        return sorted(all_results, key=lambda r: r.score, reverse=True)

    # ------------------------------------------------------------------ #
    def rerank(self, query: str, retrieved: Sequence[RetrievedChunk], course: Optional[str] = None) -> List[RerankResult]:
        """
        Rerank retrieved chunks using cross-encoder.
        
        Args:
            query: Original user query (will be rewritten if rewriter is available)
            retrieved: Retrieved chunks to rerank
            course: Optional course name for query rewriting
        """
        # Use rewritten query for reranking if available (consistent with retrieval)
        rerank_query = query
        if self.query_rewriter and self.query_rewriter.is_available:
            rerank_query = self.query_rewriter.rewrite(query, course=course)
        
        top_candidates = list(retrieved)[: self.rerank_k]
        passages = [
            (chunk.chunk_id, chunk.text, chunk.metadata) for chunk in top_candidates if chunk.text
        ]
        return self.reranker.score(rerank_query, passages)

    # ------------------------------------------------------------------ #
    def answer(self, query: str, course: Optional[str] = None) -> Dict:
        retrieved = self.retrieve(query, course=course)
        if not retrieved:
            return {
                "status": "no_info",
                "message": NO_INFO_MESSAGE,
                "reason": "no_chunks",
                "retrieved": [],
            }

        reranked = self.rerank(query, retrieved, course=course)
        if not reranked:
            return {
                "status": "no_info",
                "message": NO_INFO_MESSAGE,
                "reason": "reranker_empty",
                "retrieved": retrieved,
            }

        best = reranked[0]
        if best.confidence < self.confidence_threshold:
            return {
                "status": "no_info",
                "message": NO_INFO_MESSAGE,
                "reason": "low_confidence",
                "confidence": best.confidence,
                "retrieved": retrieved,
                "reranked": reranked,
            }

        return {
            "status": "ok",
            "message": "context_ready",
            "retrieved": retrieved,
            "reranked": reranked,
            "best_chunk": best,
            "confidence": best.confidence,
        }

