"""
Wrapper for the BAAI bge reranker model with normalized scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import math

from .load_model import load_model


@dataclass
class RerankResult:
    """Container for reranker outputs."""

    chunk_id: str
    text: str
    score: float
    confidence: float
    metadata: dict | None = None


class Reranker:
    """
    Thin wrapper around FlagEmbedding's FlagReranker.

    Provides softmax-normalized confidences for downstream filtering.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self.model = load_model("reranker", model_name=model_name)

    def score(
        self,
        query: str,
        passages: Sequence[tuple[str, str, dict | None]],
    ) -> List[RerankResult]:
        """
        Score passages relative to the query.

        Args:
            query: User query text.
            passages: Sequence of (chunk_id, text, metadata)

        Returns:
            List of RerankResult sorted by descending score, including
            softmax-normalized confidences (sum to 1.0).
        """
        if not passages:
            return []

        pairs = [[query, text] for _, text, _ in passages]
        raw_scores = self.model.compute_score(pairs) 
        
        # Softmax normalize for confidence estimates
        max_score = max(raw_scores)
        exp_scores = [math.exp(s - max_score) for s in raw_scores]
        normalizer = sum(exp_scores) or 1.0

        confidences = [val / normalizer for val in exp_scores]

        results = []    
        for (chunk_id, text, metadata), score, conf in zip(passages, raw_scores, confidences):
            results.append(
                RerankResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(score),
                    confidence=float(conf),
                    metadata=metadata,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

