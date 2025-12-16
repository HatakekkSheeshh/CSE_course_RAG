#!/usr/bin/env python3
"""
Evaluation script for RAG system.

Tests two baselines:
1. RAG Baseline (FAISS + Reranker) - Standard RAG without query rewriting
2. RAG + Query Rewriter - RAG with query rewriting enabled

Calculates metrics:
- ROUGE-1, ROUGE-2, ROUGE-L
- Exact Match (EM)
- Precision@k, Recall@k, MRR, NDCG@k (for retrieval)
- Latency and throughput
"""

import json
import time
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict

# Suppress transformers warnings about tokenizer usage
warnings.filterwarnings("ignore", message=".*tokenizer.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer parallelism warnings

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score", file=sys.stderr)

import config  # Load config first
from rag.query_pipeline import QueryPipeline
from rag.llm_client import LLMClient
from rag.query_rewriter import create_query_rewriter
from models.embedding import Embedding
import numpy as np


@dataclass
class TestQuery:
    """Test query with optional reference answer."""
    query: str
    reference_answer: Optional[str] = None  # Optional: None if reference answer unavailable
    course: Optional[str] = None  # None if course name in query, or course folder name if needs specification
    query_type: str = "general"
    note: Optional[str] = None  # Optional note about the query


@dataclass
class EvaluationResult:
    """Results for a single query."""
    query: str
    baseline: str
    answer: str
    reference_answer: str
    rewritten_query: Optional[str] = None  # Rewritten query (if rewriter was used)
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    exact_match: bool = False
    latency: float = 0.0
    retrieved_chunks: List[Dict] = None
    retrieval_metrics: Dict = None
    query_answer_similarity: float = 0.0  # Semantic similarity between query and answer
    query_chunk_similarity: float = 0.0    # Average semantic similarity between query and retrieved chunks
    
    def __post_init__(self):
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []
        if self.retrieval_metrics is None:
            self.retrieval_metrics = {}


class RAGEvaluator:
    """Evaluator for RAG system."""
    
    def __init__(self, data_dir: Path = None, index_dir: Path = None):
        self.data_dir = data_dir or config.get_rag_data_dir()
        self.index_dir = index_dir or config.get_rag_index_dir()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None
        
        # Initialize embedding model for retrieval metrics
        print("Loading embedding model...", flush=True)
        self.embedding = Embedding()
        
        # Initialize LLM client for closed book baseline
        print("Initializing LLM client...", flush=True)
        try:
            self.llm_client = LLMClient()
        except Exception as e:
            print(f"Warning: LLM client initialization failed: {e}", file=sys.stderr)
            self.llm_client = None
        
        # Pre-create query rewriter if LLM is available (reuse for all queries)
        self.query_rewriter = None
        if self.llm_client and self.llm_client.enabled:
            try:
                self.query_rewriter = create_query_rewriter(llm_client=self.llm_client)
                if self.query_rewriter.is_available:
                    print("Query rewriter initialized.", flush=True)
            except Exception as e:
                print(f"Warning: Query rewriter initialization failed: {e}", file=sys.stderr)
        
        # Pre-create pipelines (reuse for all queries to avoid reloading models)
        print("Loading QueryPipeline (this may take a moment - loading models and indices)...", flush=True)
        # Pipeline without query rewriter
        self.pipeline_baseline = QueryPipeline(
            data_dir=self.data_dir,
            index_dir=self.index_dir,
            retrieval_k=10,
            rerank_k=5,
            confidence_threshold=0.1,
            query_rewriter=None,  # No rewriter for baseline
        )
        
        # Pipeline with query rewriter
        self.pipeline_rewriter = QueryPipeline(
            data_dir=self.data_dir,
            index_dir=self.index_dir,
            retrieval_k=10,
            rerank_k=5,
            confidence_threshold=0.1,
            query_rewriter=self.query_rewriter,  # With rewriter
        )
        
        print("All models loaded. Starting evaluation...", flush=True)
    
    
    def test_rag_baseline(self, query: str, course: Optional[str] = None, use_rewriter: bool = False) -> Tuple[str, float, List[Dict], List[Dict], Optional[str]]:
        """Test RAG baseline with or without query rewriter.
        
        Returns:
            Tuple of (answer, latency, retrieved_chunks, reranked_chunks, rewritten_query)
        """
        # Use pre-created pipeline (reuse to avoid reloading models)
        pipeline = self.pipeline_rewriter if use_rewriter else self.pipeline_baseline
        
        # Get rewritten query if rewriter is used
        rewritten_query = None
        if use_rewriter and self.query_rewriter and self.query_rewriter.is_available:
            try:
                rewritten_query = self.query_rewriter.rewrite(query, course=course)
            except Exception as e:
                print(f"Warning: Query rewriting failed: {e}, using original query", file=sys.stderr)
                rewritten_query = query  # Fallback to original
        
        start_time = time.time()
        
        # Retrieve and rerank
        # Note: QueryPipeline doesn't support only_course filter in answer() method,
        # but it will search across all courses. The course filter is handled at retrieval level.
        result = pipeline.answer(query, course=course)
        
        if result.get("status") != "ok":
            latency = time.time() - start_time
            return result.get("message", "No info"), latency, [], [], rewritten_query
        
        retrieved = result.get("retrieved", [])
        reranked = result.get("reranked", [])
        
        # Format retrieved chunks
        retrieved_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200],  # Truncate for storage
                "score": chunk.score,
                "course": chunk.course
            }
            for chunk in retrieved[:10]
        ]
        
        reranked_chunks = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:200],
                "score": r.score,
                "confidence": r.confidence
            }
            for r in reranked[:5]
        ]
        
        # Generate answer using LLM
        contexts = [r.text for r in reranked[:5] if r.text.strip()]
        answer = ""
        if self.llm_client and self.llm_client.enabled and contexts:
            try:
                answer = self.llm_client.generate_answer(
                    query=query,
                    contexts=contexts,
                    conversation_history=None
                )
            except Exception as e:
                print(f"Error generating answer: {e}", file=sys.stderr)
                answer = contexts[0] if contexts else "No answer generated"
        else:
            answer = contexts[0] if contexts else "No answer generated"
        
        latency = time.time() - start_time
        
        return answer, latency, retrieved_chunks, reranked_chunks, rewritten_query
    
    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    
    def calculate_exact_match(self, generated: str, reference: str) -> bool:
        """Calculate exact match (case-insensitive, whitespace-normalized)."""
        gen_norm = " ".join(generated.lower().split())
        ref_norm = " ".join(reference.lower().split())
        return gen_norm == ref_norm
    
    def identify_relevant_chunks(self, retrieved_chunks: List[Dict], query: str, reference_answer: Optional[str] = None, similarity_threshold: float = 0.5) -> List[int]:
        """
        Identify relevant chunks by computing semantic similarity.
        
        If reference_answer is provided, uses reference-chunk similarity.
        Otherwise, uses query-chunk similarity (for evaluation without reference answers).
        
        Args:
            retrieved_chunks: List of retrieved chunks with 'text' field
            query: Original query text
            reference_answer: Optional reference answer text (if None, uses query instead)
            similarity_threshold: Minimum cosine similarity to consider chunk as relevant
            
        Returns:
            List of indices of relevant chunks
        """
        if not retrieved_chunks:
            return []
        
        # Use reference answer if available, otherwise use query
        comparison_text = reference_answer if reference_answer else query
        
        # Embed comparison text
        comp_embedding = np.array(self.embedding.embed(comparison_text), dtype="float32")
        comp_embedding = comp_embedding / (np.linalg.norm(comp_embedding) + 1e-8)  # L2 normalize
        
        relevant_indices = []
        for idx, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            
            # Embed chunk text
            chunk_embedding = np.array(self.embedding.embed(chunk_text), dtype="float32")
            chunk_embedding = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)  # L2 normalize
            
            # Compute cosine similarity
            similarity = np.dot(comp_embedding, chunk_embedding)
            
            if similarity >= similarity_threshold:
                relevant_indices.append(idx)
        
        return relevant_indices
    
    def calculate_precision_at_k(self, retrieved_chunks: List[Dict], relevant_indices: List[int], k: int) -> float:
        """Calculate Precision@k."""
        if not retrieved_chunks or k == 0:
            return 0.0
        
        top_k = retrieved_chunks[:k]
        relevant_in_top_k = sum(1 for i in range(min(k, len(retrieved_chunks))) if i in relevant_indices)
        return relevant_in_top_k / k
    
    def calculate_recall_at_k(self, retrieved_chunks: List[Dict], relevant_indices: List[int], k: int) -> float:
        """Calculate Recall@k."""
        if not relevant_indices:
            return 0.0
        
        top_k = retrieved_chunks[:k]
        relevant_in_top_k = sum(1 for i in range(min(k, len(retrieved_chunks))) if i in relevant_indices)
        return relevant_in_top_k / len(relevant_indices) if relevant_indices else 0.0
    
    def calculate_mrr(self, retrieved_chunks: List[Dict], relevant_indices: List[int]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not relevant_indices:
            return 0.0
        
        # Find the rank of the first relevant chunk (1-indexed)
        for rank, idx in enumerate(range(len(retrieved_chunks)), 1):
            if idx in relevant_indices:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved_chunks: List[Dict], relevant_indices: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@k."""
        if not retrieved_chunks or k == 0 or not relevant_indices:
            return 0.0
        
        # Calculate DCG@k
        dcg = 0.0
        for rank in range(1, min(k + 1, len(retrieved_chunks) + 1)):
            idx = rank - 1
            if idx in relevant_indices:
                # Relevance score = 1 for relevant, 0 for non-relevant
                relevance = 1.0
                dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG@k (ideal DCG - all relevant chunks at top)
        idcg = 0.0
        num_relevant = min(len(relevant_indices), k)
        for rank in range(1, num_relevant + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        # Normalize
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def calculate_retrieval_metrics(self, retrieved_chunks: List[Dict], query: str, reference_answer: Optional[str] = None) -> Dict:
        """Calculate all retrieval metrics."""
        if not retrieved_chunks:
            return {
                "precision_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "ndcg_at_5": 0.0
            }
        
        # Identify relevant chunks (uses reference_answer if available, otherwise uses query)
        relevant_indices = self.identify_relevant_chunks(retrieved_chunks, query, reference_answer, similarity_threshold=0.5)
        
        # Calculate metrics
        metrics = {
            "precision_at_5": self.calculate_precision_at_k(retrieved_chunks, relevant_indices, k=5),
            "recall_at_10": self.calculate_recall_at_k(retrieved_chunks, relevant_indices, k=10),
            "mrr": self.calculate_mrr(retrieved_chunks, relevant_indices),
            "ndcg_at_5": self.calculate_ndcg_at_k(retrieved_chunks, relevant_indices, k=5)
        }
        
        return metrics
    
    def calculate_query_answer_similarity(self, query: str, answer: str) -> float:
        """Calculate semantic similarity between query and generated answer."""
        if not answer or not answer.strip():
            return 0.0
        
        try:
            query_embedding = np.array(self.embedding.embed(query), dtype="float32")
            answer_embedding = np.array(self.embedding.embed(answer), dtype="float32")
            
            # L2 normalize
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            answer_embedding = answer_embedding / (np.linalg.norm(answer_embedding) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, answer_embedding)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating query-answer similarity: {e}", file=sys.stderr)
            return 0.0
    
    def calculate_query_chunk_similarity(self, query: str, retrieved_chunks: List[Dict], top_k: int = 5) -> float:
        """Calculate average semantic similarity between query and top-k retrieved chunks."""
        if not retrieved_chunks:
            return 0.0
        
        try:
            query_embedding = np.array(self.embedding.embed(query), dtype="float32")
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            similarities = []
            for chunk in retrieved_chunks[:top_k]:
                chunk_text = chunk.get("text", "")
                if not chunk_text:
                    continue
                
                chunk_embedding = np.array(self.embedding.embed(chunk_text), dtype="float32")
                chunk_embedding = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)
                
                similarity = np.dot(query_embedding, chunk_embedding)
                similarities.append(float(similarity))
            
            return np.mean(similarities) if similarities else 0.0
        except Exception as e:
            print(f"Error calculating query-chunk similarity: {e}", file=sys.stderr)
            return 0.0
    
    def evaluate_query(self, test_query: TestQuery, baseline: str) -> EvaluationResult:
        """Evaluate a single query for a specific baseline."""
        query = test_query.query
        reference = test_query.reference_answer
        course = test_query.course
        
        if baseline == "rag_baseline":
            answer, latency, retrieved_chunks, reranked_chunks, rewritten_query = self.test_rag_baseline(query, course, use_rewriter=False)
        elif baseline == "rag_rewriter":
            answer, latency, retrieved_chunks, reranked_chunks, rewritten_query = self.test_rag_baseline(query, course, use_rewriter=True)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        
        # Calculate metrics (skip ROUGE/EM if reference_answer is not available)
        if reference:
            rouge_scores = self.calculate_rouge(answer, reference)
            exact_match = self.calculate_exact_match(answer, reference)
        else:
            rouge_scores = {"rouge1": None, "rouge2": None, "rougeL": None}
            exact_match = None
        
        # Calculate retrieval metrics (works with or without reference_answer)
        retrieval_metrics = self.calculate_retrieval_metrics(retrieved_chunks, query, reference)
        
        # Calculate semantic similarity metrics
        query_answer_sim = self.calculate_query_answer_similarity(query, answer)
        query_chunk_sim = self.calculate_query_chunk_similarity(query, retrieved_chunks, top_k=5)
        
        return EvaluationResult(
            query=query,
            baseline=baseline,
            answer=answer,
            reference_answer=reference or "",
            rewritten_query=rewritten_query,
            rouge_1=rouge_scores["rouge1"] or 0.0,
            rouge_2=rouge_scores["rouge2"] or 0.0,
            rouge_l=rouge_scores["rougeL"] or 0.0,
            exact_match=exact_match if exact_match is not None else False,
            latency=latency,
            retrieved_chunks=retrieved_chunks,
            retrieval_metrics=retrieval_metrics,
            query_answer_similarity=query_answer_sim,
            query_chunk_similarity=query_chunk_sim
        )
    
    def evaluate_all(self, test_queries: List[TestQuery]) -> Dict[str, List[EvaluationResult]]:
        """Evaluate all queries for all baselines."""
        baselines = ["rag_baseline", "rag_rewriter"]
        results = {baseline: [] for baseline in baselines}
        
        print(f"Evaluating {len(test_queries)} queries across {len(baselines)} baselines...")
        print("="*80)
        
        for i, test_query in enumerate(test_queries, 1):
            course_info = f" [course: {test_query.course}]" if test_query.course else " [cross-course]"
            print(f"\n[{i}/{len(test_queries)}] Query: {test_query.query[:60]}...{course_info}")
            if test_query.note:
                print(f"  Note: {test_query.note}")
            
            for baseline in baselines:
                print(f"  Testing {baseline:15s}...", end=" ", flush=True)
                try:
                    result = self.evaluate_query(test_query, baseline)
                    results[baseline].append(result)
                    
                    # Show rewritten query if available
                    rewrite_info = ""
                    if result.rewritten_query and result.rewritten_query != test_query.query:
                        rewrite_info = f" [Rewritten: {result.rewritten_query[:50]}...]"
                    
                    if test_query.reference_answer:
                        print(f"✓ (ROUGE-L: {result.rouge_l:.3f}, Latency: {result.latency:.2f}s){rewrite_info}")
                    else:
                        print(f"✓ (Latency: {result.latency:.2f}s, Retrieval metrics only){rewrite_info}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    # Add empty result to maintain structure
                    results[baseline].append(EvaluationResult(
                        query=test_query.query,
                        baseline=baseline,
                        answer="",
                        reference_answer=test_query.reference_answer or "",
                        latency=0.0
                    ))
        
        return results
    
    def calculate_aggregate_metrics(self, results: Dict[str, List[EvaluationResult]]) -> Dict:
        """Calculate aggregate metrics across all queries."""
        metrics = {}
        
        for baseline, baseline_results in results.items():
            if not baseline_results:
                continue
            
            valid_results = [r for r in baseline_results if r.answer and r.answer != "LLM not available"]
            if not valid_results:
                continue
            
            n = len(valid_results)
            
            # Calculate average retrieval metrics
            retrieval_results = [r for r in valid_results if r.retrieval_metrics]
            if retrieval_results:
                avg_retrieval = {
                    "precision_at_5": sum(r.retrieval_metrics.get("precision_at_5", 0) for r in retrieval_results) / len(retrieval_results),
                    "recall_at_10": sum(r.retrieval_metrics.get("recall_at_10", 0) for r in retrieval_results) / len(retrieval_results),
                    "mrr": sum(r.retrieval_metrics.get("mrr", 0) for r in retrieval_results) / len(retrieval_results),
                    "ndcg_at_5": sum(r.retrieval_metrics.get("ndcg_at_5", 0) for r in retrieval_results) / len(retrieval_results)
                }
            else:
                avg_retrieval = {
                    "precision_at_5": None,
                    "recall_at_10": None,
                    "mrr": None,
                    "ndcg_at_5": None
                }
            
            # Check if we have reference answers for ROUGE/EM calculation
            results_with_ref = [r for r in valid_results if r.reference_answer]
            has_references = len(results_with_ref) > 0
            
            if has_references:
                rouge_1 = sum(r.rouge_1 for r in results_with_ref) / len(results_with_ref)
                rouge_2 = sum(r.rouge_2 for r in results_with_ref) / len(results_with_ref)
                rouge_l = sum(r.rouge_l for r in results_with_ref) / len(results_with_ref)
                exact_match = sum(r.exact_match for r in results_with_ref) / len(results_with_ref)
            else:
                rouge_1 = rouge_2 = rouge_l = exact_match = None
            
            # Calculate average semantic similarity metrics
            avg_query_answer_sim = sum(r.query_answer_similarity for r in valid_results) / n
            avg_query_chunk_sim = sum(r.query_chunk_similarity for r in valid_results) / n
            
            metrics[baseline] = {
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "exact_match": exact_match,
                "avg_latency": sum(r.latency for r in valid_results) / n,
                "query_answer_similarity": avg_query_answer_sim,
                "query_chunk_similarity": avg_query_chunk_sim,
                "total_queries": n,
                "valid_queries": len(valid_results),
                "queries_with_reference": len(results_with_ref),
                **avg_retrieval
            }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print evaluation results in table format."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # ====================================================================
        # TABLE: tab:answer_quality (Section: Answer Generation Quality)
        # Fill in: results.tex -> \subsection{Answer Generation Quality}
        # ====================================================================
        print("\nAnswer Generation Quality:")
        print(f"{'Baseline':<35} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'EM':<10}")
        print("-" * 80)
        
        baseline_names = {
            "rag_baseline": "RAG Baseline (FAISS + Reranker)",
            "rag_rewriter": "RAG + Query Rewriter"
        }
        
        for baseline in ["rag_baseline", "rag_rewriter"]:
            if baseline not in metrics:
                continue
            name = baseline_names.get(baseline, baseline)
            m = metrics[baseline]
            
            # Format metrics (show N/A if reference answers not available)
            rouge_1_str = f"{m['rouge_1']:.4f}" if m['rouge_1'] is not None else "N/A"
            rouge_2_str = f"{m['rouge_2']:.4f}" if m['rouge_2'] is not None else "N/A"
            rouge_l_str = f"{m['rouge_l']:.4f}" if m['rouge_l'] is not None else "N/A"
            em_str = f"{m['exact_match']:.4f}" if m['exact_match'] is not None else "N/A"
            
            print(f"{name:<35} "
                  f"{rouge_1_str:<10} "
                  f"{rouge_2_str:<10} "
                  f"{rouge_l_str:<10} "
                  f"{em_str:<10}")
        
        # Show warning if no reference answers
        if any(m.get('queries_with_reference', 0) == 0 for m in metrics.values()):
            print("\nNote: Some queries don't have reference answers. ROUGE/EM metrics are N/A.")
            print("      Retrieval metrics (Precision@k, NDCG@k) are still calculated.")
        
        # ====================================================================
        # FIGURE: fig:semantic_similarity (Section: Answer Generation Quality)
        # File: results.tex -> \subsection{Answer Generation Quality}
        # Metrics: query_answer_similarity, query_chunk_similarity
        # ====================================================================
        print("\nSemantic Similarity Metrics:")
        print(f"{'Baseline':<35} {'Query-Answer Sim':<18} {'Query-Chunk Sim':<18}")
        print("-" * 80)
        
        for baseline in ["rag_baseline", "rag_rewriter"]:
            if baseline not in metrics:
                continue
            name = baseline_names.get(baseline, baseline)
            m = metrics[baseline]
            print(f"{name:<35} "
                  f"{m['query_answer_similarity']:<18.4f} "
                  f"{m['query_chunk_similarity']:<18.4f}")
        
        # ====================================================================
        # TABLE: tab:retrieval_metrics (Section: Retrieval Performance)
        # File: results.tex -> \subsection{Retrieval Performance}
        # Metrics: precision_at_5, recall_at_10, mrr, ndcg_at_5
        # ====================================================================
        print("\nRetrieval Performance Metrics:")
        print(f"{'Baseline':<35} {'Precision@5':<15} {'Recall@10':<15} {'MRR':<12} {'NDCG@5':<12}")
        print("-" * 90)
        
        for baseline in ["rag_baseline", "rag_rewriter"]:
            if baseline not in metrics:
                continue
            name = baseline_names.get(baseline, baseline)
            m = metrics[baseline]
            prec_str = f"{m.get('precision_at_5', 0):.4f}" if m.get('precision_at_5') is not None else "N/A"
            recall_str = f"{m.get('recall_at_10', 0):.4f}" if m.get('recall_at_10') is not None else "N/A"
            mrr_str = f"{m.get('mrr', 0):.4f}" if m.get('mrr') is not None else "N/A"
            ndcg_str = f"{m.get('ndcg_at_5', 0):.4f}" if m.get('ndcg_at_5') is not None else "N/A"
            print(f"{name:<35} {prec_str:<15} {recall_str:<15} {mrr_str:<12} {ndcg_str:<12}")
        
        # ====================================================================
        # FIGURE: fig:precision_at_k (Section: Retrieval Performance)
        # File: results.tex -> \subsection{Retrieval Performance}
        # Note: Precision@k values for k=1,3,5,10 (calculated from detailed_results)
        # ====================================================================
        
        # ====================================================================
        # TABLE: tab:query_rewriter_impact (Section: Impact of Query Rewriting)
        # File: results.tex -> \subsection{Impact of Query Rewriting}
        # Metrics: ROUGE-1, ROUGE-2, ROUGE-L, EM (same as tab:answer_quality)
        # Also calculate: Improvement row (delta values)
        # ====================================================================
        
        # ====================================================================
        # FIGURE: fig:query_rewriting_improvement (Section: Impact of Query Rewriting)
        # File: results.tex -> \subsection{Impact of Query Rewriting}
        # Metrics: Precision@5, NDCG@5, Query-Answer Similarity, Query-Chunk Similarity improvements
        # ====================================================================
        
        # Show example rewritten queries for report
        print("\n" + "="*80)
        print("QUERY REWRITING EXAMPLES (for report)")
        print("="*80)
        if "rag_rewriter" in results and "rag_baseline" in results:
            rewriter_results = results["rag_rewriter"]
            baseline_results = results["rag_baseline"]
            
            print("\nComparison: Original Query vs Rewritten Query")
            print("-"*80)
            shown_count = 0
            for i, (r_rewriter, r_baseline) in enumerate(zip(rewriter_results, baseline_results), 1):
                if r_rewriter.rewritten_query and r_rewriter.rewritten_query != r_rewriter.query:
                    print(f"\n[{i}] Original Query:")
                    print(f"    {r_baseline.query}")
                    print(f"    → Rewritten Query:")
                    print(f"    {r_rewriter.rewritten_query}")
                    shown_count += 1
                    if shown_count >= 8:  # Show all queries
                        break
            
            if shown_count == 0:
                print("No query rewriting examples available (rewriter may not be enabled or queries unchanged)")
        else:
            print("Rewriter results not available")
        
        # Calculate and show query rewriting improvement
        if "rag_baseline" in metrics and "rag_rewriter" in metrics:
            baseline_metrics = metrics["rag_baseline"]
            rewriter_metrics = metrics["rag_rewriter"]
            
            print("\nQuery Rewriting Improvement:")
            print(f"{'Metric':<30} {'Baseline':<12} {'With Rewriter':<15} {'Improvement':<15} {'% Change':<12}")
            print("-" * 90)
            
            improvements = []
            if baseline_metrics.get('precision_at_5') is not None:
                baseline_val = baseline_metrics['precision_at_5']
                rewriter_val = rewriter_metrics['precision_at_5']
                delta = rewriter_val - baseline_val
                pct = (delta / baseline_val * 100) if baseline_val > 0 else 0
                improvements.append(("Precision@5", baseline_val, rewriter_val, delta, pct))
            
            if baseline_metrics.get('ndcg_at_5') is not None:
                baseline_val = baseline_metrics['ndcg_at_5']
                rewriter_val = rewriter_metrics['ndcg_at_5']
                delta = rewriter_val - baseline_val
                pct = (delta / baseline_val * 100) if baseline_val > 0 else 0
                improvements.append(("NDCG@5", baseline_val, rewriter_val, delta, pct))
            
            baseline_val = baseline_metrics['query_answer_similarity']
            rewriter_val = rewriter_metrics['query_answer_similarity']
            delta = rewriter_val - baseline_val
            pct = (delta / baseline_val * 100) if baseline_val > 0 else 0
            improvements.append(("Query-Answer Similarity", baseline_val, rewriter_val, delta, pct))
            
            baseline_val = baseline_metrics['query_chunk_similarity']
            rewriter_val = rewriter_metrics['query_chunk_similarity']
            delta = rewriter_val - baseline_val
            pct = (delta / baseline_val * 100) if baseline_val > 0 else 0
            improvements.append(("Query-Chunk Similarity", baseline_val, rewriter_val, delta, pct))
            
            for metric_name, baseline_val, rewriter_val, delta, pct in improvements:
                print(f"{metric_name:<30} "
                      f"{baseline_val:<12.4f} "
                      f"{rewriter_val:<15.4f} "
                      f"{delta:+.4f}{'':<11} "
                      f"{pct:+.2f}%")
        
        # ====================================================================
        # TABLE: tab:performance (Section: System Performance)
        # File: results.tex -> \subsection{System Performance}
        # Metrics: avg_latency, throughput (throughput = 1/avg_latency)
        # ====================================================================
        print("\nSystem Performance:")
        print(f"{'Baseline':<35} {'Avg Latency (s)':<15} {'Throughput (qps)':<15}")
        print("-" * 80)
        
        for baseline in ["rag_baseline", "rag_rewriter"]:
            if baseline not in metrics:
                continue
            name = baseline_names.get(baseline, baseline)
            m = metrics[baseline]
            throughput = 1.0 / m['avg_latency'] if m['avg_latency'] > 0 else 0.0
            print(f"{name:<35} "
                  f"{m['avg_latency']:<15.2f} "
                  f"{throughput:<15.2f}")
    
    def save_results(self, results: Dict[str, List[EvaluationResult]], metrics: Dict, output_file: Path):
        """Save detailed results to JSON file."""
        output_data = {
            "metrics": metrics,
            "detailed_results": {}
        }
        
        for baseline, baseline_results in results.items():
            output_data["detailed_results"][baseline] = [
                {
                    "query": r.query,
                    "rewritten_query": r.rewritten_query,  # Include rewritten query
                    "answer": r.answer,
                    "reference_answer": r.reference_answer,
                    "rouge_1": r.rouge_1,
                    "rouge_2": r.rouge_2,
                    "rouge_l": r.rouge_l,
                    "exact_match": r.exact_match,
                    "latency": r.latency,
                    "query_answer_similarity": r.query_answer_similarity,
                    "query_chunk_similarity": r.query_chunk_similarity,
                    "retrieved_chunks": r.retrieved_chunks or [],
                    "retrieval_metrics": r.retrieval_metrics or {}
                }
                for r in baseline_results
            ]
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")


def load_test_queries(query_file: Path) -> List[TestQuery]:
    """Load test queries from JSON file. Reference answers are optional."""
    with open(query_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    for item in data:
        queries.append(TestQuery(
            query=item["query"],
            reference_answer=item.get("reference_answer"),  # Optional field
            course=item.get("course"),
            query_type=item.get("query_type", "general"),
            note=item.get("note")
        ))
    
    return queries


def create_template_queries_file(output_file: Path):
    """Create a template test queries file. Reference answers are optional."""
    template = [
        {
            "query": "What are the prerequisites for Database Systems?",
            "reference_answer": "The prerequisites for Database Systems typically include Data Structures and Algorithms, and Programming Fundamentals.",
            "course": "Database_Systems",
            "query_type": "course_info"
        },
        {
            "query": "Explain how B-tree indexing works",
            "reference_answer": "B-tree is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.",
            "course": None,
            "query_type": "technical"
        },
        {
            "query": "What is the grading policy for this course?",
            # "reference_answer": null,  # Optional: omit if reference answer unavailable
            "course": None,
            "query_type": "course_info"
        }
    ]
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--queries",
        type=str,
        default="scripts/test_queries.json",
        help="Path to test queries JSON file (default: scripts/test_queries.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/evaluation_results.json",
        help="Path to save detailed results (default: scripts/evaluation_results.json)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: from config)"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Index directory (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Check if rouge-score is available
    if not ROUGE_AVAILABLE:
        print("Error: rouge-score package is required for evaluation.", file=sys.stderr)
        print("Install with: pip install rouge-score", file=sys.stderr)
        sys.exit(1)
    
    # Load test queries
    query_file = Path(args.queries)
    if not query_file.exists():
        print(f"Test queries file not found: {query_file}")
        print("Creating template file...")
        create_template_queries_file(query_file)
        print(f"Template created at {query_file}. Please fill in test queries and reference answers.")
        sys.exit(1)
    
    test_queries = load_test_queries(query_file)
    print(f"Loaded {len(test_queries)} test queries")
    
    # Initialize evaluator
    data_dir = Path(args.data_dir) if args.data_dir else None
    index_dir = Path(args.index_dir) if args.index_dir else None
    
    evaluator = RAGEvaluator(data_dir=data_dir, index_dir=index_dir)
    
    # Run evaluation
    results = evaluator.evaluate_all(test_queries)
    
    # Calculate aggregate metrics
    metrics = evaluator.calculate_aggregate_metrics(results)
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save detailed results
    output_file = Path(args.output)
    evaluator.save_results(results, metrics, output_file)
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print("\nNext steps:")
    print("1. Review the detailed results in:", output_file)
    print("\n2. Fill in LaTeX tables in results.tex:")
    print("   - tab:answer_quality -> ROUGE-1, ROUGE-2, ROUGE-L, EM")
    print("   - tab:retrieval_metrics -> Precision@5, Recall@10, MRR, NDCG@5")
    print("   - tab:query_rewriter_impact -> ROUGE scores + Improvement row")
    print("   - tab:performance -> Avg Latency, Throughput (1/latency)")
    print("\n3. Generate visualizations:")
    print(f"   python3 scripts/create_visualizations.py --results {args.output}")


if __name__ == "__main__":
    main()
