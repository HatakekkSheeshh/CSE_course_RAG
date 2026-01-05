#!/usr/bin/env python3
"""
Simplified RAG Evaluation Script - Focus on Query Rewriter Impact.

Compares two baselines:
1. RAG Baseline (no query rewriting)
2. RAG + Query Rewriter

Metrics designed for datasets WITHOUT ground truth reference answers:
- Retrieval Quality: Query-Chunk Semantic Similarity
- Answer Quality: Query-Answer Semantic Similarity  
- Answer Faithfulness: Answer-Chunk Overlap (is answer grounded in retrieved context?)
- Latency: Response time comparison

Usage:
    python testing/evaluate_rewriter_impact.py --queries testing/test_queries_extended.json
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import warnings

warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import config
from rag.query_pipeline import QueryPipeline
from rag.llm_client import LLMClient
from rag.query_rewriter import create_query_rewriter
from models.embedding import Embedding


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query: str
    course: str
    query_type: str
    rewritten_query: Optional[str]
    answer: str
    latency: float
    # Semantic similarity metrics (0-1, higher is better)
    query_answer_similarity: float  # How relevant is answer to query?
    query_chunk_similarity: float   # How relevant are retrieved chunks to query?
    answer_chunk_similarity: float  # Is answer grounded in chunks? (faithfulness)
    num_chunks_retrieved: int
    top_chunk_score: float


class RewriterEvaluator:
    """Evaluator comparing RAG with and without query rewriter."""
    
    def __init__(self):
        print("Initializing evaluator...")
        self.embedding = Embedding()
        
        # Initialize LLM
        try:
            self.llm_client = LLMClient()
            print(f"LLM: {self.llm_client.provider_name} - {self.llm_client.model_name}")
        except Exception as e:
            print(f"Warning: LLM init failed: {e}")
            self.llm_client = None
        
        # Query rewriter
        self.query_rewriter = None
        if self.llm_client and self.llm_client.enabled:
            try:
                self.query_rewriter = create_query_rewriter(llm_client=self.llm_client)
                print("Query rewriter: enabled")
            except Exception as e:
                print(f"Query rewriter: disabled ({e})")
        
        # Pipelines
        print("Loading indices...")
        self.pipeline_baseline = QueryPipeline(
            data_dir=config.get_rag_data_dir(),
            index_dir=config.get_rag_index_dir(),
            retrieval_k=10,
            rerank_k=5,
            query_rewriter=None,
        )
        
        self.pipeline_rewriter = QueryPipeline(
            data_dir=config.get_rag_data_dir(),
            index_dir=config.get_rag_index_dir(),
            retrieval_k=10,
            rerank_k=5,
            query_rewriter=self.query_rewriter,
        )
        print("Ready!\n")
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        try:
            emb1 = np.array(self.embedding.embed(text1), dtype="float32")
            emb2 = np.array(self.embedding.embed(text2), dtype="float32")
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
            return float(np.dot(emb1, emb2))
        except:
            return 0.0
    
    def _avg_similarity(self, text: str, texts: List[str], top_k: int = 5) -> float:
        """Compute average similarity between text and list of texts."""
        if not texts:
            return 0.0
        sims = [self._compute_similarity(text, t) for t in texts[:top_k]]
        return np.mean(sims) if sims else 0.0
    
    def evaluate_query(self, query: str, course: str, query_type: str, use_rewriter: bool) -> QueryResult:
        """Evaluate a single query."""
        pipeline = self.pipeline_rewriter if use_rewriter else self.pipeline_baseline
        
        # Get rewritten query
        rewritten_query = None
        if use_rewriter and self.query_rewriter and self.query_rewriter.is_available:
            try:
                rewritten_query = self.query_rewriter.rewrite(query, course=course)
            except:
                rewritten_query = query
        
        # Query pipeline
        start = time.time()
        result = pipeline.answer(query, course=course)
        
        if result.get("status") != "ok":
            return QueryResult(
                query=query, course=course, query_type=query_type,
                rewritten_query=rewritten_query, answer="No results",
                latency=time.time() - start,
                query_answer_similarity=0, query_chunk_similarity=0,
                answer_chunk_similarity=0, num_chunks_retrieved=0, top_chunk_score=0
            )
        
        reranked = result.get("reranked", [])
        chunk_texts = [r.text for r in reranked[:5] if r.text.strip()]
        
        # Generate answer
        answer = ""
        if self.llm_client and self.llm_client.enabled and chunk_texts:
            try:
                answer = self.llm_client.generate_answer(query=query, contexts=chunk_texts)
            except:
                answer = chunk_texts[0] if chunk_texts else ""
        else:
            answer = chunk_texts[0] if chunk_texts else ""
        
        latency = time.time() - start
        
        # Calculate metrics
        query_answer_sim = self._compute_similarity(query, answer)
        query_chunk_sim = self._avg_similarity(query, chunk_texts)
        answer_chunk_sim = self._avg_similarity(answer, chunk_texts)  # Faithfulness
        top_score = reranked[0].score if reranked else 0.0
        
        return QueryResult(
            query=query,
            course=course,
            query_type=query_type,
            rewritten_query=rewritten_query,
            answer=answer,
            latency=latency,
            query_answer_similarity=query_answer_sim,
            query_chunk_similarity=query_chunk_sim,
            answer_chunk_similarity=answer_chunk_sim,
            num_chunks_retrieved=len(reranked),
            top_chunk_score=top_score
        )
    
    def run_evaluation(self, queries: List[Dict]) -> Dict:
        """Run full evaluation on all queries."""
        results = {"baseline": [], "rewriter": []}
        
        print(f"Evaluating {len(queries)} queries...\n")
        print("=" * 80)
        
        for i, q in enumerate(queries, 1):
            query = q["query"]
            course = q.get("course", "")
            query_type = q.get("query_type", "general")
            
            print(f"[{i}/{len(queries)}] {query[:40]}... [{course[:20]}]")
            
            # Baseline
            print("  Baseline...", end=" ", flush=True)
            r_base = self.evaluate_query(query, course, query_type, use_rewriter=False)
            results["baseline"].append(r_base)
            print(f"✓ ({r_base.latency:.1f}s)")
            
            # Rewriter
            print("  Rewriter...", end=" ", flush=True)
            r_rewrite = self.evaluate_query(query, course, query_type, use_rewriter=True)
            results["rewriter"].append(r_rewrite)
            rewrite_info = f" → {r_rewrite.rewritten_query[:30]}..." if r_rewrite.rewritten_query != query else ""
            print(f"✓ ({r_rewrite.latency:.1f}s){rewrite_info}")
            print()
        
        return results
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate aggregate metrics."""
        metrics = {}
        
        for baseline_name, baseline_results in results.items():
            n = len(baseline_results)
            if n == 0:
                continue
            
            metrics[baseline_name] = {
                "num_queries": n,
                "avg_latency": np.mean([r.latency for r in baseline_results]),
                "avg_query_answer_sim": np.mean([r.query_answer_similarity for r in baseline_results]),
                "avg_query_chunk_sim": np.mean([r.query_chunk_similarity for r in baseline_results]),
                "avg_answer_chunk_sim": np.mean([r.answer_chunk_similarity for r in baseline_results]),
                "avg_top_chunk_score": np.mean([r.top_chunk_score for r in baseline_results]),
            }
            
            # Per query type breakdown
            by_type = {}
            for r in baseline_results:
                if r.query_type not in by_type:
                    by_type[r.query_type] = []
                by_type[r.query_type].append(r)
            
            metrics[baseline_name]["by_query_type"] = {
                qtype: {
                    "count": len(rs),
                    "avg_query_answer_sim": np.mean([r.query_answer_similarity for r in rs]),
                    "avg_latency": np.mean([r.latency for r in rs]),
                }
                for qtype, rs in by_type.items()
            }
        
        return metrics
    
    def print_results(self, metrics: Dict, results: Dict):
        """Print evaluation results organized by report sections."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        base = metrics["baseline"]
        rewrite = metrics["rewriter"]
        
        # ============================================================
        # SECTION 5.1: Answer Generation Quality
        # ============================================================
        print("\n" + "=" * 80)
        print("5.1 ANSWER GENERATION QUALITY")
        print("=" * 80)
        print(f"{'Metric':<30} {'Baseline':<15} {'+ Rewriter':<15} {'Δ Change':<15}")
        print("-" * 75)
        
        answer_metrics = [
            ("Query-Answer Similarity", "avg_query_answer_sim", True),
            ("Answer Faithfulness", "avg_answer_chunk_sim", True),
        ]
        
        for name, key, higher_better in answer_metrics:
            b_val = base[key]
            r_val = rewrite[key]
            delta = r_val - b_val
            pct = (delta / b_val * 100) if b_val != 0 else 0
            indicator = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"{name:<30} {b_val:<15.4f} {r_val:<15.4f} {delta:+.4f} ({pct:+.1f}%) {indicator}")
        
        # ============================================================
        # SECTION 5.2: Retrieval Performance
        # ============================================================
        print("\n" + "=" * 80)
        print("5.2 RETRIEVAL PERFORMANCE")
        print("=" * 80)
        print(f"{'Metric':<30} {'Baseline':<15} {'+ Rewriter':<15} {'Δ Change':<15}")
        print("-" * 75)
        
        retrieval_metrics = [
            ("Query-Chunk Similarity", "avg_query_chunk_sim", True),
            ("Top Chunk Score (Reranker)", "avg_top_chunk_score", True),
        ]
        
        for name, key, higher_better in retrieval_metrics:
            b_val = base[key]
            r_val = rewrite[key]
            delta = r_val - b_val
            pct = (delta / b_val * 100) if b_val != 0 else 0
            indicator = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"{name:<30} {b_val:<15.4f} {r_val:<15.4f} {delta:+.4f} ({pct:+.1f}%) {indicator}")
        
        # ============================================================
        # SECTION 5.3: Impact of Query Rewriting
        # ============================================================
        print("\n" + "=" * 80)
        print("5.3 IMPACT OF QUERY REWRITING")
        print("=" * 80)
        
        # Overall improvement summary
        print("\nOverall Improvement Summary:")
        print(f"{'Metric':<30} {'Improvement':<15} {'% Change':<15}")
        print("-" * 60)
        
        improvement_metrics = [
            ("Query-Answer Similarity", "avg_query_answer_sim"),
            ("Query-Chunk Similarity", "avg_query_chunk_sim"),
            ("Answer Faithfulness", "avg_answer_chunk_sim"),
        ]
        
        for name, key in improvement_metrics:
            b_val = base[key]
            r_val = rewrite[key]
            delta = r_val - b_val
            pct = (delta / b_val * 100) if b_val != 0 else 0
            print(f"{name:<30} {delta:+.4f}{'':<10} {pct:+.1f}%")
        
        # Query type breakdown
        print("\nBy Query Type (Query-Answer Similarity):")
        print(f"{'Type':<20} {'Baseline':<12} {'+ Rewriter':<12} {'Δ':<12}")
        print("-" * 60)
        
        all_types = set(base["by_query_type"].keys()) | set(rewrite["by_query_type"].keys())
        for qtype in sorted(all_types):
            b_type = base["by_query_type"].get(qtype, {})
            r_type = rewrite["by_query_type"].get(qtype, {})
            b_sim = b_type.get("avg_query_answer_sim", 0)
            r_sim = r_type.get("avg_query_answer_sim", 0)
            delta = r_sim - b_sim
            print(f"{qtype:<20} {b_sim:<12.4f} {r_sim:<12.4f} {delta:+.4f}")
        
        # ============================================================
        # SECTION 5.4: System Performance
        # ============================================================
        print("\n" + "=" * 80)
        print("5.4 SYSTEM PERFORMANCE")
        print("=" * 80)
        print(f"{'Metric':<30} {'Baseline':<15} {'+ Rewriter':<15} {'Δ Change':<15}")
        print("-" * 75)
        
        b_latency = base["avg_latency"]
        r_latency = rewrite["avg_latency"]
        delta_latency = r_latency - b_latency
        pct_latency = (delta_latency / b_latency * 100) if b_latency != 0 else 0
        
        b_throughput = 1.0 / b_latency if b_latency > 0 else 0
        r_throughput = 1.0 / r_latency if r_latency > 0 else 0
        delta_throughput = r_throughput - b_throughput
        pct_throughput = (delta_throughput / b_throughput * 100) if b_throughput != 0 else 0
        
        print(f"{'Avg Latency (seconds)':<30} {b_latency:<15.2f} {r_latency:<15.2f} {delta_latency:+.2f} ({pct_latency:+.1f}%)")
        print(f"{'Throughput (queries/sec)':<30} {b_throughput:<15.4f} {r_throughput:<15.4f} {delta_throughput:+.4f} ({pct_throughput:+.1f}%)")
        
        # ============================================================
        # SECTION 5.5: Qualitative Analysis
        # ============================================================
        print("\n" + "=" * 80)
        print("5.5 QUALITATIVE ANALYSIS")
        print("=" * 80)
        
        # Query Rewriting Examples
        print("\n[A] Query Rewriting Examples:")
        print("-" * 80)
        shown = 0
        for i, (r_base, r_rewrite) in enumerate(zip(results["baseline"], results["rewriter"]), 1):
            if r_rewrite.rewritten_query and r_rewrite.rewritten_query != r_base.query:
                delta_sim = r_rewrite.query_answer_similarity - r_base.query_answer_similarity
                print(f"Example {shown + 1}:")
                print(f"  Original Query:  \"{r_base.query}\"")
                print(f"  Rewritten Query: \"{r_rewrite.rewritten_query}\"")
                print(f"  Course: {r_base.course}")
                print(f"  Similarity Change: {delta_sim:+.4f} ({'improved' if delta_sim > 0 else 'decreased'})")
                print()
                shown += 1
                if shown >= 5:
                    break
        
        if shown == 0:
            print("  No query rewriting examples available.")
        
        # Case Studies - Best and Worst improvements
        print("\n[B] Case Studies:")
        print("-" * 80)
        
        # Calculate improvements for each query
        improvements = []
        for r_base, r_rewrite in zip(results["baseline"], results["rewriter"]):
            delta = r_rewrite.query_answer_similarity - r_base.query_answer_similarity
            improvements.append({
                "query": r_base.query,
                "course": r_base.course,
                "baseline_sim": r_base.query_answer_similarity,
                "rewriter_sim": r_rewrite.query_answer_similarity,
                "delta": delta,
                "rewritten": r_rewrite.rewritten_query
            })
        
        # Sort by improvement
        improvements.sort(key=lambda x: x["delta"], reverse=True)
        
        # Best improvement
        if improvements:
            best = improvements[0]
            print(f"Best Improvement (+{best['delta']:.4f}):")
            print(f"  Query: \"{best['query']}\"")
            print(f"  Course: {best['course']}")
            print(f"  Baseline Similarity: {best['baseline_sim']:.4f}")
            print(f"  With Rewriter: {best['rewriter_sim']:.4f}")
            print()
            
            # Worst improvement (or biggest decrease)
            worst = improvements[-1]
            print(f"Worst Case ({worst['delta']:+.4f}):")
            print(f"  Query: \"{worst['query']}\"")
            print(f"  Course: {worst['course']}")
            print(f"  Baseline Similarity: {worst['baseline_sim']:.4f}")
            print(f"  With Rewriter: {worst['rewriter_sim']:.4f}")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 80)
        print("SUMMARY FOR REPORT")
        print("=" * 80)
        
        qa_improve = rewrite["avg_query_answer_sim"] - base["avg_query_answer_sim"]
        qa_pct = (qa_improve / base["avg_query_answer_sim"] * 100) if base["avg_query_answer_sim"] != 0 else 0
        latency_increase = rewrite["avg_latency"] - base["avg_latency"]
        latency_pct = (latency_increase / base["avg_latency"] * 100) if base["avg_latency"] != 0 else 0
        
        print(f"\nKey Findings:")
        print(f"• Query Rewriter {'improves' if qa_improve > 0 else 'decreases'} Answer Quality by {abs(qa_improve):.4f} ({abs(qa_pct):.1f}%)")
        print(f"• Latency {'increases' if latency_increase > 0 else 'decreases'} by {abs(latency_increase):.1f}s ({abs(latency_pct):.1f}%)")
        print(f"• Number of queries evaluated: {base['num_queries']}")
        
        # Count improved queries
        improved_count = sum(1 for imp in improvements if imp["delta"] > 0)
        print(f"• Queries with improved similarity: {improved_count}/{len(improvements)} ({improved_count/len(improvements)*100:.1f}%)")
    
    def save_results(self, results: Dict, metrics: Dict, output_file: Path):
        """Save results to JSON."""
        output = {
            "metrics": metrics,
            "detailed_results": {
                baseline: [asdict(r) for r in rs]
                for baseline, rs in results.items()
            }
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Query Rewriter Impact")
    parser.add_argument("--queries", default="testing/test_queries_extended.json")
    parser.add_argument("--output", default="testing/rewriter_evaluation_results.json")
    args = parser.parse_args()
    
    # Load queries
    with open(args.queries, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries from {args.queries}")
    
    # Run evaluation
    evaluator = RewriterEvaluator()
    results = evaluator.run_evaluation(queries)
    metrics = evaluator.calculate_metrics(results)
    
    # Print and save
    evaluator.print_results(metrics, results)
    evaluator.save_results(results, metrics, Path(args.output))


if __name__ == "__main__":
    main()

