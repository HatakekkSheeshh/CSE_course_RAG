#!/usr/bin/env python3
"""
Conversation Context Memory Evaluation Script.

Evaluates conversation context memory effectiveness with three metrics:
1. Context Utilization Rate: Percentage of follow-up queries that successfully reference previous context
2. Context Window Effectiveness: Answer quality across different conversation lengths
3. Follow-up Query Quality: Comparison of answer quality with/without context

Usage:
    python evaluation/evaluate_conversation_context.py --conversations evaluation/test_conversations.json
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
from rag.conversation import ConversationManager
from models.embedding import Embedding


@dataclass
class FollowUpResult:
    """Result for a follow-up query evaluation."""
    conversation_id: str
    follow_up_query: str
    turn_index: int
    conversation_length: str
    course: str
    # With context
    answer_with_context: str
    query_answer_sim_with_context: float
    answer_faithfulness_with_context: float
    latency_with_context: float
    # Without context (standalone)
    answer_without_context: str
    query_answer_sim_without_context: float
    answer_faithfulness_without_context: float
    latency_without_context: float
    # Context utilization
    context_utilization_similarity: float  # Similarity between query and previous messages
    context_utilization_rate: float  # 1.0 if similarity > threshold, else 0.0
    # Configuration
    use_rewriter: bool


@dataclass
class ConversationResult:
    """Result for a full conversation."""
    conversation_id: str
    course: str
    conversation_length: str
    num_turns: int
    follow_up_results: List[FollowUpResult]
    baseline_results: List[FollowUpResult]
    rewriter_results: List[FollowUpResult]


class ConversationContextEvaluator:
    """Evaluator for conversation context memory."""
    
    def __init__(self, context_similarity_threshold: float = 0.3):
        print("Initializing conversation context evaluator...")
        self.embedding = Embedding()
        self.context_similarity_threshold = context_similarity_threshold
        
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
        
        # Conversation manager
        self.conversation_manager = ConversationManager(max_history_per_conversation=10)
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
    
    def _compute_context_utilization(self, query: str, previous_messages: List[str]) -> Tuple[float, float]:
        """
        Compute context utilization metrics.
        Returns: (similarity_score, utilization_rate)
        """
        if not previous_messages:
            return 0.0, 0.0
        
        # Compute similarity with all previous messages
        similarities = [self._compute_similarity(query, msg) for msg in previous_messages]
        max_similarity = max(similarities) if similarities else 0.0
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Use max similarity as the context utilization similarity
        utilization_rate = 1.0 if max_similarity > self.context_similarity_threshold else 0.0
        
        return max_similarity, utilization_rate
    
    def _evaluate_query_with_context(
        self,
        query: str,
        course: str,
        conversation_history: str,
        use_rewriter: bool
    ) -> Tuple[str, float, float, float]:
        """
        Evaluate a query with conversation context.
        Returns: (answer, query_answer_sim, answer_faithfulness, latency)
        """
        pipeline = self.pipeline_rewriter if use_rewriter else self.pipeline_baseline
        
        start = time.time()
        result = pipeline.answer(query, course=course)
        
        if result.get("status") != "ok":
            return ("No results", 0.0, 0.0, time.time() - start)
        
        reranked = result.get("reranked", [])
        chunk_texts = [r.text for r in reranked[:5] if r.text.strip()]
        
        # Generate answer with conversation history
        answer = ""
        if self.llm_client and self.llm_client.enabled and chunk_texts:
            try:
                answer = self.llm_client.generate_answer(
                    query=query,
                    contexts=chunk_texts,
                    conversation_history=conversation_history
                )
            except:
                answer = chunk_texts[0] if chunk_texts else ""
        else:
            answer = chunk_texts[0] if chunk_texts else ""
        
        latency = time.time() - start
        
        # Calculate metrics
        query_answer_sim = self._compute_similarity(query, answer)
        answer_faithfulness = self._avg_similarity(answer, chunk_texts)
        
        return (answer, query_answer_sim, answer_faithfulness, latency)
    
    def _evaluate_query_without_context(
        self,
        query: str,
        course: str,
        use_rewriter: bool
    ) -> Tuple[str, float, float, float]:
        """
        Evaluate a query without conversation context (standalone).
        Returns: (answer, query_answer_sim, answer_faithfulness, latency)
        """
        pipeline = self.pipeline_rewriter if use_rewriter else self.pipeline_baseline
        
        start = time.time()
        result = pipeline.answer(query, course=course)
        
        if result.get("status") != "ok":
            return ("No results", 0.0, 0.0, time.time() - start)
        
        reranked = result.get("reranked", [])
        chunk_texts = [r.text for r in reranked[:5] if r.text.strip()]
        
        # Generate answer without conversation history
        answer = ""
        if self.llm_client and self.llm_client.enabled and chunk_texts:
            try:
                answer = self.llm_client.generate_answer(
                    query=query,
                    contexts=chunk_texts,
                    conversation_history=None
                )
            except:
                answer = chunk_texts[0] if chunk_texts else ""
        else:
            answer = chunk_texts[0] if chunk_texts else ""
        
        latency = time.time() - start
        
        # Calculate metrics
        query_answer_sim = self._compute_similarity(query, answer)
        answer_faithfulness = self._avg_similarity(answer, chunk_texts)
        
        return (answer, query_answer_sim, answer_faithfulness, latency)
    
    def evaluate_follow_up_query(
        self,
        conversation_id: str,
        follow_up_query: str,
        course: str,
        conversation_length: str,
        turn_index: int,
        previous_messages: List[str],
        use_rewriter: bool
    ) -> FollowUpResult:
        """Evaluate a follow-up query with and without context."""
        
        # Format conversation history
        conversation_history = self.conversation_manager.format_history_for_llm(
            session_id=conversation_id,
            max_messages=10
        )
        
        # Evaluate with context
        (answer_with_ctx, qa_sim_with_ctx, faithfulness_with_ctx, latency_with_ctx) = \
            self._evaluate_query_with_context(
                query=follow_up_query,
                course=course,
                conversation_history=conversation_history,
                use_rewriter=use_rewriter
            )
        
        # Evaluate without context
        (answer_without_ctx, qa_sim_without_ctx, faithfulness_without_ctx, latency_without_ctx) = \
            self._evaluate_query_without_context(
                query=follow_up_query,
                course=course,
                use_rewriter=use_rewriter
            )
        
        # Compute context utilization
        context_sim, context_rate = self._compute_context_utilization(
            follow_up_query,
            previous_messages
        )
        
        return FollowUpResult(
            conversation_id=conversation_id,
            follow_up_query=follow_up_query,
            turn_index=turn_index,
            conversation_length=conversation_length,
            course=course,
            answer_with_context=answer_with_ctx,
            query_answer_sim_with_context=qa_sim_with_ctx,
            answer_faithfulness_with_context=faithfulness_with_ctx,
            latency_with_context=latency_with_ctx,
            answer_without_context=answer_without_ctx,
            query_answer_sim_without_context=qa_sim_without_ctx,
            answer_faithfulness_without_context=faithfulness_without_ctx,
            latency_without_context=latency_without_ctx,
            context_utilization_similarity=context_sim,
            context_utilization_rate=context_rate,
            use_rewriter=use_rewriter
        )
    
    def evaluate_conversation(self, conversation: Dict) -> ConversationResult:
        """Evaluate a full conversation."""
        conversation_id = conversation["conversation_id"]
        course = conversation["course"]
        conversation_length = conversation["conversation_length"]
        turns = conversation["turns"]
        follow_up_queries = conversation.get("follow_up_queries", [])
        
        # Clear conversation history
        self.conversation_manager.clear_history(conversation_id)
        
        # Run conversation from the beginning
        previous_messages = []
        for i, turn in enumerate(turns):
            if turn["role"] == "user":
                # Simulate user query - get answer
                query = turn["content"]
                
                # Get conversation history up to this point
                history_str = self.conversation_manager.format_history_for_llm(
                    session_id=conversation_id,
                    max_messages=10
                )
                
                result = self.pipeline_baseline.answer(query, course=course)
                
                if result.get("status") == "ok":
                    reranked = result.get("reranked", [])
                    chunk_texts = [r.text for r in reranked[:5] if r.text.strip()]
                    
                    # Generate answer with conversation history
                    answer = ""
                    if self.llm_client and self.llm_client.enabled and chunk_texts:
                        try:
                            answer = self.llm_client.generate_answer(
                                query=query,
                                contexts=chunk_texts,
                                conversation_history=history_str if history_str else None
                            )
                        except:
                            answer = chunk_texts[0] if chunk_texts else ""
                    else:
                        answer = chunk_texts[0] if chunk_texts else ""
                else:
                    answer = "No results"
                
                # Add to conversation history
                self.conversation_manager.add_message(conversation_id, "user", query)
                self.conversation_manager.add_message(conversation_id, "assistant", answer)
                
                # Track previous messages for context utilization
                previous_messages.append(query)
                if answer:
                    previous_messages.append(answer)
        
        # Evaluate follow-up queries
        baseline_results = []
        rewriter_results = []
        
        for fu_query in follow_up_queries:
            query_text = fu_query["query"]
            turn_idx = fu_query["turn_index"]
            
            # Get previous messages up to this turn
            messages_up_to_turn = previous_messages[:turn_idx * 2] if turn_idx > 0 else []
            
            # Evaluate with baseline
            print(f"  Baseline follow-up: {query_text[:40]}...", end=" ", flush=True)
            result_baseline = self.evaluate_follow_up_query(
                conversation_id=conversation_id,
                follow_up_query=query_text,
                course=course,
                conversation_length=conversation_length,
                turn_index=turn_idx,
                previous_messages=messages_up_to_turn,
                use_rewriter=False
            )
            baseline_results.append(result_baseline)
            print(f"✓")
            
            # Evaluate with rewriter
            print(f"  Rewriter follow-up: {query_text[:40]}...", end=" ", flush=True)
            result_rewriter = self.evaluate_follow_up_query(
                conversation_id=conversation_id,
                follow_up_query=query_text,
                course=course,
                conversation_length=conversation_length,
                turn_index=turn_idx,
                previous_messages=messages_up_to_turn,
                use_rewriter=True
            )
            rewriter_results.append(result_rewriter)
            print(f"✓")
        
        return ConversationResult(
            conversation_id=conversation_id,
            course=course,
            conversation_length=conversation_length,
            num_turns=len(turns),
            follow_up_results=baseline_results + rewriter_results,
            baseline_results=baseline_results,
            rewriter_results=rewriter_results
        )
    
    def run_evaluation(self, conversations: List[Dict]) -> Dict:
        """Run full evaluation on all conversations."""
        results = []
        
        print(f"Evaluating {len(conversations)} conversations...\n")
        print("=" * 80)
        
        for i, conv in enumerate(conversations, 1):
            conv_id = conv["conversation_id"]
            course = conv["course"]
            length = conv["conversation_length"]
            
            print(f"[{i}/{len(conversations)}] {conv_id} - {course} ({length})")
            
            result = self.evaluate_conversation(conv)
            results.append(result)
            print()
        
        return {"conversations": results}
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate aggregate metrics."""
        conversations = results["conversations"]
        
        # Collect all follow-up results
        all_baseline = []
        all_rewriter = []
        
        for conv_result in conversations:
            all_baseline.extend(conv_result.baseline_results)
            all_rewriter.extend(conv_result.rewriter_results)
        
        # Context Utilization Rate
        baseline_utilization_rate = np.mean([r.context_utilization_rate for r in all_baseline]) if all_baseline else 0.0
        rewriter_utilization_rate = np.mean([r.context_utilization_rate for r in all_rewriter]) if all_rewriter else 0.0
        
        # Context Window Effectiveness (by conversation length)
        by_length = {}
        for length in ["short", "medium", "long"]:
            baseline_length = [r for r in all_baseline if r.conversation_length == length]
            rewriter_length = [r for r in all_rewriter if r.conversation_length == length]
            
            by_length[length] = {
                "baseline": {
                    "count": len(baseline_length),
                    "avg_query_answer_sim": np.mean([r.query_answer_sim_with_context for r in baseline_length]) if baseline_length else 0.0,
                    "avg_answer_faithfulness": np.mean([r.answer_faithfulness_with_context for r in baseline_length]) if baseline_length else 0.0,
                },
                "rewriter": {
                    "count": len(rewriter_length),
                    "avg_query_answer_sim": np.mean([r.query_answer_sim_with_context for r in rewriter_length]) if rewriter_length else 0.0,
                    "avg_answer_faithfulness": np.mean([r.answer_faithfulness_with_context for r in rewriter_length]) if rewriter_length else 0.0,
                }
            }
        
        # Follow-up Query Quality Improvement
        baseline_improvements = []
        rewriter_improvements = []
        
        for r in all_baseline:
            if r.query_answer_sim_without_context > 0:
                qa_improvement = (r.query_answer_sim_with_context - r.query_answer_sim_without_context) / r.query_answer_sim_without_context * 100
                baseline_improvements.append(qa_improvement)
        
        for r in all_rewriter:
            if r.query_answer_sim_without_context > 0:
                qa_improvement = (r.query_answer_sim_with_context - r.query_answer_sim_without_context) / r.query_answer_sim_without_context * 100
                rewriter_improvements.append(qa_improvement)
        
        avg_baseline_improvement = np.mean(baseline_improvements) if baseline_improvements else 0.0
        avg_rewriter_improvement = np.mean(rewriter_improvements) if rewriter_improvements else 0.0
        
        # Faithfulness improvement
        baseline_faithfulness_improvements = []
        rewriter_faithfulness_improvements = []
        
        for r in all_baseline:
            if r.answer_faithfulness_without_context > 0:
                faithfulness_improvement = (r.answer_faithfulness_with_context - r.answer_faithfulness_without_context) / r.answer_faithfulness_without_context * 100
                baseline_faithfulness_improvements.append(faithfulness_improvement)
        
        for r in all_rewriter:
            if r.answer_faithfulness_without_context > 0:
                faithfulness_improvement = (r.answer_faithfulness_with_context - r.answer_faithfulness_without_context) / r.answer_faithfulness_without_context * 100
                rewriter_faithfulness_improvements.append(faithfulness_improvement)
        
        avg_baseline_faithfulness_improvement = np.mean(baseline_faithfulness_improvements) if baseline_faithfulness_improvements else 0.0
        avg_rewriter_faithfulness_improvement = np.mean(rewriter_faithfulness_improvements) if rewriter_faithfulness_improvements else 0.0
        
        return {
            "context_utilization_rate": {
                "baseline": baseline_utilization_rate,
                "rewriter": rewriter_utilization_rate,
            },
            "context_window_effectiveness": by_length,
            "follow_up_quality_improvement": {
                "baseline": {
                    "query_answer_similarity_improvement": avg_baseline_improvement,
                    "answer_faithfulness_improvement": avg_baseline_faithfulness_improvement,
                },
                "rewriter": {
                    "query_answer_similarity_improvement": avg_rewriter_improvement,
                    "answer_faithfulness_improvement": avg_rewriter_faithfulness_improvement,
                }
            },
            "summary": {
                "total_conversations": len(conversations),
                "total_follow_up_queries": len(all_baseline),
            }
        }
    
    def print_results(self, metrics: Dict):
        """Print evaluation results."""
        print("\n" + "=" * 80)
        print("CONVERSATION CONTEXT MEMORY EVALUATION RESULTS")
        print("=" * 80)
        
        # Context Utilization Rate
        print("\n" + "=" * 80)
        print("CONTEXT UTILIZATION RATE")
        print("=" * 80)
        print(f"{'Configuration':<30} {'Rate':<15}")
        print("-" * 45)
        print(f"{'Baseline':<30} {metrics['context_utilization_rate']['baseline']:<15.4f}")
        print(f"{'Rewriter':<30} {metrics['context_utilization_rate']['rewriter']:<15.4f}")
        
        # Context Window Effectiveness
        print("\n" + "=" * 80)
        print("CONTEXT WINDOW EFFECTIVENESS")
        print("=" * 80)
        print(f"{'Length':<15} {'Config':<15} {'QA Sim':<15} {'Faithfulness':<15}")
        print("-" * 60)
        
        for length in ["short", "medium", "long"]:
            data = metrics["context_window_effectiveness"][length]
            print(f"{length.capitalize():<15} {'Baseline':<15} {data['baseline']['avg_query_answer_sim']:<15.4f} {data['baseline']['avg_answer_faithfulness']:<15.4f}")
            print(f"{'':<15} {'Rewriter':<15} {data['rewriter']['avg_query_answer_sim']:<15.4f} {data['rewriter']['avg_answer_faithfulness']:<15.4f}")
        
        # Follow-up Query Quality
        print("\n" + "=" * 80)
        print("FOLLOW-UP QUERY QUALITY IMPROVEMENT")
        print("=" * 80)
        print(f"{'Configuration':<30} {'QA Sim Improvement':<25} {'Faithfulness Improvement':<25}")
        print("-" * 80)
        baseline_qa = metrics["follow_up_quality_improvement"]["baseline"]["query_answer_similarity_improvement"]
        baseline_faith = metrics["follow_up_quality_improvement"]["baseline"]["answer_faithfulness_improvement"]
        rewriter_qa = metrics["follow_up_quality_improvement"]["rewriter"]["query_answer_similarity_improvement"]
        rewriter_faith = metrics["follow_up_quality_improvement"]["rewriter"]["answer_faithfulness_improvement"]
        print(f"{'Baseline':<30} {baseline_qa:<25.2f}% {baseline_faith:<25.2f}%")
        print(f"{'Rewriter':<30} {rewriter_qa:<25.2f}% {rewriter_faith:<25.2f}%")
        
        print("\n" + "=" * 80)
        print(f"Total Conversations: {metrics['summary']['total_conversations']}")
        print(f"Total Follow-up Queries: {metrics['summary']['total_follow_up_queries']}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate conversation context memory")
    parser.add_argument(
        "--conversations",
        type=str,
        default="evaluation/test_conversations.json",
        help="Path to test conversations JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/conversation_context_results.json",
        help="Path to output results JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Context utilization similarity threshold (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    # Load conversations
    conv_path = Path(args.conversations)
    if not conv_path.exists():
        print(f"Error: Conversations file not found: {conv_path}")
        return 1
    
    with open(conv_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    # Run evaluation
    evaluator = ConversationContextEvaluator(context_similarity_threshold=args.threshold)
    results = evaluator.run_evaluation(conversations)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclasses to dict for JSON serialization
    def convert_to_dict(obj):
        if hasattr(obj, "__dict__"):
            return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        else:
            return obj
    
    results_dict = convert_to_dict(results)
    results_dict["metrics"] = metrics
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

