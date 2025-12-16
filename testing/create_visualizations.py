#!/usr/bin/env python3
"""
Create visualizations for RAG evaluation results.

Generates six plots:
1. Bar Chart: Semantic Similarity Scores
2. Line Plot: Precision@k Curves
3. Bar Chart: Query Rewriting Improvement
4. Bar Chart: ROUGE Scores Comparison
5. Scatter Plot: Quality-Latency Trade-off
6. Box Plot: Query Rewriting Improvement Distribution
"""

import json
import sys
from pathlib import Path
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib numpy", file=sys.stderr)


def load_evaluation_results(results_file: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_semantic_similarity_chart(metrics: dict, output_file: Path):
    """Create bar chart comparing semantic similarity scores."""
    baseline_metrics = metrics.get("rag_baseline", {})
    rewriter_metrics = metrics.get("rag_rewriter", {})
    
    # Extract semantic similarity metrics (handle None values)
    baseline_qa_sim = baseline_metrics.get("query_answer_similarity") or 0.0
    baseline_qc_sim = baseline_metrics.get("query_chunk_similarity") or 0.0
    rewriter_qa_sim = rewriter_metrics.get("query_answer_similarity") or 0.0
    rewriter_qc_sim = rewriter_metrics.get("query_chunk_similarity") or 0.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data for plotting
    categories = ['Query-Answer\nSimilarity', 'Query-Chunk\nSimilarity']
    baseline_values = [baseline_qa_sim, baseline_qc_sim]
    rewriter_values = [rewriter_qa_sim, rewriter_qc_sim]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_values, width, label='RAG Baseline', color='#4A90E2', alpha=0.8)
    bars2 = ax.bar(x + width/2, rewriter_values, width, label='RAG + Query Rewriter', color='#50C878', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_ylabel('Semantic Similarity Score', fontsize=11, fontweight='bold')
    ax.set_title('Semantic Similarity Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, max(max(baseline_values), max(rewriter_values)) * 1.2])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created semantic similarity chart: {output_file}")


def create_precision_at_k_plot(metrics: dict, detailed_results: dict, output_file: Path):
    """Create line plot showing Precision@k curves."""
    # Calculate Precision@k for different k values
    k_values = [1, 3, 5, 10]
    
    baseline_precisions = []
    rewriter_precisions = []
    
    # For each k, calculate precision from detailed results
    for k in k_values:
        baseline_prec = calculate_precision_at_k(detailed_results.get("rag_baseline", []), k)
        rewriter_prec = calculate_precision_at_k(detailed_results.get("rag_rewriter", []), k)
        baseline_precisions.append(baseline_prec)
        rewriter_precisions.append(rewriter_prec)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot lines
    ax.plot(k_values, baseline_precisions, marker='o', linewidth=2, markersize=8, 
            label='RAG Baseline', color='#4A90E2')
    ax.plot(k_values, rewriter_precisions, marker='s', linewidth=2, markersize=8, 
            label='RAG + Query Rewriter', color='#50C878')
    
    # Add value labels
    for k, baseline_val, rewriter_val in zip(k_values, baseline_precisions, rewriter_precisions):
        ax.text(k, baseline_val + 0.01, f'{baseline_val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(k, rewriter_val + 0.01, f'{rewriter_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_xlabel('k (Number of Retrieved Chunks)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision@k', fontsize=11, fontweight='bold')
    ax.set_title('Precision@k Curves', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(k_values)
    ax.set_ylim([0, max(max(baseline_precisions), max(rewriter_precisions)) * 1.2])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created Precision@k plot: {output_file}")


def calculate_precision_at_k(results: list, k: int) -> float:
    """Calculate average Precision@k from detailed results."""
    if not results:
        return 0.0
    
    precisions = []
    for result in results:
        retrieval_metrics = result.get("retrieval_metrics", {})
        # Try to get precision_at_k directly
        if f"precision_at_{k}" in retrieval_metrics:
            precisions.append(retrieval_metrics[f"precision_at_{k}"])
        # Fallback: use precision_at_5 if k <= 5
        elif k <= 5 and "precision_at_5" in retrieval_metrics:
            precisions.append(retrieval_metrics["precision_at_5"])
        # Fallback: estimate from available metrics
        elif "precision_at_5" in retrieval_metrics:
            # Use precision_at_5 as approximation for k > 5
            precisions.append(retrieval_metrics["precision_at_5"])
        else:
            # If no precision metrics, use 0
            precisions.append(0.0)
    
    return np.mean(precisions) if precisions else 0.0


def create_rouge_comparison_chart(metrics: dict, output_file: Path):
    """Create bar chart comparing ROUGE scores."""
    baseline_metrics = metrics.get("rag_baseline", {})
    rewriter_metrics = metrics.get("rag_rewriter", {})
    
    # Extract ROUGE metrics (handle None values)
    baseline_rouge1 = baseline_metrics.get("rouge_1") or 0.0
    baseline_rouge2 = baseline_metrics.get("rouge_2") or 0.0
    baseline_rougel = baseline_metrics.get("rouge_l") or 0.0
    baseline_em = baseline_metrics.get("exact_match") or 0.0
    
    rewriter_rouge1 = rewriter_metrics.get("rouge_1") or 0.0
    rewriter_rouge2 = rewriter_metrics.get("rouge_2") or 0.0
    rewriter_rougel = rewriter_metrics.get("rouge_l") or 0.0
    rewriter_em = rewriter_metrics.get("exact_match") or 0.0
    
    # Check if we have reference answers
    has_references = (baseline_metrics.get("queries_with_reference", 0) > 0 or 
                     rewriter_metrics.get("queries_with_reference", 0) > 0)
    
    if not has_references:
        print("Warning: No reference answers available. Skipping ROUGE comparison chart.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for plotting
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'EM']
    baseline_values = [baseline_rouge1, baseline_rouge2, baseline_rougel, baseline_em]
    rewriter_values = [rewriter_rouge1, rewriter_rouge2, rewriter_rougel, rewriter_em]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_values, width, label='RAG Baseline', color='#4A90E2', alpha=0.8)
    bars2 = ax.bar(x + width/2, rewriter_values, width, label='RAG + Query Rewriter', color='#50C878', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('ROUGE Scores Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, max(max(baseline_values), max(rewriter_values)) * 1.3])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created ROUGE comparison chart: {output_file}")


def create_quality_latency_tradeoff(metrics: dict, detailed_results: dict, output_file: Path):
    """Create scatter plot showing quality-latency trade-off."""
    # Extract data for each baseline
    baselines = ["rag_baseline", "rag_rewriter"]
    baseline_names = {
        "rag_baseline": "RAG Baseline",
        "rag_rewriter": "RAG + Query Rewriter"
    }
    
    # Collect per-query data
    data_points = {baseline: {"latency": [], "quality": []} for baseline in baselines}
    
    for baseline in baselines:
        results = detailed_results.get(baseline, [])
        for result in results:
            latency = result.get("latency", 0)
            # Use ROUGE-L as quality metric if available, otherwise use query-answer similarity
            rouge_l = result.get("rouge_l")
            if rouge_l is not None and rouge_l > 0:
                quality = rouge_l
            else:
                quality = result.get("query_answer_similarity", 0)
            
            if latency > 0 and quality >= 0:
                data_points[baseline]["latency"].append(latency)
                data_points[baseline]["quality"].append(quality)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"rag_baseline": "#4A90E2", "rag_rewriter": "#50C878"}
    markers = {"rag_baseline": "o", "rag_rewriter": "s"}
    
    for baseline in baselines:
        if data_points[baseline]["latency"]:
            ax.scatter(
                data_points[baseline]["latency"],
                data_points[baseline]["quality"],
                label=baseline_names[baseline],
                color=colors[baseline],
                marker=markers[baseline],
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidths=1
            )
    
    # Add average points
    for baseline in baselines:
        baseline_metrics = metrics.get(baseline, {})
        avg_latency = baseline_metrics.get("avg_latency", 0)
        # Use ROUGE-L if available, otherwise query-answer similarity
        avg_quality = baseline_metrics.get("rouge_l") or baseline_metrics.get("query_answer_similarity", 0)
        
        if avg_latency > 0 and avg_quality >= 0:
            ax.scatter(
                avg_latency,
                avg_quality,
                color=colors[baseline],
                marker=markers[baseline],
                s=300,
                edgecolors='black',
                linewidths=2,
                label=f"{baseline_names[baseline]} (Avg)",
                zorder=5
            )
    
    # Customize plot
    ax.set_xlabel('Latency (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality Score (ROUGE-L / Query-Answer Similarity)', fontsize=11, fontweight='bold')
    ax.set_title('Quality-Latency Trade-off', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created quality-latency trade-off plot: {output_file}")


def create_improvement_distribution_boxplot(detailed_results: dict, output_file: Path):
    """Create box plot showing distribution of improvements across queries."""
    baseline_results = detailed_results.get("rag_baseline", [])
    rewriter_results = detailed_results.get("rag_rewriter", [])
    
    if not baseline_results or not rewriter_results:
        print("Warning: Insufficient data for improvement distribution. Skipping box plot.")
        return
    
    # Match results by query (assuming same order)
    improvements = {
        "Precision@5": [],
        "NDCG@5": [],
        "Query-Answer\nSimilarity": [],
        "Query-Chunk\nSimilarity": []
    }
    
    min_len = min(len(baseline_results), len(rewriter_results))
    for i in range(min_len):
        baseline = baseline_results[i]
        rewriter = rewriter_results[i]
        
        # Precision@5 improvement
        baseline_prec = baseline.get("retrieval_metrics", {}).get("precision_at_5", 0) or 0
        rewriter_prec = rewriter.get("retrieval_metrics", {}).get("precision_at_5", 0) or 0
        improvements["Precision@5"].append(rewriter_prec - baseline_prec)
        
        # NDCG@5 improvement
        baseline_ndcg = baseline.get("retrieval_metrics", {}).get("ndcg_at_5", 0) or 0
        rewriter_ndcg = rewriter.get("retrieval_metrics", {}).get("ndcg_at_5", 0) or 0
        improvements["NDCG@5"].append(rewriter_ndcg - baseline_ndcg)
        
        # Query-Answer Similarity improvement
        baseline_qa = baseline.get("query_answer_similarity", 0) or 0
        rewriter_qa = rewriter.get("query_answer_similarity", 0) or 0
        improvements["Query-Answer\nSimilarity"].append(rewriter_qa - baseline_qa)
        
        # Query-Chunk Similarity improvement
        baseline_qc = baseline.get("query_chunk_similarity", 0) or 0
        rewriter_qc = rewriter.get("query_chunk_similarity", 0) or 0
        improvements["Query-Chunk\nSimilarity"].append(rewriter_qc - baseline_qc)
    
    # Filter out empty lists
    improvements = {k: v for k, v in improvements.items() if v}
    
    if not improvements:
        print("Warning: No improvement data available. Skipping box plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    data = [improvements[k] for k in improvements.keys()]
    labels = list(improvements.keys())
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    
    # Color boxes
    colors_box = ['#4A90E2', '#50C878', '#FFA500', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Customize plot
    ax.set_ylabel('Improvement (Absolute Change)', fontsize=11, fontweight='bold')
    ax.set_title('Query Rewriting Improvement Distribution', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created improvement distribution box plot: {output_file}")


def create_improvement_chart(metrics: dict, output_file: Path):
    """Create bar chart showing query rewriting improvement."""
    baseline_metrics = metrics.get("rag_baseline", {})
    rewriter_metrics = metrics.get("rag_rewriter", {})
    
    # Calculate improvements
    improvements = []
    metric_names = []
    
    # Precision@5 improvement
    if baseline_metrics.get("precision_at_5") is not None:
        baseline_val = baseline_metrics["precision_at_5"]
        rewriter_val = rewriter_metrics.get("precision_at_5") or 0.0
        improvement = rewriter_val - baseline_val
        improvements.append(improvement)
        metric_names.append("Precision@5")
    
    # NDCG@5 improvement
    if baseline_metrics.get("ndcg_at_5") is not None:
        baseline_val = baseline_metrics["ndcg_at_5"]
        rewriter_val = rewriter_metrics.get("ndcg_at_5") or 0.0
        improvement = rewriter_val - baseline_val
        improvements.append(improvement)
        metric_names.append("NDCG@5")
    
    # Query-Answer Similarity improvement
    baseline_val = baseline_metrics.get("query_answer_similarity") or 0.0
    rewriter_val = rewriter_metrics.get("query_answer_similarity") or 0.0
    if baseline_val > 0 or rewriter_val > 0:  # Only add if at least one has value
        improvement = rewriter_val - baseline_val
        improvements.append(improvement)
        metric_names.append("Query-Answer\nSimilarity")
    
    # Query-Chunk Similarity improvement
    baseline_val = baseline_metrics.get("query_chunk_similarity") or 0.0
    rewriter_val = rewriter_metrics.get("query_chunk_similarity") or 0.0
    if baseline_val > 0 or rewriter_val > 0:  # Only add if at least one has value
        improvement = rewriter_val - baseline_val
        improvements.append(improvement)
        metric_names.append("Query-Chunk\nSimilarity")
    
    if not improvements:
        print("Warning: No improvement data available. Skipping improvement chart.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on positive/negative improvement
    colors = ['#50C878' if imp >= 0 else '#E74C3C' for imp in improvements]
    
    bars = ax.bar(metric_names, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        label = f'{imp:+.4f}' if abs(imp) >= 0.0001 else f'{imp:+.2e}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label,
               ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9, fontweight='bold')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Customize plot
    ax.set_ylabel('Improvement (Absolute Change)', fontsize=11, fontweight='bold')
    ax.set_title('Query Rewriting Improvement', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis limits to show improvements clearly
    max_abs_imp = max(abs(imp) for imp in improvements) if improvements else 0.1
    ax.set_ylim([-max_abs_imp * 1.3, max_abs_imp * 1.3])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created improvement chart: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create visualizations from evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        default="scripts/evaluation_results.json",
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="overleaf-report/AI_PROJECT_REPORT/figures",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib and numpy are required for visualization.")
        print("Install with: pip install matplotlib numpy")
        sys.exit(1)
    
    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    data = load_evaluation_results(results_file)
    metrics = data.get("metrics", {})
    detailed_results = data.get("detailed_results", {})
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("Creating visualizations...")
    print("=" * 60)
    
    # 1. Semantic Similarity Chart
    create_semantic_similarity_chart(
        metrics,
        output_dir / "semantic_similarity.pdf"
    )
    
    # 2. Precision@k Plot
    create_precision_at_k_plot(
        metrics,
        detailed_results,
        output_dir / "precision_at_k.pdf"
    )
    
    # 3. Improvement Chart
    create_improvement_chart(
        metrics,
        output_dir / "query_rewriting_improvement.pdf"
    )
    
    # 4. ROUGE Scores Comparison
    create_rouge_comparison_chart(
        metrics,
        output_dir / "rouge_comparison.pdf"
    )
    
    # 5. Quality-Latency Trade-off
    create_quality_latency_tradeoff(
        metrics,
        detailed_results,
        output_dir / "quality_latency_tradeoff.pdf"
    )
    
    # 6. Improvement Distribution Box Plot
    create_improvement_distribution_boxplot(
        detailed_results,
        output_dir / "improvement_distribution.pdf"
    )
    
    print("=" * 60)
    print("All visualizations created successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

