#!/usr/bin/env python3
"""
Script to automatically fill LaTeX tables in results.tex from evaluation results.

Usage:
    python scripts/fill_results_tables.py --results scripts/evaluation_results.json
"""

import json
import re
import sys
from pathlib import Path
import argparse


def load_evaluation_results(results_file: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_number(value: float, decimals: int = 4) -> str:
    """Format number for LaTeX table."""
    if value is None:
        return "To be filled"
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage for LaTeX table."""
    if value is None:
        return "To be filled"
    return f"{value * 100:.{decimals}f}\\%"


def update_table_in_file(file_path: Path, table_label: str, new_content: str):
    """Update a specific table in LaTeX file."""
    content = file_path.read_text(encoding='utf-8')
    
    # Find the table with the given label
    pattern = rf'\\label\{{{re.escape(table_label)}\}}.*?\\end\{{tabular\}}'
    
    # Replace the table content
    new_table = f"\\label{{{table_label}}}{new_content}\\end{{tabular}}"
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_table, content, flags=re.DOTALL)
        file_path.write_text(content, encoding='utf-8')
        return True
    else:
        print(f"Warning: Table {table_label} not found in {file_path}", file=sys.stderr)
        return False


def generate_answer_quality_table(metrics: dict) -> str:
    """Generate LaTeX table for answer quality."""
    baseline_map = {
        "rag_baseline": "RAG Baseline (FAISS + Reranker)",
        "rag_rewriter": "RAG + Query Rewriter"
    }
    
    lines = [
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Configuration} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} & \\textbf{ROUGE-L} & \\textbf{EM} \\\\",
        "\\hline"
    ]
    
    for baseline_key in ["rag_baseline", "rag_rewriter"]:
        if baseline_key not in metrics:
            continue
        m = metrics[baseline_key]
        name = baseline_map[baseline_key]
        # Handle None values for ROUGE/EM (when reference answers unavailable)
        rouge_1 = m.get('rouge_1') if m.get('rouge_1') is not None else 0
        rouge_2 = m.get('rouge_2') if m.get('rouge_2') is not None else 0
        rouge_l = m.get('rouge_l') if m.get('rouge_l') is not None else 0
        exact_match = m.get('exact_match') if m.get('exact_match') is not None else 0
        lines.append(f"{name} & {format_number(rouge_1)} & {format_number(rouge_2)} & {format_number(rouge_l)} & {format_number(exact_match)} \\\\")
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def generate_performance_table(metrics: dict) -> str:
    """Generate LaTeX table for system performance."""
    baseline_map = {
        "rag_baseline": "RAG Baseline (FAISS + Reranker)",
        "rag_rewriter": "RAG + Query Rewriter"
    }
    
    lines = [
        "\\begin{tabular}{|l|c|c|}",
        "\\hline",
        "\\textbf{Configuration} & \\textbf{Avg Latency (s)} & \\textbf{Throughput (qps)} \\\\",
        "\\hline"
    ]
    
    for baseline_key in ["rag_baseline", "rag_rewriter"]:
        if baseline_key not in metrics:
            continue
        m = metrics[baseline_key]
        name = baseline_map[baseline_key]
        avg_latency = m.get('avg_latency', 0)
        # Calculate throughput (queries per second)
        throughput = 1.0 / avg_latency if avg_latency > 0 else 0
        lines.append(f"{name} & {format_number(avg_latency, 2)} & {format_number(throughput, 2)} \\\\")
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def generate_retrieval_table(metrics: dict) -> str:
    """Generate LaTeX table for retrieval performance metrics."""
    baseline_map = {
        "rag_baseline": "RAG Baseline (FAISS + Reranker)",
        "rag_rewriter": "RAG + Query Rewriter"
    }
    
    lines = [
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Configuration} & \\textbf{Precision@5} & \\textbf{Recall@10} & \\textbf{MRR} & \\textbf{NDCG@5} \\\\",
        "\\hline"
    ]
    
    for baseline_key in ["rag_baseline", "rag_rewriter"]:
        if baseline_key not in metrics:
            continue
        m = metrics[baseline_key]
        name = baseline_map[baseline_key]
        # Check if retrieval metrics exist
        precision = m.get('precision_at_5', None)
        recall = m.get('recall_at_10', None)
        mrr = m.get('mrr', None)
        ndcg = m.get('ndcg_at_5', None)
        
        if precision is None or recall is None or mrr is None or ndcg is None:
            # No retrieval metrics available
            lines.append(f"{name} & To be filled & To be filled & To be filled & To be filled \\\\")
        else:
            lines.append(f"{name} & {format_number(precision)} & {format_number(recall)} & {format_number(mrr)} & {format_number(ndcg)} \\\\")
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def generate_rewriter_impact_table(metrics: dict) -> str:
    """Generate LaTeX table for query rewriter impact."""
    if "rag_baseline" not in metrics or "rag_rewriter" not in metrics:
        return ""
    
    baseline = metrics["rag_baseline"]
    rewriter = metrics["rag_rewriter"]
    
    def calc_improvement(baseline_val, rewriter_val):
        if baseline_val is None or rewriter_val is None:
            return 0.0
        if baseline_val == 0:
            return 0.0
        return ((rewriter_val - baseline_val) / baseline_val) * 100
    
    # Handle None values (when reference answers unavailable)
    baseline_rouge1 = baseline.get('rouge_1') if baseline.get('rouge_1') is not None else 0
    baseline_rouge2 = baseline.get('rouge_2') if baseline.get('rouge_2') is not None else 0
    baseline_rougeL = baseline.get('rouge_l') if baseline.get('rouge_l') is not None else 0
    baseline_em = baseline.get('exact_match') if baseline.get('exact_match') is not None else 0
    
    rewriter_rouge1 = rewriter.get('rouge_1') if rewriter.get('rouge_1') is not None else 0
    rewriter_rouge2 = rewriter.get('rouge_2') if rewriter.get('rouge_2') is not None else 0
    rewriter_rougeL = rewriter.get('rouge_l') if rewriter.get('rouge_l') is not None else 0
    rewriter_em = rewriter.get('exact_match') if rewriter.get('exact_match') is not None else 0
    
    rouge1_imp = calc_improvement(baseline_rouge1, rewriter_rouge1)
    rouge2_imp = calc_improvement(baseline_rouge2, rewriter_rouge2)
    rougeL_imp = calc_improvement(baseline_rougeL, rewriter_rougeL)
    em_imp = calc_improvement(baseline_em, rewriter_em)
    
    lines = [
        "\\begin{tabular}{|l|c|c|c|c|}",
        "\\hline",
        "\\textbf{Configuration} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} & \\textbf{ROUGE-L} & \\textbf{EM} \\\\",
        "\\hline",
        f"RAG Baseline (FAISS + Reranker) & {format_number(baseline_rouge1)} & {format_number(baseline_rouge2)} & {format_number(baseline_rougeL)} & {format_number(baseline_em)} \\\\",
        "\\hline",
        f"RAG + Query Rewriter & {format_number(rewriter_rouge1)} & {format_number(rewriter_rouge2)} & {format_number(rewriter_rougeL)} & {format_number(rewriter_em)} \\\\",
        "\\hline",
        f"\\textbf{{Improvement}} & \\textbf{{{format_number(rouge1_imp / 100, 4)}}} & \\textbf{{{format_number(rouge2_imp / 100, 4)}}} & \\textbf{{{format_number(rougeL_imp / 100, 4)}}} & \\textbf{{{format_percentage(em_imp / 100)}}} \\\\",
        "\\hline",
        "\\end{tabular}"
    ]
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Fill LaTeX tables from evaluation results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overleaf-report/AI_PROJECT_REPORT/experiements/results.tex",
        help="Path to results.tex file (default: overleaf-report/AI_PROJECT_REPORT/experiements/results.tex)"
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}", file=sys.stderr)
        sys.exit(1)
    
    results = load_evaluation_results(results_file)
    metrics = results.get("metrics", {})
    
    if not metrics:
        print("Error: No metrics found in results file", file=sys.stderr)
        sys.exit(1)
    
    output_file = Path(args.output)
    if not output_file.exists():
        print(f"Error: Results file not found: {output_file}", file=sys.stderr)
        sys.exit(1)
    
    # Generate tables
    answer_quality_table = generate_answer_quality_table(metrics)
    retrieval_table = generate_retrieval_table(metrics)
    performance_table = generate_performance_table(metrics)
    rewriter_impact_table = generate_rewriter_impact_table(metrics)
    
    # Update tables in file
    content = output_file.read_text(encoding='utf-8')
    
    # Helper function to extract table body (content between \begin{tabular} and \end{tabular})
    def extract_table_body(full_table: str) -> str:
        """Extract only the content inside tabular environment."""
        # Remove \begin{tabular} and \end{tabular} lines, keep only the body
        lines = full_table.split('\n')
        body_lines = []
        skip_begin = True
        for line in lines:
            if '\\begin{tabular}' in line:
                skip_begin = False
                continue
            if '\\end{tabular}' in line:
                break
            if not skip_begin:
                body_lines.append(line)
        return '\n'.join(body_lines)
    
    # Helper function to escape backslashes for regex replacement
    def escape_for_replacement(text):
        """Escape backslashes in replacement text to avoid regex errors."""
        return text.replace('\\', '\\\\')
    
    # Update answer quality table
    # Match: \begin{tabular}{...} ... content ... \end{tabular}
    # Replace only the content part, keep \begin{tabular}{...} and \end{tabular}
    pattern1 = r'(\\begin\{table\}.*?\\caption\{Answer Generation Quality.*?\}.*?\\label\{tab:answer_quality\}.*?\\begin\{tabular\}\{[^}]+\}\n?)(.*?)(\n?\\end\{tabular\}.*?\\end\{table\})'
    table_body1 = extract_table_body(answer_quality_table)
    replacement1 = r'\1' + escape_for_replacement(table_body1) + r'\3'
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    
    # Update retrieval performance table
    pattern2 = r'(\\begin\{table\}.*?\\caption\{Retrieval Performance.*?\}.*?\\label\{tab:retrieval_metrics\}.*?\\begin\{tabular\}\{[^}]+\}\n?)(.*?)(\n?\\end\{tabular\}.*?\\end\{table\})'
    table_body2 = extract_table_body(retrieval_table)
    replacement2 = r'\1' + escape_for_replacement(table_body2) + r'\3'
    content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    
    # Update performance table
    pattern3 = r'(\\begin\{table\}.*?\\caption\{System Performance.*?\}.*?\\label\{tab:performance\}.*?\\begin\{tabular\}\{[^}]+\}\n?)(.*?)(\n?\\end\{tabular\}.*?\\end\{table\})'
    table_body3 = extract_table_body(performance_table)
    replacement3 = r'\1' + escape_for_replacement(table_body3) + r'\3'
    content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)
    
    # Update query rewriter impact table
    pattern4 = r'(\\begin\{table\}.*?\\caption\{Impact of Query Rewriting.*?\}.*?\\label\{tab:query_rewriter_impact\}.*?\\begin\{tabular\}\{[^}]+\}\n?)(.*?)(\n?\\end\{tabular\}.*?\\end\{table\})'
    table_body4 = extract_table_body(rewriter_impact_table)
    replacement4 = r'\1' + escape_for_replacement(table_body4) + r'\3'
    content = re.sub(pattern4, replacement4, content, flags=re.DOTALL)
    
    output_file.write_text(content, encoding='utf-8')
    
    print(f"Updated tables in {output_file}")
    print("\nUpdated tables:")
    print("  - Answer Generation Quality (tab:answer_quality)")
    print("  - Retrieval Performance (tab:retrieval_metrics)")
    print("  - System Performance (tab:performance)")
    print("  - Impact of Query Rewriting (tab:query_rewriter_impact)")


if __name__ == "__main__":
    main()

