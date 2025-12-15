# Evaluation System - Complete Guide

## ✅ System Status: READY

All evaluation components have been configured and tested. The system is ready to run.

## Quick Start

```bash
# Step 1: Run evaluation
    python3 scripts/evaluate_rag_system.py \
      --queries scripts/test_queries.json \
      --output scripts/evaluation_results.json

# Step 2: Generate visualizations
python3 scripts/create_visualizations.py \
  --results scripts/evaluation_results.json \
  --output-dir overleaf-report/AI_PROJECT_REPORT/figures

# Step 3: Fill LaTeX tables
python3 scripts/fill_results_tables.py \
  --results scripts/evaluation_results.json \
  --output overleaf-report/AI_PROJECT_REPORT/experiements/results.tex
```

## What's Configured

### ✅ Test Queries (`scripts/test_queries.json`)
- 8 queries total
- 3 queries with `reference_answer` (for ROUGE/EM)
- 5 queries without `reference_answer` (retrieval metrics only)
- All queries have course and query_type specified

### ✅ Evaluation Script (`scripts/evaluate_rag_system.py`)
- Supports optional reference answers
- Calculates all metrics:
  - ROUGE-1, ROUGE-2, ROUGE-L, EM (when reference available)
  - Precision@5, Recall@10, MRR, NDCG@5 (for all queries)
  - Query-Answer Semantic Similarity (for all queries)
  - Query-Chunk Semantic Similarity (for all queries)
- Handles missing metrics gracefully
- Outputs formatted tables with section markers
- Saves detailed JSON results

### ✅ Visualization Script (`scripts/create_visualizations.py`)
- Creates 3 PDF charts:
  1. Semantic Similarity Scores (bar chart)
  2. Precision@k Curves (line plot)
  3. Query Rewriting Improvement (bar chart)
- Handles missing metrics (defaults to 0.0)
- Creates output directory automatically

### ✅ Fill Tables Script (`scripts/fill_results_tables.py`)
- Automatically fills 4 LaTeX tables:
  - `tab:answer_quality`
  - `tab:retrieval_metrics`
  - `tab:query_rewriter_impact`
  - `tab:performance`
- Handles None values properly
- Preserves LaTeX formatting

### ✅ LaTeX Integration (`results.tex`)
- All tables have correct labels
- 3 figures properly referenced and placed:
  - `fig:semantic_similarity` (Answer Generation Quality section)
  - `fig:precision_at_k` (Retrieval Performance section)
  - `fig:query_rewriting_improvement` (Impact of Query Rewriting section)

## Expected Output

### Console Output
The evaluation script will print:
1. **Answer Generation Quality** → Fill `tab:answer_quality`
2. **Semantic Similarity Metrics** → Used for `fig:semantic_similarity`
3. **Retrieval Performance Metrics** → Fill `tab:retrieval_metrics`
4. **Query Rewriting Improvement** → Fill `tab:query_rewriter_impact` + `fig:query_rewriting_improvement`
5. **System Performance** → Fill `tab:performance`

### Files Created
- `scripts/evaluation_results.json` - Detailed results
- `overleaf-report/AI_PROJECT_REPORT/figures/semantic_similarity.pdf`
- `overleaf-report/AI_PROJECT_REPORT/figures/precision_at_k.pdf`
- `overleaf-report/AI_PROJECT_REPORT/figures/query_rewriting_improvement.pdf`
- Updated `overleaf-report/AI_PROJECT_REPORT/experiements/results.tex`

## Dependencies

Install required packages:
```bash
pip install rouge-score matplotlib numpy
```

## Notes

- **"N/A" values are expected** for queries without reference answers
- Evaluation may take 10-30 minutes depending on LLM response time
- All scripts handle missing metrics gracefully
- Scripts create necessary directories automatically

## Troubleshooting

See `docs/evaluation_checklist.md` for detailed troubleshooting guide.

## Documentation

- `docs/evaluation_checklist.md` - Pre-flight checks and troubleshooting
- `docs/evaluation_ready_summary.md` - Component status
- `docs/metrics_to_tables_mapping.md` - Metrics to LaTeX tables mapping
- `docs/figure_placement_guide.md` - Figure placement guide
- `docs/evaluation_without_references.md` - Evaluation without reference answers
- `docs/evaluation_metrics_and_visualizations.md` - Metrics and visualizations

---

**Status**: All systems ready. You can now run the evaluation workflow.

