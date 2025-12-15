# Metrics to LaTeX Tables Mapping

This document maps evaluation metrics to their corresponding LaTeX tables/sections in `results.tex`.

## Table Mappings

### 1. **tab:answer_quality** (Section: Answer Generation Quality)
**Location**: `results.tex` → `\subsection{Answer Generation Quality}`

**Metrics to fill**:
- `rouge_1` → **ROUGE-1** column
- `rouge_2` → **ROUGE-2** column  
- `rouge_l` → **ROUGE-L** column
- `exact_match` → **EM** column

**Script location**: `print_results()` → "Answer Generation Quality" section

---

### 2. **tab:retrieval_metrics** (Section: Retrieval Performance)
**Location**: `results.tex` → `\subsection{Retrieval Performance}`

**Metrics to fill**:
- `precision_at_5` → **Precision@5** column
- `recall_at_10` → **Recall@10** column
- `mrr` → **MRR** column
- `ndcg_at_5` → **NDCG@5** column

**Script location**: `print_results()` → "Retrieval Performance Metrics" section

---

### 3. **tab:query_rewriter_impact** (Section: Impact of Query Rewriting)
**Location**: `results.tex` → `\subsection{Impact of Query Rewriting}`

**Metrics to fill**:
- Same as `tab:answer_quality` (ROUGE-1, ROUGE-2, ROUGE-L, EM)
- **Improvement row**: Calculate delta = (RAG + Query Rewriter) - (RAG Baseline)
- **% Change**: (delta / baseline) × 100

**Script location**: `print_results()` → "Query Rewriting Improvement" section (shows delta values)

---

### 4. **tab:performance** (Section: System Performance)
**Location**: `results.tex` → `\subsection{System Performance}`

**Metrics to fill**:
- `avg_latency` → **Avg Latency (s)** column
- `throughput` = 1.0 / `avg_latency` → **Throughput (qps)** column

**Script location**: `print_results()` → "System Performance" section

---

## Figure Mappings

### 1. **fig:semantic_similarity** (Section: Answer Generation Quality)
**Location**: `results.tex` → `\subsection{Answer Generation Quality}`

**Metrics used**:
- `query_answer_similarity` → Query-Answer Similarity bar
- `query_chunk_similarity` → Query-Chunk Similarity bar

**Script location**: `print_results()` → "Semantic Similarity Metrics" section

**Generate with**: `python3 scripts/create_visualizations.py`

---

### 2. **fig:precision_at_k** (Section: Retrieval Performance)
**Location**: `results.tex` → `\subsection{Retrieval Performance}`

**Metrics used**:
- Precision@k for k = 1, 3, 5, 10 (calculated from `detailed_results`)

**Generate with**: `python3 scripts/create_visualizations.py`

---

### 3. **fig:query_rewriting_improvement** (Section: Impact of Query Rewriting)
**Location**: `results.tex` → `\subsection{Impact of Query Rewriting}`

**Metrics used**:
- Precision@5 improvement (delta)
- NDCG@5 improvement (delta)
- Query-Answer Similarity improvement (delta)
- Query-Chunk Similarity improvement (delta)

**Script location**: `print_results()` → "Query Rewriting Improvement" section

**Generate with**: `python3 scripts/create_visualizations.py`

---

## Quick Reference

When you run the evaluation script, look for these sections in the output:

1. **"Answer Generation Quality"** → Fill `tab:answer_quality`
2. **"Semantic Similarity Metrics"** → Used for `fig:semantic_similarity`
3. **"Retrieval Performance Metrics"** → Fill `tab:retrieval_metrics`
4. **"Query Rewriting Improvement"** → Fill `tab:query_rewriter_impact` (Improvement row) + `fig:query_rewriting_improvement`
5. **"System Performance"** → Fill `tab:performance`

All metrics are also saved in the JSON output file for programmatic access.

