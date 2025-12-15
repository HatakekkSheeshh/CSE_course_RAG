# Figure Placement Guide

This document explains where each visualization figure should be placed in the LaTeX report.

## Figure Locations in `results.tex`

### 1. **Semantic Similarity Chart** (`semantic_similarity.pdf`)
- **Location**: `\subsection{Answer Generation Quality}`
- **Placement**: After Table~\ref{tab:answer_quality} and the paragraph describing ROUGE results
- **Purpose**: Shows query-answer alignment and query-chunk matching using semantic similarity
- **Figure Reference**: `\ref{fig:semantic_similarity}`

**Why here?** This figure complements the ROUGE metrics table by showing semantic-level alignment, which is particularly relevant for evaluating answer quality.

---

### 2. **Precision@k Curves** (`precision_at_k.pdf`)
- **Location**: `\subsection{Retrieval Performance}`
- **Placement**: After Table~\ref{tab:retrieval_metrics} and the paragraph describing retrieval results
- **Purpose**: Shows how retrieval quality (Precision@k) varies with different k values
- **Figure Reference**: `\ref{fig:precision_at_k}`

**Why here?** This figure visualizes the retrieval performance metrics discussed in the table, showing the effectiveness across different retrieval depths.

---

### 3. **Query Rewriting Improvement** (`query_rewriting_improvement.pdf`)
- **Location**: `\subsection{Impact of Query Rewriting}`
- **Placement**: After Table~\ref{tab:query_rewriter_impact} and the paragraph describing improvements
- **Purpose**: Quantifies the absolute improvement from query rewriting across multiple metrics
- **Figure Reference**: `\ref{fig:query_rewriting_improvement}`

**Why here?** This figure directly visualizes the impact analysis, showing improvements in a clear, comparative format.

---

## Steps to Add Figures

1. **Generate the figures**:
   ```bash
   python3 scripts/create_visualizations.py \
     --results scripts/evaluation_results.json \
     --output-dir overleaf-report/AI_PROJECT_REPORT/figures
   ```

2. **Ensure figures directory exists**:
   ```bash
   mkdir -p overleaf-report/AI_PROJECT_REPORT/figures
   ```

3. **Verify figures are created**:
   - `overleaf-report/AI_PROJECT_REPORT/figures/semantic_similarity.pdf`
   - `overleaf-report/AI_PROJECT_REPORT/figures/precision_at_k.pdf`
   - `overleaf-report/AI_PROJECT_REPORT/figures/query_rewriting_improvement.pdf`

4. **Uncomment results.tex in main.tex** (if not already):
   ```latex
   \input{experiements/results}  % Remove the % comment
   ```

5. **Compile LaTeX** to see the figures in your document.

---

## Figure Captions

Each figure has been added with:
- Appropriate caption describing what it shows
- Reference label for cross-referencing
- Placement near relevant tables/text for context

The figures are already integrated into `results.tex` with proper LaTeX figure environments.

