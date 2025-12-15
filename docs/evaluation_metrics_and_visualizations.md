# Evaluation Metrics and Visualizations for RAG Project

## Proposed Evaluation Metrics

### 1. Query-Answer Semantic Similarity
- **Purpose**: Measure how well generated answers address the original query intent
- **Calculation**: Cosine similarity between query embedding and answer embedding
- **Why**: ROUGE measures lexical overlap, but semantic similarity captures meaning alignment
- **Formula**: `similarity = cosine(query_embedding, answer_embedding)`

### 2. Query Rewriting Improvement
- **Purpose**: Quantify the benefit of query rewriting
- **Metrics**:
  - Retrieval improvement: `ΔPrecision@k = Precision@k(rewritten) - Precision@k(original)`
  - Answer alignment improvement: `ΔSimilarity = Similarity(rewritten) - Similarity(original)`
  - Relative improvement: `%Improvement = (Δ / baseline) × 100`

### 3. Query-Chunk Semantic Similarity
- **Purpose**: Evaluate retrieval quality without reference answers
- **Calculation**: Average cosine similarity between query and top-k retrieved chunks
- **Why**: Enables evaluation for internal datasets without exact reference answers

## Proposed Visualizations

### 1. Bar Chart: ROUGE Scores Comparison
```
Purpose: Compare answer quality between configurations
X-axis: ROUGE-1, ROUGE-2, ROUGE-L, EM
Y-axis: Score (0-1)
Bars: RAG Baseline vs RAG + Query Rewriter
Colors: Different colors for each configuration
```

### 2. Bar Chart: Semantic Similarity Scores
```
Purpose: Show query-answer alignment improvement
X-axis: Query-Answer Similarity, Query-Chunk Similarity
Y-axis: Similarity Score (0-1)
Bars: RAG Baseline vs RAG + Query Rewriter
```

### 3. Line Plot: Precision@k Curves
```
Purpose: Show retrieval quality at different k values
X-axis: k (1, 3, 5, 10)
Y-axis: Precision@k
Lines: RAG Baseline vs RAG + Query Rewriter
```

### 4. Bar Chart: Query Rewriting Improvement
```
Purpose: Quantify improvement from query rewriting
X-axis: Metrics (Precision@5, Recall@10, NDCG@5, Query-Answer Similarity)
Y-axis: Improvement (absolute or percentage)
Bars: Show improvement values (positive = better)
```

### 5. Heatmap: Query Type vs Performance
```
Purpose: Identify where query rewriting is most effective
X-axis: Query Types (course_info, technical, factual, comparative)
Y-axis: Metrics (ROUGE-L, Semantic Similarity, Precision@5)
Colors: Performance scores (darker = better)
```

### 6. Scatter Plot: Quality-Latency Trade-off
```
Purpose: Visualize performance vs quality trade-off
X-axis: Average Latency (seconds)
Y-axis: ROUGE-L or Semantic Similarity
Points: Different configurations
Size: Number of queries (optional)
```

### 7. Box Plot: Query Rewriting Improvement Distribution
```
Purpose: Show distribution of improvements across queries
X-axis: Metrics (Precision@5, Semantic Similarity, etc.)
Y-axis: Improvement value
Boxes: Show median, quartiles, outliers
```

## Implementation in Evaluation Script

The evaluation script should calculate:
1. Query-Answer Semantic Similarity for each query
2. Query-Chunk Semantic Similarity (average of top-k)
3. Query Rewriting Improvement (delta metrics)
4. Query type breakdown

## LaTeX Figure Examples

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/rouge_comparison.pdf}
\caption{ROUGE scores comparison between RAG baseline and RAG with query rewriting.}
\label{fig:rouge_comparison}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/semantic_similarity.pdf}
\caption{Query-answer semantic similarity scores showing improved alignment with query rewriting.}
\label{fig:semantic_similarity}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/query_rewriting_improvement.pdf}
\caption{Improvement metrics demonstrating the effectiveness of query rewriting across different evaluation dimensions.}
\label{fig:query_rewriting_improvement}
\end{figure}
```

