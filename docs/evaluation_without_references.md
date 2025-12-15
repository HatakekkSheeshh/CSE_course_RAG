# Evaluation Without Reference Answers

For internal HCMUT datasets where exact reference answers are unavailable, here are practical evaluation approaches.

**Note**: The evaluation script (`scripts/evaluate_rag_system.py`) now supports optional reference answers. You can omit `reference_answer` in `test_queries.json` - the script will:
- Skip ROUGE/EM calculations (shows "N/A")
- Still calculate retrieval metrics (Precision@k, Recall@k, NDCG@k) using query-chunk similarity
- Work seamlessly with mixed queries (some with references, some without)

## 1. Retrieval-Focused Metrics (Recommended)

Focus on **retrieval quality** rather than answer generation. These metrics don't require reference answers:

### Query-Chunk Relevance Scoring
- Use semantic similarity between query and retrieved chunks
- Higher similarity = better retrieval
- Metrics: Precision@k, Recall@k, NDCG@k based on similarity scores

### Human Relevance Judgments
- Manually label retrieved chunks as "relevant" or "not relevant" for each query
- Small sample (20-50 queries) is sufficient
- Calculate metrics based on human labels

## 2. Human Evaluation (Subjective Assessment)

### Evaluation Criteria
Rate each answer on:
- **Relevance** (1-5): Does it answer the question?
- **Accuracy** (1-5): Is information factually correct?
- **Completeness** (1-5): Does it cover all aspects?
- **Clarity** (1-5): Is it easy to understand?

### Process
1. Create evaluation form with queries and generated answers
2. Have 2-3 evaluators rate each answer
3. Calculate inter-annotator agreement
4. Report average scores per configuration

## 3. Source Grounding Evaluation

Check if answers are **grounded in retrieved chunks**:

- **Source Coverage**: % of answer sentences that can be traced to retrieved chunks
- **Hallucination Detection**: Flag answers with information not in retrieved chunks
- **Citation Quality**: Verify that cited sources actually contain the information

## 4. Semantic Similarity to Retrieved Chunks

Instead of comparing to reference answers, compare generated answers to:
- **Top retrieved chunks**: High similarity = answer is grounded
- **All retrieved chunks**: Average similarity = overall grounding quality

## 5. Hybrid Approach (Recommended for Papers)

Combine multiple methods:

1. **Quantitative**: Retrieval metrics (Precision@k, NDCG@k) - no reference needed
2. **Qualitative**: Human evaluation on sample (20-30 queries)
3. **Case Studies**: Detailed analysis of representative queries
4. **Ablation Studies**: Compare with/without query rewriting, reranking, etc.

## Implementation Example

```python
# Evaluate retrieval quality without reference answers
def evaluate_retrieval_quality(query, retrieved_chunks, embedding_model):
    """Evaluate retrieval based on query-chunk similarity."""
    query_embedding = embedding_model.embed(query)
    
    similarities = []
    for chunk in retrieved_chunks:
        chunk_embedding = embedding_model.embed(chunk['text'])
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append(similarity)
    
    # Higher average similarity = better retrieval
    return {
        'avg_similarity': np.mean(similarities),
        'top_k_similarity': np.mean(similarities[:k]),
        'coverage': len([s for s in similarities if s > threshold]) / len(similarities)
    }
```

## For Your Paper

Since you're writing a paper, I recommend:

1. **Primary Metrics**: Retrieval metrics (Precision@5, Recall@10, NDCG@5) - objective, no reference needed
2. **Secondary Metrics**: Human evaluation on 30-50 representative queries
3. **Qualitative Analysis**: Case studies showing query rewriting impact
4. **Ablation Studies**: Compare configurations systematically

This approach is common in RAG papers when reference answers aren't available.

