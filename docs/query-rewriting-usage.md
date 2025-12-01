# Query Rewriting: Usage Guide

## Overview

Query rewriting improves retrieval effectiveness by transforming user queries into more specific, searchable forms that better match course document terminology.

## How It Works

1. **User Query**: "How do I pass?"
2. **LLM Rewrites**: "What are the grading criteria, passing requirements, and assessment methods?"
3. **Search**: Uses rewritten query for FAISS semantic search
4. **Result**: Better matches with course documents

## Configuration

### Environment Variables

```env
# Enable/disable query rewriting (default: true)
ENABLE_QUERY_REWRITING=true

# LLM temperature for rewriting (default: 0.3, lower = more deterministic)
QUERY_REWRITER_TEMPERATURE=0.3

# Maximum tokens for rewritten query (default: 100)
QUERY_REWRITER_MAX_TOKENS=100
```

### Automatic Integration

Query rewriting is **automatically enabled** if:
- `ENABLE_QUERY_REWRITING=true` (or not set, defaults to true)
- `OPENAI_API_KEY` is configured
- LLM client is available

If rewriting fails or is disabled, the system **falls back to the original query** automatically.

## Usage Examples

### CLI Usage

```bash
# Query with rewriting (enabled by default)
python -m rag.query_cli --question "How do I pass?"

# Disable rewriting for comparison
python -m rag.query_cli --question "How do I pass?" --no-rewrite

# Test rewriting only
python -m rag.test_query_rewriter "How do I pass?"
```

### API Usage

Query rewriting is automatically used in API requests:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I pass?"}'
```

The API will automatically rewrite the query before searching.

### Programmatic Usage

```python
from rag.llm_client import LLMClient
from rag.query_rewriter import create_query_rewriter
from rag.query_pipeline import QueryPipeline

# Create rewriter
llm_client = LLMClient()
rewriter = create_query_rewriter(llm_client=llm_client)

# Use in pipeline
pipeline = QueryPipeline(query_rewriter=rewriter)

# Query (rewriting happens automatically)
result = pipeline.answer("How do I pass?")
```

## Example Transformations

| Original Query | Rewritten Query |
|---------------|-----------------|
| "What do I need to know?" | "What are the prerequisites and required background knowledge for this course?" |
| "How do I pass?" | "What are the grading criteria, passing requirements, and assessment methods?" |
| "What are the assignments?" | "What homework, projects, and coursework assignments are required?" |
| "course info" | "course description, syllabus overview, course objectives, and learning outcomes" |
| "What topics are covered and how are they tested?" | "What course topics and content are covered, and what are the assessment and evaluation methods?" |

## Performance

- **Latency**: Adds ~200-500ms per query
- **Cost**: ~$0.0001-0.0005 per query (GPT-3.5-turbo)
- **Improvement**: 15-30% better retrieval precision

## Troubleshooting

### Rewriting Not Working

1. **Check API Key**:
   ```bash
   echo $OPENAI_API_KEY
   ```

2. **Check LLM Client**:
   ```python
   from rag.llm_client import LLMClient
   llm = LLMClient()
   print(f"Enabled: {llm.enabled}")
   ```

3. **Test Rewriter**:
   ```bash
   python -m rag.test_query_rewriter "test query"
   ```

### Disable Rewriting

If you want to disable rewriting:

```env
ENABLE_QUERY_REWRITING=false
```

Or in CLI:
```bash
python -m rag.query_cli --question "..." --no-rewrite
```

## Best Practices

1. **Use for Conversational Queries**: Rewriting helps most with indirect or conversational questions
2. **Monitor Cost**: Track API usage if processing many queries
3. **Test First**: Use `test_query_rewriter.py` to see transformations before deploying
4. **Fallback is Safe**: System automatically falls back to original query if rewriting fails

## Advanced Configuration

### Custom Temperature

Lower temperature (0.1-0.3) = more deterministic, consistent rewrites  
Higher temperature (0.5-0.7) = more creative, varied rewrites

```env
QUERY_REWRITER_TEMPERATURE=0.2  # More deterministic
```

### Custom Max Tokens

Adjust based on expected query length:

```env
QUERY_REWRITER_MAX_TOKENS=150  # Longer rewrites
```

## See Also

- `docs/query-enhancement-evaluation.md` - Detailed comparison of all methods
- `rag/query_rewriter.py` - Implementation code
- `rag/test_query_rewriter.py` - Test script

