# Query Rewriter Issues & Solutions

## 🔴 Vấn đề hiện tại

### 1. **ROUGE-L giảm: 0.179 → 0.092**
**Nguyên nhân:**
- Reranker đang dùng **original query** thay vì **rewritten query** → mismatch với retrieval
- Llama2 có thể không đủ tốt cho query rewriting
- Rewritten queries có thể mất thông tin quan trọng

**Đã sửa:**
- ✅ **Fixed:** Reranker giờ cũng dùng rewritten query (consistent với retrieval)
- ✅ Cải thiện prompt để preserve ALL key information
- ✅ Giảm temperature: 0.3 → 0.1 (deterministic hơn)
- ✅ Tăng max_tokens: 200 → 250 (đảm bảo đủ dài)

### 2. **Latency tăng: 46.84s → 105.52s**
**Nguyên nhân:**
- Phải gọi LLM **2 lần**: 1 lần để rewrite query, 1 lần để generate answer
- Llama2 chạy local nên chậm hơn Gemini API

**Giải pháp:**
- ⚠️ **Expected behavior** - không thể tránh được nếu dùng query rewriting
- 💡 **Có thể tối ưu:** Cache rewritten queries, hoặc chỉ rewrite khi query quá ngắn/ambiguous
- 💡 **Alternative:** Dùng Gemini cho rewriting (nhanh hơn) nhưng tốn API quota

### 3. **Query rewriting có thể làm mất thông tin**
**Nguyên nhân:**
- Llama2 có thể không hiểu đúng yêu cầu
- Prompt chưa đủ mạnh để enforce preservation
- Temperature quá cao → không deterministic

**Đã sửa:**
- ✅ Cải thiện prompt với CRITICAL RULES rõ ràng
- ✅ Nhấn mạnh "PRESERVE ALL KEY INFORMATION", "DO NOT SHORTEN"
- ✅ Giảm temperature xuống 0.1
- ✅ Tăng max_tokens lên 250

## 🔧 Các thay đổi đã thực hiện

### 1. **Fix Reranker Mismatch** (`rag/query_pipeline.py`)
```python
# TRƯỚC: Reranker dùng original query
def rerank(self, query: str, retrieved: ...):
    return self.reranker.score(query, passages)  # ❌ Mismatch!

# SAU: Reranker cũng dùng rewritten query
def rerank(self, query: str, retrieved: ..., course: Optional[str] = None):
    rerank_query = query
    if self.query_rewriter and self.query_rewriter.is_available:
        rerank_query = self.query_rewriter.rewrite(query, course=course)  # ✅ Consistent!
    return self.reranker.score(rerank_query, passages)
```

### 2. **Cải thiện Prompt** (`rag/query_rewriter.py`)
- Thêm CRITICAL RULES rõ ràng
- Nhấn mạnh preserve ALL key information
- Không được shorten query

### 3. **Tối ưu Config** (`config/config.py`)
- Temperature: 0.3 → 0.1 (deterministic hơn)
- Max tokens: 200 → 250 (đủ dài)

## 📊 Kết quả mong đợi

Sau khi fix:
1. **ROUGE-L sẽ cải thiện** vì reranker giờ consistent với retrieval
2. **Latency vẫn cao** (expected - phải gọi LLM 2 lần)
3. **Information loss sẽ giảm** nhờ prompt tốt hơn và temperature thấp hơn

## 🧪 Test lại

Chạy lại evaluation để xem kết quả:
```bash
docker-compose exec backend python3 scripts/evaluate_rag_system.py \
  --queries scripts/test_queries.json \
  --output scripts/evaluation_results.json
```

## 💡 Đề xuất thêm (nếu vẫn chưa tốt)

1. **Dùng Gemini cho rewriting** (nhanh hơn, tốt hơn Llama2)
   - Set `MODEL_USING=gemini` trong `.env`
   - Gemini tốt hơn cho query rewriting nhưng tốn API quota

2. **Conditional rewriting** - chỉ rewrite khi:
   - Query quá ngắn (< 5 words)
   - Query không có course name nhưng cần course-specific answer

3. **Query expansion thay vì rewriting** - thêm synonyms thay vì thay thế hoàn toàn


