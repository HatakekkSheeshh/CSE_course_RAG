# Tại sao Query Rewriting vẫn có giá trị với Temperature thấp?

## ❓ Câu hỏi: "Nếu temperature thấp giữ nguyên thông tin, thì có khác gì so với không rewrite?"

## ✅ Trả lời: Query Rewriting vẫn có giá trị vì:

### 1. **Expand & Thêm Terminology** (Quan trọng nhất!)

**Không rewrite:**
```
Query: "grading"
Embedding: [grading] → Tìm chunks có từ "grading"
```

**Có rewrite (temperature 0.1):**
```
Query: "grading"
Rewritten: "What is the grading policy and assessment criteria for this course?"
Embedding: [grading, policy, assessment, criteria, course] → Tìm chunks có nhiều từ khóa liên quan hơn
```

**Kết quả:** Retrieval tốt hơn vì embedding có nhiều từ khóa hơn!

---

### 2. **Normalize & Chuẩn hóa câu hỏi**

**Không rewrite:**
```
Query: "prerequisites?"
→ Embedding có thể không rõ ràng (câu hỏi không đầy đủ)
```

**Có rewrite (temperature 0.1):**
```
Query: "prerequisites?"
Rewritten: "What are the prerequisites for this course?"
→ Embedding rõ ràng hơn, semantic search tốt hơn
```

**Kết quả:** Câu hỏi được chuẩn hóa, retrieval ổn định hơn!

---

### 3. **Thêm Course Context**

**Không rewrite:**
```
Query: "What topics are covered?"
Course: Introduction_to_Computing
→ Embedding không có course context
```

**Có rewrite (temperature 0.1):**
```
Query: "What topics are covered?"
Course: Introduction_to_Computing
Rewritten: "What topics and course content are covered in Introduction to Computing?"
→ Embedding có course context rõ ràng
```

**Kết quả:** Retrieval tập trung vào course cụ thể hơn!

---

### 4. **Disambiguate (Làm rõ từ ngữ mơ hồ)**

**Không rewrite:**
```
Query: "assignments"
→ Có thể hiểu là: homework? projects? lab work? exams?
```

**Có rewrite (temperature 0.1):**
```
Query: "assignments"
Rewritten: "What are the course assignments, homework, and project requirements?"
→ Rõ ràng hơn, bao gồm nhiều khái niệm liên quan
```

**Kết quả:** Retrieval chính xác hơn, không bỏ sót thông tin!

---

## 📊 So sánh cụ thể

### Ví dụ 1: Query ngắn
```
Original: "grading policy"
```

**Không rewrite:**
- Embedding: `["grading", "policy"]`
- Semantic search: Tìm chunks có 2 từ này
- **Vấn đề:** Có thể bỏ sót chunks dùng từ "assessment", "evaluation", "scoring"

**Có rewrite (temperature 0.1):**
- Rewritten: "What is the grading policy, assessment criteria, and evaluation method for this course?"
- Embedding: `["grading", "policy", "assessment", "criteria", "evaluation", "method", "course"]`
- Semantic search: Tìm chunks có nhiều từ khóa liên quan
- **Kết quả:** Retrieval tốt hơn, không bỏ sót!

---

### Ví dụ 2: Query không có course context
```
Original: "What are the prerequisites?"
Course: Introduction_to_Computing (detected from query context)
```

**Không rewrite:**
- Embedding: `["prerequisites"]`
- Search: Tìm trong TẤT CẢ courses
- **Vấn đề:** Có thể trả về prerequisites của course khác

**Có rewrite (temperature 0.1):**
- Rewritten: "What are the prerequisites for Introduction to Computing course?"
- Embedding: `["prerequisites", "Introduction", "Computing", "course"]`
- Search: Tập trung vào course cụ thể
- **Kết quả:** Retrieval chính xác hơn!

---

### Ví dụ 3: Query mơ hồ
```
Original: "assignments"
```

**Không rewrite:**
- Embedding: `["assignments"]`
- **Vấn đề:** Không rõ là homework, projects, hay lab work?

**Có rewrite (temperature 0.1):**
- Rewritten: "What are the course assignments, homework, projects, and lab work requirements?"
- Embedding: `["assignments", "homework", "projects", "lab", "work", "requirements"]`
- **Kết quả:** Bao phủ nhiều khái niệm liên quan, retrieval đầy đủ hơn!

---

## 🎯 Temperature thấp (0.1) vs Temperature cao (0.7)

| Aspect | Temperature 0.1 | Temperature 0.7 |
|--------|------------------|-----------------|
| **Expansion** | ✅ Có kiểm soát, giữ nguyên thông tin | ⚠️ Có thể quá sáng tạo, mất thông tin |
| **Consistency** | ✅ Nhất quán mỗi lần | ❌ Khác nhau mỗi lần |
| **Information Loss** | ✅ Không mất thông tin | ❌ Có thể mất thông tin |
| **Retrieval Quality** | ✅ Ổn định, tốt | ⚠️ Không ổn định |

**Kết luận:** Temperature 0.1 vẫn **có expansion và improvement**, nhưng **có kiểm soát và nhất quán**!

---

## 💡 Tại sao vẫn cần Rewriting?

### 1. **Semantic Embedding tốt hơn**
- Query dài hơn → embedding có nhiều dimensions hơn
- Nhiều từ khóa → semantic search chính xác hơn
- Context rõ ràng → retrieval tập trung hơn

### 2. **Vocabulary Mismatch**
- User dùng từ: "grading"
- Document dùng từ: "assessment criteria"
- **Không rewrite:** Có thể không match
- **Có rewrite:** "grading policy and assessment criteria" → Match tốt hơn!

### 3. **Query Quality**
- User query: "prerequisites?" (không đầy đủ)
- **Không rewrite:** Embedding không rõ ràng
- **Có rewrite:** "What are the prerequisites for this course?" → Embedding rõ ràng hơn

---

## 🔬 Thử nghiệm để chứng minh

Bạn có thể test để thấy sự khác biệt:

```python
# Test 1: Query ngắn
query = "grading"
# Không rewrite: embedding chỉ có 1-2 từ
# Có rewrite: embedding có 5-7 từ → retrieval tốt hơn

# Test 2: Query không có course context
query = "What topics are covered?"
# Không rewrite: search trong tất cả courses
# Có rewrite: "What topics are covered in Introduction to Computing?" → tập trung hơn

# Test 3: Query mơ hồ
query = "assignments"
# Không rewrite: chỉ tìm "assignments"
# Có rewrite: "assignments, homework, projects, lab work" → bao phủ hơn
```

---

## ✅ Kết luận

**Temperature thấp (0.1) KHÔNG có nghĩa là không có thay đổi!**

Nó chỉ đảm bảo:
- ✅ Thay đổi **có kiểm soát** (không mất thông tin)
- ✅ Thay đổi **nhất quán** (mỗi lần giống nhau)
- ✅ Thay đổi **hữu ích** (expand, normalize, disambiguate)

**Query rewriting vẫn có giá trị vì:**
1. Expand vocabulary → retrieval tốt hơn
2. Normalize queries → consistency tốt hơn
3. Add context → precision tốt hơn
4. Disambiguate → recall tốt hơn

**Nếu không có rewriting:** Chỉ dựa vào từ khóa gốc → có thể bỏ sót thông tin liên quan!

