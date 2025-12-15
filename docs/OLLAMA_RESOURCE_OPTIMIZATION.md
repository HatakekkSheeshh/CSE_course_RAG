# Ollama Resource Optimization Guide

## Vấn đề: Máy bị crash khi chạy Ollama

Ollama có thể sử dụng rất nhiều GPU/RAM, đặc biệt với model lớn như `llama2` (7B parameters). Đây là hướng dẫn để tối ưu hóa và tránh crash.

## Giải pháp 1: Giới hạn Resource trong Docker (Đã thêm vào docker-compose.yml)

Docker Compose đã được cấu hình với resource limits:
- **CPU**: Tối đa 4 cores
- **RAM**: Tối đa 8GB
- **GPU**: Giới hạn 1 GPU

Bạn có thể điều chỉnh trong `docker-compose.yml`:
```yaml
mem_limit: 4g      # Giảm xuống 4GB nếu RAM ít (mặc định: 8g)
cpus: 2.0          # Giảm xuống 2 cores nếu máy yếu (mặc định: 4.0)
```

**Lưu ý**: Sau khi thay đổi, cần rebuild container:
```bash
docker-compose up -d --force-recreate ollama
```

## Giải pháp 2: Sử dụng Model Nhỏ Hơn (Khuyến nghị)

### Option A: Quantized Model (Nhẹ hơn, nhanh hơn)
```bash
# Thay vì llama2 (7B, ~13GB), dùng quantized version
docker exec cse_course_rag_ollama_1 ollama pull llama2:7b-q4_0  # ~4GB
# Hoặc
docker exec cse_course_rag_ollama_1 ollama pull llama2:7b-q8_0  # ~7GB, chất lượng tốt hơn
```

Sau đó cập nhật `.env`:
```env
OLLAMA_MODEL=llama2:7b-q4_0
```

### Option B: Model Nhỏ Hơn
```bash
# Phi-3 Mini (3.8B parameters, ~2.3GB)
docker exec cse_course_rag_ollama_1 ollama pull phi3:mini

# TinyLlama (1.1B parameters, ~700MB) - Rất nhẹ nhưng chất lượng thấp hơn
docker exec cse_course_rag_ollama_1 ollama pull tinyllama
```

Cập nhật `.env`:
```env
OLLAMA_MODEL=phi3:mini
# hoặc
OLLAMA_MODEL=tinyllama
```

## Giải pháp 3: Chạy Ollama ở CPU Mode (Nếu GPU quá yếu)

Nếu GPU của bạn quá yếu hoặc không có đủ VRAM:

1. **Cập nhật docker-compose.yml**:
```yaml
environment:
  - OLLAMA_NUM_GPU=0   # Force CPU-only mode
```

2. **Hoặc chạy Ollama ngoài Docker** (khuyến nghị cho Windows):
   - Tải Ollama Desktop: https://ollama.ai/download
   - Chạy Ollama trên host (không qua Docker)
   - Cập nhật `.env`: `OLLAMA_BASE_URL=http://host.docker.internal:11434`

## Giải pháp 4: Giảm Context Window và Batch Size

Tạo file `~/.ollama/config.json` (hoặc trong container):
```json
{
  "num_ctx": 2048,      # Giảm từ mặc định 4096
  "num_batch": 256,     # Giảm batch size
  "num_gpu": 1          # Số GPU (0 = CPU only)
}
```

## Giải pháp 5: Monitor Resource Usage

### Kiểm tra RAM/GPU usage:
```bash
# Trong container
docker stats cse_course_rag_ollama_1

# Hoặc trên Windows
# Mở Task Manager > Performance tab
```

### Kiểm tra model size:
```bash
docker exec cse_course_rag_ollama_1 ollama list
docker exec cse_course_rag_ollama_1 ollama show llama2
```

## So sánh Model Sizes

| Model | Parameters | Size (RAM) | Quality | Speed |
|-------|-----------|------------|---------|-------|
| `llama2` | 7B | ~13GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `llama2:7b-q4_0` | 7B (quantized) | ~4GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `llama2:7b-q8_0` | 7B (quantized) | ~7GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `phi3:mini` | 3.8B | ~2.3GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `tinyllama` | 1.1B | ~700MB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Khuyến nghị

1. **Nếu RAM < 8GB**: Dùng `phi3:mini` hoặc `tinyllama`
2. **Nếu RAM 8-16GB**: Dùng `llama2:7b-q4_0` hoặc `llama2:7b-q8_0`
3. **Nếu RAM > 16GB**: Có thể dùng `llama2` full version
4. **Nếu GPU < 4GB VRAM**: Dùng CPU mode hoặc quantized model

## Troubleshooting

### Lỗi: "Out of memory"
- Giảm `memory` limit trong docker-compose.yml
- Chuyển sang model nhỏ hơn
- Giảm `num_ctx` trong config

### Lỗi: "GPU out of memory"
- Set `OLLAMA_NUM_GPU=0` để dùng CPU
- Hoặc dùng model quantized nhỏ hơn

### Máy vẫn crash
- Kiểm tra Windows Memory Diagnostic
- Đóng các ứng dụng khác đang dùng RAM/GPU
- Xem xét nâng cấp RAM hoặc dùng Gemini API thay vì Ollama

## Chuyển sang Gemini (Nếu Ollama quá nặng)

Nếu Ollama vẫn gây crash, chuyển sang Gemini (miễn phí, không tốn tài nguyên máy):

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Gemini chạy trên cloud, không tốn GPU/RAM của máy bạn.

