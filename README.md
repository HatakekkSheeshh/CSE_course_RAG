flowchart TB
  %% Lanes
  subgraph L0[Inputs]
    I0[Page Images (PNG/JPG)]
  end

  subgraph L1[OCR / Parsing]
    A1[1) OCR\n• PaddleOCR (DBNet+SVTR)\n• Hoặc Donut OCR-free]
    A2[[Output: per-page JSONL\n{source_file, page_index, lines:[{text,bbox,conf}], doc_type}]]
  end

  subgraph L2[Text Processing]
    B1[2) Normalize & Layout\n• clean text, unicode norm\n• merge token→line, giữ bbox\n• language detect]
    B2[3) Chunking\n• page/semantic split\n• target≈350 tok, overlap≈60]
    B3[[Output: chunks.jsonl\n{text, metadata: {doc_type, course_id, source_file, slide_index, heading, language}}]]
  end

  subgraph L3[Embedding & Indexing]
    C1[4) Embedding\n• Sentence-Transformers\n(all-MiniLM-L6-v2)]
    C2[5) Build Index\n• FAISS (cosine)\n• meta.parquet song hành]
    C3[[Artifacts:\nfaiss.index + meta.parquet]]
  end

  subgraph L4[Retrieval & Generation]
    D1[6) Retrieval\n• encode(query) → top-k từ FAISS\n• (optional) cross-encoder rerank]
    D2[7) Answer Synthesis\n• LLM (RAG) + cited context\n• format: text/markdown/JSON]
    D3[[Logs & Eval\n• Recall@k, MRR\n• query, hits, latency]]
  end

  I0 --> A1 --> A2 --> B1 --> B2 --> B3 --> C1 --> C2 --> C3 --> D1 --> D2 --> D3
