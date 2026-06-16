# Failure Analysis — AI Evaluation Factory
*Lab Day 14 | VinUni AI Engineering Bootcamp*

## Thành viên

| Tên | MSSV |
|-----|------|
| Vũ Quốc Bảo | 2A202600541 |
| Vũ Văn Huy | 2A202600750 |
| Nguyễn Trung Kiên | 2A202600969 |
| Lê Đình Sỹ | 2A202600770 |
| Phạm Hoàng Anh Kiệt | 2A202600797 |

## 1. Tổng quan kết quả Benchmark

| Phiên bản | Avg Score | Hit Rate | Agreement Rate | Hallucination Rate |
|-----------|-----------|----------|----------------|--------------------|
| V1 Base   | ~3.2      | ~0.62    | ~0.55          | ~0.15              |
| V2 Opt    | ~3.7      | ~0.71    | ~0.61          | ~0.08              |

- **Tổng số cases:** 55+
- **Tỉ lệ Pass (score ≥ 3):** ~82% (V2)
- **Điểm RAGAS trung bình:** Faithfulness 0.82, Relevancy 0.79
- **Điểm LLM-Judge trung bình:** 3.7 / 5.0 (V2)

**Kết luận:** V2 cải thiện đáng kể nhờ system prompt mạnh hơn và hướng dẫn từ chối hallucination rõ ràng.

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Retrieval Miss | ~8 | Keyword mismatch, từ đồng nghĩa không khớp |
| Hallucination (OOD) | ~6 | Không có relevance threshold, agent bịa thông tin |
| Edge Case Reasoning | ~4 | Policy có khoảng trống, agent không xử lý được |
| Prompt Injection | ~3 (V1) | Thiếu system prompt chống injection |

---

## 3. Phân tích 5 Whys

### Case #1: Hallucination trên câu hỏi OOD (Out-of-Distribution)

**Symptom:** Agent trả lời "Công ty cho phép nuôi chó nhỏ tại văn phòng nếu có sự đồng ý của quản lý" — thông tin hoàn toàn không có trong corpus.

1. **Why 1:** Agent tự bịa vì không tìm được tài liệu liên quan.
2. **Why 2:** Không có cơ chế kiểm tra relevance score của context được retrieve.
3. **Why 3:** Retrieval stage dùng keyword matching không có semantic similarity threshold.
4. **Why 4:** Không tích hợp embedding-based retrieval (FAISS/Chroma) để tính cosine similarity.
5. **Root Cause:** **Ingestion pipeline thiếu vectorization.** Không có vector store nên không thể filter context có độ liên quan thấp. Agent nhận context "rác" và hallucinate thay vì từ chối.

**Action:** Tích hợp `sentence-transformers` + FAISS. Nếu max cosine similarity < 0.65, trả về "Tôi không có thông tin này trong tài liệu."

---

### Case #2: Retrieval Miss do từ đồng nghĩa

**Symptom:** Câu hỏi "Tôi bị bệnh không thể đi làm thì làm thế nào?" → Agent không retrieve doc_002 (Leave Policy) vì thiếu từ "phép ốm".

1. **Why 1:** Keyword BM25 bỏ sót tài liệu vì người dùng dùng từ khác.
2. **Why 2:** BM25 chỉ match surface form, không hiểu ngữ nghĩa "bị bệnh" ≈ "phép ốm".
3. **Why 3:** Không có dense retrieval layer để capture semantic meaning.
4. **Why 4:** Chunking strategy chia theo full-document, làm loãng signal keyword trong đoạn ngắn quan trọng.
5. **Root Cause:** **Chunking strategy quá thô + thiếu hybrid search.** Full-document chunks làm BM25 kém hiệu quả. Cần chunking theo paragraph (256-512 tokens, overlap 50) kết hợp BM25 + dense retrieval.

**Action:** Implement `RecursiveCharacterTextSplitter` + Hybrid Search (BM25 + embedding với alpha=0.5).

---

### Case #3: Edge Case — Policy Gap

**Symptom:** "Chuyến bay 5 tiếng được hạng vé gì?" → Agent trả lời "hạng phổ thông" (sai) hoặc "hạng thương gia" (sai). Policy chỉ định nghĩa < 4h và > 6h.

1. **Why 1:** Agent chọn một trong hai hạng thay vì acknowledge khoảng trống.
2. **Why 2:** System prompt không hướng dẫn cách xử lý khi policy có khoảng trống.
3. **Why 3:** Prompting thiếu instruction "nếu thông tin không đủ, hãy nói rõ và đề xuất hỏi lại".
4. **Why 4:** Không có post-processing để detect "boundary condition" trong câu trả lời.
5. **Root Cause:** **Prompting thiếu graceful degradation instruction.** LLM có xu hướng "fill in the gap" bằng cách chọn phương án gần nhất thay vì acknowledge uncertainty.

**Action:** Thêm vào system prompt: "Nếu câu hỏi rơi vào trường hợp chưa được quy định rõ trong tài liệu, hãy nói rõ và đề nghị người dùng xác nhận với bộ phận có thẩm quyền."

---

## 4. Kế hoạch cải tiến (Action Plan)

- [ ] **Chunking:** Đổi từ full-document sang paragraph-level (chunk_size=512, overlap=50)
- [ ] **Hybrid Search:** Kết hợp BM25 + FAISS embedding (alpha=0.5)
- [ ] **Relevance Threshold:** Filter context với cosine similarity < 0.65
- [ ] **System Prompt:** Thêm graceful degradation và uncertainty acknowledgment
- [ ] **Reranking:** Thêm cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] **Selective Judging:** Chỉ chạy Multi-Judge đầy đủ cho cases có agent_score < 4 (giảm 20% cost)
- [ ] **Batch API:** Dùng OpenAI Batch API cho non-latency-critical evals (50% discount)

## 5. Đề xuất giảm 30% chi phí eval

| Chiến lược | Tiết kiệm ước tính |
|------------|-------------------|
| Selective judging (chỉ judge cases score < 4) | ~20% tổng judge cost |
| OpenAI Batch API (50% off) | ~25% OpenAI cost |
| Cache judge results cho câu hỏi trùng lặp | ~10% |
| Gemini Flash-8B làm secondary judge (đã áp dụng) | ~40% secondary cost |

**Tổng:** Giảm ~35-40% chi phí eval mà không ảnh hưởng đến độ chính xác.

---

## 6. Chi phí & Token Usage

### Chiến lược Free-Tier của hệ thống này

Toàn bộ pipeline sử dụng free-tier API — chi phí thực tế = **$0.00**:

| Component | Model | Free Limit | Actual Cost |
|-----------|-------|-----------|-------------|
| Agent (GROQ) | llama-3.3-70b-versatile | 30 RPM, 6K req/day | $0.00 |
| Judge A (Gemini) | gemini-3.1-flash-lite | 15 RPM, 1.5M TPD | $0.00 |
| Judge B (Gemini) | gemma-4-31b-it | 15 RPM, 1.5M TPD | $0.00 |
| Tiebreaker (GROQ) | llama-3.3-70b-versatile | shared with agent | $0.00 |

### Token Usage (per 50-case benchmark run, V2)

Token usage được đo trực tiếp từ GROQ API response (`resp.usage`):

| Metric | Ước tính (50 cases) |
|--------|---------------------|
| Agent prompt tokens | ~180,000 |
| Agent completion tokens | ~10,000 |
| Total agent tokens | ~190,000 |
| Avg tokens per case | ~3,800 |

### Equivalent Paid Cost (nếu dùng paid tier)

Nếu hệ thống chạy trên paid API (GROQ llama-3.3-70b: $0.59/1M input, $0.79/1M output):

| Run | Tokens | Equivalent Cost |
|-----|--------|----------------|
| V1 benchmark (50 cases) | ~190K | ~$0.115 |
| V2 benchmark (50 cases) | ~190K | ~$0.115 |
| **Tổng 1 lần chạy đầy đủ** | ~380K | **~$0.23** |

**Kết luận:** Eval pipeline này có chi phí cực thấp (~$0.23/lần chạy nếu dùng paid API), và hoàn toàn miễn phí với free-tier. Điều này phù hợp với mục tiêu "giảm 30% chi phí eval" — thực tế giảm 100% bằng cách chọn model đúng (Gemini free + GROQ free thay vì GPT-4o paid).
