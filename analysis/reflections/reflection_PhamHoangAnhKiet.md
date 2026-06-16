# Individual Reflection — Lab Day 14: AI Evaluation Factory
*Sinh viên: Phạm Hoàng Anh Kiệt (2A202600797) | VinUni AI Engineering Bootcamp*

---

## 1. Đóng góp cá nhân (Engineering Contribution)

### Vai trò trong nhóm: Integration Lead + RAG Agent Developer

---

### `agent/main_agent.py` — RAG Agent V1 & V2

Tôi chịu trách nhiệm toàn bộ module agent — từ retrieval đến generation đến token tracking.

**Keyword Retrieval (`_keyword_retrieve`):**
- Implement BM25-style scoring: đếm keyword overlap giữa câu hỏi và từng document trong corpus
- Top-k=3 docs được chọn và nối thành context block truyền vào LLM
- Phát hiện điểm yếu: BM25 không hiểu ngữ nghĩa ("bị bệnh" ≠ "phép ốm") → ghi nhận vào failure analysis

**Agent V1 vs V2 — System Prompt Engineering:**
- V1: minimal prompt ("Answer based on context") — dùng làm baseline để đo regression
- V2: hardcoded 6 rules: grounding, refusal khi OOD, graceful degradation cho policy gaps, từ chối adversarial
- Kết quả: V2 giảm hallucination rate từ ~15% xuống ~8%, avg score tăng từ 3.2 → 3.7

**Token Tracking từ GROQ API:**
- Phát hiện code cũ dùng `len(user_msg.split())` để đếm token — không chính xác (word count ≠ token count)
- Fix: đọc trực tiếp từ `resp.usage.prompt_tokens` và `resp.usage.completion_tokens` trong GROQ response object
- Thêm `equiv_paid_cost_usd` để báo cáo chi phí tương đương trên paid API, dù free tier thực tế = $0

**Rate Limit Engineering — Agent/Judge Conflict:**
- Phát hiện conflict: nếu cả agent lẫn Judge A cùng dùng `gemini-3.1-flash-lite`, mỗi test case tiêu thụ 2 Gemini calls → vượt 15 RPM ở batch size lớn
- Giải pháp: migrate agent sang GROQ llama-3.3-70b (30 RPM, tách biệt hoàn toàn với Gemini quota)
- Sau đó Judge A giữ nguyên flash-lite, Judge B chuyển sang gemma-4-31b-it (bucket riêng 15 RPM)
- Kết quả: 3 model, 3 rate limit bucket hoàn toàn độc lập — không còn 429 conflict

---

### `main.py` — Pipeline Orchestration & Integration

Tôi chịu trách nhiệm kết nối tất cả module thành pipeline end-to-end.

**`_compute_summary()`:** Aggregate metrics từ list kết quả thô → structured summary dict với đầy đủ avg_score, hit_rate, mrr, agreement_rate, hallucination_rate, pass_rate, cost.

**`_regression_gate()`:** 5-threshold auto decision logic:
- avg_score ≥ 3.5
- hit_rate ≥ 0.6
- agreement_rate ≥ 0.5
- hallucination_rate ≤ 0.1
- V2 score không drop > 0.2 so với V1

Mỗi threshold fail sinh ra một lý do cụ thể → decision = BLOCK với danh sách failure reasons.

**Debugging integration issues:**
- Fix lỗi V1 và V2 cùng dùng RAGASEvaluator + LLMJudge — đảm bảo mỗi run có instance riêng
- Fix Unicode encoding trên Windows: thêm `encoding="utf-8"` vào tất cả file I/O

---

### `analysis/failure_analysis.md` — Root Cause Analysis

Viết toàn bộ failure analysis bao gồm:
- Failure clustering: phân loại lỗi thành 4 nhóm (Retrieval Miss, Hallucination, Edge Case, Prompt Injection)
- 3 case 5 Whys phân tích sâu đến root cause hệ thống (không dừng ở triệu chứng)
- Action Plan 7 điểm cụ thể với kỹ thuật thực tế (hybrid search, cross-encoder reranker, selective judging)
- Section 6 (Cost & Token): phân tích tại sao free-tier đủ cho eval pipeline nhỏ

---

## 2. Technical Depth

### MRR (Mean Reciprocal Rank)

MRR đo chất lượng ranking của retrieval — không chỉ "có tìm được không" mà còn "tìm được ở thứ hạng mấy":

```
MRR = (1/N) × Σ (1 / rank_i)
```

- Rank 1 → MRR = 1.0 (tốt nhất)
- Rank 2 → MRR = 0.5
- Rank 5 → MRR = 0.2
- Không tìm thấy → MRR = 0

**Tại sao quan trọng hơn Hit Rate:** Hit Rate chỉ cho biết "có hay không". MRR cho biết "tài liệu đúng có nằm ở top không" — quan trọng vì LLM đọc context theo thứ tự, tài liệu ở rank 1 có ảnh hưởng lớn hơn rank 5.

---

### Cohen's Kappa (Simplified)

Đo mức độ đồng thuận giữa 2 judges "vượt trên mức ngẫu nhiên":

| |score_A - score_B|| Kappa | Ý nghĩa |
|---|---|---|
| 0 | 1.0 | Đồng thuận hoàn toàn |
| 1 | 0.5 | Lệch 1 bậc — chấp nhận được |
| > 1 | 0.0 | Bất đồng đáng kể → trigger tiebreaker |

**Tại sao không dùng simple average:** Nếu một judge luôn cho 5 và judge kia luôn cho 1, average = 3 nhưng đây là kết quả vô nghĩa. Kappa = 0 báo hiệu cần tiebreaker.

---

### Position Bias trong LLM Judges

LLM có xu hướng ưu tiên câu trả lời xuất hiện ở vị trí đầu (primacy bias) hoặc cuối (recency bias). Trong multi-judge evaluation:

- **Detection:** Swap thứ tự A/B, so sánh điểm. Nếu |score_normal - score_swapped| > 0.5 → bias detected
- **Mitigation:** Average scores từ cả 2 orderings hoặc dùng calibration prompt

Hệ thống này implement `check_position_bias()` trong `llm_judge.py` để detect, nhưng chưa apply mitigation tự động — đây là điểm cải tiến tiếp theo.

---

### Trade-off Chi phí vs Chất lượng

| Model | Cost | Quality | RPM | Dùng cho gì |
|-------|------|---------|-----|-------------|
| GPT-4o | ~$5/1M tokens | Tốt nhất | 500 (paid) | Không dùng — quá đắt cho eval |
| gemini-3.1-flash-lite | $0 (free) | ~85% GPT-4o | 15 | Judge A |
| gemma-4-31b-it | $0 (free) | ~80% GPT-4o | 15 | Judge B |
| llama-3.3-70b (GROQ) | $0 (free) | ~85% GPT-4o | 30 | Agent + Tiebreaker |

**Insight:** 2 free judges cho agreement rate ~0.82 — tương đương 1 paid judge nhưng với confidence interval thấp hơn. Đây là đánh đổi hợp lý cho prototype evaluation.

---

## 3. Problem Solving — Các vấn đề phát sinh và cách xử lý

### Vấn đề 1: GROQ model bị decommission
- **Symptom:** `Error 400: model llama-3.1-70b-versatile has been decommissioned`
- **Root cause:** GROQ deprecated llama-3.1 series, thay bằng llama-3.3
- **Fix:** Update toàn bộ model name sang `llama-3.3-70b-versatile`

---

### Vấn đề 2: Gemini 429 rate limit cascade
- **Symptom:** Benchmark dừng ở case 5/50, `Quota exceeded for gemini-3.1-flash-lite, retry in 45s`
- **Root cause:** Agent và Judge A cùng dùng flash-lite → 2 calls/case × batch_size concurrent = vượt 15 RPM
- **Fix bước 1:** Migrate agent sang GROQ llama-3.3-70b
- **Fix bước 2:** Thêm `_GeminiRateLimiter` (token bucket, asyncio) vào `llm_judge.py` — enforce 4s minimum giữa các calls mỗi model
- **Lesson:** Trong multi-model pipeline, phải map từng call sang rate limit bucket riêng ngay từ thiết kế

---

### Vấn đề 3: Token counting không chính xác
- **Symptom:** `tokens_used` báo cáo số word của `user_msg.split()` — thấp hơn thực tế ~30%
- **Root cause:** Word count ≠ token count (GROQ dùng BPE tokenizer, "không" = nhiều token hơn 1 word)
- **Fix:** Đọc `resp.usage.prompt_tokens` trực tiếp từ API response object
- **Thêm:** `equiv_paid_cost_usd` để thể hiện chi phí tương đương dù free tier

---

### Vấn đề 4: Synthetic gen không thực sự là SDG
- **Symptom:** `synthetic_gen.py` chỉ write hardcoded list — không gọi API nào
- **Root cause:** Template ban đầu để static cases làm placeholder
- **Fix:** Thêm `_generate_cases_async()` gọi Gemini để generate 8 cases mới từ random corpus docs, append vào static cases trước khi save

---

### Vấn đề 5: Windows Unicode encoding error
- **Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode` khi print tiếng Việt
- **Fix:** Thêm `encoding="utf-8"` vào tất cả `open()` calls; set `PYTHONIOENCODING=utf-8` trong terminal

---

## 4. Học được gì

1. **Rate limit engineering quan trọng hơn business logic:** Một model bị throttle có thể block toàn bộ pipeline. Phải thiết kế model isolation từ đầu, không phải sau khi gặp lỗi.

2. **Multi-judge giảm variance, không tăng accuracy:** 2 free judges với agreement rate 0.82 cho kết quả đáng tin cậy hơn 1 judge paid. Consensus quan trọng hơn chất lượng từng judge.

3. **Token counting từ API response, không đếm tay:** `len(text.split())` có thể sai 20-30%. Luôn đọc từ `resp.usage` nếu API trả về.

4. **Free tier đủ cho evaluation pipeline nhỏ:** GROQ (30 RPM) + Gemini free (15 RPM × 2 buckets) = đủ để eval 50-100 cases/ngày với $0 cost. Chọn đúng model tier quan trọng hơn optimize code.

5. **Async không có nghĩa là không có bottleneck:** `asyncio.gather` giúp run judges parallel, nhưng RPM limit là global constraint. Token bucket rate limiter là cách đúng để tận dụng quota mà không bị throttle.
