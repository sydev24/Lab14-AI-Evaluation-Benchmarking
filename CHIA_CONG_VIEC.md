# Phân công công việc — Lab Day 14: AI Evaluation Factory
*Nhóm 5 thành viên | VinUni AI Engineering Bootcamp*

---

## Thành viên

| Tên | MSSV |
|-----|------|
| Vũ Quốc Bảo | 2A202600541 |
| Vũ Văn Huy | 2A202600750 |
| Nguyễn Trung Kiên | 2A202600969 |
| Lê Đình Sỹ | 2A202600770 |
| Phạm Hoàng Anh Kiệt | 2A202600797 |

---

## Phân công theo module

---

### Thành viên 1 — Vũ Quốc Bảo
**Module: Dataset & Synthetic Data Generation (SDG)**

| Nhiệm vụ | File | Mô tả |
|----------|------|-------|
| Thiết kế Golden Dataset | `data/synthetic_gen.py` | Viết 50+ test cases chất lượng cao với Ground Truth IDs |
| Red Teaming Cases | `data/synthetic_gen.py` | Thiết kế adversarial, OOD, policy-gap, multi-turn cases |
| LLM-based SDG | `data/synthetic_gen.py` | Implement `_generate_cases_async()` dùng Gemini để tổng hợp thêm cases |
| HARD_CASES_GUIDE | `data/HARD_CASES_GUIDE.md` | Đọc và implement đúng 4 loại hard case: adversarial, edge, multi-turn, stress |
| Kiểm tra phân phối | `data/golden_set.jsonl` | Đảm bảo dataset đủ đa dạng: easy/medium/hard, factual/procedural/adversarial/multi_turn |

**Deliverable:** File `data/golden_set.jsonl` với 60+ cases, phân phối cân bằng, sẵn sàng cho benchmark.

---

### Thành viên 2 — Vũ Văn Huy
**Module: Retrieval Evaluation (Hit Rate & MRR)**

| Nhiệm vụ | File | Mô tả |
|----------|------|-------|
| Implement Hit Rate | `engine/retrieval_eval.py` | Tính Hit Rate@3: 1.0 nếu expected_id có trong top-3 retrieved |
| Implement MRR | `engine/retrieval_eval.py` | Tính Mean Reciprocal Rank: 1/rank của relevant doc đầu tiên |
| Batch evaluation | `engine/retrieval_eval.py` | `evaluate_batch()` chạy toàn bộ dataset và trả về avg metrics |
| Tích hợp vào runner | `engine/runner.py` | Đảm bảo mỗi test case đều có `hit_rate` và `mrr` trong kết quả |
| Phân tích Retrieval Quality | `analysis/failure_analysis.md` | Giải thích mối liên hệ Retrieval Quality → Answer Quality (phần "Retrieval Miss") |

**Deliverable:** `retrieval_eval.py` hoạt động đúng, metrics xuất hiện trong `reports/summary.json` với `hit_rate` và `mrr`.

---

### Thành viên 3 — Nguyễn Trung Kiên
**Module: Multi-Judge Consensus Engine**

| Nhiệm vụ | File | Mô tả |
|----------|------|-------|
| Judge A (Gemini flash-lite) | `engine/llm_judge.py` | Implement `_judge_gemini()` với rubric 1-5, JSON output |
| Judge B (Gemma 4-31b) | `engine/llm_judge.py` | Implement `_judge_gemma()` — separate Gemini RPM bucket |
| Cohen's Kappa | `engine/llm_judge.py` | Implement `_cohen_kappa_simplified()`: agreement rate giữa 2 judges |
| Conflict resolution | `engine/llm_judge.py` | Tiebreaker bằng GROQ llama-3.3-70b khi |score_A - score_B| > 1 |
| Position Bias detection | `engine/llm_judge.py` | Implement `check_position_bias()`: swap A/B order, compare scores |
| Rate limiter | `engine/llm_judge.py` | `_GeminiRateLimiter` class (15 RPM) cho từng model bucket |

**Deliverable:** `llm_judge.py` với multi-judge + kappa + tiebreaker, `agreement_rate` xuất hiện trong `reports/summary.json`.

---

### Thành viên 4 — Lê Đình Sỹ
**Module: Async Runner & Regression Release Gate**

| Nhiệm vụ | File | Mô tả |
|----------|------|-------|
| Async Benchmark Runner | `engine/runner.py` | `BenchmarkRunner` với `asyncio.Semaphore`, batch parallel execution |
| Throughput optimization | `engine/runner.py` | `batch_size=5`, `sleep=3s`, đảm bảo pipeline < 2 phút cho 50 cases |
| Cost tracking | `engine/runner.py` | Aggregate cost per case (agent + judge), tính total & per-eval cost |
| Regression Gate logic | `main.py` | `_regression_gate()`: so sánh V1 vs V2, check 5 thresholds |
| Auto Release/Rollback | `main.py` | Quyết định APPROVE/BLOCK, in lý do từng threshold bị vi phạm |
| Report generation | `main.py` | Lưu `reports/summary.json` và `reports/benchmark_results.json` |

**Deliverable:** Pipeline chạy end-to-end, `reports/summary.json` có đầy đủ `regression` block với `decision` APPROVE/BLOCK.

---

### Thành viên 5 — Phạm Hoàng Anh Kiệt
**Module: RAG Agent (V1/V2) + Integration Lead + Failure Analysis**

| Nhiệm vụ | File | Mô tả |
|----------|------|-------|
| RAG Agent V1 | `agent/main_agent.py` | Baseline agent: keyword retrieval + minimal system prompt |
| RAG Agent V2 | `agent/main_agent.py` | Optimized agent: hallucination guardrails, graceful degradation |
| Token tracking | `agent/main_agent.py` | Đọc `resp.usage` từ GROQ, tính `equiv_paid_cost_usd` |
| Keyword retrieval | `agent/main_agent.py` | `_keyword_retrieve()`: BM25-style scoring trên corpus |
| Failure Analysis | `analysis/failure_analysis.md` | 5 Whys cho 3 failure cases, Action Plan, Cost strategy |
| Integration & debugging | `main.py` | Đảm bảo toàn bộ pipeline hoạt động, fix rate limit conflicts |
| Individual reflection | `analysis/reflections/reflection_minionphak.md` | Reflection cá nhân về đóng góp kỹ thuật |

**Deliverable:** Agent V1 vs V2 chạy được, `failure_analysis.md` đầy đủ, pipeline tích hợp thành công.

---

## Lịch trình thực hiện (4 tiếng)

| Giai đoạn | Thời gian | Nội dung | Người thực hiện |
|-----------|-----------|----------|----------------|
| Giai đoạn 1 | 0–45 phút | Thiết kế corpus + viết golden dataset 50+ cases | **Vũ Quốc Bảo** (lead) + cả nhóm review |
| Giai đoạn 2a | 45–90 phút | Implement Retrieval Eval (Hit Rate, MRR) | **Vũ Văn Huy** |
| Giai đoạn 2b | 45–90 phút | Implement Multi-Judge Engine | **Nguyễn Trung Kiên** |
| Giai đoạn 2c | 45–90 phút | Build RAG Agent V1/V2 | **Phạm Hoàng Anh Kiệt** |
| Giai đoạn 2d | 90–135 phút | Async Runner + Regression Gate | **Lê Đình Sỹ** |
| Giai đoạn 3 | 135–195 phút | Integration, debug, chạy benchmark | **Phạm Hoàng Anh Kiệt** (lead) + cả nhóm |
| Giai đoạn 4 | 195–240 phút | Failure Analysis, reflections, `check_lab.py` | Cả nhóm |

---

## Checklist nộp bài

- [ ] `data/golden_set.jsonl` — Vũ Quốc Bảo
- [ ] `reports/summary.json` — Lê Đình Sỹ
- [ ] `reports/benchmark_results.json` — Lê Đình Sỹ
- [ ] `analysis/failure_analysis.md` — Phạm Hoàng Anh Kiệt
- [ ] `analysis/reflections/reflection_VuQuocBao.md` — Vũ Quốc Bảo
- [ ] `analysis/reflections/reflection_VuVanHuy.md` — Vũ Văn Huy
- [ ] `analysis/reflections/reflection_NguyenTrungKien.md` — Nguyễn Trung Kiên
- [ ] `analysis/reflections/reflection_LeDinhSy.md` — Lê Đình Sỹ
- [ ] `analysis/reflections/reflection_minionphak.md` — Phạm Hoàng Anh Kiệt
- [ ] Chạy `python check_lab.py` — pass ✅
- [ ] Không commit `.env` hoặc API keys
