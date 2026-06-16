# Individual Reflection — Lab Day 14: AI Evaluation Factory
*Sinh viên: Lê Đình Sỹ (2A202600770) | VinUni AI Engineering Bootcamp*

---

## 1. Đóng góp cá nhân (Engineering Contribution)

### Vai trò trong nhóm: Async Runner & Regression Release Gate Developer

Tôi chịu trách nhiệm thiết kế, phát triển và tối ưu hóa module chạy benchmark song song bất đồng bộ, theo dõi chi phí tài chính, và xây dựng cổng kiểm soát cập nhật tự động (Regression Release Gate).

---

### `engine/runner.py` — Async Benchmark Runner & Cost Tracking

Tôi đã phát triển toàn bộ module thực thi benchmark bất đồng bộ nhằm tối đa hóa hiệu năng nhưng vẫn đảm bảo an toàn cho hạn ngạch (quota) API:

**Async Concurrency với Semaphore (`BenchmarkRunner`):**
- Thiết kế cơ chế điều khiển lượng request đồng thời bằng `asyncio.Semaphore` (đặt mặc định là `concurrency=5` hoặc `max_concurrent=8`).
- Giúp hệ thống gửi câu hỏi đến Agent và kích hoạt các Judges chấm điểm song song, rút ngắn thời gian chạy toàn bộ 50 test cases xuống dưới 2 phút (đáp ứng tiêu chuẩn tối ưu hóa hiệu năng của giảng viên).

**Throughput & Rate Limit Optimization:**
- Thiết lập cơ chế chia lô (batching) với `batch_size=5` và dừng nghỉ giữa các đợt `asyncio.sleep(3)` để khống chế throughput ở mức khoảng ~20 cases/phút.
- Phối hợp với lớp điều phối rate limiter (`_GeminiRateLimiter`) của nhóm AI để phân bổ tải đều đặn, giải quyết triệt để lỗi nghẽn hoặc trả về mã lỗi 429 (Too Many Requests).

**Cost Tracking & Token Usage Aggregation:**
- Tích hợp bộ gom chỉ số chi phí trực tiếp trên từng test case (`agent_cost` từ metadata của Agent và `judge_cost` từ LLMJudge).
- Tính toán tổng chi phí chạy benchmark (`total_usd`) và chi phí trung bình trên mỗi lượt đánh giá (`per_eval_usd`), cho phép báo cáo trực quan trong summary report.

---

### `main.py` — Regression Gate & Report Generation

Tôi thiết lập cổng kiểm soát chất lượng tự động trước khi release, so sánh chi tiết phiên bản mới (V2) với phiên bản nền tảng (V1):

**`_regression_gate()` — Cổng kiểm soát chất lượng 5 Tiêu chuẩn:**
Tôi thiết kế hàm kiểm tra tự động xem phiên bản mới có đủ điều kiện để release không bằng cách đối chiếu 5 ngưỡng sau:
1.  **Avg Judge Score (≥ 3.5):** Điểm chất lượng trung bình của Judge phải đạt mức đạt yêu cầu trở lên.
2.  **Hit Rate (≥ 0.60):** Tỷ lệ tìm kiếm chính xác của Retrieval phải tối thiểu đạt 60%.
3.  **Agreement Rate (≥ 0.50):** Tỷ lệ đồng thuận giữa các Judge phải lớn hơn 50% để đảm bảo kết quả khách quan.
4.  **Hallucination Rate (≤ 0.10):** Tỷ lệ bịa đặt thông tin của Agent không được vượt quá 10%.
5.  **Regression Delta (≥ -0.20):** Điểm của V2 không được sụt giảm quá 0.2 điểm so với V1 để tránh lỗi hồi quy nghiêm trọng.

**Quyết định Tự động (Auto Release/Rollback):**
- Nếu bất kỳ tiêu chuẩn nào bị vi phạm, hệ thống tự động trả về trạng thái `BLOCK` kèm danh sách cụ thể các lỗi vi phạm (ví dụ: `hit_rate 0.76 < threshold 0.80`).
- Nếu tất cả vượt qua, hệ thống trả về trạng thái `APPROVE`.

**Ghi và Xuất Báo cáo:**
- Tổng hợp toàn bộ dữ liệu so sánh V1 vs V2 vào cấu trúc JSON chuẩn hóa.
- Tự động tạo và lưu trữ hai file báo cáo bắt buộc: [summary.json](file:///c:/Users/Asus/Data-interview-quesions/Lab14-AI-Evaluation-Benchmarking/reports/summary.json) và [benchmark_results.json](file:///c:/Users/Asus/Data-interview-quesions/Lab14-AI-Evaluation-Benchmarking/reports/benchmark_results.json).

---

## 2. Technical Depth

### MRR (Mean Reciprocal Rank)
MRR đánh giá chất lượng của Retrieval dựa trên vị trí xuất hiện đầu tiên của tài liệu chính xác (vị trí càng cao điểm càng lớn):
$$MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$
Khác với Hit Rate (chỉ quan tâm có tìm thấy hay không), MRR phạt nặng các hệ thống tìm kiếm đưa tài liệu đúng xuống vị trí dưới sâu (ví dụ: rank 3 chỉ được 0.33 điểm). Điều này vô cùng quan trọng trong RAG vì LLM sẽ bị loãng thông tin và tốn nhiều token khi đọc các tài liệu không liên quan ở các vị trí đầu.

### Cohen's Kappa (Độ đồng thuận Judges)
Cohen's Kappa là công cụ đo lường mức độ đồng ý giữa các Judges sau khi đã loại trừ yếu tố đồng thuận do ngẫu nhiên:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
Trong dự án này, chúng tôi áp dụng một phiên bản đơn giản hóa dựa trên khoảng cách điểm: bằng nhau thì $\kappa=1.0$, lệch 1 điểm thì $\kappa=0.5$, lệch trên 1 điểm thì $\kappa=0.0$ và kích hoạt Tie-breaker. Điều này giúp chúng tôi loại bỏ tính chủ quan khi chỉ tin vào một mô hình đơn lẻ.

### Position Bias (Thiên vị Vị trí)
LLM thường có xu hướng ưu ái phương án đứng đầu tiên trong prompt (Primacy bias) hoặc phương án cuối cùng (Recency bias). Chúng tôi phát hiện ra điều này bằng cách chạy so sánh hai lần với vị trí câu trả lời đảo ngược (`check_position_bias`). Kết quả cho thấy nếu điểm số thay đổi đáng kể khi đổi chỗ, mô hình Judge đang bị thiên vị và cần sử dụng trung bình cộng của cả hai lượt đổi chỗ để hiệu chuẩn.

### Đánh đổi Chất lượng và Chi phí (Cost vs Quality Trade-off)
Mặc dù sử dụng mô hình Gemini 2.5 Flash tốn chi phí hơn ($0.075/1M tokens) so với mô hình Llama 3.2 3B miễn phí, chất lượng câu trả lời đã cải thiện từ 1.93 lên 3.62. Tuy nhiên, việc chạy đánh giá (Evaluation) tiêu tốn lượng token gấp nhiều lần so với việc chạy Agent (vì phải gửi kèm rubrics dài và cả Ground Truth). Vì vậy, chiến lược sử dụng các mô hình nhỏ/miễn phí làm Judge sơ bộ và chỉ chuyển các case phức tạp cho mô hình lớn làm Tie-breaker giúp giảm 30% chi phí chạy evaluation thực tế.

---

## 3. Problem Solving — Các vấn đề phát sinh và hướng xử lý

### Vấn đề 1: Nghẽn và Treo kết nối SSL Socket trên Windows
- **Triệu chứng:** Pipeline benchmark sử dụng `httpx.AsyncClient` liên tục bị treo ở các case đầu tiên, không in ra logs và không hoàn thành được.
- **Nguyên nhân:** Thư viện `asyncio` mặc định của Python trên Windows (`ProactorEventLoop`) gặp xung đột với cơ chế bắt tay SSL bất đồng bộ của `httpx.AsyncClient`.
- **Giải pháp:** Chuyển đổi mã nguồn của Agent và Judges sang các cuộc gọi đồng bộ bằng `httpx.Client`, sau đó bọc ngoài bằng `asyncio.to_thread`. Cách này đẩy việc bắt tay SSL sang các luồng hệ điều hành độc lập (Thread Pool) chạy cực kỳ ổn định và song song hoàn toàn trên Windows.

### Vấn đề 2: Lỗi 402 (Không đủ tín dụng) từ OpenRouter
- **Triệu chứng:** API trả về lỗi 402 mặc dù tài khoản vẫn còn hạn ngạch miễn phí hoặc số dư nhỏ.
- **Nguyên nhân:** OpenRouter mặc định kiểm tra số dư dựa trên giới hạn token tối đa của model (lên tới 65,535 tokens). Nếu tài khoản không đủ tiền thanh toán cho 65k tokens, request sẽ bị từ chối trước khi chạy.
- **Giải pháp:** Thiết lập cứng trường `"max_tokens": 1000` trong tất cả các payload gửi đi. OpenRouter chỉ kiểm tra số dư tối thiểu cho 1000 tokens (chưa tới $0.0001) nên cuộc gọi thành công ngay lập tức.

---

## 4. Bài học kinh nghiệm

1.  **Thiết kế bất đồng bộ phải tính đến giới hạn vật lý:** Dù viết code async có thể gửi hàng trăm request cùng lúc, hạn ngạch API (RPM/TPM) luôn là nút thắt cổ chai. Việc sử dụng Semaphore kết hợp Sleep-delay là bắt buộc để duy trì hệ thống chạy ổn định.
2.  **Concurrency an toàn trên Windows:** Không nên lạm dụng thư viện async thuần cho các kết nối SSL trên Windows. Kỹ thuật đưa các luồng I/O đồng bộ vào `asyncio.to_thread` là một giải pháp thay thế rất thực tế và mạnh mẽ.
3.  **Hồi quy chất lượng (Regression) khó nhận biết nếu không có Eval Factory:** Bản V2 mặc dù cải thiện rõ rệt về điểm số trung bình, nhưng nếu không có Regression Gate tự động phát hiện Hit Rate giảm dưới 80%, chúng tôi đã có thể tự tin release một phiên bản có module tìm kiếm không đạt yêu cầu.
