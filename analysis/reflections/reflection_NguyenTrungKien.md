# Reflection cá nhân - Nguyễn Trung Kiên

**Họ tên:** Nguyễn Trung Kiên  
**MSSV:** 2A202600696  
**Vai trò:** Thành viên 3  
**Module phụ trách:** Multi-Judge Consensus Engine

## 1. Phần việc đã thực hiện

Trong Lab Day 14 - AI Evaluation Factory, tôi phụ trách xây dựng cơ chế đánh giá đa mô hình (Multi-Judge Consensus Engine) trong file `engine/llm_judge.py`. Mục tiêu là tạo ra một bộ chấm điểm tự động khách quan, ổn định và có khả năng giải quyết xung đột khi các mô hình chấm điểm khác nhau.

Các phần tôi thực hiện gồm:
- Implement **Judge A** sử dụng `gemini-3.1-flash-lite` với thang điểm rubric 1-5 và đầu ra chuẩn JSON.
- Implement **Judge B** sử dụng `gemma-4-31b-it` qua Gemini API, tách biệt bucket giới hạn tốc độ (RPM).
- Implement **Cohen's Kappa Simplified** để tính toán tỷ lệ đồng thuận (`agreement_rate`) giữa hai Judge.
- Xây dựng cơ chế **Conflict Resolution (Tiebreaker)** bằng `llama-3.3-70b-versatile` qua GROQ khi điểm số chênh lệch lớn hơn 1 (|score_A - score_B| > 1).
- Implement **Position Bias Detection** (`check_position_bias()`) để kiểm tra thiên kiến vị trí bằng cách hoán đổi thứ tự câu trả lời A/B.
- Thiết kế **Rate Limiter** (`_GeminiRateLimiter`) để kiểm soát số lượng request (15 RPM) cho từng model bucket riêng biệt, tránh bị lỗi 429 Too Many Requests.

## 2. Chi tiết kỹ thuật

### Hệ thống Multi-Judge và Rate Limiting
Do giới hạn API của phiên bản miễn phí (15 RPM), tôi đã áp dụng thuật toán token-bucket (`_GeminiRateLimiter`) kết hợp `asyncio.Lock` để đảm bảo khoảng thời gian tối thiểu giữa các requests. Tôi chia Judge A và Judge B thành hai bucket riêng, giúp tối đa hóa throughput bằng cách chạy song song mà không vi phạm hạn mức của từng mô hình.

### Cohen's Kappa Simplified
Tỷ lệ đồng thuận được tính đơn giản hóa:
- Chênh lệch = 0: Đồng thuận hoàn toàn (Kappa = 1.0)
- Chênh lệch = 1: Đồng thuận một phần (Kappa = 0.5)
- Chênh lệch > 1: Xung đột (Kappa = 0.0)

### Conflict Resolution
Khi xảy ra xung đột (chênh lệch > 1), hệ thống gọi đến một mô hình thứ ba độc lập (`llama-3.3-70b` qua GROQ) để làm trọng tài (Tiebreaker). Quyết định dùng GROQ cho tiebreaker là để tiết kiệm quota của Gemini cho các luồng chấm chính, đảm bảo hệ thống không bị nghẽn.

## 3. Kết quả đạt được

Hệ thống Multi-Judge đã hoạt động ổn định trong đường ống đánh giá. Tỷ lệ đồng thuận (Agreement Rate) tổng thể của các test cases đạt mức khá cao (~82.0%), cho thấy prompt rubric được thiết kế tốt và các mô hình có độ tin cậy nhất định. Các trường hợp xung đột đã được Tiebreaker xử lý thành công mà không gây gián đoạn pipeline. Kết quả điểm trung bình và tỷ lệ đồng thuận đã được xuất đúng vào file `reports/summary.json`.

## 4. Những vấn đề phát hiện

Trong quá trình phát triển, tôi nhận thấy:
- **Rate Limit là nút thắt cổ chai lớn nhất:** Nếu không có `asyncio` và custom rate limiter, API sẽ liên tục báo lỗi 429. Việc chia bucket đã giải quyết triệt để vấn đề này.
- **Position Bias:** Trong một số trường hợp, mô hình có xu hướng thiên vị câu trả lời được đưa ra đầu tiên (A) bất kể chất lượng thực tế. Tính năng `check_position_bias()` giúp định lượng được vấn đề này, mặc dù cần chạy nhiều lần hơn để có dữ liệu chính xác.
- **Parse JSON:** Mô hình đôi khi sinh ra markdown có chứa code block (```json) khiến `json.loads` bị lỗi. Hàm `_extract_json` với regex đã giúp bóc tách JSON an toàn hơn.

## 5. Bài học rút ra

Qua phần việc này, tôi học được cách xử lý các bài toán async/await phức tạp trong Python, đặc biệt là cách quản lý concurrency với các third-party API có giới hạn nghiêm ngặt. Tôi cũng hiểu rõ hơn về các khái niệm đánh giá mô hình LLM-as-a-Judge, từ việc xây dựng Rubric đến việc tính toán độ đồng thuận (Cohen's Kappa).

## 6. Hướng cải tiến

Nếu có thêm thời gian, tôi sẽ:
- Triển khai Exponential Backoff linh hoạt hơn trong quá trình retry thay vì chờ cố định.
- Lưu lại toàn bộ log của Tiebreaker để phân tích sâu hơn lý do tại sao hai Judge ban đầu lại bất đồng quan điểm.
- Hỗ trợ nhiều format đầu ra thay vì chỉ ép kiểu JSON, ví dụ sử dụng thư viện Structured Output của LLM (như `instructor` hoặc `pydantic`).

## 7. Kết luận

Module Multi-Judge Consensus Engine là một phần quan trọng để đảm bảo tính khách quan của hệ thống AI Evaluation Factory. Việc kết hợp nhiều mô hình và có cơ chế xử lý xung đột giúp kết quả đánh giá (Benchmarking) trở nên đáng tin cậy hơn, đáp ứng yêu cầu của bài lab.
