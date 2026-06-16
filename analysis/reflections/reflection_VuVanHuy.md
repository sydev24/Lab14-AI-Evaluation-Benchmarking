# Reflection cá nhân - Vũ Văn Huy

**Họ tên:** Vũ Văn Huy  
**MSSV:** 2A202600750  
**Vai trò:** Thành viên 2  
**Module phụ trách:** Retrieval Evaluation - Hit Rate & MRR

## 1. Phần việc đã thực hiện

Trong Lab Day 14 - AI Evaluation Factory, tôi phụ trách phần đánh giá chất lượng retrieval của hệ thống RAG. Mục tiêu chính của module này là kiểm tra xem agent có truy xuất đúng tài liệu cần thiết trước khi sinh câu trả lời hay không.

Các phần tôi thực hiện gồm:

- Xây dựng `RetrievalEvaluator` trong `engine/retrieval_eval.py`.
- Implement metric `Hit Rate@3` để kiểm tra expected document có xuất hiện trong top-3 retrieved documents không.
- Implement metric `MRR` (Mean Reciprocal Rank) để đánh giá thứ hạng của tài liệu đúng đầu tiên.
- Implement `evaluate_batch()` để chạy đánh giá retrieval trên toàn bộ dataset và trả về các chỉ số trung bình.
- Tích hợp retrieval metrics vào `engine/runner.py` để mỗi test case đều có `hit_rate`, `mrr`, `expected_ids`, `retrieved_ids`.
- Đảm bảo `hit_rate` và `mrr` được tổng hợp vào `reports/summary.json`.
- Đóng góp phần phân tích Retrieval Miss trong `analysis/failure_analysis.md`, tập trung vào mối liên hệ giữa retrieval quality và answer quality.

## 2. Chi tiết kỹ thuật

### Hit Rate@3

Metric này trả về `1.0` nếu ít nhất một document id trong `expected_retrieval_ids` xuất hiện trong top-3 `retrieved_ids`, ngược lại trả về `0.0`.

Ý nghĩa của metric này là đo khả năng hệ thống có đưa đúng tài liệu vào context cho LLM hay không. Nếu retrieval miss, câu trả lời phía sau rất dễ bị thiếu thông tin hoặc hallucinate.

### Mean Reciprocal Rank

MRR được tính bằng công thức:

```text
MRR = 1 / rank
```

Trong đó `rank` là vị trí đầu tiên của tài liệu đúng trong danh sách retrieved documents. Nếu tài liệu đúng đứng đầu, MRR = `1.0`; nếu đứng thứ 2, MRR = `0.5`; nếu không tìm thấy, MRR = `0.0`.

Metric này giúp đánh giá không chỉ việc retrieval có đúng hay không, mà còn đánh giá tài liệu đúng được xếp hạng tốt đến mức nào.

## 3. Kết quả đạt được

Kết quả trong `reports/summary.json` cho Agent V2:

| Metric | Giá trị |
|--------|---------|
| Hit Rate | 0.9 |
| MRR | 0.8433 |
| Tổng số test cases | 50 |
| Số cases retrieve đúng | 45 |
| Số cases retrieval miss | 5 |

Kết quả này cho thấy retrieval pipeline hoạt động tương đối tốt: phần lớn câu hỏi đều lấy được tài liệu đúng trong top-3, và đa số tài liệu đúng nằm ở rank cao.

## 4. Những vấn đề phát hiện

Qua failure analysis, tôi nhận thấy một số lỗi retrieval chủ yếu đến từ:

- Keyword mismatch: người dùng hỏi bằng từ khác với từ xuất hiện trong tài liệu.
- Synonym mismatch: ví dụ "bị bệnh" và "phép ốm" có liên quan về nghĩa nhưng keyword retrieval không bắt được tốt.
- Chunking còn thô: nếu document quá dài, tín hiệu keyword quan trọng có thể bị loãng.
- Chưa có dense retrieval hoặc embedding similarity threshold để kiểm tra độ liên quan ngữ nghĩa.

Các lỗi này ảnh hưởng trực tiếp đến answer quality. Khi retrieval không lấy đúng context, LLM có thể trả lời thiếu thông tin hoặc tự suy diễn.

## 5. Bài học rút ra

Qua phần này, tôi hiểu rõ hơn rằng trong hệ thống RAG, chất lượng câu trả lời không chỉ phụ thuộc vào LLM mà phụ thuộc rất mạnh vào retrieval stage. Một câu trả lời tốt cần bắt đầu từ việc lấy đúng tài liệu, đúng thứ tự ưu tiên, và có cơ chế phát hiện khi không có context phù hợp.

Tôi cũng học được cách thiết kế metric đơn giản nhưng hữu ích cho evaluation pipeline:

- Hit Rate giúp đánh giá khả năng "có lấy đúng tài liệu không".
- MRR giúp đánh giá "tài liệu đúng được xếp hạng tốt đến đâu".
- Khi kết hợp với judge score và hallucination rate, retrieval metrics giúp giải thích nguyên nhân gốc của nhiều failure cases.

## 6. Hướng cải tiến

Nếu tiếp tục phát triển module này, tôi sẽ ưu tiên:

- Kết hợp BM25 với dense retrieval để hỗ trợ semantic search.
- Thêm FAISS hoặc Chroma làm vector store.
- Áp dụng chunking theo paragraph thay vì full-document.
- Thêm relevance threshold để agent biết khi nào nên từ chối trả lời vì không có tài liệu phù hợp.
- Bổ sung reranking để cải thiện MRR, đưa tài liệu đúng lên rank cao hơn.

## 7. Kết luận

Phần Retrieval Evaluation giúp hệ thống AI Evaluation Factory có khả năng đo lường chất lượng retrieval một cách rõ ràng và định lượng. Module này không chỉ tạo ra `hit_rate` và `mrr` trong report, mà còn giúp nhóm hiểu được vì sao một số câu trả lời thất bại và cần cải thiện retrieval pipeline ở đâu.
