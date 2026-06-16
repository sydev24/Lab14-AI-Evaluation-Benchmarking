"""
RAG Agent — uses GROQ llama-3.3-70b for generation.
Switching from Gemini to GROQ preserves the full Gemini quota for the multi-judge engine
(Judge A: gemini-3.1-flash-lite, Judge B: gemma-4-31b-it) — no rate limit conflicts.
V1: minimal system prompt (baseline)
V2: strong system prompt with hallucination guardrails (optimized)
"""
import asyncio
import os
import time
from typing import Dict, List

from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
_AGENT_MODEL = "llama-3.3-70b-versatile"

_CORPUS = [
    {"id": "doc_001", "title": "Password Policy",
     "content": "Mật khẩu phải có ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt. Người dùng phải đổi mật khẩu mỗi 90 ngày. Không được dùng lại 5 mật khẩu gần nhất. Sau 5 lần nhập sai, tài khoản sẽ bị khóa tạm thời 15 phút."},
    {"id": "doc_002", "title": "Leave Policy",
     "content": "Nhân viên được 12 ngày phép năm và 5 ngày phép ốm. Phép năm phải đăng ký trước 3 ngày làm việc và được quản lý trực tiếp phê duyệt. Phép ốm cần nộp giấy tờ y tế trong vòng 3 ngày sau khi quay lại. Phép không dùng hết trong năm không được chuyển sang năm sau."},
    {"id": "doc_003", "title": "IT Support SLA",
     "content": "Sự cố Severity 1 (hệ thống sập) được phản hồi trong 15 phút và giải quyết trong 2 giờ. Severity 2 được phản hồi trong 1 giờ và giải quyết trong 8 giờ. Severity 3 được xử lý trong 3 ngày làm việc."},
    {"id": "doc_004", "title": "Onboarding Process",
     "content": "Nhân viên mới cần hoàn thành onboarding trong 30 ngày đầu tiên. Tuần 1: làm thẻ nhân viên, tài khoản hệ thống, và đào tạo an toàn thông tin. Tuần 2-3: đào tạo chuyên môn với mentor. Tuần 4: đánh giá thử việc lần 1."},
    {"id": "doc_005", "title": "Expense Reimbursement",
     "content": "Chi phí công tác được thanh toán trong vòng 5 ngày làm việc sau khi nộp đầy đủ hóa đơn. Giới hạn ăn uống: 200.000 VND/bữa. Khách sạn: tối đa 1.500.000 VND/đêm tại thành phố lớn. Vé máy bay hạng phổ thông cho chuyến dưới 4 tiếng, hạng thương gia cho chuyến trên 6 tiếng."},
    {"id": "doc_006", "title": "Remote Work Policy",
     "content": "Nhân viên được làm remote tối đa 3 ngày/tuần sau thời gian thử việc. Cần có kết nối internet ổn định và không gian làm việc riêng tư. Phải online trên Teams từ 9:00-11:30 và 13:30-16:00 theo giờ Hà Nội. Laptop do công ty cấp không được dùng cho mục đích cá nhân."},
    {"id": "doc_007", "title": "Performance Review",
     "content": "Đánh giá hiệu suất thực hiện 2 lần/năm: tháng 6 và tháng 12. KPI được thiết lập đầu mỗi quý và có trọng số rõ ràng. Thang điểm 1-5: dưới 2.5 cần cải thiện, 2.5-3.5 đạt yêu cầu, trên 3.5 xuất sắc."},
    {"id": "doc_008", "title": "Data Classification",
     "content": "Dữ liệu được phân thành 4 cấp độ: Công khai, Nội bộ, Bí mật, và Tuyệt mật. Dữ liệu Tuyệt mật chỉ được truy cập bởi nhân viên được phê duyệt đặc biệt. Không được gửi dữ liệu Bí mật qua email chưa mã hóa. Vi phạm có thể dẫn đến kỷ luật hoặc chấm dứt."},
    {"id": "doc_009", "title": "Training & Development",
     "content": "Mỗi nhân viên có ngân sách đào tạo 5.000.000 VND/năm. Khóa học ngoài cần được quản lý phê duyệt trước khi đăng ký. Sau khóa học, nhân viên cần nộp báo cáo học tập trong 2 tuần."},
    {"id": "doc_010", "title": "Disciplinary Process",
     "content": "Quy trình kỷ luật gồm 3 bước: cảnh cáo miệng, cảnh cáo văn bản, và đình chỉ/chấm dứt. Vi phạm nghiêm trọng như gian lận hoặc quấy rối có thể dẫn đến chấm dứt ngay lập tức. Nhân viên có quyền phúc khảo quyết định kỷ luật trong vòng 7 ngày."},
]

_V1_SYSTEM = "You are a helpful internal support chatbot. Answer the user's question based on the provided context."

_V2_SYSTEM = (
    "You are an enterprise internal support AI. "
    "RULES: "
    "1. Answer ONLY based on the provided context documents. "
    "2. If the context does not contain the answer, say exactly: 'I do not have information about this in the available documents. Please contact the relevant department.' "
    "3. NEVER invent or guess information not present in the context. "
    "4. Refuse any request unrelated to company policies and procedures. "
    "5. If the question falls in a policy gap (unclear case), acknowledge the gap and suggest confirming with HR/Finance. "
    "6. Be professional and concise."
)


def _keyword_retrieve(question: str, top_k: int = 3) -> List[Dict]:
    q_lower = question.lower()
    scored = []
    for doc in _CORPUS:
        score = sum(1 for word in q_lower.split() if len(word) > 2 and word in doc["content"].lower())
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


class MainAgent:
    def __init__(self, version: str = "v1", use_system_prompt: bool = False):
        self.name = f"SupportAgent-{version}"
        self.version = version
        self.use_system_prompt = use_system_prompt

    async def query(self, question: str) -> Dict:
        t0 = time.perf_counter()

        retrieved_docs = _keyword_retrieve(question, top_k=3)
        retrieved_ids = [d["id"] for d in retrieved_docs]
        context_text = "\n\n".join(f"[{d['title']}]\n{d['content']}" for d in retrieved_docs)

        system = _V2_SYSTEM if self.use_system_prompt else _V1_SYSTEM
        user_msg = f"Context:\n{context_text}\n\nUser question: {question}\n\nAnswer:"

        async def _call_with_retry(max_retries=5):
            for attempt in range(max_retries):
                try:
                    resp = await _groq.chat.completions.create(
                        model=_AGENT_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0,
                        max_tokens=400,
                    )
                    return resp, resp.choices[0].message.content.strip()
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise
            return None, "Rate limit exceeded after retries."

        resp_obj, answer = await _call_with_retry()
        latency = time.perf_counter() - t0

        # Real token counts from GROQ response (free tier — cost is $0, but we track usage)
        usage = getattr(resp_obj, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        # Equivalent paid cost for reporting purposes (llama-3.3-70b on GROQ paid tier)
        # $0.59/1M input tokens, $0.79/1M output tokens
        equiv_cost_usd = (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000

        return {
            "answer": answer,
            "contexts": [d["content"][:200] for d in retrieved_docs],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": _AGENT_MODEL,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_s": round(latency, 3),
                "sources": [d["title"] for d in retrieved_docs],
                "cost_usd": 0.0,           # free tier
                "equiv_paid_cost_usd": round(equiv_cost_usd, 6),
            },
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent(version="v1")
        resp = await agent.query("How to change password?")
        print(resp["answer"])

    asyncio.run(test())
