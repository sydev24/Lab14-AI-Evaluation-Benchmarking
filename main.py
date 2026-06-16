"""
AI Evaluation Factory — Main Pipeline
Phases:
  1. Load golden dataset
  2. Benchmark V1 (base agent)
  3. Benchmark V2 (optimized agent with better system prompt)
  4. Regression Release Gate (delta analysis)
  5. Save reports
"""
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from agent.main_agent import MainAgent
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator


# ── Release gate thresholds ───────────────────────────────────────────────────
GATE_MIN_AVG_SCORE = 3.5        # avg judge score must be ≥ 3.5
GATE_MIN_HIT_RATE = 0.6         # retrieval hit rate must be ≥ 60%
GATE_MIN_AGREEMENT = 0.5        # judge agreement rate must be ≥ 50%
GATE_MAX_HALLUCINATION_RATE = 0.1  # hallucination in ≤ 10% of cases
GATE_MAX_REGRESSION_DELTA = -0.2   # V2 score must not drop > 0.2 vs V1


class RAGASEvaluator:
    """Wraps RetrievalEvaluator to produce RAGAS-style faithfulness/relevancy scores."""

    def __init__(self):
        self._retrieval = RetrievalEvaluator()

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        hit = self._retrieval.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self._retrieval.calculate_mrr(expected_ids, retrieved_ids)

        # Faithfulness heuristic: 1.0 if retrieved the right doc, 0.5 otherwise
        faithfulness = 0.9 if hit == 1.0 else 0.5
        # Relevancy heuristic: based on MRR rank quality
        relevancy = 0.7 + 0.3 * mrr

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
            "retrieval": {"hit_rate": hit, "mrr": mrr},
        }


def _compute_summary(results: List[Dict], version: str) -> Dict:
    total = len(results)
    if total == 0:
        return {}

    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    hit_rate = sum(r["retrieval"]["hit_rate"] for r in results) / total
    mrr = sum(r["retrieval"]["mrr"] for r in results) / total
    agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    hallucination_rate = sum(1 for r in results if r.get("hallucination")) / total
    avg_latency = sum(r["latency"] for r in results) / total
    total_cost = sum(r["cost"]["total_usd"] for r in results)
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total

    return {
        "metadata": {
            "version": version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "agreement_rate": round(agreement, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "pass_rate": round(pass_rate, 4),
            "avg_latency_s": round(avg_latency, 3),
        },
        "cost": {
            "total_usd": round(total_cost, 4),
            "per_eval_usd": round(total_cost / total, 5),
        },
    }


def _regression_gate(v1_summary: Dict, v2_summary: Dict, v2_results: List[Dict]) -> Tuple[str, List[str]]:
    """
    Auto Release Gate.
    Returns ("APPROVE" | "BLOCK", [list of failure reasons]).
    """
    m = v2_summary["metrics"]
    v1_score = v1_summary["metrics"]["avg_score"]
    v2_score = m["avg_score"]
    delta = v2_score - v1_score

    failures = []

    if m["avg_score"] < GATE_MIN_AVG_SCORE:
        failures.append(f"avg_score {m['avg_score']:.2f} < threshold {GATE_MIN_AVG_SCORE}")

    if m["hit_rate"] < GATE_MIN_HIT_RATE:
        failures.append(f"hit_rate {m['hit_rate']:.2f} < threshold {GATE_MIN_HIT_RATE}")

    if m["agreement_rate"] < GATE_MIN_AGREEMENT:
        failures.append(f"agreement_rate {m['agreement_rate']:.2f} < threshold {GATE_MIN_AGREEMENT}")

    if m["hallucination_rate"] > GATE_MAX_HALLUCINATION_RATE:
        failures.append(f"hallucination_rate {m['hallucination_rate']:.2f} > threshold {GATE_MAX_HALLUCINATION_RATE}")

    if delta < GATE_MAX_REGRESSION_DELTA:
        failures.append(f"regression: V2 score dropped {delta:.2f} vs V1")

    decision = "BLOCK" if failures else "APPROVE"
    return decision, failures


async def _run_benchmark(version: str, agent, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
    print(f"\n{'='*50}")
    print(f"🔬 Benchmarking {version}...")
    evaluator = RAGASEvaluator()
    judge = LLMJudge()
    runner = BenchmarkRunner(agent, evaluator, judge, concurrency=5)
    results = await runner.run_all(dataset)
    summary = _compute_summary(results, version)
    return results, summary


async def main():
    print("🚀 AI Evaluation Factory — khởi động\n")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Chạy: python data/synthetic_gen.py")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if len(dataset) < 10:
        print(f"❌ Dataset quá nhỏ ({len(dataset)} cases). Cần ít nhất 10 cases.")
        return

    print(f"📋 Loaded {len(dataset)} test cases")

    # V1: base agent (minimal system prompt)
    v1_agent = MainAgent(version="v1", use_system_prompt=False)
    v1_results, v1_summary = await _run_benchmark("Agent_V1_Base", v1_agent, dataset)

    # V2: optimized agent (better system prompt + same retrieval)
    v2_agent = MainAgent(version="v2", use_system_prompt=True)
    v2_results, v2_summary = await _run_benchmark("Agent_V2_Optimized", v2_agent, dataset)

    if not v1_results or not v2_results:
        print("ERROR: One or both benchmark runs produced 0 valid results. Check API keys and model names.")
        return

    # Regression gate
    decision, failures = _regression_gate(v1_summary, v2_summary, v2_results)

    print(f"\n{'='*50}")
    print("📊 KÊTS QUẢ SO SÁNH (REGRESSION ANALYSIS)")
    print(f"{'='*50}")

    m1, m2 = v1_summary["metrics"], v2_summary["metrics"]
    delta_score = m2["avg_score"] - m1["avg_score"]

    print(f"{'Metric':<25} {'V1':>8} {'V2':>8} {'Delta':>10}")
    print(f"{'-'*55}")
    print(f"{'avg_score':<25} {m1['avg_score']:>8.3f} {m2['avg_score']:>8.3f} {delta_score:>+10.3f}")
    print(f"{'hit_rate':<25} {m1['hit_rate']:>8.3f} {m2['hit_rate']:>8.3f} {m2['hit_rate']-m1['hit_rate']:>+10.3f}")
    print(f"{'agreement_rate':<25} {m1['agreement_rate']:>8.3f} {m2['agreement_rate']:>8.3f} {m2['agreement_rate']-m1['agreement_rate']:>+10.3f}")
    print(f"{'hallucination_rate':<25} {m1['hallucination_rate']:>8.3f} {m2['hallucination_rate']:>8.3f} {m2['hallucination_rate']-m1['hallucination_rate']:>+10.3f}")
    print(f"{'pass_rate':<25} {m1['pass_rate']:>8.3f} {m2['pass_rate']:>8.3f} {m2['pass_rate']-m1['pass_rate']:>+10.3f}")
    print(f"\nTotal cost V1: ${v1_summary['cost']['total_usd']:.4f} | V2: ${v2_summary['cost']['total_usd']:.4f}")
    print(f"Per-eval V2:  ${v2_summary['cost']['per_eval_usd']:.5f}")

    print(f"\n{'='*50}")
    if decision == "APPROVE":
        print(f"✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print(f"❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")
        for reason in failures:
            print(f"   • {reason}")

    # Save reports
    os.makedirs("reports", exist_ok=True)

    # Merge V2 summary with regression info
    v2_summary["regression"] = {
        "v1_avg_score": m1["avg_score"],
        "v2_avg_score": m2["avg_score"],
        "delta": round(delta_score, 4),
        "decision": decision,
        "failure_reasons": failures,
    }
    v2_summary["metadata"]["version"] = "Agent_V2_Optimized"

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print("\n💾 Đã lưu: reports/summary.json, reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
