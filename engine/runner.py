"""
Async Benchmark Runner
- Runs all test cases in parallel batches (controlled concurrency)
- Tracks per-case cost (agent + judge)
- Reports total cost and per-eval cost
"""
import asyncio
import time
from typing import List, Dict

from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, concurrency: int = 1):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.semaphore = asyncio.Semaphore(concurrency)
        self.retrieval_eval = RetrievalEvaluator()

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self.semaphore:
            start = time.perf_counter()

            # 1. Agent generates answer
            response = await self.agent.query(test_case["question"])
            agent_latency = time.perf_counter() - start

            # 2. RAGAS-style metrics via evaluator
            ragas_scores = await self.evaluator.score(test_case, response)

            # 3. Multi-Judge evaluation
            judge_result = await self.judge.evaluate_multi_judge(
                test_case["question"],
                response["answer"],
                test_case.get("expected_answer", ""),
            )

            # 4. Retrieval metrics (per-case)
            expected_ids = test_case.get("expected_retrieval_ids", [])
            retrieved_ids = response.get("retrieved_ids", [])
            hit_rate = self.retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.retrieval_eval.calculate_mrr(expected_ids, retrieved_ids)

            agent_cost = response.get("metadata", {}).get("cost_usd", 0)
            judge_cost = judge_result.get("cost_usd", 0)
            total_cost = agent_cost + judge_cost

            return {
                "test_case": test_case["question"],
                "difficulty": test_case.get("difficulty", "medium"),
                "type": test_case.get("type", "factual"),
                "agent_response": response["answer"],
                "latency": round(agent_latency, 3),
                "ragas": ragas_scores,
                "judge": judge_result,
                "retrieval": {
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                },
                "cost": {
                    "agent_usd": round(agent_cost, 6),
                    "judge_usd": round(judge_cost, 6),
                    "total_usd": round(total_cost, 6),
                },
                "status": "fail" if judge_result["final_score"] < 3 else "pass",
                "hallucination": judge_result.get("hallucination_detected", False),
            }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        """
        Run test cases with concurrency=3 batching.
        Model split: agent→GROQ (30 RPM), Judge A→gemini-3.1-flash-lite (15 RPM),
        Judge B→gemma-4-31b-it (15 RPM separate bucket).
        5 concurrent cases + 3s inter-batch sleep + per-model rate limiter → ~20 cases/min.
        """
        batch_size = 5
        print(f"  ▶ Running {len(dataset)} test cases (batch={batch_size}, async parallel judges)...")
        t0 = time.perf_counter()
        all_results = []

        for batch_start in range(0, len(dataset), batch_size):
            batch = dataset[batch_start:batch_start + batch_size]
            batch_results = await asyncio.gather(
                *[self.run_single_test(case) for case in batch],
                return_exceptions=True,
            )
            all_results.extend(batch_results)
            done = min(batch_start + batch_size, len(dataset))
            print(f"    ... {done}/{len(dataset)} done")
            if done < len(dataset):
                await asyncio.sleep(3)

        elapsed = time.perf_counter() - t0

        valid = []
        errors = 0
        for r in all_results:
            if isinstance(r, Exception):
                errors += 1
                print(f"  [ERR] {r}")
            else:
                valid.append(r)

        total_cost = sum(r["cost"]["total_usd"] for r in valid)
        print(f"  ✅ Completed {len(valid)}/{len(dataset)} cases in {elapsed:.1f}s")
        print(f"  💰 Total cost: ${total_cost:.4f} | Per eval: ${total_cost/max(len(valid),1):.5f}")
        if errors:
            print(f"  ⚠️  {errors} cases failed")

        return valid
