"""
Retrieval Evaluator
Calculates Hit Rate and MRR (Mean Reciprocal Rank) for RAG retrieval quality.
"""
import asyncio
from typing import List, Dict


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """1.0 if any expected_id appears in the top_k retrieved docs, else 0.0."""
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of the first relevant document, or 0 if none found."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Evaluate retrieval over an entire dataset.
        Each case must have 'expected_retrieval_ids' and agent response must include 'retrieved_ids'.
        Falls back to heuristic scoring when agent doesn't return retrieved_ids.
        """
        hit_rates = []
        mrr_scores = []

        for case in dataset:
            expected = case.get("expected_retrieval_ids", [])
            if not expected:
                continue

            retrieved = case.get("agent_retrieved_ids", [])
            if not retrieved:
                # heuristic: treat as miss when agent returns no retrieval metadata
                hit_rates.append(0.0)
                mrr_scores.append(0.0)
                continue

            hit_rates.append(self.calculate_hit_rate(expected, retrieved))
            mrr_scores.append(self.calculate_mrr(expected, retrieved))

        avg_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        return {
            "avg_hit_rate": round(avg_hit, 4),
            "avg_mrr": round(avg_mrr, 4),
            "evaluated_cases": len(hit_rates),
        }
