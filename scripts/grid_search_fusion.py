#!/usr/bin/env python3
"""
Grid search fusion weights (alpha, beta, gamma) for MRR@5 optimization.

Collects raw vector+keyword scores from /recall_debug for all dataset cases,
then does offline grid search to find optimal alpha/beta/gamma.

Usage:
    python scripts/grid_search_fusion.py

Requires hawk-memory-api running and dataset populated via benchmark_hawk.py.
"""

import json
import requests
from collections import defaultdict
from itertools import product


API = "http://127.0.0.1:18360"
DATASET = "datasets/hawk_memory/conversational_qa.jsonl"
TOP_K = 5
RRF_K = 60


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_recall_debug(question: str, top_k: int = TOP_K) -> list[dict]:
    """Recall with debug mode, return raw scores for all fetched results."""
    payload = {
        "query": question,
        "agent_id": "eval",
        "top_k": top_k,
        "min_score": 0.0,  # 接受低分结果，用于 grid search
    }
    resp = requests.post(f"{API}/recall_debug", json=payload, timeout=30)
    if resp.status_code != 200:
        print(f"  WARNING: recall failed with {resp.status_code}: {resp.text[:200]}")
        return []
    data = resp.json()
    return data.get("memories", [])


def fused_score(vec_raw, kw_raw, kw_rank, alpha, beta, gamma):
    """Compute fused score with given weights."""
    import math
    # RRF bonus
    rrf_bonus = 1.0 / (RRF_K + kw_rank) if kw_rank > 0 else 0.0
    return alpha * vec_raw + beta * kw_raw + gamma * rrf_bonus


def _text_overlap(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """Check if two texts have significant token overlap."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return False
    overlap = len(tokens1 & tokens2)
    jaccard = overlap / len(tokens1 | tokens2)
    return jaccard > threshold


def compute_mrr_at_k(dataset, results_by_case, alpha, beta, gamma):
    """
    Compute MRR@K given fusion weights.
    dataset: list of {id, question, answer}
    results_by_case: dict mapping case_id -> list of recall results with raw scores
    """
    reciprocal_ranks = []

    for case in dataset:
        case_id = case["id"]
        answer = case["answer"]
        memories = results_by_case.get(case_id, [])
        if not memories:
            reciprocal_ranks.append(0.0)
            continue

        # Sort by fused score
        scored = [
            (mem, fused_score(
                mem.get("vector_score_raw", 0.0),
                mem.get("keyword_score_raw", 0.0),
                mem.get("keyword_rank", -1),
                alpha, beta, gamma
            ))
            for mem in memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Find rank of ground truth using text similarity
        for rank, (mem, _) in enumerate(scored, 1):
            if _text_overlap(mem.get("text", ""), answer, threshold=0.5):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def main():
    print(f"Loading dataset: {DATASET}")
    dataset = load_dataset(DATASET)
    print(f"Dataset size: {len(dataset)}")

    # Collect raw scores for all cases
    print("\nFetching recall_debug for all cases...")
    results_by_case = {}
    for i, case in enumerate(dataset):
        case_id = case["id"]
        question = case["question"]
        print(f"  [{i+1}/{len(dataset)}] {case_id}: {question[:40]}...", end=" ", flush=True)

        memories = fetch_recall_debug(question, top_k=TOP_K)
        # Attach case_id to each memory for tracking
        for m in memories:
            m["case_id"] = case_id
        results_by_case[case_id] = memories
        print(f"→ {len(memories)} results")

    # Save raw scores for offline grid search
    raw_path = "reports/fusion_grid_search_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results_by_case, f, ensure_ascii=False)
    print(f"\nRaw scores saved to {raw_path}")

    # Grid search
    print("\nRunning grid search...")
    best_mrr = 0.0
    best_weights = None

    # Alpha range: 0.0 to 1.0 step 0.1
    # Beta range: 0.0 to 1.0 step 0.1
    # Gamma = 1 - alpha - beta (normalized)
    grid_points = []
    for alpha in [round(x * 0.1, 1) for x in range(0, 11)]:  # 0.0 to 1.0
        for beta in [round(x * 0.1, 1) for x in range(0, int((1.0 - alpha) * 10) + 1)]:
            gamma = round(1.0 - alpha - beta, 2)
            if gamma < 0:
                gamma = 0.0
            grid_points.append((round(alpha, 1), round(beta, 1), gamma))

    print(f"Grid points: {len(grid_points)}")
    for alpha, beta, gamma in grid_points:
        mrr = compute_mrr_at_k(dataset, results_by_case, alpha, beta, gamma)
        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = (alpha, beta, gamma)
            print(f"  NEW BEST: α={alpha:.1f} β={beta:.1f} γ={gamma:.2f} → MRR@{TOP_K}={mrr:.4f}")

    print(f"\n{'='*60}")
    print(f"Best weights: α={best_weights[0]:.1f} β={best_weights[1]:.1f} γ={best_weights[2]:.2f}")
    print(f"Best MRR@{TOP_K}: {best_mrr:.4f}")
    print(f"{'='*60}")

    # Save results
    output = {
        "best_weights": {"alpha": best_weights[0], "beta": best_weights[1], "gamma": best_weights[2]},
        "best_mrr": best_mrr,
        "grid_size": len(grid_points),
        "dataset_size": len(dataset),
    }
    with open("reports/fusion_grid_search_result.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to reports/fusion_grid_search_result.json")

    # Also compute baseline MRR with current weights
    print("\nBaseline comparison (current weights):")
    for alpha, beta, gamma in [(0.75, 0.15, 0.10), (0.60, 0.30, 0.10), (0.50, 0.50, 0.00)]:
        mrr = compute_mrr_at_k(dataset, results_by_case, alpha, beta, gamma)
        print(f"  α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} → MRR@{TOP_K}={mrr:.4f}")


if __name__ == "__main__":
    main()
