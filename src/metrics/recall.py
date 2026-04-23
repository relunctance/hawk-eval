"""
Recall Metrics — MRR / Recall@K / NDCG
"""

from typing import Any


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """计算 MRR。ranks 是每个 query 的目标排名（1-indexed），未命中为 None。"""
    score = 0.0
    valid = 0
    for r in ranks:
        if r is not None and r > 0:
            score += 1.0 / r
            valid += 1
    return score / valid if valid > 0 else 0.0


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Recall@K：命中数 / 总数。"""
    hit = sum(1 for r in ranks if r is not None and r <= k)
    return hit / len(ranks) if ranks else 0.0


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """NDCG@K。理想序为前 k 个全命中。"""
    def dcg(ranks, k):
        score = 0.0
        for i, r in enumerate(ranks[:k]):
            if r is not None and r <= k:
                score += 1.0 / (i + 1)
        return score

    ideal = list(range(1, k + 1))
    d = dcg(ranks, k)
    i = dcg(ideal, k)
    return d / i if i > 0 else 0.0


def compute_recall_metrics(
    results: list[dict],
    k_values: list[int] = None,
) -> dict[str, Any]:
    """
    批量计算 recall 指标。

    results: list of {
        "query_id": str,
        "target_id": str,
        "retrieved_ids": list[str],  # 按排名排序
    }
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    ranks = []
    for item in results:
        ret = item.get("retrieved_ids", [])
        try:
            rank = ret.index(item["target_id"]) + 1
        except ValueError:
            rank = None
        ranks.append(rank)

    metrics = {
        f"mrr@{k}": mean_reciprocal_rank([r if r and r <= k else None for r in ranks])
        for k in k_values
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(ranks, k)

    return metrics
