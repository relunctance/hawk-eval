"""
Recall Metrics — MRR / Recall@K / NDCG
"""

from typing import Any


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """计算 MRR。ranks 是每个 query 的目标排名（1-indexed），未命中为 None。
    Standard MRR: sum(1/rank) over all queries / N (including unmatched=0).
    """
    if not ranks:
        return 0.0
    score = 0.0
    for r in ranks:
        if r is not None and r > 0:
            score += 1.0 / r
        # unmatched (None or 0) contributes 0
    return score / len(ranks)


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


def _strip_prefix(t: str) -> str:
    """去掉 capture 存储格式的前缀，只保留核心内容（answer）。

    所有格式统一处理：取最后一个换行之后的内容，再去掉角色前缀。
    - "用户: question\\n助手: answer" → "助手: answer" → "answer"
    - "question\\nanswer" → "answer"
    - "助手: answer" → "answer"
    - "用户: 用户: question\\n助手: answer" → "answer"
    """
    # 取最后一个换行之后的内容（去掉 question 行）
    if "\n" in t:
        t = t.split("\n")[-1]
    # 去掉角色前缀
    for p in ("用户: ", "助手: ", "User: ", "Assistant: "):
        if t.startswith(p):
            t = t[len(p):]
    return t.strip()


def _text_similar_match(target: str, retrieved: list[str], threshold: float = 0.6) -> int | None:
    """用 text_similar 找 target 在 retrieved 中的排名（1-indexed），未命中返回 None。"""
    target_stripped = _strip_prefix(target)
    for i, ret_text in enumerate(retrieved):
        ret_stripped = _strip_prefix(ret_text)
        # Token overlap matching (same as benchmark_hawk.text_similar)
        words1 = set(target_stripped.split())
        words2 = set(ret_stripped.split())
        if not words1 or not words2:
            continue
        overlap = len(words1 & words2)
        if overlap / max(len(words1), len(words2)) >= threshold:
            return i + 1  # 1-indexed
    return None


def compute_recall_metrics(
    results: list[dict],
    k_values: list[int] = None,
) -> dict[str, Any]:
    """
    批量计算 recall 指标。

    results: list of {
        "query_id": str,
        "target_id": str,         # 原始 target 文本（或 memory_id）
        "retrieved_ids": list[str],  # 原始 retrieved 文本列表（按排名排序）
        "use_text_similarity": bool,  # 可选：用 text_similar 匹配代替精确字符串匹配
    }
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    ranks = []
    for item in results:
        target = item.get("target_id", "")
        retrieved = item.get("retrieved_ids", [])
        use_text_sim = item.get("use_text_similarity", False)

        if use_text_sim:
            rank = _text_similar_match(target, retrieved)
        else:
            # Exact string match (legacy behavior)
            try:
                rank = retrieved.index(target) + 1
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
