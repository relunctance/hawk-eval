"""
BLEU Score — 简单实现 BLEU-1/2/3/4
"""

import math
from typing import Any


def ngram_count(text: str, n: int) -> dict[tuple, int]:
    """统计 n-gram 出现次数。"""
    tokens = text.split()
    counts = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def bleu_score(pred: str, ref: str, max_n: int = 4) -> dict[str, float]:
    """
    计算 BLEU-1 ~ BLEU-4。

    pred: 预测文本
    ref: 参考文本（ground truth）
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()

    # 短文本惩罚
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))

    scores = {}
    for n in range(1, max_n + 1):
        pred_counts = ngram_count(pred, n)
        ref_counts = ngram_count(ref, n)

        # 截断匹配
        matches = sum(
            min(pred_counts.get(k, 0), ref_counts.get(k, 0))
            for k in pred_counts
        )
        total = sum(pred_counts.values())

        if total == 0:
            scores[f"bleu{n}"] = 0.0
        else:
            p_n = matches / total
            scores[f"bleu{n}"] = p_n

    # 几何平均（简化版：算术平均）
    if scores:
        avg = sum(scores.values()) / len(scores)
        scores["bleu"] = bp * avg
    else:
        scores["bleu"] = 0.0

    return scores


def f1_score(pred: str, ref: str) -> float:
    """简单的 word-level F1。"""
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_text_metrics(pred: str, ref: str) -> dict[str, Any]:
    """批量计算文本质量指标。"""
    bleu = bleu_score(pred, ref)
    f1 = f1_score(pred, ref)
    return {
        "bleu1": bleu.get("bleu1", 0.0),
        "bleu2": bleu.get("bleu2", 0.0),
        "bleu3": bleu.get("bleu3", 0.0),
        "bleu4": bleu.get("bleu4", 0.0),
        "f1": f1,
    }
