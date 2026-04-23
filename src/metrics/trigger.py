"""
Trigger / Procedural Memory Metrics
对标 m_flow 的 recall/accuracy/fp_rate
"""

from typing import Any


def trigger_accuracy(
    results: list[dict],
) -> dict[str, Any]:
    """
    计算触发准确性。

    results: list of {
        "query_id": str,
        "expected_trigger": bool,
        "actual_triggered": bool,
        "expected_episode_ids": list[str],
        "retrieved_episode_ids": list[str],
        "expected_procedure_keys": list[str],
        "retrieved_procedure_keys": list[str],
    }
    """
    total = len(results)
    if total == 0:
        return {"trigger_accuracy": 0.0, "n": 0}

    trigger_hit = sum(
        1 for r in results
        if r.get("expected_trigger") == r.get("actual_triggered")
    )

    # procedural recall: 命中的 procedure key 数
    proc_recalls = []
    for r in results:
        exp = set(r.get("expected_procedure_keys", []))
        ret = set(r.get("retrieved_procedure_keys", []))
        if exp:
            proc_recalls.append(len(ret & exp) / len(exp))
        else:
            proc_recalls.append(1.0 if not ret else 0.0)

    # episodic recall
    epi_recalls = []
    for r in results:
        exp = set(r.get("expected_episode_ids", []))
        ret = set(r.get("retrieved_episode_ids", []))
        if exp:
            epi_recalls.append(len(ret & exp) / len(exp))
        else:
            epi_recalls.append(1.0 if not ret else 0.0)

    # false positive rate（negative case 中错误触发的比例）
    negatives = [r for r in results if not r.get("expected_trigger")]
    fp = 0
    if negatives:
        fp = sum(1 for r in negatives if r.get("actual_triggered"))

    return {
        "trigger_accuracy": trigger_hit / total,
        "procedural_recall": sum(proc_recalls) / len(proc_recalls),
        "episodic_recall": sum(epi_recalls) / len(epi_recalls),
        "fp_inject_rate": fp / len(negatives) if negatives else 0.0,
        "n": total,
        "n_negative": len(negatives),
    }
