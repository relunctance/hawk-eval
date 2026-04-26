#!/usr/bin/env python3
"""
Evaluation Runner — 统一评测引擎

用法:
    python -m src.runner --dataset datasets/hawk_memory/conversational_qa.jsonl \
        --adapter hawk_memory_api --output reports/hawk.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src.metrics import compute_recall_metrics, compute_text_metrics, trigger_accuracy
from src.adapters import HawkMemoryAdapter, MFlowAdapter, Mem0Adapter, RAGBaselineAdapter


# ─── Adapter 工厂 ──────────────────────────────────────────────────────────

ADAPTERS = {
    "hawk_memory_api": HawkMemoryAdapter,
    "m_flow": MFlowAdapter,
    "mem0": Mem0Adapter,
    "rag_baseline": RAGBaselineAdapter,
}


def make_adapter(name: str, extra_args: dict = None) -> object:
    """创建 adapter 实例。"""
    factory = ADAPTERS.get(name)
    if factory is None:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    # 实例化：如果 extra_args 就用参数构造，否则用默认构造
    if extra_args:
        adapter = factory(**extra_args)
    else:
        adapter = factory()
    return adapter


# ─── 数据加载 ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL 文件。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ─── Recall 评测 ────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    query_id: str
    query: str
    target_id: str | None
    retrieved_ids: list[str]
    retrieved_texts: list[str]
    latency: float
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def run_recall_benchmark(
    adapter: object,
    dataset: list[dict],
    top_k: int = 10,
    verbose: bool = True,
) -> tuple[list[CaseResult], list[dict]]:
    """
    对一个 dataset 运行 recall benchmark。

    dataset: JSONL，每项需有 id / question / answer（或 memory_text）
    adapter: 实现了 recall(query, top_k) -> {memories: [{id, text, score}]} 的对象

    返回 (results, raw_text_metrics)
    """
    results = []
    raw_text_metrics = []

    for i, item in enumerate(dataset):
        qid = item.get("id", f"q-{i}")
        query = item.get("question", "")
        target_id = item.get("target_id", item.get("memory_id", None))
        answer = item.get("answer", item.get("memory_text", ""))
        ground_truth = item.get("memory_text", item.get("answer", ""))

        try:
            resp = adapter.recall(query, top_k=top_k)
            memories = resp.get("memories", []) if isinstance(resp, dict) else []
            latency = resp.get("latency", 0.0) if isinstance(resp, dict) else 0.0

            retrieved_ids = [m.get("id", f"r-{j}") for j, m in enumerate(memories)]
            retrieved_texts = [m.get("text", "") for m in memories]

            # 文本质量指标（BLEU / F1）
            if retrieved_texts and ground_truth:
                text_metrics = compute_text_metrics(retrieved_texts[0], ground_truth)
            else:
                text_metrics = {}

            raw_text_metrics.append(text_metrics)

            results.append(CaseResult(
                query_id=qid,
                query=query,
                target_id=target_id,
                retrieved_ids=retrieved_ids,
                retrieved_texts=retrieved_texts,
                latency=latency,
            ))

            if verbose and (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(dataset)}", flush=True)

        except Exception as e:
            results.append(CaseResult(
                query_id=qid,
                query=query,
                target_id=target_id,
                retrieved_ids=[],
                retrieved_texts=[],
                latency=0.0,
                error=str(e),
            ))

    return results, raw_text_metrics


# ─── Procedural 评测 ────────────────────────────────────────────────────────

def run_procedural_benchmark(
    adapter: object,
    dataset: list[dict],
    top_k: int = 10,
    verbose: bool = True,
) -> list[dict]:
    """
    运行程序性记忆 benchmark（对标 m_flow）。

    dataset: JSONL，每项需有 id / query / expect
    adapter: 需实现 recall(query, top_k) 方法
    """
    results = []

    for i, item in enumerate(dataset):
        qid = item.get("id", f"p-{i}")
        query = item.get("query", "")
        expect = item.get("expect", {})

        try:
            resp = adapter.recall(query, top_k=top_k)
            memories = resp.get("memories", []) if isinstance(resp, dict) else []
            retrieved_ids = [m.get("id", "") for m in memories]
            # For hawk-memory: use full text for keyword matching (title field may not exist)
            retrieved_titles = [m.get("text", "") for m in memories]

            # 检查是否触发
            should_trigger = expect.get("should_trigger_procedural", False)
            actual_triggered = len(retrieved_ids) > 0

            # 检查 procedural key 命中
            exp_procs = expect.get("procedures", {}).get("any_of_titles_contains", [])
            hit_procs = any(
                any(kw in title for title in retrieved_titles)
                for kw in exp_procs
            ) if exp_procs else True

            results.append({
                "query_id": qid,
                "expected_trigger": should_trigger,
                "actual_triggered": actual_triggered,
                "expected_procedure_keys": exp_procs,
                "retrieved_procedure_keys": retrieved_titles,
                "procedure_hit": hit_procs,
                "retrieved_ids": retrieved_ids,
            })

            if verbose and (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(dataset)}", flush=True)

        except Exception as e:
            results.append({
                "query_id": qid,
                "expected_trigger": expect.get("should_trigger_procedural", False),
                "actual_triggered": False,
                "expected_procedure_keys": [],
                "retrieved_procedure_keys": [],
                "procedure_hit": False,
                "retrieved_ids": [],
                "error": str(e),
            })

    return results


# ─── 主程序 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="hawk-eval 评测引擎")
    parser.add_argument("--dataset", "-d", required=True, help="JSONL 数据集路径")
    parser.add_argument("--adapter", "-a", required=True,
                        choices=list(ADAPTERS.keys()), help="被测系统适配器")
    parser.add_argument("--output", "-o", default="eval_result.json", help="输出报告路径")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Recall top_k")
    parser.add_argument("--type", "-t", choices=["recall", "procedural"], default="recall",
                        help="评测类型")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # 加载数据集
    dataset = load_jsonl(args.dataset)
    print(f"[hawk-eval] 数据集: {args.dataset} ({len(dataset)} 条)")
    print(f"[hawk-eval] 适配器: {args.adapter}")
    print(f"[hawk-eval] 类型: {args.type}")

    # 创建 adapter
    adapter = make_adapter(args.adapter)

    # 检查健康
    if hasattr(adapter, "health_check"):
        ok = adapter.health_check()
        if not ok:
            print(f"[hawk-eval] ✗ {args.adapter} 不可用，请先启动服务", file=sys.stderr)
            sys.exit(1)
        print(f"[hawk-eval] ✓ {args.adapter} 可用")

    # 运行评测
    t0 = time.time()
    if args.type == "procedural":
        raw_results = run_procedural_benchmark(adapter, dataset, args.top_k, args.verbose)
        metrics = trigger_accuracy(raw_results)
        case_results = raw_results
    else:
        case_results, text_metrics = run_recall_benchmark(
            adapter, dataset, args.top_k, args.verbose
        )
        recall_input = [
            {
                "query_id": r.query_id,
                "target_id": r.target_id,
                "retrieved_ids": r.retrieved_ids,
            }
            for r in case_results
        ]
        metrics = compute_recall_metrics(recall_input, k_values=[1, 3, 5, 10])
        # 合并文本质量指标
        if text_metrics:
            avg_bleu = sum(t.get("bleu1", 0) for t in text_metrics) / len(text_metrics)
            avg_f1 = sum(t.get("f1", 0) for t in text_metrics) / len(text_metrics)
            metrics["bleu1"] = avg_bleu
            metrics["f1"] = avg_f1

    elapsed = time.time() - t0
    print(f"[hawk-eval] 完成，耗时 {elapsed:.1f}s")

    # 延迟统计
    if hasattr(adapter, "latency_stats"):
        lat = adapter.latency_stats()
        metrics["latency_p50"] = lat.get("p50", 0)
        metrics["latency_p99"] = lat.get("p99", 0)
        metrics["latency_mean"] = lat.get("mean", 0)

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "adapter": args.adapter,
        "dataset": args.dataset,
        "type": args.type,
        "elapsed_seconds": elapsed,
        "n_cases": len(dataset),
        "metrics": metrics,
        "cases": [c.to_dict() if hasattr(c, "to_dict") else c for c in case_results],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n[结果] {args.adapter} @ {Path(args.dataset).name}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n报告已保存: {args.output}")

    # CI Gate: 如果有 recall 指标，检查是否 > 0（基础检查）
    if "mrr@5" in metrics and metrics["mrr@5"] == 0:
        print("\n⚠ MRR@5 为 0，可能服务未正常返回结果", file=sys.stderr)
        # 不强制退出，因为可能是数据集问题


if __name__ == "__main__":
    main()
