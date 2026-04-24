#!/usr/bin/env python3
"""
KR3.4: hawk-eval 多 agent 评测
- 按 agent 分别 capture 数据集
- 按 agent 分别 recall + 计算 MRR
- 输出汇总报告

用法:
  python -m src.benchmark_multi_agent --agents user1,user2,user3 --dataset datasets/hawk_memory/conversational_qa.jsonl
"""
import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_hawk import HawkMemoryBenchmark, load_cache, DEFAULT_CACHE


def compute_mrr_and_recall(cases: list) -> dict:
    """从 case 结果计算 MRR 和 Recall 指标。"""
    mrr = {"mrr@1": 0, "mrr@3": 0, "mrr@5": 0, "mrr@10": 0}
    recall = {"recall@1": 0, "recall@3": 0, "recall@5": 0, "recall@10": 0}
    bleu1_sum = 0
    f1_sum = 0
    latencies = []

    for c in cases:
        rank = c.get("rank")
        latencies.append(c.get("latency", 0))
        bleu1_sum += c.get("bleu1", 0)
        f1_sum += c.get("f1", 0)

        for k in [1, 3, 5, 10]:
            if rank is not None and rank <= k:
                mrr[f"mrr@{k}"] += 1.0 / rank
            recall[f"recall@{k}"] += 1 if (rank is not None and rank <= k) else 0

    n = len(cases)
    if n == 0:
        return {"mrr@1": 0, "mrr@3": 0, "mrr@5": 0, "mrr@10": 0,
                "recall@1": 0, "recall@3": 0, "recall@5": 0, "recall@10": 0,
                "bleu1_avg": 0, "f1_avg": 0, "latency_avg": 0, "latency_p50": 0}

    for k in [1, 3, 5, 10]:
        mrr[f"mrr@{k}"] /= n
        recall[f"recall@{k}"] = recall[f"recall@{k}"] / n

    latencies.sort()
    p50_idx = int(len(latencies) * 0.5)

    return {
        **mrr,
        **recall,
        "bleu1_avg": bleu1_sum / n,
        "f1_avg": f1_sum / n,
        "latency_avg": sum(latencies) / len(latencies),
        "latency_p50": latencies[p50_idx] if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="hawk-eval 多 agent 评测（KR3.4）")
    parser.add_argument("--dataset", default="datasets/hawk_memory/conversational_qa.jsonl",
                        help="JSONL 数据集路径")
    parser.add_argument("--agents", default="user1,user2,user3",
                        help="逗号分隔的 agent ID 列表")
    parser.add_argument("--output", default="reports/multi_agent.json",
                        help="输出报告路径")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rewrite", action="store_true", help="启用 Query Rewrite")
    parser.add_argument("--limit", type=int, default=0, help="只跑前N条（0=全部）")
    args = parser.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    if not agents:
        print("❌ 至少需要 1 个 agent")
        return

    def log(*a, **kw):
        print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **kw)

    # Load dataset
    dataset = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dataset.append(json.loads(line))
                except:
                    pass

    if args.limit > 0:
        dataset = dataset[:args.limit]

    log(f"📊 多 agent 评测: {len(dataset)} 条 × {len(agents)} agents")

    bm = HawkMemoryBenchmark()
    if not bm.health_check():
        log("❌ hawk-memory-api health check failed")
        return
    log("✅ hawk-memory-api health OK")

    all_reports = {}
    all_results = {}  # agent -> list of case results

    for agent_id in agents:
        log(f"\n{'='*50}")
        log(f"🤖 Agent: {agent_id}")
        log(f"{'='*50}")

        # 1. Clean DB for this agent
        log(f"[{agent_id}] 清理 eval 数据...")
        try:
            bm.api.post("/admin/cleanup", {"agent_id": agent_id})
        except Exception:
            pass
        time.sleep(2)

        # 2. Capture dataset with this agent_id
        log(f"[{agent_id}] Capture {len(dataset)} 条记忆 (agent={agent_id})...")
        session_data = bm.capture_dataset(dataset, top_k=args.top_k, log_fn=log,
                                          use_llm=False, rewrite=args.rewrite, agent_id=agent_id)

        time.sleep(3)  # Wait for index

        # 3. Recall + evaluate
        log(f"[{agent_id}] Recall 评测...")
        results, metrics = bm.recall_eval(dataset, top_k=args.top_k, log_fn=log,
                                           rewrite=args.rewrite, agent_id=agent_id)

        log(f"[{agent_id}] MRR@5={metrics.get('mrr@5', 0):.3f}  Recall@5={metrics.get('recall@5', 0):.1%}  "
            f"Latency={metrics.get('latency_p50', 0):.2f}s")

        all_results[agent_id] = results
        all_reports[agent_id] = {
            "agent": agent_id,
            "count": len(results),
            "metrics": metrics,
            "cases": [r.to_dict() for r in results],
        }

    # 汇总所有 agent 结果（不带 agent 过滤）
    log(f"\n{'='*50}")
    log("📊 汇总报告（全部 agent 混合）")
    log(f"{'='*50}")

    # Aggregate all cases
    all_cases = []
    for agent_id, results in all_results.items():
        all_cases.extend([(agent_id, r) for r in results])

    # Compute aggregate metrics (no agent filter = all memories)
    agg_metrics = compute_mrr_and_recall([r for _, r in all_cases])
    log(f"Aggregate MRR@5={agg_metrics['mrr@5']:.3f}  Recall@5={agg_metrics['recall@5']:.1%}")

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "dataset_count": len(dataset),
        "agents": agents,
        "top_k": args.top_k,
        "rewrite": args.rewrite,
        "per_agent": all_reports,
        "aggregate": {
            "metrics": agg_metrics,
        }
    }

    with open(args.output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print per-agent summary table
    log(f"\n{'Agent':<20} {'MRR@5':>8} {'Recall@5':>10} {'Latency':>10}")
    log("-" * 52)
    for agent_id in agents:
        m = all_reports[agent_id]["metrics"]
        lat = m.get("latency_p50", 0)
        log(f"{agent_id:<20} {m.get('mrr@5', 0):>8.3f} {m.get('recall@5', 0):>10.1%} {lat:>9.2f}s")
    log("-" * 52)
    log(f"{'AGGREGATE':<20} {agg_metrics['mrr@5']:>8.3f} {agg_metrics['recall@5']:>10.1%} {agg_metrics['latency_p50']:>9.2f}s")

    log(f"\n✅ 报告已保存: {args.output}")


if __name__ == "__main__":
    main()
