#!/usr/bin/env python3
"""
Evolving Events Recall Benchmark

评测 multi-hop reasoning 记忆能力。

评测逻辑（与 benchmark_locomo.py 一致）：
1. capture 一条记忆（用 answer text）
2. 等索引就绪
3. 用 question 做 recall
4. 用 top-1 召回结果与 gold answer 计算 BLEU-1 / F1
5. 按 multi-hop vs single-hop 分类汇总

用法:
    PYTHONPATH=src python -m src.benchmark_evolving_events \\
        --dataset datasets/evolving_events/evolving_events_qa.jsonl \\
        --output reports/evolving_events_recall.json
"""

import argparse
import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from benchmark_locomo import HawkMemoryLocomoBenchmark, calculate_bleu1, calculate_f1


# ─── HTTP ───────────────────────────────────────────────────────────────────

HAWK_BASE = os.getenv("HAWK_API_BASE", "http://127.0.0.1:18368")


def hawk_req(method, path, body=None, timeout=30):
    url = HAWK_BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    req_ = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req_, timeout=timeout) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read()), e.code
        except Exception:
            return str(e), e.code
    except Exception as e:
        return str(e), -1


def text_similar(t1: str, t2: str, threshold: float = 0.5) -> bool:
    words1 = set(t1.lower().split())
    words2 = set(t2.lower().split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    return overlap / max(len(words1), len(words2)) >= threshold


@dataclass
class CaseResult:
    query_id: str
    is_multi_hop: bool
    query: str
    gold_answer: str
    coarse: str
    retrieved_texts: list[str]
    rank: int | None
    latency: float
    bleu1: float = 0.0
    f1: float = 0.0

    def to_dict(self):
        return asdict(self)


class HawkMemoryEEBenchmark:
    """Evolving Events benchmark for hawk-memory-api."""

    def __init__(self):
        self.latencies: list[float] = []

    def health_check(self) -> bool:
        data, s = hawk_req("GET", "/health")
        return s == 200 and data.get("status") == "ok"

    def capture(self, text: str) -> bool:
        session = f"ee-{uuid.uuid4().hex[:8]}"
        body = {
            "session_id": session,
            "user_id": "benchmark",
            "message": text,
            "response": "",
            "platform": "ee-benchmark",
        }
        data, s = hawk_req("POST", "/v1/capture", body)
        return s in (200, 201)

    def recall(self, query: str, top_k: int = 10) -> tuple[list[dict], float]:
        body = {"query": query, "top_k": top_k, "platform": "ee-benchmark"}
        t0 = time.perf_counter()
        data, s = hawk_req("POST", "/v1/recall", body)
        latency = time.perf_counter() - t0
        self.latencies.append(latency)
        if s != 200:
            return [], latency
        return data.get("memories", []), latency

    def run(self, dataset: list[dict], top_k: int = 10) -> tuple[list[CaseResult], dict]:
        # 1. Capture
        print(f"  [1] Capture {len(dataset)} memories...")
        def do_capture(item: dict) -> bool:
            text = item.get("answer", "")
            return bool(text and self.capture(text))

        with ThreadPoolExecutor(max_workers=5) as ex:
            caps = list(ex.map(do_capture, dataset))
        print(f"      captured {sum(caps)}/{len(dataset)}")

        print(f"  [2] Wait for index (3s)...")
        time.sleep(3)

        # 2. Recall
        print(f"  [3] Recall {len(dataset)} queries...")

        def do_recall(item: dict, i: int) -> CaseResult:
            qid = item.get("id", f"ee-{i}")
            query = item.get("question", "")
            gold = item.get("answer", "")
            coarse = item.get("coarse", "")
            is_multi = item.get("is_multi_hop", False)

            memories, latency = self.recall(query, top_k=top_k)
            retrieved = [m.get("text", "") for m in memories]

            rank = None
            for pos, txt in enumerate(retrieved):
                if text_similar(txt, gold):
                    rank = pos + 1
                    break

            bleu1 = calculate_bleu1(retrieved[0] if retrieved else "", gold)
            f1 = calculate_f1(retrieved[0] if retrieved else "", gold)

            return CaseResult(
                query_id=qid, is_multi_hop=is_multi,
                query=query, gold_answer=gold, coarse=coarse,
                retrieved_texts=retrieved, rank=rank,
                latency=latency, bleu1=bleu1, f1=f1,
            )

        results: list[CaseResult] = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(do_recall, item, i): i for i, item in enumerate(dataset)}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 20 == 0:
                    print(f"      progress: {done}/{len(dataset)}")

        results.sort(key=lambda r: r.query_id)

        # 3. Aggregate
        metrics = self._aggregate(results)
        print(f"  [4] Done")
        return results, metrics

    def _aggregate(self, results: list[CaseResult]) -> dict:
        total = len(results)
        if total == 0:
            return {}

        def safe_avg(values):
            v = [x for x in values if x is not None and x >= 0]
            return sum(v) / len(v) if v else 0.0

        ranks = [r.rank for r in results]
        hits = sum(1 for r in ranks if r is not None and r <= 10)
        mrr = sum(1/r for r in ranks if r is not None) / total

        metrics = {
            "n": total,
            "mrr@10": mrr,
            "recall@10": hits / total,
            "bleu1_avg": safe_avg([r.bleu1 for r in results]),
            "f1_avg": safe_avg([r.f1 for r in results]),
            "latency_avg": safe_avg([r.latency for r in results]),
        }

        # Multi-hop vs single-hop
        from collections import defaultdict
        by_type: dict[str, list[CaseResult]] = defaultdict(list)
        for r in results:
            key = "multi-hop" if r.is_multi_hop else "single-hop"
            by_type[key].append(r)

        cat_metrics = {}
        for label, cat_results in sorted(by_type.items()):
            cat_ranks = [r.rank for r in cat_results]
            cat_hits = sum(1 for r in cat_ranks if r is not None and r <= 10)
            cat_mrr = sum(1/r for r in cat_ranks if r is not None) / len(cat_results)
            cat_metrics[label] = {
                "n": len(cat_results),
                "mrr@10": cat_mrr,
                "recall@10": cat_hits / len(cat_results),
                "bleu1_avg": safe_avg([r.bleu1 for r in cat_results]),
                "f1_avg": safe_avg([r.f1 for r in cat_results]),
            }

        metrics["per_type"] = cat_metrics
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evolving Events recall benchmark")
    parser.add_argument("--dataset", "-d",
                        default="datasets/evolving_events/evolving_events_qa.jsonl")
    parser.add_argument("--output", "-o",
                        default="reports/evolving_events_recall.json")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    args = parser.parse_args()

    dataset = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dataset.append(json.loads(line))
                except Exception:
                    pass

    print(f"[EvolvingEvents] dataset={args.dataset} n={len(dataset)}")

    bm = HawkMemoryEEBenchmark()
    if not bm.health_check():
        print("❌ hawk-memory-api health check failed")
        sys.exit(1)
    print("✅ hawk-memory-api health OK")

    t0 = time.time()
    results, metrics = bm.run(dataset, top_k=args.top_k)
    elapsed = time.time() - t0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "elapsed_seconds": elapsed,
        "n_cases": len(results),
        "metrics": metrics,
        "cases": [r.to_dict() for r in results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  Overall (n={metrics['n']})")
    print(f"  MRR@10:     {metrics.get('mrr@10', 0):.4f}")
    print(f"  Recall@10:  {metrics.get('recall@10', 0):.1%}")
    print(f"  BLEU-1 avg: {metrics.get('bleu1_avg', 0):.4f}")
    print(f"  F1 avg:     {metrics.get('f1_avg', 0):.4f}")
    print(f"\n  Per-Type:")
    for label, cm in metrics.get("per_type", {}).items():
        print(f"    {label:12s} n={cm['n']:3d}  MRR@10={cm['mrr@10']:.4f}  "
              f"Recall@10={cm['recall@10']:.1%}  BLEU1={cm['bleu1_avg']:.4f}")
    print(f"\n  Baseline Comparison (Human-like Correctness):")
    print(f"    mflow (k=5,gpt5mini):  95.8%")
    print(f"    mflow (k=10,gpt5.4):   97.7%")
    print(f"    cognee (k=10,gpt5.4):  93.0%")
    print(f"    graphiti (k=10,gpt5.4): 68.4%")
    print(f"{'='*55}")
    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
