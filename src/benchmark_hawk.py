#!/usr/bin/env python3
"""
hawk-memory-api Recall Benchmark

评测逻辑（文本匹配代替 ID 匹配）：
1. capture 一条记忆（用 answer text）
2. 等索引就绪
3. 用 question 做 recall
4. 检查 target answer 是否在返回结果中（文本相似度匹配）
5. 计算 MRR / Recall@K / BLEU / F1 / Latency

用法:
    python -m src.benchmark_hawk \
        --dataset datasets/hawk_memory/conversational_qa.jsonl \
        --output reports/hawk_recall.json
"""

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from src.metrics import compute_recall_metrics, compute_text_metrics


# ─── HTTP ───────────────────────────────────────────────────────────────────

BASE = "http://127.0.0.1:18360"


def req(method, path, body=None, timeout=10):
    url = BASE + path
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


# ─── Text Similarity ─────────────────────────────────────────────────────────

def text_similar(t1: str, t2: str, threshold: float = 0.6) -> bool:
    """判断两个文本是否相似（token overlap > threshold）。"""
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    return overlap / max(len(words1), len(words2)) >= threshold


# ─── Benchmark ───────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    query_id: str
    query: str
    target_text: str
    retrieved_texts: list[str]
    rank: int | None
    latency: float
    bleu1: float = 0.0
    f1: float = 0.0

    def to_dict(self):
        return asdict(self)


class HawkMemoryBenchmark:
    """hawk-memory-api recall benchmark。"""

    def __init__(self, platform: str = "benchmark"):
        self.platform = platform
        self.latencies: list[float] = []

    def health_check(self) -> bool:
        data, s = req("GET", "/health")
        return s == 200 and data.get("status") == "ok"

    def capture(self, text: str) -> bool:
        """存入一条记忆。"""
        session = f"bm-{uuid.uuid4().hex[:8]}"
        body = {
            "session_id": session,
            "user_id": "benchmark",
            "message": text,
            "response": "",
            "platform": self.platform,
        }
        data, s = req("POST", "/capture", body)
        return s in (200, 201)

    def recall(self, query: str, top_k: int = 10) -> tuple[list[dict], float]:
        """recall，返回 (memories, latency)。"""
        body = {"query": query, "top_k": top_k, "platform": self.platform}
        t0 = time.perf_counter()
        data, s = req("POST", "/recall", body)
        latency = time.perf_counter() - t0
        self.latencies.append(latency)
        if s != 200:
            return [], latency
        return data.get("memories", []), latency

    def run(self, dataset: list[dict], top_k: int = 10) -> tuple[list[CaseResult], dict]:
        """
        运行 benchmark。

        dataset: list of {id, question, answer/memory_text}
        """
        results: list[CaseResult] = []

        # 1. 预先 capture 所有 target 记忆
        print(f"  [1] Capture {len(dataset)} 条记忆...")
        captured = 0
        for item in dataset:
            text = item.get("answer") or item.get("memory_text") or ""
            if text and self.capture(text):
                captured += 1
            time.sleep(0.05)
        print(f"      已 capture {captured}/{len(dataset)} 条")
        print(f"  [2] 等待索引就绪 (3s)...")
        time.sleep(3)

        # 2. 对每个 query 做 recall
        print(f"  [3] 开始 recall 评测...")
        for i, item in enumerate(dataset):
            qid = item.get("id", f"q-{i}")
            query = item.get("question", "")
            target_text = item.get("answer") or item.get("memory_text", "")

            memories, latency = self.recall(query, top_k=top_k)
            retrieved_texts = [m.get("text", "") for m in memories]

            # 文本匹配找 rank
            rank = None
            for pos, txt in enumerate(retrieved_texts):
                if text_similar(txt, target_text):
                    rank = pos + 1
                    break

            # BLEU/F1（top-1 vs target）
            bleu1 = 0.0
            f1 = 0.0
            if retrieved_texts and target_text:
                tm = compute_text_metrics(retrieved_texts[0], target_text)
                bleu1 = tm.get("bleu1", 0.0)
                f1 = tm.get("f1", 0.0)

            results.append(CaseResult(
                query_id=qid, query=query,
                target_text=target_text,
                retrieved_texts=retrieved_texts,
                rank=rank, latency=latency,
                bleu1=bleu1, f1=f1,
            ))

            if (i + 1) % 10 == 0:
                print(f"      进度: {i+1}/{len(dataset)}")

        # 3. 计算汇总指标
        metrics = compute_recall_metrics([
            {"query_id": r.query_id, "target_id": r.target_text,
             "retrieved_ids": r.retrieved_texts}
            for r in results
        ], k_values=[1, 3, 5, 10])

        # 用文本替代 ID 做 recall 输入
        # compute_recall_metrics 内部用 target_id 匹配 retrieved_ids
        # 由于 retrieved_ids 是文本列表，我们重新用 ranks 计算
        ranks = [r.rank for r in results]
        for k in [1, 3, 5, 10]:
            hit = sum(1 for r in ranks if r is not None and r <= k)
            metrics[f"recall@{k}"] = hit / len(ranks) if ranks else 0.0
            if hit > 0:
                mrr = sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)
                metrics[f"mrr@{k}"] = mrr

        # 文本质量
        metrics["bleu1"] = sum(r.bleu1 for r in results) / len(results)
        metrics["f1"] = sum(r.f1 for r in results) / len(results)

        # 延迟
        if self.latencies:
            sorted_lat = sorted(self.latencies)
            n = len(sorted_lat)
            metrics["latency_p50"] = sorted_lat[int(n * 0.5)]
            metrics["latency_p99"] = sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0]
            metrics["latency_mean"] = sum(sorted_lat) / n

        return results, metrics


# ─── 主程序 ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="hawk-memory-api Recall Benchmark")
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--output", "-o", default="reports/hawk_recall.json")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    print(f"[benchmark] 数据集: {args.dataset} ({len(dataset)} 条)")

    bm = HawkMemoryBenchmark()
    if not bm.health_check():
        print("[benchmark] ✗ hawk-memory-api 不可用 (http://127.0.0.1:18360)")
        print("             请先启动: cd ~/repos/hawk-memory-api && ./run.sh")
        sys.exit(1)
    print("[benchmark] ✓ hawk-memory-api 可用")

    t0 = time.time()
    results, metrics = bm.run(dataset, top_k=args.top_k)
    elapsed = time.time() - t0

    print(f"\n[结果]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "adapter": "hawk_memory_api",
        "dataset": args.dataset,
        "elapsed_seconds": elapsed,
        "n_cases": len(dataset),
        "metrics": metrics,
        "cases": [r.to_dict() for r in results],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {args.output}")

    # 打印 rank > 10 的样例
    missed = [r for r in results if r.rank is None]
    if missed:
        print(f"\n未命中 ({len(missed)} 条):")
        for r in missed[:3]:
            print(f"  Q: {r.query}")
            print(f"  T: {r.target_text[:60]}")
            print(f"  R: {r.retrieved_texts[:2]}")
            print()


if __name__ == "__main__":
    import urllib.request
    import urllib.error
    main()
