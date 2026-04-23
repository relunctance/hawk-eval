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
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from metrics import compute_recall_metrics, compute_text_metrics


# ─── HTTP ───────────────────────────────────────────────────────────────────

BASE = "http://127.0.0.1:18360"


def req(method, path, body=None, timeout=30):
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

def _strip_prefix(t: str) -> str:
    """去掉 capture 存储格式的前缀，只保留核心内容（answer）。

    所有格式统一处理：取最后一个换行之后的内容，再去掉角色前缀。
    - "用户: question\n助手: answer" → "助手: answer" → "answer"
    - "question\nanswer" → "answer"
    - "助手: answer" → "answer"
    - "用户: 用户: question\n助手: answer" → "answer"
    """
    # 取最后一个换行之后的内容（去掉 question 行）
    if "\n" in t:
        t = t.split("\n")[-1]
    # 去掉角色前缀
    for p in ("用户: ", "助手: ", "User: ", "Assistant: "):
        if t.startswith(p):
            t = t[len(p):]
    return t.strip()


def text_similar(t1: str, t2: str, threshold: float = 0.6) -> bool:
    """判断两个文本是否相似（token overlap > threshold）。capture 前缀会在比对前自动去掉。"""
    t1 = _strip_prefix(t1)
    t2 = _strip_prefix(t2)
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

    def capture_qa(self, question: str, answer: str) -> bool:
        """存入一组 QA 记忆（question → message, answer → response，建立问答关联）。"""
        session = f"bm-{uuid.uuid4().hex[:8]}"
        body = {
            "session_id": session,
            "user_id": "benchmark",
            "message": question,
            "response": answer,
            "platform": self.platform,
        }
        data, s = req("POST", "/capture", body)
        return s in (200, 201)

    def capture(self, text: str) -> bool:
        """存入一条记忆（兼容旧接口）。"""
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

    def run(self, dataset: list[dict], top_k: int = 10,
             log_fn=None) -> tuple[list[CaseResult], dict]:
        """
        运行 benchmark。

        dataset: list of {id, question, answer/memory_text}
        log_fn: 日志函数，默认 print
        """
        if log_fn is None:
            log_fn = print

        # 1. 预先 capture 所有 target 记忆（并发=1，避免 API 过载）
        log_fn(f"  [1] Capture {len(dataset)} 条记忆...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def do_capture(item: dict) -> bool:
            question = item.get("question", "")
            answer = item.get("answer") or item.get("memory_text") or ""
            return bool(question and answer and self.capture_qa(question, answer))

        with ThreadPoolExecutor(max_workers=1) as ex:
            caps = list(ex.map(do_capture, dataset))
        captured = sum(caps)
        log_fn(f"      已 capture {captured}/{len(dataset)} 条")
        log_fn(f"  [2] 等待索引就绪 (3s)...")
        time.sleep(3)

        # 2. 对每个 query 做 recall（并发=1，每批<30s）
        log_fn(f"  [3] 开始 recall 评测（并发=1）...")

        def do_recall(item: dict, i: int) -> CaseResult:
            qid = item.get("id", f"q-{i}")
            query = item.get("question", "")
            target_text = item.get("answer") or item.get("memory_text", "")
            memories, latency = self.recall(query, top_k=top_k)
            retrieved_texts = [m.get("text", "") for m in memories]
            rank = None
            for pos, txt in enumerate(retrieved_texts):
                if text_similar(txt, target_text):
                    rank = pos + 1
                    break
            bleu1 = 0.0
            f1 = 0.0
            if retrieved_texts and target_text:
                tm = compute_text_metrics(retrieved_texts[0], target_text)
                bleu1 = tm.get("bleu1", 0.0)
                f1 = tm.get("f1", 0.0)
            return CaseResult(query_id=qid, query=query, target_text=target_text,
                              retrieved_texts=retrieved_texts, rank=rank,
                              latency=latency, bleu1=bleu1, f1=f1)

        results: list[CaseResult] = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(do_recall, item, i): i
                       for i, item in enumerate(dataset)}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 20 == 0:
                    log_fn(f"      进度: {done}/{len(dataset)}")

        # 保证顺序
        results.sort(key=lambda r: int(r.query_id.split("-")[1]) if "-" in r.query_id else 0)

        # 3. 计算汇总指标（retrieved_texts 有 "用户: " / "助手: " 前缀，需要去掉才能匹配 target_text）
        def _strip(t: str) -> str:
            for p in ("用户: ", "助手: "):
                if t.startswith(p):
                    return t[len(p):]
            return t

        metrics = compute_recall_metrics([
            {"query_id": r.query_id, "target_id": r.target_text,
             "retrieved_ids": [_strip(t) for t in r.retrieved_texts]}
            for r in results
        ], k_values=[1, 3, 5, 10])

        # 添加文本指标汇总
        metrics["bleu1_avg"] = sum(r.bleu1 for r in results) / len(results) if results else 0
        metrics["f1_avg"] = sum(r.f1 for r in results) / len(results) if results else 0
        metrics["latency_avg"] = sum(r.latency for r in results) / len(results) if results else 0
        metrics["latency_p50"] = sorted(r.latency for r in results)[len(results)//2] if results else 0

        log_fn(f"  [4] 完成")
        return results, metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="hawk-memory-api recall benchmark")
    parser.add_argument("--dataset", default="datasets/hawk_memory/conversational_qa.jsonl",
                       help="JSONL dataset path")
    parser.add_argument("--output", default="reports/hawk_recall.json",
                       help="Output report JSON path")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0,
                       help="只跑前N条（0=全部，用于快速迭代验证）")
    parser.add_argument("--offset", type=int, default=0,
                       help="从第几条开始跳过（0=从头，用于分批跑）")
    parser.add_argument("--host", default="http://127.0.0.1:18360")
    parser.add_argument("--log", default=None,
                       help="日志文件路径（追加），不指定则只输出到 stdout")
    args = parser.parse_args()

    # 设置日志输出（同时写文件 + stdout）
    import logging, sys
    log_handler = None
    if args.log:
        log_handler = logging.FileHandler(args.log, mode="a")
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.addHandler(log_handler)
        root.setLevel(logging.INFO)
        # 把 print 重定向到日志
        def log_print(*a, **kw):
            sep = kw.pop("sep", " ")
            msg = sep.join(str(x) for x in a)
            logging.info(msg)
            print(*a, **kw)
    else:
        log_print = print

    import json
    from pathlib import Path

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

    if args.offset > 0:
        dataset = dataset[args.offset:]
    if args.limit > 0:
        dataset = dataset[:args.limit]

    log_print(f"[benchmark] 数据集: {args.dataset} ({len(dataset)} 条)")

    # Run
    bm = HawkMemoryBenchmark()
    bm.host = args.host

    if not bm.health_check():
        log_print("❌ hawk-memory-api health check failed")
        if log_handler:
            log_handler.close()
        return

    log_print("✅ hawk-memory-api health OK")
    results, metrics = bm.run(dataset, top_k=args.top_k, log_fn=log_print)

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "count": len(results),
        "metrics": metrics,
        "cases": [r.to_dict() for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    log_print(f"\n{'='*50}")
    log_print(f"  MRR@1:  {metrics.get('mrr@1', 0):.3f}")
    log_print(f"  MRR@3:  {metrics.get('mrr@3', 0):.3f}")
    log_print(f"  MRR@5:  {metrics.get('mrr@5', 0):.3f}")
    log_print(f"  MRR@10: {metrics.get('mrr@10', 0):.3f}")
    log_print(f"  Recall@1:  {metrics.get('recall@1', 0):.1%}")
    log_print(f"  Recall@3:  {metrics.get('recall@3', 0):.1%}")
    log_print(f"  Recall@5:  {metrics.get('recall@5', 0):.1%}")
    log_print(f"  Recall@10: {metrics.get('recall@10', 0):.1%}")
    log_print(f"  BLEU-1 avg: {metrics.get('bleu1_avg', 0):.3f}")
    log_print(f"  F1 avg:      {metrics.get('f1_avg', 0):.3f}")
    log_print(f"  Latency avg: {metrics.get('latency_avg', 0):.3f}s")
    log_print(f"  Latency P50: {metrics.get('latency_p50', 0):.3f}s")
    log_print(f"{'='*50}")
    log_print(f"\n报告已保存: {args.output}")

    if log_handler:
        log_handler.close()
        logging.getLogger().removeHandler(log_handler)

if __name__ == "__main__":
    main()
