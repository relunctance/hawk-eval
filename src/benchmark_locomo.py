#!/usr/bin/env python3
"""
LoCoMo-10 Recall Benchmark

评测逻辑（文本匹配代替 ID）：
1. capture 一条记忆（用 answer text）
2. 等索引就绪
3. 用 question 做 recall
4. 用 top-1 召回结果与 gold answer 计算 BLEU-1 / F1
5. （可选）LLM-Judge 判断 CORRECT / WRONG
6. 按 Multi-hop / Temporal / Open-domain / Single-hop 分类汇总

用法:
    PYTHONPATH=src python -m src.benchmark_locomo \\
        --dataset datasets/locomo/locomo_qa.jsonl \\
        --output reports/locomo_recall.json

环境变量:
    OPENAI_API_KEY   LLM-Judge 使用 GPT-4o-mini（可选）
    HAWK_API_BASE    hawk-memory-api 地址（默认 http://127.0.0.1:18360）
"""

import argparse
import json
import os
import sys
import time
import uuid
import urllib.request
import urllib.error
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# nltk BLEU (与 mflow-benchmarks 完全对齐)
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


# ─── HTTP ───────────────────────────────────────────────────────────────────

HAWK_BASE = os.getenv("HAWK_API_BASE", "http://127.0.0.1:18360")


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


# ─── BLEU-1 / F1（与 mflow-benchmarks metrics.py 完全对齐）──────────────────

def _tokenize(text: str) -> list[str]:
    """简单的 whitespace + punctuation tokenization。"""
    text = str(text).lower()
    text = text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
    return text.split()


def calculate_bleu1(prediction: str, reference: str) -> float:
    """
    计算 BLEU-1，与 mflow-benchmarks/benchmarks/locomo-mflow/scripts/metrics.py 完全对齐。
    使用 nltk word_tokenize + SmoothingFunction.method1。
    """
    if not prediction or not reference:
        return 0.0

    try:
        if _HAS_NLTK:
            pred_tokens = nltk.word_tokenize(prediction.lower())
            ref_tokens = [nltk.word_tokenize(reference.lower())]
        else:
            pred_tokens = prediction.lower().split()
            ref_tokens = [reference.lower().split()]
    except Exception:
        pred_tokens = prediction.lower().split()
        ref_tokens = [reference.lower().split()]

    if len(pred_tokens) == 0:
        return 0.0

    # Smoothing to avoid 0 for short predictions
    from nltk.translate.bleu_score import SmoothingFunction
    smooth = SmoothingFunction().method1

    try:
        score = nltk.translate.bleu_score.sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1, 0, 0, 0),  # BLEU-1 only
            smoothing_function=smooth,
        )
    except Exception:
        score = 0.0

    return score


def calculate_f1(prediction: str, reference: str) -> float:
    """
    计算 word-level F1，与 mflow-benchmarks calculate_f1() 完全对齐。
    """
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── LLM Judge（与 mflow-benchmarks evaluate_llm_judge 完全对齐）──────────────

LLM_JUDGE_PROMPT = """You are evaluating a question-answering system.

Question: {question}

Gold Answer: {gold_answer}

Generated Answer: {generated_answer}

You must respond with a JSON object with a single key "label" and value either "CORRECT" or "WRONG".

- "CORRECT" if the generated answer contains the key information from the gold answer.
- "WRONG" if the generated answer is missing or contradicts the key information.

Respond with only the JSON object."""


def llm_judge(question: str, gold_answer: str, generated_answer: str,
              model: str = "gpt-4o-mini") -> int:
    """
    LLM-Judge 判断，与 mflow-benchmarks evaluate_llm_judge() 完全对齐。
    返回 1 (CORRECT) 或 0 (WRONG)。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return -1  # 无 key 跳过

    try:
        import openai
    except ImportError:
        return -1

    client = openai.OpenAI(api_key=api_key)

    def extract_json(text: str) -> str:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            return match.group()
        return text

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": LLM_JUDGE_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(extract_json(resp.choices[0].message.content))
        label = result.get("label", "WRONG").upper()
        return 1 if label == "CORRECT" else 0
    except Exception as e:
        print(f"LLM Judge error: {e}", file=sys.stderr)
        return -1


# ─── Benchmark ───────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    query_id: str
    category: str
    query: str
    gold_answer: str
    retrieved_texts: list[str]
    rank: int | None  # 1-indexed, None = miss
    latency: float
    bleu1: float = 0.0
    f1: float = 0.0
    llm_correct: int = -1  # 1/0/-1

    def to_dict(self):
        return asdict(self)


def text_similar(t1: str, t2: str, threshold: float = 0.5) -> bool:
    """简单的 token overlap 判断两个 answer 是否匹配。"""
    words1 = set(t1.lower().split())
    words2 = set(t2.lower().split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2)
    return overlap / max(len(words1), len(words2)) >= threshold


class HawkMemoryLocomoBenchmark:
    """LoCoMo-10 recall benchmark for hawk-memory-api."""

    def __init__(self):
        self.latencies: list[float] = []

    def health_check(self) -> bool:
        data, s = hawk_req("GET", "/health")
        return s == 200 and data.get("status") == "ok"

    def capture(self, text: str) -> bool:
        session = f"locomo-{uuid.uuid4().hex[:8]}"
        body = {
            "session_id": session,
            "user_id": "benchmark",
            "message": text,
            "response": "",
            "platform": "locomo-benchmark",
        }
        data, s = hawk_req("POST", "/capture", body)
        return s in (200, 201)

    def recall(self, query: str, top_k: int = 10) -> tuple[list[dict], float]:
        body = {"query": query, "top_k": top_k, "platform": "locomo-benchmark"}
        t0 = time.perf_counter()
        data, s = hawk_req("POST", "/recall", body)
        latency = time.perf_counter() - t0
        self.latencies.append(latency)
        if s != 200:
            return [], latency
        return data.get("memories", []), latency

    def run(self, dataset: list[dict], top_k: int = 10,
            use_llm_judge: bool = False) -> tuple[list[CaseResult], dict]:
        # 1. Capture all memories
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
            qid = item.get("id", f"q-{i}")
            cat = item.get("category", "Unknown")
            query = item.get("question", "")
            gold = item.get("answer", "")

            memories, latency = self.recall(query, top_k=top_k)
            retrieved = [m.get("text", "") for m in memories]

            # Find rank via text similarity
            rank = None
            for pos, txt in enumerate(retrieved):
                if text_similar(txt, gold):
                    rank = pos + 1
                    break

            bleu1 = calculate_bleu1(retrieved[0] if retrieved else "", gold)
            f1 = calculate_f1(retrieved[0] if retrieved else "", gold)

            llm_correct = -1
            if use_llm_judge and retrieved:
                llm_correct = llm_judge(query, gold, retrieved[0])

            return CaseResult(
                query_id=qid, category=cat, query=query,
                gold_answer=gold, retrieved_texts=retrieved,
                rank=rank, latency=latency,
                bleu1=bleu1, f1=f1, llm_correct=llm_correct,
            )

        results: list[CaseResult] = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(do_recall, item, i): i for i, item in enumerate(dataset)}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 50 == 0:
                    print(f"      progress: {done}/{len(dataset)}")

        # Sort by id number
        def sort_key(r: CaseResult) -> int:
            try:
                return int(r.query_id.split("-")[1])
            except Exception:
                return 0
        results.sort(key=sort_key)

        # 3. Aggregate metrics
        metrics = self._aggregate(results)
        print(f"  [4] Done")
        return results, metrics

    def _aggregate(self, results: list[CaseResult]) -> dict:
        """计算整体 + 分类汇总指标。"""
        total = len(results)
        if total == 0:
            return {}

        def safe_avg(values):
            v = [x for x in values if x is not None and x >= 0]
            return sum(v) / len(v) if v else 0.0

        # Overall
        ranks = [r.rank for r in results]
        recall_hits = sum(1 for r in ranks if r is not None and r <= 10)
        mrr = sum(1/r for r in ranks if r is not None) / total

        metrics = {
            "n": total,
            "mrr@10": mrr,
            "recall@10": recall_hits / total,
            "bleu1_avg": safe_avg([r.bleu1 for r in results]),
            "f1_avg": safe_avg([r.f1 for r in results]),
            "latency_avg": safe_avg([r.latency for r in results]),
        }

        # Per-category
        from collections import defaultdict
        by_cat: dict[str, list[CaseResult]] = defaultdict(list)
        for r in results:
            by_cat[r.category].append(r)

        cat_metrics = {}
        for cat, cat_results in sorted(by_cat.items()):
            cat_ranks = [r.rank for r in cat_results]
            cat_hits = sum(1 for r in cat_ranks if r is not None and r <= 10)
            cat_mrr = sum(1/r for r in cat_ranks if r is not None) / len(cat_results)
            cat_metrics[cat] = {
                "n": len(cat_results),
                "mrr@10": cat_mrr,
                "recall@10": cat_hits / len(cat_results),
                "bleu1_avg": safe_avg([r.bleu1 for r in cat_results]),
                "f1_avg": safe_avg([r.f1 for r in cat_results]),
            }

        metrics["per_category"] = cat_metrics

        # LLM Judge summary (if available)
        llm_results = [r for r in results if r.llm_correct >= 0]
        if llm_results:
            llm_acc = sum(r.llm_correct for r in llm_results) / len(llm_results)
            metrics["llm_judge_accuracy"] = llm_acc
            metrics["llm_judge_n"] = len(llm_results)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="LoCoMo-10 recall benchmark for hawk-memory-api")
    parser.add_argument("--dataset", "-d",
                        default="datasets/locomo/locomo_qa.jsonl",
                        help="JSONL dataset path")
    parser.add_argument("--output", "-o",
                        default="reports/locomo_recall.json",
                        help="Output report path")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable LLM-Judge (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    # Load dataset
    dataset = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dataset.append(json.loads(line))
                except Exception:
                    pass

    print(f"[LoCoMo] dataset={args.dataset} n={len(dataset)} top_k={args.top_k}")

    bm = HawkMemoryLocomoBenchmark()
    if not bm.health_check():
        print("❌ hawk-memory-api health check failed")
        sys.exit(1)
    print("✅ hawk-memory-api health OK")

    t0 = time.time()
    results, metrics = bm.run(dataset, top_k=args.top_k, use_llm_judge=args.llm_judge)
    elapsed = time.time() - t0

    # Save report
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

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Overall (n={metrics['n']})")
    print(f"  MRR@10:     {metrics.get('mrr@10', 0):.4f}")
    print(f"  Recall@10:  {metrics.get('recall@10', 0):.1%}")
    print(f"  BLEU-1 avg: {metrics.get('bleu1_avg', 0):.4f}")
    print(f"  F1 avg:     {metrics.get('f1_avg', 0):.4f}")
    print(f"  Latency avg:{metrics.get('latency_avg', 0):.3f}s")
    if "llm_judge_accuracy" in metrics:
        print(f"  LLM-Judge:  {metrics['llm_judge_accuracy']:.1%} ({metrics['llm_judge_n']} cases)")
    print(f"\n  Per-Category:")
    for cat, cm in metrics.get("per_category", {}).items():
        print(f"    {cat:20s} n={cm['n']:3d}  MRR@10={cm['mrr@10']:.4f}  "
              f"Recall@10={cm['recall@10']:.1%}  BLEU1={cm['bleu1_avg']:.4f}")
    print(f"{'='*55}")
    print(f"\nReport saved: {args.output}")

    # Comparison with baselines
    print(f"\n[Baseline Comparison]")
    print(f"  M-flow (LLM-Judge):    81.8%")
    print(f"  Cognee (LLM-Judge):    79.4%")
    print(f"  Zep Cloud (LLM-Judge): 73.4%")
    if "llm_judge_accuracy" in metrics:
        print(f"  hawk-memory-api:       {metrics['llm_judge_accuracy']:.1%}")


if __name__ == "__main__":
    main()
