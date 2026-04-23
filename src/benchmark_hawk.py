#!/usr/bin/env python3
"""
hawk-memory-api Recall Benchmark

评测逻辑（文本匹配代替 ID 匹配）：
1. capture 一条记忆（用 answer text）
2. 等索引就绪
3. 用 question 做 recall
4. 检查 target answer 是否在返回结果中（文本相似度匹配）
5. 计算 MRR / Recall@K / BLEU / F1 / Latency

两阶段用法（推荐，用于快速迭代）：
  # Phase 1: capture 数据集（只跑一次）
  python -m src.benchmark_hawk --mode capture --dataset ... --session-file sessions/my.jsonl

  # Phase 2: recall 评测（可跑多次，修改 top-k 等参数）
  python -m src.benchmark_hawk --mode recall --dataset ... --session-file sessions/my.jsonl --top-k 5

  # 合并模式（保留原有行为）
  python -m src.benchmark_hawk --mode both --dataset ... --output reports/...
"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import httpx

from metrics import compute_recall_metrics, compute_text_metrics

# ─── Embedding Precompute ───────────────────────────────────────────────────

# xinference endpoint (same as hawk-memory-api uses)
_XINFERENCE_BASE = "http://127.0.0.1:9997/v1"
_XINFERENCE_MODEL = "bge-m3"

# SQLite cache path
_CACHE_DB = Path.home() / ".hermes" / "hawk_eval_cache.db"


def _dataset_hash(dataset: list[dict]) -> str:
    """Dataset fingerprint — includes all question/answer pairs."""
    canonical = json.dumps(
        [{"id": d.get("id", ""), "question": d.get("question", ""), "answer": d.get("answer", "") or d.get("memory_text", "")}
         for d in dataset],
        sort_keys=True,
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


class EmbedCache:
    """
    SQLite-backed embedding cache.
    Schema:
      cache(dataset_hash TEXT, query_id TEXT, query_text TEXT,
            embedding BLOB, created_at REAL, PRIMARY KEY(dataset_hash, query_id))
    """

    def __init__(self, db_path: Path = _CACHE_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embed_cache (
                    dataset_hash TEXT,
                    query_id    TEXT,
                    query_text  TEXT,
                    embedding   BLOB,
                    created_at  REAL,
                    PRIMARY KEY (dataset_hash, query_id)
                )
            """)

    def get(self, dataset_hash: str, query_id: str) -> list[float] | None:
        """Return embedding list[float] or None if not cached."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embed_cache WHERE dataset_hash=? AND query_id=?",
                (dataset_hash, query_id)
            ).fetchone()
        if row:
            import pickle
            return pickle.loads(row[0])
        return None

    def get_batch(self, dataset_hash: str, query_ids: list[str]) -> dict[str, list[float] | None]:
        """Return {query_id: embedding or None} for the given ids."""
        placeholders = ",".join("?" * len(query_ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT query_id, embedding FROM embed_cache WHERE dataset_hash=? AND query_id IN ({placeholders})",
                [dataset_hash] + query_ids
            ).fetchall()
        result = {qid: None for qid in query_ids}
        import pickle
        for qid, emb in rows:
            result[qid] = pickle.loads(emb)
        return result

    def put(self, dataset_hash: str, query_id: str, query_text: str, embedding: list[float]):
        """Store one embedding."""
        import pickle
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embed_cache
                    (dataset_hash, query_id, query_text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (dataset_hash, query_id, query_text, pickle.dumps(embedding), time.time()))

    def put_batch(self, dataset_hash: str, items: list[dict]):
        """Store multiple embeddings. items = [{query_id, query_text, embedding}, ...]"""
        import pickle
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO embed_cache
                    (dataset_hash, query_id, query_text, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, [(dataset_hash, it["query_id"], it["query_text"],
                   pickle.dumps(it["embedding"]), now) for it in items])

    def has_all(self, dataset_hash: str, query_ids: list[str]) -> bool:
        """Check whether all query_ids are cached for this dataset_hash."""
        placeholders = ",".join("?" * len(query_ids))
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM embed_cache WHERE dataset_hash=? AND query_id IN ({placeholders})",
                [dataset_hash] + query_ids
            ).fetchone()[0]
        return count == len(query_ids)

    def clear(self, dataset_hash: str):
        """Delete all entries for a dataset_hash."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM embed_cache WHERE dataset_hash=?", (dataset_hash,))


async def precompute_query_embeddings(
    queries: list[str],
    dataset_hash: str,
    query_ids: list[str],
    log_fn=None,
) -> list[list[float] | None]:
    """
    Pre-compute embeddings for a list of query strings using xinference directly.
    Uses SQLite cache — skips already-cached queries.
    Returns list of embedding vectors (None for skipped/already-cached).
    """
    if log_fn is None:
        log_fn = lambda *a, **k: None

    cache = EmbedCache()

    # Check which are already cached
    cached = cache.get_batch(dataset_hash, query_ids)
    missing_indices = [i for i, qid in enumerate(query_ids) if cached.get(qid) is None]
    missing_ids = [query_ids[i] for i in missing_indices]

    log_fn(f"  [embed] {len(queries)} 条 query，SQLite 缓存命中 {len(queries)-len(missing_ids)}/{len(queries)}")

    results: list[list[float] | None] = [None] * len(queries)

    # Fill in cached values
    for i, qid in enumerate(query_ids):
        if cached.get(qid) is not None:
            results[i] = cached[qid]

    if missing_ids:
        log_fn(f"  [embed] 需预计算 {len(missing_ids)} 条 embedding...")

        async def _embed_one(text: str, qid: str, idx: int) -> tuple[int, list[float]]:
            async with httpx.AsyncClient(timeout=30.0) as client:
                t0 = time.time()
                resp = await client.post(
                    f"{_XINFERENCE_BASE}/embeddings",
                    json={"model": _XINFERENCE_MODEL, "input": text},
                )
                vec = resp.json()["data"][0]["embedding"]
                log_fn(f"      [{idx+1}/{len(missing_ids)}] {text[:20]:20s} {time.time()-t0:.2f}s")
                return idx, vec

        tasks = [_embed_one(queries[i], query_ids[i], j) for j, i in enumerate(missing_indices)]
        pending = {j: query_ids[i] for j, i in enumerate(missing_indices)}
        store_items = []

        for coro in asyncio.as_completed(tasks):
            idx, vec = await coro
            orig_i = missing_indices[idx]
            results[orig_i] = vec
            store_items.append({
                "query_id": pending[idx],
                "query_text": queries[orig_i],
                "embedding": vec,
            })

        # Batch persist to SQLite
        cache.put_batch(dataset_hash, store_items)
        log_fn(f"      已写入 SQLite {len(store_items)} 条")

    return results


# ─── HTTP ───────────────────────────────────────────────────────────────────

BASE = "http://127.0.0.1:18360"  # default for HawkMemoryBenchmark.__init__


def req(base_url, method, path, body=None, timeout=30):
    url = base_url.rstrip("/") + path
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

    def __init__(self, platform: str = "benchmark", base_url: str = BASE):
        self.platform = platform
        self.base_url = base_url
        self.latencies: list[float] = []

    def health_check(self) -> bool:
        data, s = req(self.base_url, "GET", "/health")
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
        data, s = req(self.base_url, "POST", "/capture", body)
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
        data, s = req(self.base_url, "POST", "/capture", body)
        return s in (200, 201)

    def recall(self, query: str, top_k: int = 10,
               query_vector: list[float] = None) -> tuple[list[dict], float]:
        """recall，返回 (memories, latency)。query_vector 有则跳过 embed。"""
        body = {"query": query, "top_k": top_k, "platform": self.platform}
        if query_vector is not None:
            body["query_vector"] = query_vector
        t0 = time.perf_counter()
        data, s = req(self.base_url, "POST", "/recall", body)
        latency = time.perf_counter() - t0
        self.latencies.append(latency)
        if s != 200:
            return [], latency
        return data.get("memories", []), latency

    def capture_dataset(self, dataset: list[dict], log_fn=None) -> str:
        """Phase 1: capture 数据集，返回 dataset_hash（embed 已写入 SQLite）。

        dataset_hash 用于 recall 阶段加载 embeddings。
        """
        if log_fn is None:
            log_fn = print

        # Build items before capture (needed for dataset_hash)
        items = [
            {"id": item.get("id", f"q-{i}"),
             "question": item.get("question", ""),
             "answer": item.get("answer") or item.get("memory_text", "")}
            for i, item in enumerate(dataset)
        ]
        dataset_hash = _dataset_hash(items)

        log_fn(f"  [capture] {len(dataset)} 条记忆...")
        ok = 0
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            answer = item.get("answer") or item.get("memory_text") or ""
            if question and answer:
                self.capture_qa(question, answer)
                ok += 1
            if (i + 1) % 20 == 0:
                log_fn(f"      进度: {i+1}/{len(dataset)}")
        log_fn(f"      已 capture {ok}/{len(dataset)} 条")
        log_fn(f"  [capture] 等待索引就绪 (3s)...")
        time.sleep(3)

        # 预计算 query embeddings（写入 SQLite）
        queries = [item.get("question", "") for item in dataset if item.get("question")]
        query_ids = [item.get("id", f"q-{i}") for i, item in enumerate(dataset) if item.get("question")]
        asyncio.run(precompute_query_embeddings(queries, dataset_hash, query_ids, log_fn=log_fn))

        return dataset_hash

    def recall_eval(self, dataset: list[dict], top_k: int = 10,
                    embeddings: list[list[float]] = None,
                    log_fn=None) -> tuple[list[CaseResult], dict]:
        """Phase 2: 只跑 recall，不 capture。embeddings 有则预计算好，直接传给 API。"""
        if log_fn is None:
            log_fn = print

        log_fn(f"  [recall] 评测 {len(dataset)} 条...")

        def do_recall(item: dict, i: int) -> CaseResult:
            qid = item.get("id", f"q-{i}")
            query = item.get("question", "")
            target_text = item.get("answer") or item.get("memory_text", "")
            query_vec = embeddings[i] if (embeddings and i < len(embeddings)) else None
            memories, latency = self.recall(query, top_k=top_k, query_vector=query_vec)
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
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(do_recall, item, i): i
                       for i, item in enumerate(dataset)}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 20 == 0:
                    log_fn(f"      进度: {done}/{len(dataset)}")

        # 保证顺序
        results.sort(key=lambda r: int(r.query_id.split("-")[1])
                     if "-" in r.query_id else 0)

        # 计算汇总指标
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

        metrics["bleu1_avg"] = sum(r.bleu1 for r in results) / len(results) if results else 0
        metrics["f1_avg"] = sum(r.f1 for r in results) / len(results) if results else 0
        metrics["latency_avg"] = sum(r.latency for r in results) / len(results) if results else 0
        metrics["latency_p50"] = sorted(r.latency for r in results)[len(results)//2] if results else 0

        return results, metrics

    def run(self, dataset: list[dict], top_k: int = 10,
             log_fn=None) -> tuple[list[CaseResult], dict]:
        """
        运行 benchmark（合并模式，等同于 capture + recall）。
        """
        if log_fn is None:
            log_fn = print

        log_fn(f"  [1] Capture {len(dataset)} 条记忆...")

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

        return self.recall_eval(dataset, top_k=top_k, log_fn=log_fn)


def main():
    parser = argparse.ArgumentParser(description="hawk-memory-api recall benchmark")
    parser.add_argument("--dataset", default="datasets/hawk_memory/conversational_qa.jsonl",
                       help="JSONL dataset path")
    parser.add_argument("--output", default="reports/hawk_recall.json",
                       help="Output report JSON path")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0,
                       help="只跑前N条（0=全部）")
    parser.add_argument("--offset", type=int, default=0,
                       help="从第几条开始跳过（0=从头）")
    parser.add_argument("--mode", default="both",
                       choices=["capture", "recall", "both"],
                       help="capture=只 capture 数据集；recall=只评测 recall；both=两者都做（默认）")
    parser.add_argument("--host", default="http://127.0.0.1:18360",
                       help="hawk-memory-api base URL")

    args = parser.parse_args()

    log_print = print

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

    # Compute dataset_hash (used for SQLite embedding cache key)
    items = [
        {"id": d.get("id", f"q-{i}"),
         "question": d.get("question", ""),
         "answer": d.get("answer") or d.get("memory_text", "")}
        for i, d in enumerate(dataset)
    ]
    dataset_hash = _dataset_hash(items)

    log_print(f"[benchmark] 数据集: {args.dataset} ({len(dataset)} 条)  mode={args.mode}")

    bm = HawkMemoryBenchmark(base_url=args.host)

    if not bm.health_check():
        log_print("❌ hawk-memory-api health check failed")
        return

    log_print("✅ hawk-memory-api health OK")

    results = []
    metrics = {}

    if args.mode in ("capture", "both"):
        bm.capture_dataset(dataset, log_fn=log_print)

    if args.mode in ("recall", "both"):
        # Load embeddings from SQLite cache
        cache = EmbedCache()
        query_ids = [item.get("id", f"q-{i}") for i, item in enumerate(dataset)]
        cached_emb = cache.get_batch(dataset_hash, query_ids)
        embeddings = [cached_emb.get(qid) for qid in query_ids]

        hit = sum(1 for e in embeddings if e is not None)
        log_print(f"[recall] SQLite embedding 缓存命中 {hit}/{len(embeddings)}")
        if hit < len(embeddings):
            # Re-compute missing embeddings
            missing = [i for i, e in enumerate(embeddings) if e is None]
            log_print(f"[recall] 缺失 {len(missing)} 条，重新预计算...")
            missing_queries = [dataset[i].get("question", "") for i in missing]
            missing_ids = [query_ids[i] for i in missing]
            import asyncio as _asyncio
            new_embs = _asyncio.run(
                precompute_query_embeddings(missing_queries, dataset_hash, missing_ids, log_fn=log_print)
            )
            for mi, idx in enumerate(missing):
                embeddings[idx] = new_embs[mi]
            cache.put_batch(dataset_hash, [
                {"query_id": missing_ids[mi], "query_text": missing_queries[mi], "embedding": new_embs[mi]}
                for mi in range(len(missing))
            ])

        results, metrics = bm.recall_eval(dataset, top_k=args.top_k,
                                          embeddings=embeddings, log_fn=log_print)

    if not results and metrics:
        log_print("\n（capture-only 模式，无评测结果）")
        return

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "mode": args.mode,
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


if __name__ == "__main__":
    main()
