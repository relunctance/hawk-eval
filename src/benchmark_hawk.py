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
import json
import sys
import hashlib
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from metrics import compute_recall_metrics, compute_text_metrics


# ─── HTTP ───────────────────────────────────────────────────────────────────

BASE = "http://127.0.0.1:18360"

DEFAULT_CACHE = "data/query_embeddings_cache.json"


def load_cache(dataset_path: str) -> dict | None:
    """检测并加载预计算缓存。"""
    cache_file = Path(DEFAULT_CACHE)
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        cache = json.load(f)
    fp = dataset_fingerprint(dataset_path)
    if cache.get("fingerprint") != fp:
        return None
    return cache


def dataset_fingerprint(dataset_path: str) -> str:
    """对数据集内容做 hash，用于判断缓存是否过期。"""
    with open(dataset_path) as f:
        content = f.read()
    return hashlib.sha256(content.encode()).hexdigest()[:12]


EMBED_URL = "http://127.0.0.1:9997/v1/embeddings"
EMBED_MODEL = "bge-m3"


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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量计算文本 embedding（直接调 xinference，不走 hawk-memory-api）。"""
    body = {"model": EMBED_MODEL, "input": texts}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    req_ = urllib.request.Request(EMBED_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req_, timeout=30) as r:
            result = json.loads(r.read())
    except Exception as e:
        print(f"  [embedding] ERROR: {e}")
        return []
    return [item["embedding"] for item in result.get("data", [])]


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
    match_method: str = "none"  # "memory_id" | "text" | "none"

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

    def capture_qa(self, question: str, answer: str, retries: int = 2) -> bool:
        """存入一组 QA 记忆（question → message, answer → response，建立问答关联）。"""
        session = f"bm-{uuid.uuid4().hex[:8]}"
        body = {
            "session_id": session,
            "user_id": "benchmark",
            "message": question,
            "response": answer,
            "platform": self.platform,
        }
        for attempt in range(retries + 1):
            data, s = req("POST", "/capture", body)
            if s in (200, 201):
                return True
            if attempt < retries:
                time.sleep(1)
        return False

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

    def capture_batch(self, items: list[dict], batch_size: int = 50, max_workers: int = 4) -> tuple[int, int]:
        """
        批量 capture QA 对，使用 /capture/batch 端点（server-side batch embedding + parallel LLM extraction）。

        相比逐条 /capture：
        - 1 个 HTTP 请求携带多个 items
        - Server 端并行 LLM extraction（asyncio.gather）
        - Server 端批量 embedding（embed_texts 一次请求 xinference）
        - 大幅减少 HTTP RTT 开销

        Args:
            items: list of dicts with 'question' and 'answer' (or 'memory_text') keys
            batch_size: items per batch request (default 50)
            max_workers: concurrent batch requests (default 4)

        Returns:
            (success_count, total_count)
        """
        # Build batch items (filter out empty question/answer)
        batch_items = []
        for item in items:
            question = item.get("question", "")
            answer = item.get("answer") or item.get("memory_text") or ""
            if question and answer:
                batch_items.append({
                    "session_id": f"bm-{uuid.uuid4().hex[:8]}",
                    "user_id": "benchmark",
                    "message": question,
                    "response": answer,
                    "platform": self.platform,
                })

        if not batch_items:
            return 0, len(items)

        total = len(batch_items)
        captured = 0

        def send_batch(batch: list[dict]) -> int:
            body = {"items": batch}
            data, s = req("POST", "/capture/batch", body)
            if s in (200, 201):
                return data.get("stored", 0)
            return 0

        # Split into batches and process concurrently
        batches = [batch_items[i:i + batch_size] for i in range(0, len(batch_items), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(send_batch, batch): len(batch) for batch in batches}
            for future in as_completed(futures):
                try:
                    captured += future.result()
                except Exception:
                    pass

        return captured, total

    def direct_capture_batch(self, items: list[dict], batch_size: int = 50, max_workers: int = 4) -> tuple[int, int]:
        """
        批量 direct capture，使用 /direct_capture 端点（完全绕过 LLM extraction）。

        用途：benchmark / 评测专用 — 直接存储 ground-truth，测 recall 能力。
        不走 LLM extraction，所以极快（只算 embedding）。

        Args:
            items: list of dicts with 'question' and 'answer' (or 'memory_text') keys
            batch_size: items per batch request (default 50)
            max_workers: concurrent HTTP requests (default 4)

        Returns:
            (success_count, total_count, memory_ids)
            memory_ids: list of all memory_ids returned by server (aligned with input order)
        """
        # Build DirectCaptureItem list (ground truth, no extraction)
        memories = []
        for item in items:
            answer = item.get("answer") or item.get("memory_text") or ""
            if not answer:
                continue
            memories.append({
                "text": answer,
                "category": "other",
                "importance": 1.0,
                "name": "",
                "description": "",
                "metadata": {
                    "question": item.get("question", ""),
                },
            })

        if not memories:
            return 0, len(items), []

        total = len(memories)

        def send_batch(batch: list[dict]) -> tuple[int, list[str]]:
            body = {
                "memories": batch,
                "session_id": f"bm-{uuid.uuid4().hex[:8]}",
                "platform": self.platform,
                "agent_id": "eval",
            }
            data, s = req("POST", "/direct_capture", body)
            if s in (200, 201):
                ids = data.get("memory_ids", [])
                return data.get("stored", 0), ids
            return 0, []

        # Split into batches and process concurrently
        batches = [memories[i:i + batch_size] for i in range(0, len(memories), batch_size)]

        stored = 0
        all_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(send_batch, batch): len(batch) for batch in batches}
            for future in as_completed(futures):
                try:
                    cnt, ids = future.result()
                    stored += cnt
                    all_ids.extend(ids)
                except Exception:
                    pass

        return stored, total, all_ids

    def direct_capture_batch_with_ids(self, items: list[dict], batch_size: int = 50, max_workers: int = 4) -> tuple[int, int, list[str]]:
        """
        批量 direct capture，返回每条记忆的 memory_id（按原 items 顺序对齐）。

        与 direct_capture_batch 的区别：返回的 memory_ids 列表与输入 items 一一对应，
        方便 hawk-eval 用 memory_id 做 ground-truth 评测。

        Returns:
            (success_count, total_count, memory_ids)
            memory_ids: list of memory_ids，顺序与原 items 对齐（answer 为空的项对应空字符串）
        """
        # Build memories in parallel with tracking original indices
        memories_by_idx: list[tuple[int, dict]] = []  # (original_idx, memory_dict)
        for i, item in enumerate(items):
            answer = item.get("answer") or item.get("memory_text") or ""
            if not answer:
                continue
            memories_by_idx.append((i, {
                "text": answer,
                "category": "other",
                "importance": 1.0,
                "name": "",
                "description": "",
                "metadata": {
                    "question": item.get("question", ""),
                },
            }))

        if not memories_by_idx:
            return 0, len(items), [""] * len(items)

        total = len(items)

        def send_batch(batch: list[tuple[int, dict]]) -> tuple[int, list[tuple[int, str]]]:
            """Returns (stored_count, list of (original_idx, memory_id))"""
            memories = [m for _, m in batch]
            body = {
                "memories": memories,
                "session_id": f"bm-{uuid.uuid4().hex[:8]}",
                "platform": self.platform,
                "agent_id": "eval",
            }
            data, s = req("POST", "/direct_capture", body)
            if s in (200, 201):
                ids = data.get("memory_ids", [])
                # Pair original indices with returned IDs (aligned with batch order)
                return data.get("stored", 0), list(zip([idx for idx, _ in batch], ids))
            return 0, []

        # Split into batches
        batches = [memories_by_idx[i:i + batch_size] for i in range(0, len(memories_by_idx), batch_size)]

        stored = 0
        # index_to_memory_id: original_idx -> memory_id
        index_to_memory_id: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(send_batch, batch): len(batch) for batch in batches}
            for future in as_completed(futures):
                try:
                    cnt, idx_id_pairs = future.result()
                    stored += cnt
                    for orig_idx, mem_id in idx_id_pairs:
                        index_to_memory_id[orig_idx] = mem_id
                except Exception:
                    pass

        # Build aligned memory_ids list (empty string for skipped items)
        memory_ids = [index_to_memory_id.get(i, "") for i in range(total)]
        return stored, total, memory_ids

    def recall(self, query: str, top_k: int = 10,
               query_vector: list[float] | None = None,
               rewrite: bool = False,
               agent_id: str = "eval") -> tuple[list[dict], float]:
        """recall，返回 (memories, latency)。可通过 query_vector 跳过 embedding 计算。

        Args:
            rewrite: if True, use LLM query rewriting before search (KR2.4)
            agent_id: filter by agent namespace (default "eval")
        """
        body = {"query": query, "top_k": top_k, "mode": "platform_only",
                "platform": self.platform, "agent_id": agent_id, "rewrite": rewrite}
        if query_vector is not None:
            body["query_vector"] = query_vector
        t0 = time.perf_counter()
        data, s = req("POST", "/recall", body)
        latency = time.perf_counter() - t0
        self.latencies.append(latency)
        if s != 200:
            return [], latency
        return data.get("memories", []), latency

    def capture_dataset(self, dataset: list[dict], log_fn=None, use_llm: bool = False) -> dict:
        """Phase 1: capture 数据集 + 预计算 question embedding，返回 session 文件内容供 recall 使用。

        Returns: {
            "platform": str,
            "count": int,
            "items": [{
                "id": str,           # dataset original id
                "memory_id": str,    # hawk-memory-api memory_id (for exact rank matching)
                "question": str,
                "answer": str,
                "query_vector": list[float],
            }, ...]
        }
        """
        if log_fn is None:
            log_fn = print

        mode_str = "LLM extraction" if use_llm else "direct (no LLM)"
        log_fn(f"  [capture] {len(dataset)} 条记忆（{mode_str}）...")

        # 1) capture (use LLM or not based on flag)
        t0 = time.time()
        if use_llm:
            ok, total = self.capture_batch(dataset, batch_size=50, max_workers=4)
            memory_ids = []  # LLM capture doesn't return per-item IDs yet
        else:
            ok, total, memory_ids = self.direct_capture_batch_with_ids(dataset, batch_size=50, max_workers=4)
        elapsed = time.time() - t0
        log_fn(f"      已 capture {ok}/{total} 条 ({elapsed:.1f}s)")

        # 2) 预计算所有 question embedding（不占用 recall 路径）
        log_fn(f"  [embedding] 预计算 {len(dataset)} 条 question 向量...")
        questions = [
            item.get("question", "") for item in dataset
        ]
        vectors = embed_texts(questions)
        log_fn(f"      向量计算完成 ({len(vectors)}/{len(questions)})")

        # 3) 等待索引就绪
        log_fn(f"  [capture] 等待索引就绪 (3s)...")
        time.sleep(3)

        return {
            "platform": self.platform,
            "count": ok,
            "items": [
                {
                    "id": item.get("id", f"q-{i}"),
                    "memory_id": memory_ids[i] if i < len(memory_ids) else "",
                    "question": item.get("question", ""),
                    "answer": item.get("answer") or item.get("memory_text", ""),
                    "query_vector": vectors[i] if i < len(vectors) else None,
                }
                for i, item in enumerate(dataset)
            ],
        }

    def recall_eval(self, dataset: list[dict], top_k: int = 10,
                    log_fn=None, rewrite: bool = False,
                    agent_id: str = "eval") -> tuple[list[CaseResult], dict]:
        """Phase 2: 只跑 recall，不 capture。用 session 中的预计算 query_vector。"""
        if log_fn is None:
            log_fn = print

        log_fn(f"  [recall] 评测 {len(dataset)} 条（query_vector 预计算模式）...")

        def do_recall(item: dict, i: int) -> CaseResult:
            qid = item.get("id", f"q-{i}")
            query = item.get("question", "")
            query_vector = item.get("query_vector")
            target_memory_id = item.get("memory_id", "")
            target_text = item.get("answer") or item.get("memory_text", "")
            memories, latency = self.recall(query, top_k=top_k, query_vector=query_vector, rewrite=rewrite, agent_id=agent_id)

            # Primary: exact memory_id match (precise, no ambiguity)
            rank = None
            match_method = "none"
            if target_memory_id:
                for pos, m in enumerate(memories):
                    if m.get("id") == target_memory_id:
                        rank = pos + 1
                        match_method = "memory_id"
                        break

            # Fallback: text similarity (for sessions created before memory_id was added)
            if rank is None and target_text:
                retrieved_texts = [m.get("text", "") for m in memories]
                for pos, txt in enumerate(retrieved_texts):
                    if text_similar(txt, target_text):
                        rank = pos + 1
                        match_method = "text"
                        break

            retrieved_texts = [m.get("text", "") for m in memories]
            bleu1 = 0.0
            f1 = 0.0
            if retrieved_texts and target_text:
                tm = compute_text_metrics(retrieved_texts[0], target_text)
                bleu1 = tm.get("bleu1", 0.0)
                f1 = tm.get("f1", 0.0)
            return CaseResult(query_id=qid, query=query, target_text=target_text,
                              retrieved_texts=retrieved_texts, rank=rank,
                              latency=latency, bleu1=bleu1, f1=f1,
                              match_method=match_method)

        results: list[CaseResult] = []
        with ThreadPoolExecutor(max_workers=3) as executor:
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
        # NOTE: must use _strip_prefix (module-level, handles \n-split + role prefix)
        # not the local _strip above (only handles role prefix → would cause
        # compute_recall_metrics exact-match to fail)
        metrics = compute_recall_metrics([
            {"query_id": r.query_id, "target_id": _strip_prefix(r.target_text),
             "retrieved_ids": [_strip_prefix(t) for t in r.retrieved_texts]}
            for r in results
        ], k_values=[1, 3, 5, 10])

        metrics["bleu1_avg"] = sum(r.bleu1 for r in results) / len(results) if results else 0
        metrics["f1_avg"] = sum(r.f1 for r in results) / len(results) if results else 0
        metrics["latency_avg"] = sum(r.latency for r in results) / len(results) if results else 0
        metrics["latency_p50"] = sorted(r.latency for r in results)[len(results)//2] if results else 0

        return results, metrics

    def run(self, dataset: list[dict], top_k: int = 10,
             log_fn=None, cache: dict | None = None,
             use_llm: bool = False,
             rewrite: bool = False,
             agent_id: str = "eval") -> tuple[list[CaseResult], dict]:
        """
        运行 benchmark（合并模式，等同于 capture + recall）。
        cache: 预计算的 query_embeddings（来自 load_cache）。
        use_llm: True=走 LLM extraction (/capture/batch)，False=direct存储 (/direct_capture)。
        """
        if log_fn is None:
            log_fn = print

        mode_str = "LLM extraction" if use_llm else "direct (no LLM)"
        log_fn(f"  [1] Capture {len(dataset)} 条记忆（{mode_str}）...")
        t0 = time.time()
        if use_llm:
            captured, total = self.capture_batch(dataset, batch_size=50, max_workers=4)
            memory_ids = []
        else:
            captured, total, memory_ids = self.direct_capture_batch_with_ids(dataset, batch_size=50, max_workers=4)
        elapsed = time.time() - t0
        log_fn(f"      已 capture {captured}/{total} 条 ({elapsed:.1f}s)")

        # 注入 query_vector（来自缓存，或实时计算）
        if cache is not None:
            log_fn(f"  [embedding] 从缓存加载 {len(dataset)} 条 question 向量...")
            for i, item in enumerate(dataset):
                cache_item = cache["items"][i] if i < len(cache.get("items", [])) else {}
                item["query_vector"] = cache_item.get("query_vector")
                # Inject memory_id from cache (for sessions saved before this change)
                if "memory_id" not in item:
                    item["memory_id"] = cache_item.get("memory_id", "")
        else:
            log_fn(f"  [embedding] 预计算 {len(dataset)} 条 question 向量...")
            questions = [item.get("question", "") for item in dataset]
            vectors = embed_texts(questions)
            log_fn(f"      向量计算完成 ({len(vectors)}/{len(questions)})")
            for i, item in enumerate(dataset):
                item["query_vector"] = vectors[i] if i < len(vectors) else None
                # Inject memory_id from capture (new sessions have this)
                if memory_ids and i < len(memory_ids):
                    item["memory_id"] = memory_ids[i]

        log_fn(f"  [2] 等待索引就绪 (3s)...")
        time.sleep(3)

        return self.recall_eval(dataset, top_k=top_k, log_fn=log_fn, rewrite=rewrite, agent_id=agent_id)


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
    parser.add_argument("--host", default="http://127.0.0.1:18360")
    parser.add_argument("--mode", default="both",
                       choices=["capture", "recall", "both"],
                       help="capture=只 capture 数据集；recall=只评测 recall；both=两者都做（默认）")
    parser.add_argument("--session-file",
                       help="capture/recall 共享的 session 文件路径（Phase 间传递数据）")
    parser.add_argument("--llm", dest="use_llm", action="store_true",
                       help="走 LLM extraction (/capture/batch)，较慢但真实")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false",
                       help="不走 LLM 直接存储 (/direct_capture)，快（默认）")
    parser.add_argument("--rewrite", action="store_true",
                       help="recall 时启用 Query Rewrite（KR2.4，MRR+0.23）")
    parser.add_argument("--agent", default="eval",
                       help="recall agent 命名空间过滤（默认 eval，用于 benchmark 隔离）")
    parser.set_defaults(use_llm=False, rewrite=False)
    args = parser.parse_args()

    from datetime import datetime
    def log_print(*args, **kwargs):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)

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

    log_print(f"[benchmark] 数据集: {args.dataset} ({len(dataset)} 条)  mode={args.mode}")

    bm = HawkMemoryBenchmark()

    if not bm.health_check():
        log_print("❌ hawk-memory-api health check failed")
        return

    log_print("✅ hawk-memory-api health OK")

    # 尝试加载预计算缓存
    cache = None
    if args.mode in ("both", "recall"):
        cache = load_cache(args.dataset)
        if cache:
            log_print(f"✅ 预计算缓存命中: {cache['count']} 条 (fingerprint: {cache.get('fingerprint', '?')})")
        else:
            log_print("⚠️  未找到预计算缓存，将实时计算 embedding（较慢）")
            log_print("   运行: python scripts/precompute_query_embeddings.py --dataset ... 预计算缓存")

    results = []
    metrics = {}

    if args.mode == "capture":
        session_data = bm.capture_dataset(dataset, log_fn=log_print, use_llm=args.use_llm)
        if args.session_file:
            Path(args.session_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.session_file, "w") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            log_print(f"✅ session 已保存: {args.session_file}")

    if args.mode == "recall":
        # recall-only: load from session file if provided
        if args.session_file:
            with open(args.session_file) as f:
                session_data = json.load(f)
            dataset = session_data["items"]
            log_print(f"[recall] 从 session 加载 {len(dataset)} 条（忽略 --offset/--limit）")
        results, metrics = bm.recall_eval(dataset, top_k=args.top_k, log_fn=log_print, rewrite=args.rewrite, agent_id=args.agent)

    if args.mode == "both":
        # run() handles capture + recall internally
        results, metrics = bm.run(dataset, top_k=args.top_k, log_fn=log_print, cache=cache, use_llm=args.use_llm, rewrite=args.rewrite, agent_id=args.agent)

    if not results and metrics:
        log_print("\n（capture-only 模式，无评测结果）")
        return

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "mode": args.mode,
        "agent": args.agent,
        "rewrite": args.rewrite,
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
