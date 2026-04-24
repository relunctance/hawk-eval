#!/usr/bin/env python3
"""
预计算 query embeddings 并缓存到文件。

用途：避免每次 benchmark 都重新计算 embedding，加快评测速度。

用法：
    # 预计算 200 条
    python scripts/precompute_query_embeddings.py \
        --dataset datasets/hawk_memory/conversational_qa.jsonl \
        --output data/query_embeddings_cache.jsonl

    # 评测时（自动检测缓存）
    python -m src.benchmark_hawk --dataset datasets/hawk_memory/conversational_qa.jsonl
"""

import argparse
import hashlib
import json
import time
import urllib.request
import urllib.error
from pathlib import Path


BASE = "http://127.0.0.1:18360"
EMBED_URL = "http://127.0.0.1:9997/v1/embeddings"
EMBED_MODEL = "bge-m3"
VERSION = "v1"


def req(method, path, body=None, timeout=30):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    req_ = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req_, timeout=timeout) as r:
            return json.loads(r.read()), r.status
    except Exception as e:
        return str(e), -1


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量计算文本 embedding（直接调 xinference）。"""
    body = {"model": EMBED_MODEL, "input": texts}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    req_ = urllib.request.Request(EMBED_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req_, timeout=60) as r:
            result = json.loads(r.read())
    except Exception as e:
        print(f"  [embedding] ERROR: {e}")
        return []
    return [item["embedding"] for item in result.get("data", [])]


def dataset_fingerprint(dataset_path: str) -> str:
    """对数据集内容做 hash，用于判断缓存是否过期。"""
    with open(dataset_path) as f:
        content = f.read()
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_dataset(path: str) -> list[dict]:
    dataset = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dataset.append(json.loads(line))
                except Exception:
                    pass
    return dataset


def precompute(dataset_path: str, output: str, batch_size: int = 32):
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    questions = [item.get("question", "") for item in dataset]

    # 检查现有缓存
    cache_file = Path(output)
    if cache_file.exists():
        with open(cache_file) as f:
            existing = json.load(f)
        fp = dataset_fingerprint(dataset_path)
        if existing.get("fingerprint") == fp and len(existing.get("items", [])) == len(questions):
            print(f"[precompute] 缓存已存在且有效 ({len(questions)} 条)，跳过。直接运行 benchmark 即可。")
            print(f"  缓存文件: {output}")
            return
        else:
            print(f"[precompute] 数据集已更新，重新计算...")

    print(f"[precompute] 数据集: {dataset_path} ({len(dataset_path)} 条)")
    print(f"[precompute] 输出: {output}")
    print(f"[precompute] batch_size: {batch_size}")

    t0 = time.time()
    all_vectors = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        vectors = embed_texts(batch)
        all_vectors.extend(vectors)
        print(f"  进度: {min(i+batch_size, len(questions))}/{len(questions)}")

    elapsed = time.time() - t0
    print(f"[precompute] 完成 {len(all_vectors)} 条，耗时 {elapsed:.1f}s ({elapsed/len(all_vectors)*1000:.1f}ms/条)")

    # 写入缓存
    fp = dataset_fingerprint(dataset_path)
    cache = {
        "version": VERSION,
        "fingerprint": fp,
        "dataset": dataset_path,
        "model": EMBED_MODEL,
        "count": len(all_vectors),
        "items": [
            {
                "id": item.get("id", f"q-{i}"),
                "question": item.get("question", ""),
                "answer": item.get("answer") or item.get("memory_text", ""),
                "query_vector": all_vectors[i] if i < len(all_vectors) else None,
            }
            for i, item in enumerate(dataset)
        ]
    }
    with open(output, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"[precompute] 缓存已保存: {output}")


def main():
    parser = argparse.ArgumentParser(description="预计算 query embeddings 并缓存")
    parser.add_argument("--dataset", default="datasets/hawk_memory/conversational_qa.jsonl",
                       help="JSONL 数据集路径")
    parser.add_argument("--output", default="data/query_embeddings_cache.json",
                       help="缓存输出路径")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="每批计算的文本数（默认 32）")
    args = parser.parse_args()

    precompute(args.dataset, args.output, args.batch_size)


if __name__ == "__main__":
    main()
