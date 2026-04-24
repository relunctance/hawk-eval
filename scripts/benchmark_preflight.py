#!/usr/bin/env python3
"""
Benchmark Preflight Checks — 运行 benchmark 前必须通过的前置检查。

检查项目：
1. hawk-memory-api health（服务可用性）
2. xinference embedding 服务可用性
3. 数据集质量（answer 长度、placeholder 检测）
4. DB 当前状态（总条数、按 agent_id 分布）
5. DB 污染检测（如果有 eval 数据残留，提示清理）

用法：
    python scripts/benchmark_preflight.py
    python scripts/benchmark_preflight.py --strict      # 失败时退出码=1
    python scripts/benchmark_preflight.py --dataset datasets/hawk_memory/conversational_qa.jsonl
    python scripts/benchmark_preflight.py --cleanup    # 发现问题时自动清理 eval namespace
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path


BASE = "http://127.0.0.1:18360"
EMBED_URL = "http://127.0.0.1:9997/v1/embeddings"
EMBED_MODEL = "bge-m3"


# ─── HTTP helpers ────────────────────────────────────────────────────────────

def req(method, path, body=None, timeout=15):
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
    """批量计算文本 embedding（直接调 xinference）。"""
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


# ─── Check 1: hawk-memory-api health ────────────────────────────────────────

def check_api_health(strict: bool = False) -> bool:
    print("\n[1/5] hawk-memory-api health check...")
    data, status = req("GET", "/health")
    if status != 200:
        msg = f"  ❌ API health check failed: HTTP {status} — {data}"
        print(msg)
        if strict:
            sys.exit(1)
        return False

    mem_count = data.get("memory_count", 0)
    api_status = data.get("status", "unknown")
    version = data.get("version", "?")

    if api_status == "ok":
        print(f"  ✅ API OK (v{version})，当前 DB 记忆条数: {mem_count}")
        return True
    else:
        print(f"  ⚠️  API status={api_status} (v{version})，DB 记忆: {mem_count}")
        return True  # degraded 仍可继续


# ─── Check 2: xinference embedding service ──────────────────────────────────

def check_embedding_service(strict: bool = False) -> bool:
    print("\n[2/5] xinference embedding service check...")
    test_texts = ["测试文本", "hello world"]
    vectors = embed_texts(test_texts)

    if not vectors or len(vectors) != 2:
        msg = f"  ❌ Embedding service failed: returned {len(vectors)} vectors (expected 2)"
        print(msg)
        if strict:
            sys.exit(1)
        return False

    # 检查向量维度
    dim = len(vectors[0])
    print(f"  ✅ Embedding service OK (model={EMBED_MODEL}, dim={dim})")
    return True


# ─── Check 3: Dataset quality ────────────────────────────────────────────────

# 已知的 placeholder / 类别名关键词（不能作为 answer 出现）
_PLACEHOLDER_KEYWORDS = frozenset({
    "查询", "配置", "设置", "管理", "查询结果",
    "service", "config", "query", "result",
    # 常见短回答模板
    "待查询", "请稍后", "稍后再试",
})


def check_dataset_quality(dataset_path: str, strict: bool = False) -> bool:
    print(f"\n[3/5] Dataset quality check: {dataset_path}")

    if not Path(dataset_path).exists():
        msg = f"  ❌ 数据集不存在: {dataset_path}"
        print(msg)
        if strict:
            sys.exit(1)
        return False

    with open(dataset_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if not lines:
        print("  ❌ 数据集为空")
        if strict:
            sys.exit(1)
        return False

    issues = []

    # 3a: placeholder 检测
    placeholder_answers = []
    for d in lines:
        answer = d.get("answer", "") or ""
        ans_lower = answer.lower()
        for kw in _PLACEHOLDER_KEYWORDS:
            if ans_lower == kw.lower() or ans_lower.startswith(kw.lower() + "："):
                placeholder_answers.append((d.get("id", "?"), answer[:30]))
                break

    # 3c: 重复 answer 检测
    answer_seen: dict[str, list[str]] = {}
    for d in lines:
        answer = d.get("answer", "") or ""
        if answer not in answer_seen:
            answer_seen[answer] = []
        answer_seen[answer].append(d.get("id", "?"))

    duplicate_answers = {a: ids for a, ids in answer_seen.items() if len(ids) > 1}

    # 汇总报告
    print(f"  总条数: {len(lines)}")

    if placeholder_answers:
        issues.append(f"  ⚠️  {len(placeholder_answers)} 条 answer 疑似 placeholder / 类别名")
        for iid, ans in placeholder_answers[:5]:
            issues.append(f"      {iid}: \"{ans}\"")
        if len(placeholder_answers) > 5:
            issues.append(f"      ... 还有 {len(placeholder_answers) - 5} 条")

    if duplicate_answers:
        issues.append(f"  ⚠️  {len(duplicate_answers)} 个 answer 有重复 ID（recall 时互相干扰）")
        for ans, ids in list(duplicate_answers.items())[:3]:
            issues.append(f"      answer=\"{ans[:30]}\": {ids}")

    if issues:
        for issue in issues:
            print(issue)
        print(f"\n  ⚠️  数据集存在 {len(placeholder_answers)} 条疑似 placeholder，"
              f"{len(duplicate_answers)} 个重复 answer")
        if strict:
            sys.exit(1)
        return False
    else:
        print(f"  ✅ 数据集质量通过（{len(lines)} 条，无 placeholder，{len(duplicate_answers)} 个重复 answer 需注意）")
        return True


# ─── Check 4: DB status ──────────────────────────────────────────────────────

def check_db_status(strict: bool = False) -> bool:
    print("\n[4/5] DB 当前状态...")
    data, status = req("GET", "/stats")

    if status != 200:
        msg = f"  ❌ /stats 请求失败: HTTP {status} — {data}"
        print(msg)
        if strict:
            sys.exit(1)
        return False

    total = data.get("total", 0)
    by_source = data.get("by_source", {})
    by_category = data.get("by_category", {})
    by_agent_id = data.get("by_agent_id", {})
    db_path = data.get("db_path", "?")

    print(f"  总记忆条数: {total}")
    print(f"  by_source: {json.dumps(by_source, ensure_ascii=False)}")
    print(f"  by_category: {json.dumps(by_category, ensure_ascii=False)}")
    print(f"  by_agent_id: {json.dumps(by_agent_id, ensure_ascii=False)}")

    if total == 0:
        print("  ✅ DB 为空，clean 状态")
        return True

    # 检测 eval namespace 残留（benchmark capture 用 agent_id='eval'）
    eval_count = by_agent_id.get("eval", 0)
    if eval_count > 0:
        print(f"\n  ⚠️  发现 {eval_count} 条 eval namespace 数据残留！")
        print(f"      建议: curl -s -X POST '{BASE}/admin/cleanup' -d '{{\"agent_id\":\"eval\"}}'")
        print(f"      或使用: python scripts/benchmark_preflight.py --cleanup")
        if strict:
            sys.exit(1)
        return False

    print("  ✅ DB 状态正常（无明显残留数据）")
    return True


# ─── Check 5: Cleanup eval namespace ─────────────────────────────────────────

def cleanup_eval_namespace() -> bool:
    print("\n[5/5] 清理 eval namespace...")
    data, status = req("POST", "/admin/cleanup", {"agent_id": "eval"})

    if status != 200:
        # 可能 admin 端点还没实现，尝试 GET stats 看能不能清理
        print(f"  ⚠️  /admin/cleanup 返回 {status}，尝试直接 soft-delete 所有 eval 记忆...")
        # Fallback: 通过 /stats 的 agent_id 信息清理
        stats_data, _ = req("GET", "/stats")
        print(f"  ⚠️  admin/cleanup 端点未实现，请手动清理或升级 hawk-memory-api")
        print(f"      当前 DB 状态: total={stats_data.get('total', 0)}")
        return False

    deleted = data.get("deleted", 0)
    print(f"  ✅ 已清理 {deleted} 条 eval 记忆")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Preflight Checks")
    parser.add_argument("--strict", action="store_true",
                       help="检查失败时 exit code = 1（CI 模式）")
    parser.add_argument("--dataset", default="datasets/hawk_memory/conversational_qa.jsonl",
                       help="数据集路径（用于质量检查）")
    parser.add_argument("--cleanup", action="store_true",
                       help="发现 eval 数据残留时自动清理")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="跳过数据集质量检查（无本地数据集时使用）")
    args = parser.parse_args()

    print("=" * 60)
    print("  Benchmark Preflight Checks")
    print("=" * 60)

    # 1. API health
    ok1 = check_api_health(args.strict)

    # 2. Embedding service
    ok2 = check_embedding_service(args.strict)

    # 3. Dataset quality
    ok3 = True
    if not args.skip_dataset:
        ok3 = check_dataset_quality(args.dataset, args.strict)

    # 4. DB status
    ok4 = check_db_status(args.strict)

    # 5. Cleanup (only if --cleanup flag and DB has issues)
    ok5 = True
    if args.cleanup and not ok4:
        ok5 = cleanup_eval_namespace()

    # Summary
    print("\n" + "=" * 60)
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5
    if all_ok:
        print("  ✅ 全部检查通过，可以开始 benchmark")
        print("=" * 60)
        sys.exit(0)
    else:
        print("  ⚠️  部分检查未通过，详见上文")
        print("=" * 60)
        if args.strict:
            sys.exit(1)
        sys.exit(0)  # non-strict: warning only


if __name__ == "__main__":
    main()
