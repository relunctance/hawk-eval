#!/usr/bin/env python3
"""
Benchmark hawk-memory-api /rules/discover endpoint.

Evaluates whether the LLM can correctly discover trigger rules from conversation history:
- explicit_procedure: user asks about deployment/config/checklist/rollback procedures
- implicit_task: user says "照旧处理" / "按老办法"
- micro_action: short operational commands (clear cache, toggle switch)
- negative: chitchat / knowledge QA should not yield rules

Usage:
    PYTHONPATH=src python src/benchmark_discover.py
    PYTHONPATH=src python src/benchmark_discover.py --max-rules 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ─── HTTP Client ─────────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:18360"


def http_post(path: str, body: dict) -> dict:
    import urllib.request
    import urllib.error

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# ─── Seed Cases ─────────────────────────────────────────────────────────────────
# Each case: conversation + expected rule type + keywords that should appear

SEED_CASES = [
    {
        "id": "disc-001",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "怎么部署服务到生产环境？"},
            {"role": "assistant", "content": "部署流程：1. 构建镜像 2. 推送仓库 3. 滚动更新"},
        ],
        "expect_rule_type": "explicit_procedure",
        "expect_keywords": ["部署", "发布", "上线"],
    },
    {
        "id": "disc-002",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "Redis 配置怎么改？"},
            {"role": "assistant", "content": "修改 Redis 配置：编辑 redis.conf，重启生效"},
        ],
        "expect_rule_type": "explicit_procedure",
        "expect_keywords": ["配置", "Redis"],
    },
    {
        "id": "disc-003",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "数据库出问题了怎么回滚？"},
            {"role": "assistant", "content": "回滚步骤：1. 停止写入 2. 导出数据 3. 执行回滚脚本"},
        ],
        "expect_rule_type": "explicit_procedure",
        "expect_keywords": ["回滚", "rollback"],
    },
    {
        "id": "disc-004",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "每次发布前要做哪些检查？"},
            {"role": "assistant", "content": "发布前检查清单：1. 单元测试 2. 集成测试 3. 告警检查"},
        ],
        "expect_rule_type": "explicit_procedure",
        "expect_keywords": ["检查", "清单", "checklist"],
    },
    {
        "id": "disc-005",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "系统故障怎么排查？"},
            {"role": "assistant", "content": "故障排查：1. 查看监控 2. 检查日志 3. 定位根因"},
        ],
        "expect_rule_type": "explicit_procedure",
        "expect_keywords": ["故障", "排查"],
    },
    {
        "id": "disc-006",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "这个操作照旧处理就行。"},
            {"role": "assistant", "content": "好的，按之前的方式处理。"},
        ],
        "expect_rule_type": "implicit_task",
        "expect_keywords": [],
    },
    {
        "id": "disc-007",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "用老办法解决。"},
            {"role": "assistant", "content": "明白，使用之前的流程。"},
        ],
        "expect_rule_type": "implicit_task",
        "expect_keywords": [],
    },
    {
        "id": "disc-008",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "帮我清一下缓存。"},
            {"role": "assistant", "content": "已清理缓存。"},
        ],
        "expect_rule_type": "micro_action",
        "expect_keywords": ["缓存", "清理"],
    },
    {
        "id": "disc-009",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "把灰度开关打开。"},
            {"role": "assistant", "content": "灰度已启用。"},
        ],
        "expect_rule_type": "micro_action",
        "expect_keywords": ["开关", "灰度"],
    },
    {
        "id": "disc-010",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "今天天气怎么样？"},
            {"role": "assistant", "content": "今天晴，气温20度。"},
        ],
        "expect_rule_type": "negative",
        "expect_keywords": [],
    },
    {
        "id": "disc-011",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "什么是微服务架构？"},
            {"role": "assistant", "content": "微服务是一种架构风格..."},
        ],
        "expect_rule_type": "negative",
        "expect_keywords": [],
    },
    {
        "id": "disc-012",
        "conversation": [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "1+1等于多少？"},
            {"role": "assistant", "content": "2"},
        ],
        "expect_rule_type": "negative",
        "expect_keywords": [],
    },
]


# ─── Evaluation ─────────────────────────────────────────────────────────────────


def keyword_match(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in text."""
    if not keywords:
        return True
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_discovered_rules(discovered: list[dict], case: dict) -> dict[str, Any]:
    """Evaluate whether discovered rules match expectations."""
    expect_type = case["expect_rule_type"]
    expect_keywords = case["expect_keywords"]

    # Find rules matching expected type
    matching_type = [r for r in discovered if r.get("type") == expect_type]

    if expect_type == "negative":
        # Negative case: should have 0 rules or only negative rules
        if len(discovered) == 0:
            return {"correct": True, "reason": "no rules discovered (expected negative)"}
        if len(matching_type) == len(discovered):
            return {"correct": True, "reason": "all rules are negative (expected)"}
        return {
            "correct": False,
            "reason": f"expected negative/no rules, got {len(discovered)} rules with types {[r.get('type') for r in discovered]}",
        }

    # Positive case: should have at least one rule of expected type
    if not matching_type:
        return {
            "correct": False,
            "reason": f"expected {expect_type} rule, got types {[r.get('type') for r in discovered]}",
        }

    # Check keywords in rule names
    for rule in matching_type:
        rule_text = json.dumps(rule, ensure_ascii=False)
        if keyword_match(rule_text, expect_keywords):
            return {"correct": True, "reason": f"matched {expect_type} with keywords"}

    # Type matched but keywords didn't — partial credit
    return {
        "correct": False,
        "reason": f"type matched but keywords {[r.get('name', '') for r in matching_type]} missing {expect_keywords}",
    }


# ─── Benchmark ─────────────────────────────────────────────────────────────────


def benchmark_discover(max_rules: int = 5) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  hawk /rules/discover benchmark — {len(SEED_CASES)} cases")
    print(f"{'='*60}\n")

    results = []
    correct = 0
    latencies = []

    for case in SEED_CASES:
        case_id = case["id"]
        conversation = case["conversation"]

        start = time.time()
        resp = http_post(
            "/rules/discover",
            {"messages": conversation, "max_rules": max_rules},
        )
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        if "error" in resp:
            result = {
                "id": case_id,
                "type": case["expect_rule_type"],
                "query": conversation[0]["content"][:30],
                "expected": case["expect_rule_type"],
                "actual": "ERROR",
                "correct": False,
                "latency_ms": round(latency_ms, 1),
                "error": resp["error"],
            }
        else:
            discovered = resp.get("rules", [])
            eval_result = evaluate_discovered_rules(discovered, case)
            is_correct = eval_result["correct"]

            if is_correct:
                correct += 1

            result = {
                "id": case_id,
                "type": case["expect_rule_type"],
                "query": conversation[0]["content"][:30],
                "expected": case["expect_rule_type"],
                "actual": list(set(r.get("type") for r in discovered)) or "none",
                "correct": is_correct,
                "latency_ms": round(latency_ms, 1),
                "reason": eval_result["reason"],
                "discovered_count": len(discovered),
            }

        results.append(result)
        status = "✓" if result["correct"] else "✗"
        print(f"  {status} [{case_id}] {result['type']:20s} | {result['query'][:25]} | {result.get('reason', '')}")

    n = len(SEED_CASES)
    accuracy = correct / n if n > 0 else 0
    p50 = sorted(latencies)[n // 2] if latencies else 0
    p95 = sorted(latencies)[int(n * 0.95)] if latencies else 0

    print(f"\n{'─'*60}")
    print(f"  Accuracy:   {accuracy:.1%}  ({correct}/{n})")
    print(f"  Latency:    P50={p50:.0f}ms  P95={p95:.0f}ms")
    print(f"{'─'*60}\n")

    # By-type breakdown
    by_type: dict[str, dict] = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1

    print("Accuracy by type:")
    for t, stats in sorted(by_type.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {t:25s}  {acc:.1%}  ({stats['correct']}/{stats['total']})")

    return {
        "total": n,
        "correct": correct,
        "accuracy": accuracy,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "by_type": by_type,
        "results": results,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark /rules/discover endpoint")
    parser.add_argument(
        "--max-rules", type=int, default=5, help="Max rules to discover per case"
    )
    parser.add_argument(
        "--output", default="", help="Write JSON results to file"
    )
    args = parser.parse_args()

    result = benchmark_discover(max_rules=args.max_rules)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.output}")

    # Exit code: 0 if accuracy >= 70%, else 1
    sys.exit(0 if result["accuracy"] >= 0.70 else 1)


if __name__ == "__main__":
    main()
