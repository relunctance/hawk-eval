#!/usr/bin/env python3
"""
Benchmark hawk-memory-api trigger rules against m_flow procedural dataset.

Evaluates whether hawk trigger rules correctly identify:
- explicit_procedure: should trigger for deployment/config/etc queries
- implicit_task: should trigger for "照旧处理" style queries
- micro_action: should trigger for short operational commands
- negative: should NOT trigger for chitchat

Usage:
    PYTHONPATH=src python src/benchmark_trigger.py
    PYTHONPATH=src python src/benchmark_trigger.py --dataset m_flow_procedural
    PYTHONPATH=src python src/benchmark_trigger.py --dataset custom --rules-seed
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
    req = urllib.request.Request(f"{API_BASE}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def seed_rules():
    """Seed trigger rules based on m_flow research."""
    rules = [
        {
            "name": "部署触发器",
            "type": "explicit_procedure",
            "procedures": {
                "any_of_keys": [],
                "any_of_titles_contains": ["部署", "发布", "上线"],
                "at_least_one_active": True,
            },
            "injection_constraints": {
                "require_context_fields": ["when", "why"],
                "require_steps": True,
            },
            "enabled": True,
            "description": "用户询问部署/发布/上线流程时触发",
        },
        {
            "name": "回滚触发器",
            "type": "explicit_procedure",
            "procedures": {
                "any_of_keys": [],
                "any_of_titles_contains": ["回滚", "rollback"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "数据库回滚步骤",
        },
        {
            "name": "配置触发器",
            "type": "explicit_procedure",
            "procedures": {
                "any_of_keys": [],
                "any_of_titles_contains": ["配置", "Redis", "缓存", "Nginx"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "配置相关流程",
        },
        {
            "name": "检查清单触发器",
            "type": "explicit_procedure",
            "procedures": {
                "any_of_keys": [],
                "any_of_titles_contains": ["检查", "checklist", "清单"],
                "at_least_one_active": True,
            },
            "injection_constraints": {"require_steps": True},
            "enabled": True,
            "description": "检查清单类流程",
        },
        {
            "name": "故障排查触发器",
            "type": "explicit_procedure",
            "procedures": {
                "any_of_keys": [],
                "any_of_titles_contains": ["故障", "排查", "紧急", "问题"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "故障排查流程",
        },
        {
            "name": "隐式任务-照旧",
            "type": "implicit_task",
            "procedures": {},
            "enabled": True,
            "description": "用户说照旧处理",
        },
        {
            "name": "隐式任务-老办法",
            "type": "implicit_task",
            "procedures": {},
            "enabled": True,
            "description": "用户说用老办法",
        },
        {
            "name": "微操作-清缓存",
            "type": "micro_action",
            "procedures": {
                "any_of_titles_contains": ["缓存", "清理", "清空"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "清缓存操作",
        },
        {
            "name": "微操作-开关",
            "type": "micro_action",
            "procedures": {
                "any_of_titles_contains": ["开关", "启用", "禁用", "灰度"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "开关操作",
        },
        {
            "name": "闲聊屏蔽",
            "type": "negative",
            "procedures": {
                "any_of_titles_contains": ["天气", "新闻", "你好", "今天怎么样", "1+1"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "闲聊不触发procedure",
            "alternative_recall": "",
        },
        {
            "name": "知识问答屏蔽",
            "type": "negative",
            "procedures": {
                "any_of_titles_contains": ["什么是", "为什么是", "解释一下"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "知识问答不触发procedural",
            "alternative_recall": "atomic",
        },
        {
            "name": "事件查询屏蔽",
            "type": "negative",
            "procedures": {
                "any_of_titles_contains": ["会议纪要", "上周", "昨天的"],
                "at_least_one_active": True,
            },
            "enabled": True,
            "description": "事件查询触发episodic而非procedural",
            "alternative_recall": "episodic",
        },
    ]

    # Get current rules and clear them
    existing = http_post("/rules", {})
    for rule in existing.get("rules", []):
        http_post(f"/rules/{rule['id']}", {"delete": True})

    # Create new rules
    created = 0
    for rule in rules:
        resp = http_post("/rules", rule)
        if "error" not in resp:
            created += 1
    print(f"[benchmark_trigger] Seeded {created}/{len(rules)} rules")
    return created


# ─── Dataset Loading ─────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> list[dict]:
    base = Path(__file__).parent.parent / "datasets"
    if name == "m_flow_procedural":
        path = base / "m_flow_procedural" / "procedural_eval_v1.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ─── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_trigger(dataset_name: str, rules_seed: bool = False) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  hawk trigger benchmark — {dataset_name}")
    print(f"{'='*60}\n")

    # Optionally seed rules
    if rules_seed:
        seed_rules()

    # Load dataset
    cases = load_dataset(dataset_name)
    print(f"Loaded {len(cases)} test cases\n")

    # Run evaluation
    results = []
    correct = 0
    false_positives = []  # triggered when shouldn't
    false_negatives = []  # didn't trigger when should
    latencies = []

    for case in cases:
        query = case["query"]
        expected_trigger = case.get("expect", {}).get("should_trigger_procedural", False)
        case_type = case.get("type", "unknown")
        case_id = case.get("id", "??")

        start = time.time()
        resp = http_post("/rules/evaluate", {"query": query, "include_negative": True})
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        actual_trigger = resp.get("should_trigger", False)
        matched_types = resp.get("matched_rule_types", [])
        is_correct = actual_trigger == expected_trigger

        if is_correct:
            correct += 1
        elif actual_trigger and not expected_trigger:
            false_positives.append({
                "id": case_id,
                "type": case_type,
                "query": query,
                "matched_rules": matched_types,
            })
        else:
            false_negatives.append({
                "id": case_id,
                "type": case_type,
                "query": query,
                "matched_rules": matched_types,
            })

        results.append({
            "id": case_id,
            "type": case_type,
            "query": query[:60],
            "expected": expected_trigger,
            "actual": actual_trigger,
            "correct": is_correct,
            "latency_ms": round(latency_ms, 1),
            "matched_rules": matched_types,
        })

    n = len(cases)
    accuracy = correct / n if n > 0 else 0
    p50 = sorted(latencies)[n // 2] if latencies else 0
    p95 = sorted(latencies)[int(n * 0.95)] if latencies else 0

    print(f"{'─'*60}")
    print(f"  Accuracy:   {accuracy:.1%}  ({correct}/{n})")
    print(f"  Latency:    P50={p50:.0f}ms  P95={p95:.0f}ms")
    print(f"  False+:     {len(false_positives)}")
    print(f"  False-:     {len(false_negatives)}")
    print(f"{'─'*60}\n")

    # Breakdown by type
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

    # Show errors
    if false_positives:
        print(f"\nFalse Positives ({len(false_positives)}):")
        for fp in false_positives[:5]:
            print(f"  [{fp['id']}] query={fp['query'][:50]}  matched={fp['matched_rules']}")

    if false_negatives:
        print(f"\nFalse Negatives ({len(false_negatives)}):")
        for fn in false_negatives[:5]:
            print(f"  [{fn['id']}] query={fn['query'][:50]}  expected_trigger")

    return {
        "dataset": dataset_name,
        "total": n,
        "correct": correct,
        "accuracy": accuracy,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "by_type": by_type,
        "results": results,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark hawk trigger rules")
    parser.add_argument("--dataset", default="m_flow_procedural", help="Dataset name")
    parser.add_argument("--rules-seed", action="store_true", help="Seed rules before benchmarking")
    parser.add_argument("--output", default="", help="Write JSON results to file")
    args = parser.parse_args()

    result = benchmark_trigger(args.dataset, rules_seed=args.rules_seed)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nResults written to {args.output}")

    # Exit code: 0 if accuracy >= 80%, else 1
    sys.exit(0 if result["accuracy"] >= 0.80 else 1)


if __name__ == "__main__":
    main()
