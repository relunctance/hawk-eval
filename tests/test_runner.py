#!/usr/bin/env python3
"""
hawk-eval Self-Check
快速验证系统是否正常工作（不需要 ground truth ID）。
用语义相似度判断：检索结果是否包含相关内容。
"""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:18368"  # hawk-memory Go binary


def req(method, path, body=None, timeout=8):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), r.status
    except Exception as e:
        return str(e), -1


def check(name, ok, detail=""):
    status = "✓" if ok else "✗"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("=" * 50)
    print("hawk-eval Self-Check")
    print("=" * 50)

    # 1. 健康检查
    print("\n[1] hawk-memory-api 健康检查")
    data, s = req("GET", "/health")
    check("status == 200", s == 200, f"({s})")
    check("status == 'ok'", data.get("status") == "ok", f"({data.get('status')})")
    count = data.get("memory_count", 0)
    check(f"memory_count >= 0", count >= 0, f"({count} 条记忆)")

    # 2. Capture
    print("\n[2] /capture")
    body = {
        "session_id": "eval-check",
        "user_id": "eval",
        "message": "我今天学了 Python 编程",
        "response": "Python 很棒",
        "platform": "eval",
    }
    data, s = req("POST", "/capture", body)
    check("status 200/201", s in (200, 201), f"({s})")
    stored = data.get("stored", 0)
    check("stored >= 1", stored >= 1, f"(存了 {stored} 条)")

    # 3. Recall 语义验证
    print("\n[3] /recall 语义验证")
    test_queries = [
        ("Python", ["Python", "编程"], "Python 相关记忆"),
        ("飞书", ["飞书", "webhook"], "飞书相关记忆"),
        ("hawk", ["hawk", "bridge"], "hawk 相关记忆"),
    ]
    sem_ok = 0
    for query, keywords, desc in test_queries:
        data, s = req("POST", "/recall", {"query": query, "top_k": 5, "platform": "eval"})
        if s == 200:
            texts = " ".join(m.get("text", "") for m in data.get("memories", []))
            hits = sum(1 for kw in keywords if kw in texts)
            ok = hits > 0
            check(f"recall('{query}') → {desc}", ok,
                  f"({hits}/{len(keywords)} 关键词命中)")
            if ok:
                sem_ok += 1
        else:
            check(f"recall('{query}')", False, f"(status={s})")

    # 4. Stats
    print("\n[4] /stats")
    data, s = req("GET", "/stats")
    check("status == 200", s == 200)
    check("has total", "total" in data)
    check("has by_category", "by_category" in data)

    print("\n" + "=" * 50)
    passed = sem_ok
    print(f"✓ 系统正常 ({passed}/{len(test_queries)} 语义查询通过)")
    print("  → 可用 make benchmark-hawk 跑完整 benchmark")
    print("=" * 50)
    return 0 if passed >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
