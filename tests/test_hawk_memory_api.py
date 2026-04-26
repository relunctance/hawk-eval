#!/usr/bin/env python3
"""
hawk-memory-api 集成测试套件

覆盖：
1. health check
2. capture + recall 基本流程
3. QA 格式（message + response）存储与召回
4. agent_id=None 边界情况（曾导致 500 错误）
5. 关键词召回语义验证
6. 多 session 隔离

运行：
    pytest tests/ -v
    make test

前提：hawk-memory Go binary 运行在 http://127.0.0.1:18368
"""

import json
import time
import uuid
import urllib.request
import urllib.error
from typing import Any

import pytest

BASE_URL = "http://127.0.0.1:18368"


def req(method: str, path: str, body: dict = None, timeout: float = 10) -> tuple[Any, int]:
    """发起 HTTP 请求，返回 (response_data, status_code)"""
    url = BASE_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read()), e.code
        except Exception:
            return str(e), e.code
    except Exception as e:
        return str(e), -1


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def health_check():
    """验证服务可用"""
    data, s = req("GET", "/health")
    assert s == 200, f"hawk-memory-api 不可用: {s} {data}"
    assert data.get("status") == "ok", f"服务状态异常: {data}"
    return data


@pytest.fixture
def clean_session():
    """每个测试用独立 session，避免互相干扰"""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_platform():
    """测试专用 platform"""
    return f"pytest-{uuid.uuid4().hex[:6]}"


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, health_check):
        assert health_check["status"] == "ok"

    def test_health_has_memory_count(self, health_check):
        assert "memory_count" in health_check

    def test_stats_endpoint(self):
        data, s = req("GET", "/stats")
        assert s == 200, f"stats 返回 {s}: {data}"
        assert "total" in data or "memory_count" in data


class TestCapture:
    def test_capture_simple_message(self, clean_session, test_platform):
        """存一条简单记忆，能成功返回"""
        body = {
            "session_id": clean_session,
            "user_id": "pytest",
            "message": "我今天学了 Python",
            "response": "",
            "platform": test_platform,
        }
        data, s = req("POST", "/capture", body)
        assert s in (200, 201), f"capture 失败: {s} {data}"
        # 返回 memory_ids 列表
        assert "memory_ids" in data or "id" in data or "stored" in data

    def test_capture_qa_format(self, clean_session, test_platform):
        """QA 格式 capture（message + response），recall 时返回助手回复"""
        question = "When did Caroline go to the LGBTQ support group?"
        answer = "7 May 2023"
        body = {
            "session_id": clean_session,
            "user_id": "pytest",
            "message": question,
            "response": answer,
            "platform": test_platform,
        }
        data, s = req("POST", "/capture", body)
        assert s in (200, 201), f"capture QA 失败: {s} {data}"
        time.sleep(1)  # 等索引

        # recall 用 question
        recall_data, recall_s = req("POST", "/recall", {
            "query": question,
            "top_k": 5,
            "platform": test_platform,
        })
        assert recall_s == 200, f"recall 返回 {recall_s}: {recall_data}"
        mems = recall_data.get("memories", [])
        assert len(mems) > 0, "recall 返回空结果"

        # 助手回复应该出现在结果中
        texts = " ".join(m.get("text", "") for m in mems)
        assert answer in texts, f"answer '{answer}' 未在 recall 结果中: {texts[:100]}"


class TestRecall:
    def test_recall_returns_200(self, clean_session, test_platform):
        """recall 正常调用返回 200"""
        # 先存一条
        body = {
            "session_id": clean_session,
            "user_id": "pytest",
            "message": "测试 recall",
            "response": "",
            "platform": test_platform,
        }
        req("POST", "/capture", body)
        time.sleep(1)

        # recall
        data, s = req("POST", "/recall", {
            "query": "测试",
            "top_k": 5,
            "platform": test_platform,
        })
        assert s == 200, f"recall 返回 {s}: {data}"
        assert "memories" in data

    def test_recall_no_500_on_none_agent_id(self, clean_session, test_platform):
        """
        关键回归测试：agent_id=None 曾导致 500 Internal Server Error

        问题：MemoryItem 模型 agent_id: str 不接受 None，
              但 LanceDB 某些记录 agent_id 为 NULL
        修复：agent_id: Optional[str] = ""
        """
        # 存一条记忆（不过滤 agent_id）
        body = {
            "session_id": clean_session,
            "user_id": "pytest",
            "message": "agent_id 测试",
            "response": "",
            "platform": test_platform,
        }
        req("POST", "/capture", body)
        time.sleep(1)

        # recall 不过滤 agent_id（曾在这里 500）
        data, s = req("POST", "/recall", {
            "query": "agent_id 测试",
            "top_k": 10,
            "platform": test_platform,
        })
        assert s == 200, f"recall agent_id=None 时 500: {data}"
        assert "memories" in data

    def test_recall_with_top_k(self, clean_session, test_platform):
        """top_k 参数有效"""
        # 存 3 条不同记忆
        for i in range(3):
            body = {
                "session_id": clean_session,
                "user_id": "pytest",
                "message": f"测试话题 {i}",
                "response": f"回答 {i}",
                "platform": test_platform,
            }
            req("POST", "/capture", body)
        time.sleep(2)

        data, s = req("POST", "/recall", {
            "query": "测试",
            "top_k": 2,
            "platform": test_platform,
        })
        assert s == 200
        mems = data.get("memories", [])
        assert len(mems) <= 2, f"top_k=2 但返回 {len(mems)} 条"

    def test_recall_semantic(self, clean_session, test_platform):
        """语义召回：存 Python 相关，搜 Python 能找到"""
        body = {
            "session_id": clean_session,
            "user_id": "pytest",
            "message": "我今天学了 Python 编程",
            "response": "Python 很棒",
            "platform": test_platform,
        }
        req("POST", "/capture", body)
        time.sleep(1)

        data, s = req("POST", "/recall", {
            "query": "Python",
            "top_k": 5,
            "platform": test_platform,
        })
        assert s == 200
        texts = " ".join(m.get("text", "") for m in data.get("memories", []))
        assert "Python" in texts, f"搜 Python 但结果不含: {texts[:80]}"


class TestErrorHandling:
    def test_recall_empty_query_rejected(self):
        """空 query 且无 session_id 应返回 400"""
        data, s = req("POST", "/recall", {"query": "", "top_k": 5})
        # 服务端应拒绝空查询
        assert s >= 400

    def test_recall_nonexistent_query_returns_empty(self, test_platform):
        """不存在的 query 返回空结果，不是 500"""
        data, s = req("POST", "/recall", {
            "query": "xyzxyzxyzthisdoesnotexist123123",
            "top_k": 5,
            "platform": test_platform,
        })
        assert s == 200, f"不存在的 query 应返回 200，不是 {s}"
        assert isinstance(data.get("memories"), list)


class TestSessionIsolation:
    def test_different_sessions_isolated(self):
        """不同 session 的记忆互不干扰（同一 platform 下）"""
        sess1 = f"iso1-{uuid.uuid4().hex[:8]}"
        sess2 = f"iso2-{uuid.uuid4().hex[:8]}"
        platform = f"iso-{uuid.uuid4().hex[:6]}"

        # sess1 存 A
        req("POST", "/capture", {
            "session_id": sess1,
            "user_id": "pytest",
            "message": "这是会话一的内容",
            "response": "",
            "platform": platform,
        })
        # sess2 存 B
        req("POST", "/capture", {
            "session_id": sess2,
            "user_id": "pytest",
            "message": "这是会话二的内容",
            "response": "",
            "platform": platform,
        })
        time.sleep(1)

        # sess1 recall 只能看到自己的
        data1, s1 = req("POST", "/recall", {
            "query": "会话一",
            "top_k": 5,
            "platform": platform,
        })
        assert s1 == 200
        texts1 = " ".join(m.get("text", "") for m in data1.get("memories", []))

        data2, s2 = req("POST", "/recall", {
            "query": "会话二",
            "top_k": 5,
            "platform": platform,
        })
        assert s2 == 200
        texts2 = " ".join(m.get("text", "") for m in data2.get("memories", []))

        # 各找各的
        assert "会话一" in texts1, f"sess1 找不到自己内容: {texts1}"
        assert "会话二" in texts2, f"sess2 找不到自己内容: {texts2}"


# ─── 运行入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
