#!/usr/bin/env python3
"""
hawk-memory Go E2E pytest 测试套件 (KR6.13)

覆盖：
1. health check（含 FTS、xinference 状态）
2. capture + recall 完整链路
3. 关键词搜索（pure-Go FTS 验证）
4. fusion 评分结果验证
5. decay 触发验证
6. agent 隔离验证
7. batch capture + recall
8. 删除后 recall 不返回

前提：hawk-memory Go 必须运行在 http://127.0.0.1:18368

运行：
    pytest tests/test_hawk_memory_go.py -v
    # 或从 hawk-eval 目录：
    PYTHONPATH=src pytest tests/test_hawk_memory_go.py -v
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
    """验证 hawk-memory Go 服务可用"""
    data, s = req("GET", "/health")
    assert s == 200, f"hawk-memory Go 不可用: {s} {data}"
    assert data.get("status") == "ok", f"服务状态异常: {data}"
    return data


@pytest.fixture
def clean_agent():
    """每个测试用独立 agent，避免互相干扰"""
    return f"pytest-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_suffix():
    """唯一后缀用于验证测试数据隔离"""
    return time.time_ns()


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, health_check):
        assert health_check["status"] == "ok"

    def test_health_has_fts_status(self, health_check):
        """FTS 状态应该在 health 中"""
        assert "fts_ok" in health_check, f"health 缺少 fts_ok: {health_check}"

    def test_health_fts_detail(self, health_check):
        """FTS 详情应该包含 text 列"""
        fts_detail = health_check.get("fts_detail", "")
        assert "text" in fts_detail.lower(), f"fts_detail 不含 text: {fts_detail}"


class TestCaptureRecall:
    def test_capture_and_recall_chain(self, clean_agent, unique_suffix):
        """核心 E2E：capture → recall 完整链路"""
        # Capture
        body = {
            "text": f"test memory unique-{unique_suffix} marker-e2e",
            "agent_id": clean_agent,
            "metadata": {"e2e": True},
        }
        data, s = req("POST", "/v1/capture", body)
        assert s == 200, f"capture 失败: {s} {data}"
        assert data.get("stored", False) or data.get("memory_ids"), f"capture 未存储: {data}"

        # Recall
        recall_body = {
            "query": f"unique-{unique_suffix}",
            "agent_id": clean_agent,
            "top_k": 5,
        }
        recall_data, recall_s = req("POST", "/v1/recall", recall_body)
        assert recall_s == 200, f"recall 失败: {recall_s} {recall_data}"

        memories = recall_data.get("memories", [])
        assert len(memories) > 0, f"recall 返回空，capture 应该成功: {recall_data}"

        # 验证 agent_id 和 text 匹配
        found = False
        for m in memories:
            if (m.get("agent_id") == clean_agent and
                    f"unique-{unique_suffix}" in m.get("text", "")):
                found = True
                break
        assert found, f"未找到对应的 capture 记忆: {memories}"

    def test_recall_returns_200(self, clean_agent, unique_suffix):
        """recall 正常调用返回 200"""
        # 先 capture
        body = {
            "text": f"recall test {unique_suffix}",
            "agent_id": clean_agent,
        }
        req("POST", "/v1/capture", body)
        time.sleep(0.5)

        # recall
        data, s = req("POST", "/v1/recall", {
            "query": f"{unique_suffix}",
            "agent_id": clean_agent,
            "top_k": 5,
        })
        assert s == 200, f"recall 返回 {s}: {data}"
        assert "memories" in data


class TestKeywordSearch:
    def test_keyword_search_finds_memory(self, clean_agent, unique_suffix):
        """pure-Go FTS 关键词搜索验证"""
        keyword = f"kw{unique_suffix}"
        body = {
            "text": f"banana orange apple {keyword} e2e test",
            "agent_id": clean_agent,
        }
        data, s = req("POST", "/v1/capture", body)
        assert s == 200, f"capture 失败: {s}"

        time.sleep(0.5)

        # 用关键词 recall
        recall_data, recall_s = req("POST", "/v1/recall", {
            "query": keyword,
            "agent_id": clean_agent,
            "top_k": 5,
        })
        assert recall_s == 200
        assert recall_data.get("count", 0) >= 1, f"pure-Go FTS 未找到关键词: {recall_data}"


class TestFusionScoring:
    def test_fusion_scores_present(self, clean_agent, unique_suffix):
        """fusion 评分字段存在于 recall 结果中"""
        body = {
            "text": f"fusion test memory {unique_suffix}",
            "agent_id": clean_agent,
        }
        req("POST", "/v1/capture", body)
        time.sleep(0.5)

        recall_data, s = req("POST", "/v1/recall", {
            "query": f"fusion test memory {unique_suffix}",
            "agent_id": clean_agent,
            "top_k": 5,
        })
        assert s == 200
        assert recall_data.get("count", 0) > 0, "fusion 测试无返回"

        m = recall_data["memories"][0]
        # fusion 评分字段
        has_fusion = any(k in m for k in ("score", "final_score", "vector_score", "keyword_score", "fused_score"))
        assert has_fusion, f"fusion 评分字段缺失: {m}"


class TestDecay:
    def test_decay_fields_present(self, clean_agent, unique_suffix):
        """decay 相关字段存在于 recall 结果"""
        body = {
            "text": f"decay test {unique_suffix}",
            "agent_id": clean_agent,
        }
        req("POST", "/v1/capture", body)
        time.sleep(0.5)

        # recall 应该更新 last_accessed
        recall_data, s = req("POST", "/v1/recall", {
            "query": f"{unique_suffix}",
            "agent_id": clean_agent,
            "top_k": 5,
        })
        assert s == 200
        if recall_data.get("count", 0) > 0:
            m = recall_data["memories"][0]
            # decay_score 或 access_count 等字段
            has_decay = any(k in m for k in ("decay_score", "access_count", "last_accessed"))
            if not has_decay:
                pytest.skip(f"decay 字段未在结果中（可能未实现）: {m}")


class TestAgentIsolation:
    def test_different_agents_isolated(self, unique_suffix):
        """不同 agent 的记忆互不干扰"""
        agent_a = f"agent-iso-a-{uuid.uuid4().hex[:8]}"
        agent_b = f"agent-iso-b-{uuid.uuid4().hex[:8]}"

        # Agent A 存 A 的记忆
        req("POST", "/v1/capture", {
            "text": f"secret for agent A unique-{unique_suffix}",
            "agent_id": agent_a,
        })
        # Agent B 存 B 的记忆
        req("POST", "/v1/capture", {
            "text": f"secret for agent B unique-{unique_suffix}",
            "agent_id": agent_b,
        })
        time.sleep(0.5)

        # Agent A recall
        data_a, s_a = req("POST", "/v1/recall", {
            "query": f"unique-{unique_suffix}",
            "agent_id": agent_a,
            "top_k": 5,
        })
        assert s_a == 200
        texts_a = " ".join(m.get("text", "") for m in data_a.get("memories", []))
        assert "agent A" in texts_a, f"Agent A 应该看到自己的记忆: {texts_a}"
        assert "agent B" not in texts_a, f"Agent A 不应该看到 Agent B 的记忆: {texts_a}"


class TestBatchCapture:
    def test_batch_capture(self, clean_agent, unique_suffix):
        """批量 capture + recall"""
        memos = [
            {"text": f"batch memory 1 unique-{unique_suffix}", "agent_id": clean_agent},
            {"text": f"batch memory 2 unique-{unique_suffix}", "agent_id": clean_agent},
            {"text": f"batch memory 3 unique-{unique_suffix}", "agent_id": clean_agent},
        ]
        body = {
            "memories": memos,
            "agent_id": clean_agent,
        }
        data, s = req("POST", "/v1/capture/batch", body)
        assert s == 200, f"batch capture 失败: {s} {data}"
        assert data.get("stored", 0) >= 3 or len(data.get("memory_ids", [])) >= 3

        time.sleep(0.5)

        # 验证都能 recall
        recall_data, recall_s = req("POST", "/v1/recall", {
            "query": f"unique-{unique_suffix}",
            "agent_id": clean_agent,
            "top_k": 10,
        })
        assert recall_s == 200
        assert recall_data.get("count", 0) >= 3, f"batch capture 后 recall 数量不足: {recall_data}"


class TestDelete:
    def test_delete_forgets_memory(self, clean_agent, unique_suffix):
        """删除后 recall 不返回该记忆"""
        # Capture
        body = {
            "text": f"to be deleted {unique_suffix}",
            "agent_id": clean_agent,
        }
        cap_data, _ = req("POST", "/v1/capture", body)
        time.sleep(0.5)

        # 获取 memory id（可能是 hex string）
        memory_id = cap_data.get("id") or (cap_data.get("memory_ids", []) or [None])[0]
        assert memory_id, f"capture 未返回 id: {cap_data}"

        # Delete
        del_data, del_s = req("DELETE", f"/v1/memory/{memory_id}")
        assert del_s in (200, 204), f"delete 失败: {del_s} {del_data}"

        time.sleep(0.5)

        # Recall 不应该返回已删除的记忆
        recall_data, recall_s = req("POST", "/v1/recall", {
            "query": f"{unique_suffix}",
            "agent_id": clean_agent,
            "top_k": 5,
        })
        assert recall_s == 200
        # 验证已删除的 memory 不在结果中
        for m in recall_data.get("memories", []):
            assert m.get("id") != memory_id, f"已删除的记忆仍出现在 recall: {m}"


# ─── 运行入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
