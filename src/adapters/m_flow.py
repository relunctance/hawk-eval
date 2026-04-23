"""
Adapter: m_flow
通过 HTTP API 调用 m_flow 的 retrieval 端点进行 recall benchmark

认证：支持 form-urlencoded login 获取 Bearer token
默认账户：default_user@example.com / default_password（容器启动时自动创建）

当 m_flow 内部无数据集导致 search 崩溃时，会降级到 MockMFlowAdapter
（返回空结果作为基准对照，不阻塞评测流程）。
"""

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MFlowAdapter:
    """
    连接 m_flow API 进行 recall 评测。

    需要先启动 m_flow（端口 8000）：
        cd ~/repos/m_flow && uv run python -m m_flow.api.client

    或通过环境变量指定地址和认证：
        M_FLOW_BASE_URL=http://127.0.0.1:8000
        M_FLOW_EMAIL=your@email.com
        M_FLOW_PASSWORD=***

    当 m_flow datasets 为空导致 search 报错时，自动降级到 MockMFlowAdapter。
    """

    base_url: str = field(
        default_factory=lambda: os.getenv("M_FLOW_BASE_URL", "http://127.0.0.1:8000")
    )
    timeout: int = 10
    _token: str = field(default=None, repr=False)
    _mock: bool = field(default=False, repr=False)

    # ─── 认证 ────────────────────────────────────────────────
    def _ensure_token(self) -> str:
        """惰性认证：首次调用时登录获取 token，之后复用。"""
        if self._token:
            return self._token

        email = os.getenv("M_FLOW_EMAIL", "default_user@example.com")
        password = os.getenv("M_FLOW_PASSWORD", "default_password")

        # form-urlencoded: FastAPI-Users 默认格式
        data = urllib.parse.urlencode({"username": email, "password": password}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/auth/login",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                result = json.loads(r.read())
                self._token = result["access_token"]
                return self._token
        except Exception as e:
            raise RuntimeError(f"m_flow 认证失败: {e}") from e

    # ─── 健康检查 ────────────────────────────────────────────
    def health_check(self) -> bool:
        if self._mock:
            return True
        try:
            req = urllib.request.Request(
                f"{self.base_url}/health",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    # ─── 核心 recall 接口（hawk-eval runner 依赖此方法）────────
    def recall(self, query: str, top_k: int = 10, mode: str = "episodic") -> dict[str, Any]:
        """
        调用 m_flow /api/v1/search，返回对齐后的 memories 列表。

        当 m_flow datasets 为空时自动降级到 MockMFlowAdapter（返回空结果）。

        对齐格式：{"memories": [{"id", "text", "score", "category"}], "latency": float}
        """
        # 降级到 mock（m_flow 无数据集时）
        if self._mock:
            return {"memories": [], "latency": 0.0}

        token = self._ensure_token()
        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "dataset_ids": None,  # 不传空列表 []，m_flow 内部对空列表有 bug
        }
        if mode != "episodic":
            body["recall_mode"] = mode.upper()

        data, status = self._req("POST", "/api/v1/search", body, token)

        # 检测 m_flow 无数据集导致的崩溃（409 或 200+error 字段）
        is_no_data = (
            status == 409
            or (status == 200 and isinstance(data, dict) and data.get("error"))
        )
        if is_no_data:
            err = str(data.get("error", "")) if isinstance(data, dict) else str(data)
            if "list index out of range" in err or "datasets" in err.lower():
                self._mock = True  # 后续全部降级
                return {"memories": [], "latency": 0.0, "_mock_fallback": True}

        if status != 200:
            return {"memories": [], "error": str(data), "latency": 0.0}
        return self._normalize(data)

    # ─── search（recall 的别名）─────────────────────────────
    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """直接调用 recall，保持向后兼容。"""
        return self.recall(query, top_k)

    # ─── 内部 ───────────────────────────────────────────────
    def _req(
        self, method: str, path: str, body: dict = None, token: str = None
    ) -> tuple[Any, int]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read()), r.status
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read()), e.code
            except Exception:
                return str(e), e.code
        except Exception as e:
            return str(e), -1

    def _normalize(self, data: dict) -> dict:
        """把 m_flow 返回格式对齐到 hawk-eval 统一格式。"""
        memories = []
        # m_flow CombinedSearchResult 结构
        result = data.get("result") or {}
        items = (
            result.get("episodes", [])
            or result.get("items", [])
            or result.get("procedures", [])
            or []
        )
        for item in items:
            memories.append({
                "id": item.get("id") or item.get("episode_id", ""),
                "text": item.get("text") or item.get("content", "")
                        or item.get("description", ""),
                "score": item.get("score", 1.0),
                "category": item.get("type") or item.get("category", "unknown"),
            })
        return {"memories": memories}


# ─── Mock Adapter（m_flow 无数据时的基准对照）───────────────────────────────


@dataclass
class MockMFlowAdapter:
    """
    Mock m_flow：当 m_flow 无数据集时，用作基准对照。

    返回空结果，隔离测试 hawk-memory-api 自身的 recall 质量。
    用于验证在没有任何干扰记忆的情况下，hawk 的 recall 表现。
    """

    def health_check(self) -> bool:
        return True

    def recall(self, query: str, top_k: int = 10, **kwargs) -> dict[str, Any]:
        """返回空结果，模拟冷启动状态。"""
        return {"memories": [], "latency": 0.0, "_mock": True}

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return self.recall(query, top_k)
