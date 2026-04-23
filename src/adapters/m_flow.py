"""
Adapter: m_flow
通过 HTTP API 调用 m_flow 的 retrieval 端点
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MFlowAdapter:
    """
    连接 m_flow API 进行 recall 评测。

    需要先启动 m_flow: cd ~/repos/m_flow && python -m m_flow.api.client
    默认端口 8000，或通过 M_FLOW_BASE_URL 环境变量指定。
    """

    base_url: str = field(
        default_factory=lambda: os.getenv("M_FLOW_BASE_URL", "http://127.0.0.1:8000")
    )
    timeout: int = 10

    def health_check(self) -> bool:
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

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """
        调用 m_flow /memorize/search 接口。
        返回格式需与 hawk-memory-api 对齐：
        {"memories": [{"id": ..., "text": ..., "score": ..., "category": ...}]}
        """
        body = {"query": query, "top_k": top_k}
        data, status = self._req("POST", "/memorize/search", body)
        if status != 200:
            return {"memories": [], "error": str(data)}
        # m_flow 返回格式可能不同，做一次转换对齐
        return self._normalize(data)

    def _normalize(self, data: dict) -> dict:
        """把 m_flow 的返回格式对齐到 hawk-eval 统一格式。"""
        memories = []
        # m_flow 可能的返回字段：items / results / episodes 等
        items = data.get("items") or data.get("results") or data.get("episodes") or []
        for item in items:
            memories.append({
                "id": item.get("id") or item.get("episode_id", ""),
                "text": item.get("text") or item.get("content", ""),
                "score": item.get("score", 1.0),
                "category": item.get("type") or item.get("category", "unknown"),
            })
        return {"memories": memories}

    def _req(self, method: str, path: str, body: dict = None) -> tuple[Any, int]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"}
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
