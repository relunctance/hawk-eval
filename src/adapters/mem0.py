"""
Adapter: Mem0
通过 Mem0 API 进行 recall 评测（竞品对标）
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "https://api.mem0.ai")


@dataclass
class Mem0Adapter:
    """
    连接 Mem0 API 进行 recall 评测。

    需要设置 MEM0_API_KEY 环境变量。
    """

    base_url: str = MEM0_BASE_URL
    api_key: str = MEM0_API_KEY
    org_id: str = os.getenv("MEM0_ORG_ID", "")
    project_id: str = os.getenv("MEM0_PROJECT_ID", "")
    timeout: int = 15

    def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/memories/search",
                data=json.dumps({"query": "test", "top_k": 1}).encode(),
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """
        Mem0 /v1/memories/search 接口。
        返回格式对齐：
        {"memories": [{"id": ..., "text": ..., "score": ...}]}
        """
        body = {
            "query": query,
            "top_k": top_k,
        }
        if self.org_id:
            body["org_id"] = self.org_id
        if self.project_id:
            body["project_id"] = self.project_id

        data, status = self._req("POST", "/v1/memories/search", body)
        if status != 200:
            return {"memories": [], "error": str(data)}

        return self._normalize(data)

    def _normalize(self, data: dict) -> dict:
        """对齐到 hawk-eval 统一格式。"""
        memories = []
        for item in data.get("results", []) or data.get("memories", []):
            memories.append({
                "id": item.get("id", ""),
                "text": item.get("text", "") or item.get("memory", ""),
                "score": item.get("score", 1.0),
                "category": item.get("category", "unknown"),
            })
        return {"memories": memories}

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _req(self, method: str, path: str, body: dict = None) -> tuple[Any, int]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
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
