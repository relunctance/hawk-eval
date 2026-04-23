"""
Adapter: hawk-memory-api
"""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HawkMemoryAdapter:
    """
    连接到本地 hawk-memory-api 进行 recall 评测。

    用法:
        adapter = HawkMemoryAdapter(base_url="http://127.0.0.1:18360")
        result = adapter.recall("Python 编程", top_k=10)
    """

    base_url: str = "http://127.0.0.1:18360"
    timeout: int = 10
    platform: str = "eval"
    latency_results: list[float] = field(default_factory=list)

    def health_check(self) -> bool:
        """检查服务是否可用。"""
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

    def capture(self, text: str, session_id: str, user_id: str = "eval",
                question: str = None, answer: str = None) -> dict:
        """
        存入一条记忆。

        支持两种模式：
        - legacy: text=待存储文本（作为 message，response 为空）
        - qa:     question + answer 作为完整对话存储
                 → 存储 "用户: {question}" 和 "助手: {answer}"
                 → recall(query=question) 时可匹配到含 answer 的记忆
        """
        body = {
            "session_id": session_id,
            "user_id": user_id,
            "platform": self.platform,
        }
        if question is not None and answer is not None:
            # QA 模式：存储完整对话
            body["message"] = question
            body["response"] = answer
        else:
            # Legacy 模式：text 作为 message
            body["message"] = text
            body["response"] = ""
        data, _ = self._req("POST", "/capture", body)
        return data

    def recall(
        self,
        query: str,
        top_k: int = 10,
        platform: str = None,
    ) -> dict[str, Any]:
        """
        执行 recall，返回 {"memories": [...], "latency": float}

        memories 每项包含 id / text / score / category
        """
        body = {"query": query, "top_k": top_k}
        if platform:
            body["platform"] = platform
        else:
            body["platform"] = self.platform

        t0 = time.perf_counter()
        data, status = self._req("POST", "/recall", body)
        latency = time.perf_counter() - t0
        self.latency_results.append(latency)

        if status != 200:
            return {"memories": [], "error": data, "latency": latency}

        return {
            "memories": data.get("memories", []),
            "count": data.get("count", 0),
            "latency": latency,
        }

    def get_stats(self) -> dict:
        """获取统计信息。"""
        data, _ = self._req("GET", "/stats")
        return data

    def latency_stats(self) -> dict[str, float]:
        """返回延迟统计。"""
        if not self.latency_results:
            return {"p50": 0.0, "p99": 0.0, "mean": 0.0}
        sorted_lat = sorted(self.latency_results)
        n = len(sorted_lat)
        return {
            "p50": sorted_lat[int(n * 0.5)],
            "p99": sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0],
            "mean": sum(sorted_lat) / n,
        }

    # ─── 内部 ──────────────────────────────────────────────

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
