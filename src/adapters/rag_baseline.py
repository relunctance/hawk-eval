"""
RAG Baseline Adapter
使用纯 embedding-based retrieval 作为对比基准
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any


EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def get_embedding(text: str) -> list[float]:
    """用 OpenAI embedding 接口获取向量。"""
    if not EMBEDDING_API_KEY:
        return [0.0] * 1536  # fallback zero vector

    body = {
        "model": EMBEDDING_MODEL,
        "input": text[:8192],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
    }
    url = f"{EMBEDDING_BASE}/embeddings"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            result = json.loads(r.read())
        return result["data"][0]["embedding"]
    except Exception:
        return [0.0] * 1536


def cosine_sim(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RAGBaselineAdapter:
    """
    纯 RAG baseline：用 embedding 相似度做 recall。
    用于和 hawk-memory-api 对比，看记忆层是否优于纯向量检索。
    """

    documents: list[dict] = field(default_factory=list)
    doc_embeddings: list[list[float]] = field(default_factory=list)

    def add_documents(self, docs: list[dict]):
        """添加文档。docs 每项需有 id / text。"""
        for doc in docs:
            emb = get_embedding(doc["text"])
            self.documents.append(doc)
            self.doc_embeddings.append(emb)

    def recall(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """基于 embedding 相似度做 recall。"""
        if not self.documents:
            return {"memories": []}

        q_emb = get_embedding(query)
        scores = [cosine_sim(q_emb, d) for d in self.doc_embeddings]

        # 按分数排序
        scored = sorted(zip(scores, self.documents), key=lambda x: -x[0])
        top = scored[:top_k]

        memories = [
            {
                "id": doc.get("id", ""),
                "text": doc.get("text", ""),
                "score": float(score),
                "category": doc.get("category", "unknown"),
            }
            for score, doc in top
        ]
        return {"memories": memories}
