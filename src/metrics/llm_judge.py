"""
LLM as Judge — 用 LLM 判断答案正确性
"""

import json
import os
import urllib.request
import urllib.error

# 支持多 provider
PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _openai_judge(question: str, reference: str, response: str) -> float:
    """用 OpenAI 判断答案是否正确（0 或 1）。"""
    prompt = f"""你是一个答案评审员。给定一个问题、参考答案和模型回答，判断回答是否正确。

问题: {question}
参考答案: {reference}
模型回答: {response}

只返回一个 JSON：{{"correct": 0 或 1, "reason": "简短原因"}}
"""

    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    url = f"{OPENAI_BASE}/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        content = result["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        return float(data.get("correct", 0))
    except Exception:
        return 0.0


def llm_judge(question: str, reference: str, response: str) -> float:
    """判断答案是否正确，返回 0.0 或 1.0。"""
    if not OPENAI_API_KEY:
        # 无 key 时跳过，返回 None 表示不支持
        return -1.0

    if PROVIDER == "openai":
        return _openai_judge(question, reference, response)
    # 其他 provider 后续扩展
    return -1.0
