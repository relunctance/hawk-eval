#!/usr/bin/env python3
"""
Grid Search Fusion Weights — KR2.5
测试不同 fusion 权重组合对 MRR@5 的影响。

权重配置:
  α (alpha) = vector/semantic weight  (当前 0.75)
  β (beta)  = keyword weight          (当前 0.15)
  γ (gamma)  = RRF rank bonus weight   (当前 0.10)
  rrf_k = 60 (固定)

测试空间（精简，去掉无效组合 α+β+γ=1）：
  α ∈ {0.50, 0.60, 0.70, 0.75, 0.80}
  β ∈ {0.10, 0.15, 0.20, 0.25}
  γ ∈ {0.05, 0.10, 0.15}

总计约 5×4×3 = 60 组合（去重后约 40 组）
"""

import json
import itertools
import requests
import time
from pathlib import Path

DATASET = Path(__file__).parent.parent / "datasets" / "hawk_memory" / "conversational_qa.jsonl"
API_URL = "http://127.0.0.1:18360/recall_debug"
TOP_K = 5


def load_dataset():
    with open(DATASET) as f:
        return [json.loads(line) for line in f]


def load_precomputed_vectors(path: str) -> dict[str, list[float]]:
    """Load precomputed query embeddings from precompute script output."""
    with open(path) as f:
        data = json.load(f)  # JSON (not JSONL): {"version":"v1","items":[...]}
    items = data.get("items", [])
    return {item["id"]: item["query_vector"] for item in items if "query_vector" in item}


def recall_with_scores(query: str, top_k: int = 20, query_vector: list[float] | None = None) -> list[dict]:
    """Call recall_debug endpoint, return raw scores for all candidates.
    
    Uses platform_only + agent_id=eval to match benchmark mode.
    Does NOT use rewrite (benchmark also doesn't use rewrite).
    """
    body = {
        "query": query,
        "top_k": top_k,
        "mode": "platform_only",
        "platform": "hermes",
        "agent_id": "eval",
        "rewrite": False,  # match benchmark mode
    }
    if query_vector is not None:
        body["query_vector"] = query_vector
    resp = requests.post(
        API_URL,
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["memories"]  # list of MemoryItemDebug


def fused_score(
    vec_raw: float,
    kw_raw: float,
    kw_rank: int,
    alpha: float,
    beta: float,
    gamma: float,
    vec_min: float,
    vec_max: float,
    kw_min: float,
    kw_max: float,
    rrf_k: int = 60,
) -> float:
    """Compute fused score given raw scores and weights."""
    vec_range = vec_max - vec_min or 1.0
    norm_vec = 1.0 - (vec_raw - vec_min) / vec_range  # distance → similarity

    kw_range = kw_max - kw_min or 1.0
    norm_kw = (kw_raw - kw_min) / kw_range

    rrf_bonus = 1.0 / (rrf_k + kw_rank + 1) if kw_rank >= 0 else 0.0
    max_rrf = 1.0 / (rrf_k + 1)

    return alpha * norm_vec + beta * norm_kw + gamma * (rrf_bonus / max_rrf)


def compute_mrr(recalls: list[list[dict]], targets: list[str]) -> float:
    """Compute MRR@5 given recall results and target answer texts."""
    mrr = 0.0
    for recall_list, target in zip(recalls, targets):
        for rank, item in enumerate(recall_list[:TOP_K], 1):
            if item.get("text", "").strip() == target.strip():
                mrr += 1.0 / rank
                break
    return mrr / len(targets)


def text_similar(a: str, b: str) -> bool:
    """Simple text similarity (exact for benchmark)."""
    return a.strip() == b.strip()


def run_grid_search():
    print("Loading dataset...")
    dataset = load_dataset()
    answers = [d["answer"] for d in dataset]

    print(f"Dataset: {len(dataset)} items")

    # Phase 0: Load precomputed vectors (same as benchmark)
    print("\n[Phase 0] Loading precomputed query vectors...")
    cache_path = Path(__file__).parent.parent / "data" / "query_embeddings_cache.jsonl"
    qvec_map = load_precomputed_vectors(str(cache_path))
    print(f"  Loaded {len(qvec_map)} precomputed vectors")

    # Phase 0b: Ensure benchmark memories are in DB via /direct_capture
    print("\n[Phase 0b] Capturing benchmark memories via /direct_capture...")
    direct_cap_url = "http://127.0.0.1:18360/direct_capture"
    captured = 0
    for d in dataset:
        answer = d.get("answer") or d.get("memory_text") or ""
        if not answer:
            continue
        body = {
            "memories": [{
                "text": answer,
                "category": "other",
                "importance": 1.0,
                "name": "",
                "description": "",
                "metadata": {"question": d.get("question", "")},
            }],
            "session_id": f"bm-gridsearch",
            "platform": "hermes",
            "agent_id": "eval",
        }
        r = requests.post(direct_cap_url, json=body, timeout=10)
        if r.status_code in (200, 201):
            captured += 1
    print(f"  ✅ Captured {captured}/{len(dataset)} memories")

    # Phase 1: collect all raw scores (using precomputed vectors like benchmark)
    print("\n[Phase 1] Fetching raw scores from recall_debug (with precomputed vectors)...")
    all_raw: list[list[dict]] = []
    for i, d in enumerate(dataset):
        q = d["question"]
        qid = d.get("id", f"q-{i}")
        qvec = qvec_map.get(qid)
        print(f"  [{i+1}/{len(dataset)}] {q[:50]}... vec={'YES' if qvec else 'MISSING'}")
        items = recall_with_scores(q, top_k=20, query_vector=qvec)
        all_raw.append(items)

    # Collect global min/max for normalization
    all_vec = [m["vector_score_raw"] for items in all_raw for m in items if m.get("vector_score_raw", -999) != -999]
    all_kw = [m["keyword_score_raw"] for items in all_raw for m in items if m.get("keyword_score_raw", -999) != -999]
    vec_min, vec_max = min(all_vec), max(all_vec)
    kw_min, kw_max = min(all_kw), max(all_kw)
    print(f"  vec range: [{vec_min:.4f}, {vec_max:.4f}]")
    print(f"  kw range:  [{kw_min:.4f}, {kw_max:.4f}]")

    # Phase 2: grid search
    print("\n[Phase 2] Grid search fusion weights...")

    # Weight grid (skip invalid combos where α+β+γ != 1.0 approximately)
    alphas = [0.50, 0.60, 0.70, 0.75, 0.80]
    betas = [0.10, 0.15, 0.20, 0.25]
    gammas = [0.05, 0.10, 0.15]

    results = []
    for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
        # Skip combos that don't sum to ~1.0 (allow small tolerance)
        if abs(alpha + beta + gamma - 1.0) > 0.01:
            continue

        # Compute fused scores for each item's candidate list
        fused_rankings = []
        for items in all_raw:
            scored = []
            for m in items:
                kw_rank = m.get("keyword_rank", -1)
                fs = fused_score(
                    vec_raw=m.get("vector_score_raw", 0.0),
                    kw_raw=m.get("keyword_score_raw", 0.0),
                    kw_rank=kw_rank,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    vec_min=vec_min,
                    vec_max=vec_max,
                    kw_min=kw_min,
                    kw_max=kw_max,
                )
                scored.append((fs, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            fused_rankings.append([m for _, m in scored[:TOP_K]])

        mrr = compute_mrr(fused_rankings, answers)
        results.append((mrr, alpha, beta, gamma))
        print(f"  α={alpha:.2f} β={beta:.2f} γ={gamma:.2f} → MRR@5={mrr:.4f}")

    # Sort by MRR
    results.sort(key=lambda x: x[0], reverse=True)

    print("\n[Phase 3] Top 5 configurations:")
    for mrr, a, b, g in results[:5]:
        print(f"  MRR@5={mrr:.4f}  α={a:.2f} β={b:.2f} γ={g:.2f}")

    best_mrr, best_alpha, best_beta, best_gamma = results[0]
    print(f"\n✅ Best: α={best_alpha:.2f} β={best_beta:.2f} γ={best_gamma:.2f} → MRR@5={best_mrr:.4f}")

    # Save results
    output = {
        "best": {"alpha": best_alpha, "beta": best_beta, "gamma": best_gamma, "mrr@5": best_mrr},
        "all_results": [
            {"alpha": a, "beta": b, "gamma": g, "mrr@5": mrr}
            for mrr, a, b, g in results
        ],
    }
    out_path = Path(__file__).parent.parent / "reports" / "fusion_grid_search.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return best_alpha, best_beta, best_gamma, best_mrr


if __name__ == "__main__":
    run_grid_search()
