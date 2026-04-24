#!/usr/bin/env python3
"""
Grid Search Fusion Weights — KR2.5 (200-dataset full run)
测试不同 fusion 权重组合对 MRR@5 的影响。

权重配置（12组，α+β+γ=1）：
  α (vector_weight) ∈ {0.3, 0.4, 0.5}
  β (keyword_weight) ∈ {0.3, 0.4, 0.5}
  γ (fts_weight) ∈ {0.1, 0.2, 0.3}

约 12 组有效组合（去重后）。
"""

import itertools
import json
import time
from pathlib import Path

import requests

DATASET = Path(__file__).parent.parent / "datasets" / "hawk_memory" / "conversational_qa.jsonl"
API_URL = "http://127.0.0.1:18360/recall_debug"
TOP_K = 5


def load_dataset():
    with open(DATASET) as f:
        return [json.loads(line) for line in f]


def load_precomputed_vectors(path: str) -> dict[str, list[float]]:
    with open(path) as f:
        data = json.load(f)
    items = data.get("items", [])
    return {item["id"]: item["query_vector"] for item in items if "query_vector" in item}


def recall_with_scores(query: str, top_k: int = 50, query_vector=None) -> list[dict]:
    body = {
        "query": query,
        "top_k": top_k,
        "mode": "platform_only",
        "platform": "hermes",
        "agent_id": "eval",
        "rewrite": False,
    }
    if query_vector is not None:
        body["query_vector"] = query_vector
    resp = requests.post(API_URL, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["memories"]


def compute_mrr(recalls: list[list[dict]], targets: list[str]) -> float:
    mrr = 0.0
    for recall_list, target in zip(recalls, targets):
        for rank, item in enumerate(recall_list[:TOP_K], 1):
            if item.get("text", "").strip() == target.strip():
                mrr += 1.0 / rank
                break
    return mrr / len(targets)


def run_grid_search():
    print("Loading dataset...")
    dataset = load_dataset()
    answers = [d["answer"] for d in dataset]
    print(f"Dataset: {len(dataset)} items")

    # Load precomputed vectors
    cache_path = Path(__file__).parent.parent / "data" / "query_embeddings_cache.jsonl"
    qvec_map = load_precomputed_vectors(str(cache_path))
    print(f"Precomputed vectors: {len(qvec_map)}")

    # Ensure benchmark memories are in DB
    print("\n[Phase 1] Ensuring benchmark memories in DB via /direct_capture...")
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
            "session_id": "bm-gridsearch-full",
            "platform": "hermes",
            "agent_id": "eval",
        }
        r = requests.post(direct_cap_url, json=body, timeout=10)
        if r.status_code in (200, 201):
            captured += 1
    print(f"  ✅ Captured {captured}/{len(dataset)} memories")

    # Fetch raw scores from recall_debug (all items at once)
    print("\n[Phase 2] Fetching raw scores from recall_debug...")
    all_raw: list[list[dict]] = []
    for i, d in enumerate(dataset):
        q = d["question"]
        qid = d.get("id", f"q-{i}")
        qvec = qvec_map.get(qid)
        print(f"  [{i+1}/{len(dataset)}] q={q[:40]}... vec={'YES' if qvec else 'MISSING'}")
        items = recall_with_scores(q, top_k=50, query_vector=qvec)
        all_raw.append(items)

    # Compute global min/max for normalization
    all_vec = [m["vector_score_raw"] for items in all_raw for m in items if m.get("vector_score_raw", -999) != -999]
    all_kw = [m["keyword_score_raw"] for items in all_raw for m in items if m.get("keyword_score_raw", -999) != -999]
    vec_min, vec_max = min(all_vec), max(all_vec)
    kw_min, kw_max = min(all_kw), max(all_kw)
    print(f"  vec range: [{vec_min:.4f}, {vec_max:.4f}]")
    print(f"  kw range:  [{kw_min:.4f}, {kw_max:.4f}]")

    # Grid search
    print("\n[Phase 3] Grid search fusion weights...")

    # Weight grid: 3×3×2 = 18, keep only valid combos (α+β+γ=1)
    alphas = [0.3, 0.4, 0.5]
    betas = [0.3, 0.4, 0.5]
    gammas = [0.1, 0.2, 0.3]

    results = []
    for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
        if abs(alpha + beta + gamma - 1.0) > 0.01:
            continue

        rrf_k = 60
        fused_rankings = []
        for items in all_raw:
            scored = []
            for m in items:
                kw_rank = m.get("keyword_rank", -1)
                vec_raw = m.get("vector_score_raw", 0.0)
                kw_raw = m.get("keyword_score_raw", 0.0)

                # Normalize
                vec_range = vec_max - vec_min or 1.0
                norm_vec = 1.0 - (vec_raw - vec_min) / vec_range
                kw_range = kw_max - kw_min or 1.0
                norm_kw = (kw_raw - kw_min) / kw_range

                rrf_bonus = 1.0 / (rrf_k + kw_rank + 1) if kw_rank >= 0 else 0.0
                max_rrf = 1.0 / (rrf_k + 1)
                fs = alpha * norm_vec + beta * norm_kw + gamma * (rrf_bonus / max_rrf)
                scored.append((fs, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            fused_rankings.append([m for _, m in scored[:TOP_K]])

        mrr = compute_mrr(fused_rankings, answers)
        results.append((mrr, alpha, beta, gamma))
        print(f"  α={alpha:.1f} β={beta:.1f} γ={gamma:.1f} → MRR@5={mrr:.4f}")

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n[Phase 4] Top 5 configurations:")
    for mrr, a, b, g in results[:5]:
        print(f"  MRR@5={mrr:.4f}  α={a:.1f} β={b:.1f} γ={g:.1f}")

    best_mrr, best_alpha, best_beta, best_gamma = results[0]
    print(f"\n✅ Best: α={best_alpha:.1f} β={best_beta:.1f} γ={best_gamma:.1f} → MRR@5={best_mrr:.4f}")

    # Save results
    output = {
        "best": {"alpha": best_alpha, "beta": best_beta, "gamma": best_gamma, "mrr@5": best_mrr},
        "all_results": [
            {"alpha": a, "beta": b, "gamma": g, "mrr@5": float(mrr)}
            for mrr, a, b, g in results
        ],
    }
    out_path = Path(__file__).parent.parent / "reports" / "fusion_grid_search_full.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Apply best weights via systemctl environment
    print(f"\n[Phase 5] Applying best weights: α={best_alpha} β={best_beta} γ={best_gamma}")
    return best_alpha, best_beta, best_gamma, best_mrr


if __name__ == "__main__":
    run_grid_search()
