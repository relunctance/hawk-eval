#!/bin/bash
# Grid search fusion weights by restarting hawk-memory-api with different HAWK_FUSION_* env vars.
# Then runs benchmark and reads MRR from JSON report.
#
# Usage: bash scripts/grid_search_fusion.sh
#
# Requires hawk-memory-api to be installed as a systemd user service.
# Restart is needed because fusion weights are baked into the lancedb.py at import time.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EVAL_DIR"

HAWK_MEMORY_API_DIR="$HOME/repos/hawk-memory-api"
API_URL="http://127.0.0.1:18360"
REPORT_FILE="reports/grid_search.json"

# Check if service is running
if ! systemctl --user is-active hawk-memory-api &>/dev/null; then
    echo "ERROR: hawk-memory-api service is not running"
    exit 1
fi

echo "=== Grid Search: Fusion Weight Optimization ==="
echo "Base MRR (alpha=0.75 beta=0.15 gamma=0.10):"
echo ""

# Clear existing eval data
curl -s -X POST "$API_URL/admin/cleanup?agent_id=eval" > /dev/null

# Grid search space (alpha + beta + gamma must sum to 1.0)
# We'll vary alpha from 0.5 to 0.9, beta from 0.05 to 0.3, gamma fixed at 0.10

declare -A results

for alpha in 0.50 0.60 0.70 0.75 0.80 0.85 0.90; do
    for beta in 0.05 0.10 0.15 0.20 0.30; do
        gamma=$(python3 -c "print(round(1.0 - $alpha - $beta, 2))")
        if (( $(echo "$gamma < 0.01" | bc -l) )); then
            continue
        fi

        echo "Testing: alpha=$alpha beta=$beta gamma=$gamma"

        # Restart with new weights
        HAWK_FUSION_ALPHA=$alpha HAWK_FUSION_BETA=$beta HAWK_FUSION_GAMMA=$gamma \
            systemctl --user restart hawk-memory-api

        # Wait for service to be ready
        sleep 2
        for i in {1..10}; do
            if curl -s "$API_URL/health" | grep -q '"status":"ok"'; then
                break
            fi
            sleep 1
        done

        # Clear eval data again after restart
        curl -s -X POST "$API_URL/admin/cleanup?agent_id=eval" > /dev/null

        # Run benchmark (5 cases only for speed)
        python3 src/benchmark_hawk.py \
            --dataset datasets/hawk_memory/conversational_qa.jsonl \
            --top-k 5 \
            --limit 5 \
            --output "reports/grid_temp.json" 2>/dev/null | tail -1

        # Extract MRR@5
        mrr=$(python3 -c "
import json
with open('reports/grid_temp.json') as f:
    r = json.load(f)
print(r['metrics'].get('mrr@5', 0))
" 2>/dev/null || echo "0")

        echo "  -> MRR@5 = $mrr"
        results["$alpha,$beta,$gamma"]=$mrr

        # Restore default weights for next iteration
        HAWK_FUSION_ALPHA=0.75 HAWK_FUSION_BETA=0.15 HAWK_FUSION_GAMMA=0.10 \
            systemctl --user restart hawk-memory-api
        sleep 2
    done
done

# Find best
echo ""
echo "=== Results ==="
best_mrr=0
best_weights=""

for key in "${!results[@]}"; do
    mrr="${results[$key]}"
    if (( $(echo "$mrr > $best_mrr" | bc -l) )); then
        best_mrr=$mrr
        best_weights=$key
    fi
    echo "alpha,beta,gamma=$key -> MRR@5=$mrr"
done

echo ""
echo "BEST: alpha,beta,gamma=$best_weights -> MRR@5=$best_mrr"
echo "$best_weights,$best_mrr" > reports/best_fusion_weights.txt
