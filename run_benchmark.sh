#!/bin/bash
cd ~/repos/hawk-eval
export PYTHONPATH=src
LOGFILE="logs/benchmark_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
python3 -m src.benchmark_hawk \
  --dataset datasets/hawk_memory/conversational_qa.jsonl \
  --mode both \
  --top-k 5 \
  --output "reports/hawk_recall_full_$(date +%Y%m%d_%H%M%S).json" \
  2>&1 | tee "$LOGFILE"
