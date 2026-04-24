.PHONY: help test benchmark-hawk benchmark-hawk-proc benchmark-m-flow benchmark-locomo benchmark-evolving benchmark-all benchmark-multi-agent compare install benchmark-preflight benchmark-hawk-clean

PYTHON := python3
PYTEST := pytest
VERSION := $$(cat VERSION 2>/dev/null || echo "0.1.0")

# 默认目标
help:
	@echo "hawk-eval — 三件套评测体系"
	@echo ""
	@echo "用法:"
	@echo "  make test             运行 pytest 测试套件（12 项集成测试，~35s）"
	@echo "  make test-smoke       快速冒烟测试（~15s，核心路径）"
	@echo "  make test-self-check  自检脚本（无需网络）"
	@echo "  make benchmark-hawk       跑 hawk-memory-api recall benchmark"
	@echo "  make precompute          预计算 query embeddings 缓存（一次性，约30s）"
	@echo "  make benchmark-hawk-quick    快速冒烟测试（20条，~2min）"
	@echo "  make benchmark-preflight    运行 benchmark 前置检查"
	@echo "  make benchmark-hawk-clean  清理 eval 残留后跑 benchmark"
	@echo "  make benchmark-hawk-proc    跑 hawk-memory-api procedural benchmark"
	@echo "  make benchmark-m-flow     跑 m_flow procedural benchmark"
	@echo "  make benchmark-all        跑完整竞品对比"
	@echo "  make compare              生成竞品对比报告"
	@echo "  make install              安装依赖"
	@echo ""
	@echo "前提条件:"
	@echo "  hawk-memory-api 必须先启动: cd ~/repos/hawk-memory-api && ./run.sh"
	@echo "  m_flow 必须先启动:         cd ~/repos/m_flow && python -m m_flow.api.client"

# ─── 依赖 ──────────────────────────────────────────────────────────────────

install:
	@echo "安装依赖...（当前为纯标准库，无需额外安装）"

# ─── 测试 ────────────────────────────────────────────────────────────────────

test:
	@echo "[test] 运行 hawk-eval 测试套件..."
	@$(PYTHON) -m pytest tests/ -v --tb=short 2>&1

test-smoke:
	@echo "[test-smoke] 快速冒烟测试（<60s）..."
	@$(PYTHON) -m pytest tests/test_hawk_memory_api.py -v --tb=short -k "health or capture_simple or recall_returns or no_500" 2>&1

test-self-check:
	@echo "[test] 自检脚本（无需网络）..."
	@$(PYTHON) tests/test_runner.py 2>&1

# ─── Hawk Memory API Benchmarks ───────────────────────────────────────────────

benchmark-hawk: benchmark-preflight
	@echo "[benchmark] hawk-memory-api recall benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_hawk \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--output reports/hawk_recall_$(VERSION).json

precompute:
	@echo "[precompute] 预计算 query embeddings (200条)..."
	@$(PYTHON) scripts/precompute_query_embeddings.py \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--output data/query_embeddings_cache.json

benchmark-preflight:
	@echo "[preflight] 运行 benchmark 前置检查..."
	@PYTHONPATH=src $(PYTHON) scripts/benchmark_preflight.py

benchmark-hawk-quick:
	@echo "[benchmark] hawk-memory-api recall benchmark (quick 20条)..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_hawk \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--limit 20 \
		--output reports/hawk_recall_quick.json

benchmark-hawk-clean:
	@echo "[clean] 清理 eval 残留数据 + 跑 benchmark..."
	@curl -s -X POST "http://127.0.0.1:18360/admin/cleanup" \
		-H "Content-Type: application/json" \
		-d '{"agent_id":"eval"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'已清理 {d.get(\"deleted\",0)} 条 eval 记忆')"
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_hawk \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--output reports/hawk_recall_$(VERSION).json

benchmark-hawk-proc:
	@echo "[benchmark] hawk-memory-api procedural benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.runner \
		-d datasets/hawk_memory/procedural.jsonl \
		-a hawk_memory_api \
		-t procedural \
		-o reports/hawk_procedural_$(VERSION).json \
		--verbose

# ─── LoCoMo-10 Benchmark ───────────────────────────────────────────────────────

benchmark-locomo:
	@echo "[benchmark] LoCoMo-10 recall benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_locomo \
		--dataset datasets/locomo/locomo_qa.jsonl \
		--output reports/locomo_recall.json

# ─── Evolving Events Benchmark ───────────────────────────────────────────────

benchmark-evolving:
	@echo "[benchmark] Evolving Events recall benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_evolving_events \
		--dataset datasets/evolving_events/evolving_events_qa.jsonl \
		--output reports/evolving_events_recall.json

# ─── m_flow Benchmark ────────────────────────────────────────────────────────

benchmark-m-flow:
	@echo "[benchmark] m_flow procedural benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.runner \
		-d datasets/m_flow_procedural/procedural_eval_v1.jsonl \
		-a m_flow \
		-t procedural \
		-o reports/m_flow_procedural_$(VERSION).json \
		--verbose

# ─── 完整竞品对比 ──────────────────────────────────────────────────────────────

benchmark-all: benchmark-hawk benchmark-m-flow
	@echo "[benchmark] 对比报告..."
	@$(PYTHON) -m src.report \
		-r reports/hawk_recall_$(VERSION).json \
		   reports/m_flow_procedural_$(VERSION).json \
		-o reports/compare_$(VERSION).md

benchmark-multi-agent:
	@echo "[benchmark] 多 agent 评测（KR3.4）..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_multi_agent \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--agents user1,user2,user3 \
		--output reports/multi_agent.json

# ─── 报告 ────────────────────────────────────────────────────────────────────

compare:
	@$(PYTHON) -m src.report \
		-r reports/hawk_recall_$(VERSION).json \
		   reports/m_flow_procedural_$(VERSION).json \
		-o reports/compare.md

# ─── 初始化基准 ────────────────────────────────────────────────────────────────

baseline:
	@echo "[baseline] 保存初始基准..."
	@mkdir -p baselines
	@$(PYTHON) -m src.runner \
		-d datasets/hawk_memory/conversational_qa.jsonl \
		-a hawk_memory_api \
		-o baselines/hawk_baseline.json \
		--verbose
	@echo "基准已保存: baselines/hawk_baseline.json"
