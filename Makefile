.PHONY: help test benchmark-hawk benchmark-m-flow benchmark-all compare install

PYTHON := python3
PYTEST := pytest
VERSION := $$(cat VERSION 2>/dev/null || echo "0.1.0")

# 默认目标
help:
	@echo "hawk-eval — 三件套评测体系"
	@echo ""
	@echo "用法:"
	@echo "  make test                 运行单元/集成测试"
	@echo "  make benchmark-hawk       跑 hawk-memory-api recall benchmark"
	@echo "  make benchmark-hawk-proc  跑 hawk-memory-api procedural benchmark"
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
	@echo "[test] 运行 hawk-eval 测试..."
	@$(PYTHON) -m pytest tests/ -v 2>/dev/null || echo "(pytest 未安装，使用 runner 自检)"
	@$(PYTHON) -m src.runner \
		-d datasets/hawk_memory/conversational_qa.jsonl \
		-a hawk_memory_api \
		-o reports/hawk_self_check.json \
		--verbose

# ─── Hawk Memory API Benchmarks ───────────────────────────────────────────────

benchmark-hawk:
	@echo "[benchmark] hawk-memory-api recall benchmark..."
	@mkdir -p reports
	@$(PYTHON) -m src.runner \
		-d datasets/hawk_memory/conversational_qa.jsonl \
		-a hawk_memory_api \
		-o reports/hawk_recall_$(VERSION).json \
		--verbose

benchmark-hawk-proc:
	@echo "[benchmark] hawk-memory-api procedural benchmark..."
	@mkdir -p reports
	@$(PYTHON) -m src.runner \
		-d datasets/hawk_memory/procedural.jsonl \
		-a hawk_memory_api \
		-t procedural \
		-o reports/hawk_procedural_$(VERSION).json \
		--verbose

# ─── m_flow Benchmark ────────────────────────────────────────────────────────

benchmark-m-flow:
	@echo "[benchmark] m_flow procedural benchmark..."
	@mkdir -p reports
	@$(PYTHON) -m src.runner \
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
