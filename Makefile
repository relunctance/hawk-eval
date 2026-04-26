.PHONY: help test benchmark-hawk benchmark-hawk-proc benchmark-m-flow benchmark-locomo benchmark-evolving benchmark-all benchmark-multi-agent compare install benchmark-preflight benchmark-hawk-clean benchmark-trigger benchmark-hawk-init

PYTHON := python3
PYTEST := pytest
VERSION := $$(cat VERSION 2>/dev/null || echo "0.1.0")

# 默认目标
help:
	@echo "hawk-eval — 评测体系"
	@echo ""
	@echo "用法:"
	@echo "  make test                    运行 pytest 测试套件"
	@echo "  make benchmark-hawk-init     ⭐ 初始化干净环境 + 跑 benchmark（推荐）"
	@echo "  make benchmark-hawk          直接跑 benchmark（不清空DB，数据可能累积）"
	@echo "  make benchmark-hawk-clean   仅清理 eval 记忆 + 跑 benchmark"
	@echo "  make benchmark-hawk-quick   快速冒烟（20条）"
	@echo "  make benchmark-preflight    前置检查"
	@echo "  make benchmark-m-flow       m_flow procedural benchmark"
	@echo "  make benchmark-all           完整竞品对比"
	@echo ""
	@echo "前提条件:"
	@echo "  hawk-memory Go 必须先启动: systemctl --user status hawk-memory"
	@echo "  xinference 必须先启动:    systemctl --user status xinference"

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

# ─── Hawk Memory Go Benchmarks ────────────────────────────────────────────────

# ⭐ 推荐：初始化干净环境 + 跑 benchmark
# 1. 检查 Go 服务健康
# 2. 删除整个 Go DB（破坏性，但保证干净）
# 3. 重启 Go 服务（自动建空 DB + FTS）
# 4. 验证 FTS 索引就绪
# 5. 跑 benchmark
benchmark-hawk-init:
	@echo "═══════════════════════════════════════"
	@echo "  benchmark-hawk-init: 干净环境初始化"
	@echo "═══════════════════════════════════════"
	@echo ""
	@echo "【1/5】检查 Go hawk-memory 服务..."
	@HEALTH=$$(curl -s --max-time 3 http://127.0.0.1:18368/health 2>/dev/null); \
	if echo "$$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then \
		echo "  ✅ hawk-memory Go 运行正常"; \
	else \
		echo "  🔴 hawk-memory Go 未响应，正在启动..."; \
		systemctl --user start hawk-memory 2>/dev/null || (cd ~/repos/hawk-memory && go run ./cmd/hawk-memory/ &); \
		sleep 5; \
	fi
	@echo ""
	@echo "【2/5】检查 xinference 服务..."
	@XINFERENCE_OK=$$(curl -s --max-time 3 http://127.0.0.1:9997/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('data') else 'down')" 2>/dev/null || echo "down"); \
	if [ "$$XINFERENCE_OK" = "ok" ]; then \
		echo "  ✅ xinference embedding 服务正常"; \
	else \
		echo "  🔴 xinference 未响应，benchmark 可能失败"; \
	fi
	@echo ""
	@echo "【3/5】停止 hawk-memory 服务（为删除 DB 做准备）..."
	@systemctl --user stop hawk-memory 2>/dev/null || true
	@sleep 2
	@echo ""
	@echo "【4/5】删除 Go DB（彻底清空，确保干净）..."
	@rm -rf /home/gql/.hawk/go-lancedb && echo "  ✅ 已删除 /home/gql/.hawk/go-lancedb" || echo "  ⚠️ DB 路径不存在，跳过"
	@echo ""
	@echo "【5/5】重启 hawk-memory 服务（自动重建空 DB + FTS）..."
	@systemctl --user start hawk-memory 2>/dev/null || (cd ~/repos/hawk-memory && go run ./cmd/hawk-memory/ &)
	@sleep 5
	@echo ""
	@echo "【验证】FTS 索引就绪检查..."
	@FTS_OK=$$(curl -s --max-time 5 http://127.0.0.1:18368/v1/index_status 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('ready' if d.get('has_fts', False) else 'not_ready')" 2>/dev/null || echo "unknown"); \
	echo "  FTS 状态: $$FTS_OK"
	@ROWS=$$(curl -s --max-time 5 http://127.0.0.1:18368/v1/stats 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('memory_count','?'))" 2>/dev/null || echo "?"); \
	echo "  DB 行数: $$ROWS (期望 0)"
	@echo ""
	@echo "═══════════════════════════════════════"
	@echo "  开始跑 benchmark..."
	@echo "═══════════════════════════════════════"
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_hawk \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--output reports/hawk_recall_$(VERSION).json

benchmark-hawk:
	@echo "[benchmark] hawk-memory Go recall benchmark..."
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
	@echo "[benchmark] hawk-memory Go recall benchmark (quick 20条)..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_hawk \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--limit 20 \
		--output reports/hawk_recall_quick.json

# 清理 eval agent_id 的记忆（不清空整个 DB，不重建 FTS）
# 仅当不想清空全量数据但想跑干净 eval 时使用
benchmark-hawk-clean:
	@echo "[clean] 清理 eval 记忆 + 跑 benchmark..."
	@for id in $$(curl -s http://127.0.0.1:18368/v1/memories/recent?agent_id=eval\&limit=100 2>/dev/null | python3 -c "import sys,json; print(' '.join([m['id'] for m in json.load(sys.stdin).get('memories',[])]))" 2>/dev/null); do \
		curl -s -X DELETE "http://127.0.0.1:18368/v1/memory/$$id" 2>/dev/null; \
	done
	@echo "已清理 eval 记忆（注意：DB 总量不变，旧数据可能影响 recall rank）"
	@echo "推荐使用: make benchmark-hawk-init（清空整个 DB，保证绝对干净）"
	@echo ""
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_hawk \
		--dataset datasets/hawk_memory/conversational_qa.jsonl \
		--output reports/hawk_recall_$(VERSION).json

benchmark-hawk-proc:
	@echo "[benchmark] hawk-memory Go procedural benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.runner \
		-d datasets/hawk_memory/procedural.jsonl \
		-a hawk_memory_api \
		-t procedural \
		-o reports/hawk_procedural_$(VERSION).json \
		--verbose

benchmark-locomo:
	@echo "[benchmark] LoCoMo-10 recall benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_locomo \
		--output reports/locomo_recall.json

benchmark-evolving:
	@echo "[benchmark] Evolving Events recall benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_evolving_events \
		--output reports/evolving_recall.json

benchmark-m-flow:
	@echo "[benchmark] m_flow procedural benchmark..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.runner \
		-d datasets/m_flow_procedural/ \
		-a m_flow \
		-t procedural \
		-o reports/m_flow_procedural.json \
		--verbose

benchmark-all: benchmark-hawk benchmark-m-flow
	@echo "[benchmark] 对比报告..."
	@PYTHONPATH=src $(PYTHON) -c "
import json
try:
    hawk = json.load(open('reports/hawk_recall_$(VERSION).json'))
    mflow = json.load(open('reports/m_flow_procedural.json'))
    print('=== MRR@5 对比 ===')
    print(f'  hawk-memory: {hawk.get(\"mrr@5\", \"N/A\"):.3f}')
    print(f'  m_flow:     {mflow.get(\"mrr@5\", \"N/A\"):.3f}')
except Exception as e:
    print(f'报告生成失败: {e}')
"

compare:
	@echo "[compare] 生成竞品对比报告..."
	@PYTHONPATH=src $(PYTHON) scripts/compare.py 2>&1 || echo "compare.py not found"

benchmark-multi-agent:
	@echo "[benchmark] 多 agent 评测..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) -m src.benchmark_multi_agent \
		--output reports/multi_agent.json \
		--verbose

benchmark-trigger:
	@echo "[benchmark] hawk trigger rules evaluation..."
	@mkdir -p reports
	@PYTHONPATH=src $(PYTHON) src/benchmark_trigger.py \
		--output reports/trigger_eval.json \
		--verbose
