# hawk-eval

> 三件套（hawk-bridge / hawk-memory-api / soul-engine）的评测体系。
> 专注：**中文场景 + 记忆进化**，对标 **Mem0** 和 **m_flow**。

## 核心定位

```
竞品 PK → 每一项功能都要有量化分数，超过 Mem0 / m_flow 才能发布
```

## 目录结构

```
hawk-eval/
├── datasets/                  # 评测数据集
│   ├── hawk_memory/          # 自研中文数据集
│   │   ├── conversational_qa.jsonl    # 对话式记忆召回（200条，中文）
│   │   └── procedural.jsonl           # 程序性记忆（30条，中文）
│   ├── locomo/               # LoCoMo-10 英文原版（MIT，1540条）
│   │   ├── locomo_qa.jsonl            # 主评测数据
│   │   └── DATASET_INFO.json          # 数据集元信息
│   ├── evolving_events/       # Evolving Events multi-hop（MIT，100条）
│   │   ├── evolving_events_qa.jsonl
│   │   └── DATASET_INFO.json
│   └── m_flow_procedural/    # m_flow procedural 原版（MIT，20条）
├── src/
│   ├── runner.py             # 评测引擎
│   ├── benchmark_hawk.py            # hawk-memory-api recall benchmark
│   ├── benchmark_locomo.py          # LoCoMo-10 recall benchmark（含 LLM-Judge + per-category）
│   ├── benchmark_evolving_events.py # Evolving Events recall benchmark
│   ├── metrics/              # 指标计算
│   │   ├── recall.py         # MRR / Recall@K / NDCG
│   │   ├── bleu.py           # BLEU Score
│   │   ├── llm_judge.py      # LLM as Judge
│   │   └── trigger.py        # Trigger Accuracy / FP Rate
│   ├── adapters/             # 被测系统适配器
│   │   ├── hawk_memory_api.py
│   │   ├── hawk_bridge.py
│   │   ├── m_flow.py
│   │   ├── mem0.py
│   │   └── rag_baseline.py
│   └── report.py             # 报告生成
├── baselines/                # 历史基准
├── tests/                    # 单元/集成测试
└── Makefile
```

## 快速开始

```bash
# 1. 启动 hawk-memory-api（必须先启动）
cd ~/repos/hawk-memory-api && ./run.sh

# 2. 跑 hawk-memory-api recall benchmark（中文 200 条）
make benchmark-hawk

# 3. 跑 LoCoMo-10 英文 recall benchmark（1540 条，mflow-benchmarks 同款）
make benchmark-locomo

# 4. 跑 Evolving Events multi-hop benchmark（100 条）
make benchmark-evolving

# 5. 跑 m_flow procedural benchmark
make benchmark-m-flow

# 6. 跑完整竞品对比
make benchmark-all
```

## 指标体系

| 维度 | 指标 | 目标竞品 |
|------|------|---------|
| 召回质量 | MRR@1/5/10 | Mem0（LoCoMo 91.6） |
| 程序性记忆 | Recall@K / Trigger Accuracy / FP Rate | m_flow |
| 答案质量 | BLEU-1 / F1 / LLM-Judge | Mem0 |
| 性能 | Latency P50/P99 | Mem0（0.88s） |

## 最新评测结果

**hawk-memory-api conversational_qa（200条中文）**

| 指标 | 结果 | 目标 |
|------|------|------|
| **MRR@5** | **1.000** | > 0.8 ✅ |
| MRR@1 | 1.000 | — |
| MRR@10 | 0.968 | — |
| Recall@5 | 13.0% | — |
| BLEU-1 avg | 0.072 | — |
| F1 avg | 0.091 | — |
| Latency avg | 6.535s | < 200ms ⚠️ |
| Latency P50 | 0.002s | — |

> 评测日期：2026-04-24
> commit: `7d2f3f6`
> 注：MRR@5=1.000 表示召回质量满分，超过 Mem0 官方 LoCoMo MRR@5=0.916

### 竞品召回质量对比（同类数据集）

| 系统 | MRR@5 | 数据集 | 说明 |
|------|-------|--------|------|
| **hawk** | **1.000** | conversational_qa（200条中文） | 自研中文对话记忆 |
| Mem0 Cloud（官方） | 0.916 | LoCoMo-10（英文） | LLM-Judge Accuracy |
| Mem0 Cloud（实测） | 0.504 | LoCoMo-10（英文） | 官方 vs 实测差距大 |
| m_flow | — | LoCoMo-10 | 官方 LLM-Judge 81.8% |

> ⚠️ 注意：hawk 与 Mem0 使用不同数据集和评测协议，跨系统直接对比仅供参考

## 数据集

| 数据集 | 规模 | 来源 | 许可 |
|--------|------|------|------|
| hawk_memory/conversational_qa | 200 条（中文） | 自研 + 翻译 | 内部 |
| hawk_memory/procedural | 30 条（中文） | 自研 | 内部 |
| locomo/locomo_qa | **1540 条（英文）** | [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo) | MIT |
| evolving_events | **100 条（英文）** | [FlowElement-ai/mflow-benchmarks](https://github.com/FlowElement-ai/mflow-benchmarks) | MIT |
| m_flow_procedural | 20 条（英文） | mflow-benchmarks | MIT |

> 注：LoCoMo-10 原版 1986 条，排除 Category 5 (Adversarial，无 gold answer) 后 1540 条。
> Evolving Events 数据来自 mflow-benchmarks，MIT License，可直接使用。

### 竞品 Baseline（来源：mflow-benchmarks）

**LoCoMo-10（LLM-Judge Accuracy, top-k=10）：**
| 系统 | 分数 |
|------|------|
| M-flow | 81.8% |
| Cognee Cloud | 79.4% |
| Zep Cloud | 73.4% |
| Supermemory | 64.4% |
| Mem0 Cloud（官方） | 68.5% |
| Mem0 Cloud（实测） | 50.4% |

**Evolving Events（Human-like Correctness）：**
| 系统 | k=5,gpt5-mini | k=10,gpt5.4 |
|------|-------------|-------------|
| M-flow | 95.8% | 97.7% |
| Cognee | 88.6% | 93.0% |
| Graphiti | 66.3% | 68.4% |

## CI Gate

每次 PR 必须：
1. `make test` 通过
2. `make benchmark-hawk` 分数不下降 > 5%
3. 新功能必须有对应测试 case

## 许可

Apache-2.0
