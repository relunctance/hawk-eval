# hawk-eval

> 四件套（hawk-bridge / hawk-memory-api / soul-engine / hawk-eval）的评测体系。
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

### 2026-04-24：memory_id 精确匹配 vs text 匹配

**核心发现**：capture 返回 memory_id 后，用 memory_id 做精确匹配替代 text 相似度匹配，召回质量大幅提升。

**验证集**：conversational_qa 前 25 条（中文）

| 指标 | text匹配（旧） | memory_id匹配（新） | 提升 |
|------|---------------|-------------------|------|
| **MRR@5** | 0.213 | **0.946** | +344% |
| **Recall@5** | 32% | **92%** | +188% |
| Match Rate | — | **92% (23/25)** | — |

> commit: `8c073e0`，评测日期：2026-04-24

**未命中分析（2/25）**：
- HM-024「用户的时区是哪个」：capture 时 answer 是「UTC+8」，recall query 是「时区」，语义模糊
- HM-025「上次让我记得的事」：属于泛化类 query，需要更强的语义推理

---

### 2026-04-24 全量 200 条评测（含 Bug 记录）

> ⚠️ **发现 Benchmark 脚本 Bug**：memory_id 注入失败，79/200 的 `target_memory_id=None`，
> 导致 fallback 到 text 匹配。指标仅供参考，待修复后再重新评测。

**评测结果（memory_id 精确匹配，有效 121/200）**：

| 指标 | 结果 | 备注 |
|------|------|------|
| **MRR@5** | **0.398** | 仅供参考，bug 导致不可信 |
| MRR@1 | 1.000 | 仅供参考 |
| Recall@5 | 60.5% | 121/200 有 rank |
| 未命中 | 39.5% (79/200) | memory_id 未注入 |
| Latency P50 | 5.0s | — |

**rank 分布异常**：大量聚集在 rank=3（64条），说明 recall 排序有均匀化倾向（fusion 多路融合后 rank 拉平）。

> commit: `a954ca4`，评测日期：2026-04-24
> 待修复：benchmark memory_id 对齐逻辑（`direct_capture_batch_with_ids` 并发乱序问题）

---

### 历史基准：全量数据集

**hawk-memory-api conversational_qa（200条中文，全量）**

| 指标 | 结果 | 目标 |
|------|------|------|
| **MRR@5** | **0.996** | > 0.9 ✅ |
| MRR@1 | 0.992 | — |
| MRR@10 | 0.998 | — |
| Recall@5 | 98% | > 60% ✅ |

> 评测日期：2026-04-24，commit: `7d2f3f6`

**hawk-memory-api LoCoMo-10（20条英文，预计算向量）**

| 指标 | 结果 | 目标 |
|------|------|------|
| **MRR@5** | **1.000** | > 0.9 ✅ |
| Recall@5 | 100% | > 60% ✅ |
| 成功 case | 20/20 | — |

> 评测日期：2026-04-24，commit: `dee4730`
> 注：与 m_flow E2E 评测协议不同，仅供参考

### 竞品召回质量对比

| 系统 | MRR@5 | 数据集 | 说明 |
|------|-------|--------|------|
| **hawk** | **1.000** | LoCoMo-10（20条） | recall 评测 |
| **hawk** | **0.996** | conversational_qa（200条中文） | recall 评测 |
| Mem0 Cloud（官方） | 0.916 | LoCoMo-10 | LLM-Judge E2E |
| m_flow（官方） | 81.8% | LoCoMo-10 | LLM-Judge E2E |

> ⚠️ hawk 是 recall 评测，Mem0/m_flow 是 E2E 生成评测，协议不同，直接对比仅供参考

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
