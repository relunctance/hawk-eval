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
│   │   ├── conversational_qa.jsonl    # 对话式记忆召回（对标 LoCoMo）
│   │   └── procedural.jsonl           # 程序性记忆（对标 m_flow）
│   ├── m_flow_procedural/    # m_flow procedural_eval_v1.jsonl
│   └── locomo/               # LoCoMo 英文原版（Mem0 论文）
├── src/
│   ├── runner.py             # 评测引擎
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
# 1. 跑 hawk-memory-api 的 recall benchmark
make benchmark-hawk

# 2. 和 m_flow 对比
make benchmark-m-flow-compare

# 3. 跑完整竞品对比
make benchmark-all

# 4. 本地服务必须先启动
# hawk-memory-api: cd ~/repos/hawk-memory-api && ./run.sh
```

## 指标体系

| 维度 | 指标 | 目标竞品 |
|------|------|---------|
| 召回质量 | MRR@1/5/10 | Mem0（LoCoMo 91.6） |
| 程序性记忆 | Recall@K / Trigger Accuracy / FP Rate | m_flow |
| 答案质量 | BLEU-1 / F1 / LLM-Judge | Mem0 |
| 性能 | Latency P50/P99 | Mem0（0.88s） |

## 数据集

| 数据集 | 规模 | 来源 |
|--------|------|------|
| conversational_qa | 50条（目标200） | 自研中文 + 翻译 LoCoMo |
| procedural | 30条（目标100） | 自研中文 + m_flow |
| m_flow_procedural | 1条（目标用完原版） | m_flow 仓库 |
| locomo | 使用原版 | Mem0 论文 |

## CI Gate

每次 PR 必须：
1. `make test` 通过
2. `make benchmark-hawk` 分数不下降 > 5%
3. 新功能必须有对应测试 case

## 许可

Apache-2.0
