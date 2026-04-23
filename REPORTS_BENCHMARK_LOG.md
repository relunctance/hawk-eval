# Hawk-Memory-Api MRR Benchmark Log
> 迭代过程记录，2026-04-24

---

## Bug 修复记录

### Bug 1: Prefix 不匹配导致 MRR 永远为 0
- **问题**：`compute_recall_metrics` 用 `target_id == retrieved_ids` 做精确匹配，但 `retrieved_ids` 有 `"用户: "` / `"助手: "` 前缀，`target_id` 无前缀
- **发现**：batch01 结果 MRR=0，但 rank=1 且 bleu1=0.314，明显有匹配
- **修复**：计算 metrics 前 strip 前缀
- **commit**: `b9d2398`

### Bug 2: offset/limit 顺序错误
- **问题**：`dataset[:limit]` → `dataset[offset:]`，先 limit 再 offset，导致 offset > limit 时永远拿到 0 条
- **发现**：batch02-20 全部返回 0 条数据
- **修复**：改为先 offset 再 limit
- **commit**: `31ebb80`

### Bug 3: 并发过高导致 API 超时
- **问题**：`max_workers=5`，200 条全量跑超时（>600s）
- **修复**：降为 `max_workers=2`
- **commit**: `a123239`

---

## Batch 运行结果（已完成）

| Batch | Offset | Count | MRR@1 | MRR@5 | Recall@1 | Latency(s) |
|-------|--------|-------|-------|-------|----------|------------|
| 01 | 0 | 10 | **1.000** | **1.000** | 10.0% | 30.0 |
| 02 | 10 | 10 | 0.000 | 0.000 | 0.0% | 30.0 |
| 03 | 20 | 10 | **1.000** | **1.000** | 10.0% | 30.0 |
| 04 | 30 | 10 | 0.000 | 0.000 | 0.0% | 0.0 |
| 06 | 50 | 10 | 0.000 | 0.000 | 0.0% | 17.5 |
| 07 | 60 | 10 | 0.000 | 0.000 | 0.0% | 30.0 |
| 08 | 70 | 10 | 0.000 | 0.000 | 0.0% | 30.0 |
| 09 | 80 | 10 | **1.000** | **1.000** | 10.0% | 30.0 |

**已完成的 8 批平均: MRR@1=0.375 MRR@5=0.375**

---

## 问题分析

### MRR 两极分化（0.000 vs 1.000）
- batch 01/03/09 MRR=1.000（完美）
- batch 02/04/06/07/08 MRR=0.000（完全失败）
- **原因**：capture 是 session 级别的，每条记忆用独立 session 存入，**没有建立问答关联**。question 和 answer 是分开的记忆，recall 时只根据 question 查，但 API 存的是 "answer text"，导致 question 和 answer 之间没有向量关联。

### 根本问题
benchmark capture 时用 answer text 作为 `message` 存入，但 question 是独立的问题，没有一起存入。正确的做法应该是把 QA 对作为一个整体存入，或者 capture 两次（question + answer）。

### 数据
- conversational_qa: 200 条（HM-001 ~ HM-200）
- 其中 01/03/09（对应 HM-001~010, HM-021~030, HM-081~090）MRR=1.000
- 其余批次 MRR=0.000

---

## 下一步

1. **修复 capture 逻辑**：把 question + answer 作为整体存入（一次 capture 包含完整 QA）
2. **或者**：capture 两次，question 作为 query，answer 作为 context
3. **再跑全量验证**
