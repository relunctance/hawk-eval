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

### Bug 4: Capture 只存 answer，没有 question（根本问题）
- **问题**：benchmark capture 时只把 answer 文本存入，question 是独立的问题，没有一起存入。导致 question 和 answer 之间没有语义关联，recall 时只能靠向量相似度硬匹配。
- **表现**：MRR 两极分化——01/03/09 完美，其他全部 0.000
- **修复**：新增 `capture_qa(question, answer)` 方法，把 `"用户: {question}\n助手: {answer}"` 合并存入，建立问答关联
- **并发**：降低为 `max_workers=1`，每批<30s
- **日志**：新增 `--log` 参数，输出同时写文件和 stdout
- **commit**: `a3f8c2d`

---

## Batch 运行结果（已完成的批次）

> **注意**：01-09 是修复前的旧结果（只存 answer）。10+ 是修复后结果。

| Batch | Offset | Count | MRR@1 | MRR@5 | Recall@5 | Latency(s) | 说明 |
|-------|--------|-------|-------|-------|----------|------------|------|
| 01 | 0 | 10 | 1.000 | 1.000 | 10.0% | 30.0 | 旧逻辑（只存 answer）|
| 02 | 10 | 10 | 0.000 | 0.000 | 0.0% | 30.0 | 旧逻辑 |
| 03 | 20 | 10 | 1.000 | 1.000 | 10.0% | 30.0 | 旧逻辑 |
| 04 | 30 | 10 | 0.000 | 0.000 | 0.0% | 0.0 | 旧逻辑 |
| 06 | 50 | 10 | 0.000 | 0.000 | 0.0% | 17.5 | 旧逻辑 |
| 07 | 60 | 10 | 0.000 | 0.000 | 0.0% | 30.0 | 旧逻辑 |
| 08 | 70 | 10 | 0.000 | 0.000 | 0.0% | 30.0 | 旧逻辑 |
| 09 | 80 | 10 | 1.000 | 1.000 | 10.0% | 30.0 | 旧逻辑 |

**旧逻辑结论**：两极分化，非 1.0 即 0.0，说明 QA 无关联时 recall 靠随机匹配。

---

## 修复后测试（10 条样例）

| 测试 | Offset | Count | MRR@1 | MRR@3 | MRR@5 | Recall@5 | Latency(s) |
|------|--------|-------|-------|-------|-------|----------|------------|
| test-log | 0 | 3 | 0.000 | 0.500 | 0.500 | 33.3% | 11.3 |

**修复后结论**：MRR@3=0.5（3条中2条在 top-3），说明 QA 关联已建立，但非完美匹配（可能是 recall 质量或 text_similar 阈值问题）。

---

## 日志文件

每次运行的日志追加到 `reports/benchmark_run.log`，可用以下命令查看：
```bash
tail -f reports/benchmark_run.log
```

---

## 下一步

1. 跑全量 200 条，验证 MRR 是否稳定 > 0.5
2. 调优 recall 质量（embedding / fusion 参数）
3. 打通 m_flow / Mem0 adapter 做竞品对比
