#!/usr/bin/env python3
"""
检查 benchmark 报告是否达到最低 MRR 阈值。
KR3.3 CI Gate 核心脚本。

用法:
    python scripts/check_mrr_threshold.py --min 0.5
    python scripts/check_mrr_threshold.py --min 0.5 --report reports/hawk_recall.json
    python scripts/check_mrr_threshold.py --min 0.5 --baseline reports/baseline.json  # 与 baseline 比较，下降 > 5% 则 fail
"""

import argparse
import json
import sys
from pathlib import Path


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_mrr(report: dict) -> float:
    metrics = report.get("metrics", {})
    return metrics.get("mrr@5", 0.0)


def get_mrr_recall(report: dict) -> float:
    metrics = report.get("metrics", {})
    return metrics.get("recall@5", 0.0)


def main():
    parser = argparse.ArgumentParser(description="检查 MRR 是否达到阈值")
    parser.add_argument("--min", type=float, default=0.5,
                        help="最低 MRR@5 阈值（默认 0.5）")
    parser.add_argument("--report", default="reports/hawk_recall.json",
                        help="评测报告路径")
    parser.add_argument("--baseline", default=None,
                        help="基线报告路径（与基线比较，下降 >5%% 则 fail）")
    parser.add_argument("--recall-min", type=float, default=0.0,
                        help="最低 Recall@5 阈值（默认 0）")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ 报告文件不存在: {report_path}")
        sys.exit(1)

    report = load_report(report_path)
    mrr = get_mrr(report)
    recall = get_mrr_recall(report)

    print(f"  MRR@5:   {mrr:.3f} (阈值: ≥{args.min:.3f})")
    print(f"  Recall@5: {recall:.1%} (阈值: ≥{args.recall_min:.1%})")

    # 与基线比较
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = load_report(baseline_path)
            baseline_mrr = get_mrr(baseline)
            drop = baseline_mrr - mrr
            drop_pct = drop / baseline_mrr * 100 if baseline_mrr else 0
            print(f"  Baseline MRR@5: {baseline_mrr:.3f}")
            print(f"  下降: {drop:.3f} ({drop_pct:.1f}%)")
            if drop_pct > 5.0:
                print(f"❌ MRR 下降 {drop_pct:.1f}% > 5%，CI Gate 失败")
                sys.exit(1)
            else:
                print(f"✅ MRR 下降 {drop_pct:.1f}% ≤ 5%，通过")

    # 阈值检查
    if mrr < args.min:
        print(f"❌ MRR@5 {mrr:.3f} < {args.min:.3f}，CI Gate 失败")
        sys.exit(1)

    if recall < args.recall_min:
        print(f"❌ Recall@5 {recall:.1%} < {args.recall_min:.1%}，CI Gate 失败")
        sys.exit(1)

    print(f"✅ MRR@5 和 Recall@5 均达标")
    sys.exit(0)


if __name__ == "__main__":
    main()
