#!/usr/bin/env python3
"""
评测报告生成 — 对比多个系统的分数，输出 Markdown + JSON 报告
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_metrics(metrics: dict) -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k}: **{v:.4f}**")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def compare_reports(reports: list[dict]) -> str:
    """生成对比表格。"""
    # 收集所有指标键
    all_keys = set()
    for r in reports:
        all_keys.update(r.get("metrics", {}).keys())

    lines = [
        "| System | " + " | ".join(all_keys) + " |",
        "|--------|" + "|".join(["--------:" for _ in all_keys]) + "|",
    ]

    for r in reports:
        name = r.get("adapter", r.get("system", "unknown"))
        vals = []
        for k in all_keys:
            v = r.get("metrics", {}).get(k)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif v is None:
                vals.append("—")
            else:
                vals.append(str(v))
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def generate_markdown(reports: list[dict], output_path: str) -> str:
    """生成 Markdown 报告。"""
    timestamp = datetime.now().isoformat()
    md = f"""# hawk-eval 评测报告

生成时间: {timestamp}

## 系统对比

{compare_reports(reports)}

## 详细指标

"""

    for r in reports:
        name = r.get("adapter", r.get("system", "unknown"))
        md += f"\n### {name}\n\n"
        md += format_metrics(r.get("metrics", {})) + "\n"

    Path(output_path).write_text(md, encoding="utf-8")
    return md


def main():
    parser = argparse.ArgumentParser(description="hawk-eval 报告生成")
    parser.add_argument("--reports", "-r", nargs="+", required=True,
                        help="评测报告 JSON 文件路径")
    parser.add_argument("--output", "-o", default="eval_report.md",
                        help="输出 Markdown 路径")
    args = parser.parse_args()

    reports = [load_report(p) for p in args.reports]
    md = generate_markdown(reports, args.output)
    print(md)
    print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()
