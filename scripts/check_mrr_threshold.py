#!/usr/bin/env python3
"""
Check if MRR@5 meets minimum threshold.
Fails with exit code 1 if below threshold.

Usage:
    python scripts/check_mrr_threshold.py --min 0.5
    python scripts/check_mrr_threshold.py --min 0.5 --report reports/hawk_recall_0.1.0.json
"""
import argparse
import json
import sys
import glob


def find_latest_report():
    """Find most recent benchmark report."""
    reports = sorted(glob.glob("reports/hawk_recall*.json"), reverse=True)
    if not reports:
        return None
    return reports[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=float, required=True, help="Minimum MRR@5 threshold")
    parser.add_argument("--report", help="Specific report path (default: latest)")
    args = parser.parse_args()

    # Find report
    if args.report:
        report_path = args.report
    else:
        report_path = find_latest_report()

    if not report_path or not glob.glob(report_path):
        print(f"ERROR: No benchmark report found")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    mrr5 = report.get("mrr_at_k", {}).get("5", 0)
    recall5 = report.get("recall_at_k", {}).get("5", 0)
    cases = report.get("total_cases", 0)

    print(f"Report: {report_path}")
    print(f"Cases: {cases}")
    print(f"MRR@5:  {mrr5:.4f} (threshold: {args.min:.4f})")
    print(f"Recall@5: {recall5:.1%}")

    if mrr5 < args.min:
        print(f"\n❌ FAILED: MRR@5 {mrr5:.4f} < {args.min:.4f}")
        sys.exit(1)
    else:
        print(f"\n✅ PASSED: MRR@5 {mrr5:.4f} >= {args.min:.4f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
